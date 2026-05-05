"""
SAC implementation using TorchRL
- TorchRL batched gpu env
- Simulation and training loop on GPU (no data transfer overhead)
- Replay buffer with backbone feature (no obs, to save memory and speed up training)
"""


from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn
from torch.optim import Adam
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.objectives import SACLoss, ValueEstimators

from avoid_everything.pretraining import PretrainingMotionPolicyTransformer
from avoid_everything.type_defs import DatasetType
from rl.architecture import MPiFormerTorchRLArchitecture
from rl.gpu_batch_env import AvoidEverythingEnv

# Device used for storing the replay buffer (can be set to torch.device("cpu") to save GPU memory)
REPLAY_BUFFER_DEVICE = torch.device("cpu")


@dataclass
class TrainMetrics:
    actor_loss: float = 0.0
    qvalue_loss: float = 0.0
    alpha_loss: float = 0.0
    alpha: float = 0.0
    replay_size: int = 0


class CriticQ1Only(nn.Module):
    def __init__(self, critic_head: nn.Module):
        super().__init__()
        self.critic_head = critic_head

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        hidden = features.transpose(0, 1).contiguous()
        q1, _ = self.critic_head(hidden, action)
        return q1


class CriticQ2Only(nn.Module):
    def __init__(self, critic_head: nn.Module):
        super().__init__()
        self.critic_head = critic_head

    def forward(self, features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        hidden = features.transpose(0, 1).contiguous()
        _, q2 = self.critic_head(hidden, action)
        return q2


class BackboneToAbstract(nn.Module):
    """Convert backbone output from [S, B, D] to replay-friendly [B, S, D]."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        point_cloud_labels: torch.Tensor,
        point_cloud: torch.Tensor,
        configuration: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.backbone(point_cloud_labels, point_cloud, configuration)
        return hidden.transpose(0, 1).contiguous()


class ActorFromAbstract(nn.Module):
    """Consume abstract feature [B, S, D] and return Gaussian params."""

    def __init__(self, actor_head: nn.Module):
        super().__init__()
        self.actor_head = actor_head

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = features.transpose(0, 1).contiguous()
        return self.actor_head(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal TorchRL SAC with feature replay on GPU")
    parser.add_argument("cfg_path", type=Path, help="Path to RL YAML config")
    parser.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=None, help="Override cfg total_timesteps")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--overfit", type=int, default=None)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cfg(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_start_lr(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, dict):
        return float(value.get("start_lr", default))
    return float(value)


def get_next_td(step_td: TensorDictBase) -> TensorDictBase:
    return step_td.get("next") if "next" in step_td.keys() else step_td


def make_env(cfg: dict[str, Any], device: torch.device, overfit: int | None) -> AvoidEverythingEnv:
    env_cfg = cfg.get("env", {})
    n_envs = int(env_cfg.get("train_n_envs", env_cfg.get("n_envs", 1)))
    max_delta_q = float(env_cfg.get("max_delta_q", 1.0))

    env = AvoidEverythingEnv(
        urdf_path=cfg["urdf_path"],
        num_robot_points=cfg["num_robot_points"],
        num_obstacle_points=cfg["num_obstacle_points"],
        num_target_points=cfg["num_target_points"],
        collision_mode=env_cfg.get("collision_mode", "torch_franka"),
        scene_buffer=float(env_cfg.get("scene_buffer", 0.0)),
        position_threshold=float(env_cfg.get("position_threshold", 0.01)),
        orientation_threshold=float(env_cfg.get("orientation_threshold", 15.0)),
        max_episode_steps=int(env_cfg.get("max_episode_steps", 100)),
        terminate_ep_on_collision=bool(env_cfg.get("terminate_ep_on_collision", True)),
        action_delta_clip=env_cfg.get("action_delta_clip", max_delta_q),
        max_delta_q=max_delta_q,
        batch_size=n_envs,
        device=device,
    )

    env.set_scene_generation_from_dataset(
        data_dir=cfg["data_dir"],
        trajectory_key=cfg["train_trajectory_key"],
        dataset_type=DatasetType.TRAIN,
        num_workers=int(env_cfg.get("num_workers", 0)),
        random_scale=0.0,
        overfit_idx=overfit,
        n_eval_episodes=None,
        shuffle=True,
    )
    return env


def load_bc_model(cfg: dict[str, Any], device: torch.device) -> PretrainingMotionPolicyTransformer:
    model = PretrainingMotionPolicyTransformer.load_from_checkpoint(
        cfg["bc_checkpoint_path"],
        urdf_path=cfg["urdf_path"],
        num_robot_points=cfg["num_robot_points"],
        action_chunk_length=cfg["action_chunk_length"],
        **cfg["bc_checkpoint_parameters"],
    ).to(device)
    model.eval()
    return model


def build_sac_loss(
    actor: ProbabilisticActor,
    critic_head: nn.Module,
    target_entropy: float,
    fixed_alpha: bool,
    alpha_init: float,
    gamma: float,
) -> SACLoss:
    q1_module = TensorDictModule(
        CriticQ1Only(critic_head),
        in_keys=["features", "action"],
        out_keys=["state_action_value"],
    )
    q2_module = TensorDictModule(
        CriticQ2Only(critic_head),
        in_keys=["features", "action"],
        out_keys=["state_action_value"],
    )

    loss_module = SACLoss(
        actor_network=actor,
        qvalue_network=[q1_module, q2_module],
        target_entropy=target_entropy,
        fixed_alpha=fixed_alpha,
        alpha_init=alpha_init,
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)
    return loss_module


def soft_update_qvalue_targets(loss_module: SACLoss, tau: float) -> None:
    with torch.no_grad():
        target_params = list(loss_module.target_qvalue_network_params.values(True, True))
        source_params = list(loss_module.qvalue_network_params.values(True, True))
        for target, source in zip(target_params, source_params):
            target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)


def train_step(
    loss_module: SACLoss,
    batch: TensorDictBase,
    actor_optim: Adam,
    qvalue_optim: Adam,
    alpha_optim: Adam | None,
    tau: float,
) -> TrainMetrics:
    # critic loss and update
    q_loss_td = loss_module(batch)
    qvalue_loss = q_loss_td["loss_qvalue"]
    qvalue_optim.zero_grad(set_to_none=True)
    qvalue_loss.backward()
    qvalue_optim.step()

    # actor loss and update
    actor_loss_td = loss_module(batch)
    actor_loss = actor_loss_td["loss_actor"]
    actor_optim.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_optim.step()

    # alpha loss and update (if learnable)
    alpha_loss_value = 0.0
    if alpha_optim is not None:
        alpha_loss_td = loss_module(batch)
        alpha_loss = alpha_loss_td["loss_alpha"]
        alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_optim.step()
        alpha_loss_value = float(alpha_loss.item())

    # update q target networks
    soft_update_qvalue_targets(loss_module, tau=tau)

    # return step metrics for logging
    return TrainMetrics(
        actor_loss=float(actor_loss.item()),
        qvalue_loss=float(qvalue_loss.item()),
        alpha_loss=alpha_loss_value,
        alpha=float(actor_loss_td["alpha"].item()),
    )


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.cfg_path)

    device = resolve_device(args.device)
    set_seed(args.seed)
    print(f"Using device: {device}")

    env = make_env(cfg, device=device, overfit=args.overfit)
    bc_model = load_bc_model(cfg, device=device)

    split_layer = int(cfg.get("freeze_transformer_layers", 7))
    architecture = MPiFormerTorchRLArchitecture(
        bc_model=bc_model,
        pc_bounds=cfg["bc_checkpoint_parameters"]["pc_bounds"],
        split_layer=split_layer,
        deep_copy=True,
        log_std_init=float(cfg.get("log_std_init", -10.0)),
    )

    backbone_module = TensorDictModule(
        BackboneToAbstract(architecture.backbone),
        in_keys=["point_cloud_labels", "point_cloud", "configuration"],
        out_keys=["features"],
    )
    actor_head_module = TensorDictModule(
        ActorFromAbstract(architecture.actor_head),
        in_keys=["features"],
        out_keys=["loc", "scale"],
    )

    action_low = -float(cfg.get("env", {}).get("max_delta_q", 1.0))
    action_high = float(cfg.get("env", {}).get("max_delta_q", 1.0))
    actor = ProbabilisticActor(
        module=actor_head_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": action_low, "high": action_high},
        return_log_prob=True,
    )

    gamma = float(cfg.get("gamma", 0.99))
    tau = float(cfg.get("tau", 0.005))
    entropy_cfg = cfg.get("entropy_coef", "auto")
    if isinstance(entropy_cfg, str) and entropy_cfg.startswith("auto"):
        fixed_alpha = False
        alpha_init = 1.0
        if "_" in entropy_cfg:
            alpha_init = float(entropy_cfg.split("_", 1)[1])
    else:
        fixed_alpha = True
        alpha_init = float(entropy_cfg)

    target_entropy = float(cfg.get("target_entropy", -float(env.robot.MAIN_DOF)))
    loss_module = build_sac_loss(
        actor=actor,
        critic_head=architecture.critic_head,
        target_entropy=target_entropy,
        fixed_alpha=fixed_alpha,
        alpha_init=alpha_init,
        gamma=gamma,
    )

    actor_lr = extract_start_lr(cfg.get("finetuning_actor_lr"), 3e-4)
    critic_lr = extract_start_lr(cfg.get("finetuning_critic_lr"), 3e-4)
    actor_optim = Adam(architecture.actor_head.parameters(), lr=actor_lr)
    qvalue_optim = Adam(architecture.critic_head.parameters(), lr=critic_lr)
    alpha_optim = None if fixed_alpha else Adam([loss_module.log_alpha], lr=actor_lr)

    batch_size = int(cfg["batch_size"])
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(int(cfg["buffer_size"]), device=REPLAY_BUFFER_DEVICE),
        batch_size=batch_size,
    )

    total_steps = int(args.total_steps if args.total_steps is not None else cfg["total_timesteps"])
    train_freq = cfg.get("train_freq", [1, "step"])
    update_every = int(train_freq[0] if isinstance(train_freq, (list, tuple)) else train_freq)
    gradient_steps = int(cfg.get("gradient_steps", 1))
    learning_starts = int(cfg.get("transitions_before_learn", batch_size))

    obs_td = env.reset()
    metrics = TrainMetrics()

    print(
        f"Starting training: steps={total_steps}, n_envs={env.num_envs}, "
        f"learning_starts={learning_starts}, update_every={update_every}, gradient_steps={gradient_steps}"
    )

    for step in range(total_steps):
        with torch.no_grad():
            backbone_module(obs_td)
            actor(obs_td)
            action = obs_td["action"]

        step_input = TensorDict({"action": action}, batch_size=env.batch_size, device=device)
        step_td = env.step(step_input)
        next_td = get_next_td(step_td)

        with torch.no_grad():
            backbone_module(next_td)

        transition = TensorDict(
            {
                "features": obs_td["features"].detach(),
                "action": action.detach(),
                "next": TensorDict(
                    {
                        "features": next_td["features"].detach(),
                        "reward": next_td["reward"].detach(),
                        "done": next_td["done"].detach(),
                        "terminated": next_td["terminated"].detach(),
                    },
                    batch_size=env.batch_size,
                    device=REPLAY_BUFFER_DEVICE,
                ),
            },
            batch_size=env.batch_size,
            device=REPLAY_BUFFER_DEVICE,
        )
        replay_buffer.extend(transition)

        # if step < learning_starts, skip learning and just populate replay buffer
        if step >= learning_starts and len(replay_buffer) >= batch_size and (step + 1) % update_every == 0:
            for _ in range(gradient_steps):
                sampled = replay_buffer.sample().to(device)
                metrics = train_step(
                    loss_module=loss_module,
                    batch=sampled,
                    actor_optim=actor_optim,
                    qvalue_optim=qvalue_optim,
                    alpha_optim=alpha_optim,
                    tau=tau,
                )

        # Partial reset: eset only the environments that are done, keep the rest as is, using a torch mask to select which ones to reset
        # Copy the "done" state to the "_reset" mask, then pass to env.reset()
        next_td.set("_reset", next_td.get("done"))
        obs_td = env.reset(next_td)

        if (step + 1) % args.log_every == 0 or step == total_steps - 1:
            metrics.replay_size = len(replay_buffer)
            print(
                f"step={step + 1} replay={metrics.replay_size} "
                f"actor_loss={metrics.actor_loss:.4f} "
                f"qvalue_loss={metrics.qvalue_loss:.4f} "
                f"alpha={metrics.alpha:.4f}"
            )

    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
