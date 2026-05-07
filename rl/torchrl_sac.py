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
from tqdm import tqdm

import torch
import yaml
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torch.optim import Adam
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.objectives import SACLoss, ValueEstimators
from torchrl.envs.utils import ExplorationType, set_exploration_type

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


# Adapters removed: use MPiFormerTorchRLArchitecture.build_backbone_module(..., batch_first=True)
# and build_head_module(..., batch_first=True) instead of local transposing adapters.


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


def make_env(cfg: dict[str, Any], device: torch.device, overfit: int | None = None) -> AvoidEverythingEnv:
    env_cfg = cfg.get("env", {})
    batch_size = int(env_cfg.get("batch_size",None))
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
        batch_size=batch_size,
        device=device,
    )

    env.set_scene_generation_from_dataset(
        data_dir=cfg["data_dir"],
        trajectory_key=cfg["train_trajectory_key"],
        dataset_type=DatasetType.TRAIN,
        num_workers=int(env_cfg.get("num_workers", 0)),
        random_scale=0.0,
        n_eval_episodes=int(cfg.get("n_eval_episodes", 0)),
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


def soft_update_qvalue_targets(loss_module: SACLoss, tau: float) -> None:
    with torch.no_grad():
        target_params = list(loss_module.target_qvalue_network_params.values(True, True))
        source_params = list(loss_module.qvalue_network_params.values(True, True))
        for target, source in zip(target_params, source_params):
            target.data.mul_(1.0 - tau).add_(source.data, alpha=tau)


def _cpuify(obj):
    """Recursively move all tensors in obj to CPU for smaller checkpoint files."""
    if torch.is_tensor(obj):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: _cpuify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpuify(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_cpuify(v) for v in obj)
    return obj


def save_checkpoint(
    save_dir: Path,
    architecture: "MPiFormerTorchRLArchitecture",
    actor_optim: Adam,
    qvalue_optim: Adam,
    alpha_optim: Adam | None,
    loss_module: SACLoss,
    cfg: dict[str, Any],
    step: int,
    save_replay: bool = False,
    replay_buffer: TensorDictReplayBuffer | None = None,
) -> Path:
    """Save actor/critic heads, optimizer states, alpha and some metadata.

    Returns the path to the saved checkpoint.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"sac_checkpoint_{step}.pt"

    ckpt: dict[str, Any] = {
        "actor_head": _cpuify(architecture.actor_head.state_dict()),
        "critic_head_q1": _cpuify(architecture.critic_head_q1.state_dict()),
        "critic_head_q2": _cpuify(architecture.critic_head_q2.state_dict()),
        "actor_optim": _cpuify(actor_optim.state_dict()),
        "critic_optim": _cpuify(qvalue_optim.state_dict()),
        "alpha_optim": _cpuify(alpha_optim.state_dict()) if alpha_optim is not None else None,
        "log_alpha": _cpuify(loss_module.log_alpha.detach().cpu()) if hasattr(loss_module, "log_alpha") else None,
        "cfg": cfg,
        "step": int(step),
    }

    if save_replay and replay_buffer is not None:
        try:
            ckpt["replay_len"] = len(replay_buffer)
        except Exception:
            ckpt["replay_len"] = None

    torch.save(ckpt, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")
    return ckpt_path


def load_checkpoint(
    ckpt_path: Path,
    architecture: "MPiFormerTorchRLArchitecture",
    actor_optim: Adam,
    qvalue_optim: Adam,
    alpha_optim: Adam | None,
    loss_module: SACLoss,
    device: torch.device,
) -> dict[str, Any]:
    """Load the checkpoint and restore model/optim state. Returns the checkpoint dict."""
    ckpt = torch.load(ckpt_path, map_location=device)

    # restore modules
    architecture.actor_head.load_state_dict(ckpt["actor_head"])
    architecture.critic_head_q1.load_state_dict(ckpt["critic_head_q1"])
    architecture.critic_head_q2.load_state_dict(ckpt["critic_head_q2"])

    # restore optimizers
    try:
        actor_optim.load_state_dict(ckpt["actor_optim"])
    except Exception as e:
        print(f"Warning: failed to load actor optimizer state: {e}")
    try:
        qvalue_optim.load_state_dict(ckpt["critic_optim"])
    except Exception as e:
        print(f"Warning: failed to load critic optimizer state: {e}")

    if alpha_optim is not None and ckpt.get("alpha_optim") is not None:
        try:
            alpha_optim.load_state_dict(ckpt["alpha_optim"])
        except Exception as e:
            print(f"Warning: failed to load alpha optimizer state: {e}")

    # restore log_alpha if present
    if hasattr(loss_module, "log_alpha") and ckpt.get("log_alpha") is not None:
        try:
            loss_module.log_alpha.data.copy_(ckpt["log_alpha"].to(loss_module.log_alpha.device))
        except Exception as e:
            print(f"Warning: failed to restore log_alpha: {e}")

    # ensure target q networks match loaded critic parameters
    try:
        with torch.no_grad():
            src = list(loss_module.qvalue_network_params.values(True, True))
            tgt = list(loss_module.target_qvalue_network_params.values(True, True))
            for s, t in zip(src, tgt):
                t.copy_(s)
    except Exception as e:
        print(f"Warning: failed to copy qvalue params to targets: {e}")

    return ckpt


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



@torch.no_grad()
def evaluate_policy(inference_policy: nn.Module, eval_env: Any, n_episodes: int = 4, max_steps: int | None = None) -> dict:
    """Run deterministic evaluation using the actor mean (via TorchRL ExplorationType.DETERMINISTIC).

    Returns a dict with 'avg_return' and 'episode_returns'.
    """
    # ensure policy and env are on the same device
    policy_dev = getattr(eval_env, "device", None)
    if policy_dev is None:
        try:
            policy_dev = next(inference_policy.parameters()).device
        except StopIteration:
            policy_dev = torch.device("cpu")
    inference_policy.to(policy_dev)
    inference_policy.eval()

    episode_returns: list[float] = []
    cum_returns = torch.zeros(eval_env.batch_size, device=policy_dev)
    dones = torch.zeros(eval_env.batch_size, dtype=torch.bool, device=policy_dev)

    with set_exploration_type(ExplorationType.DETERMINISTIC):
        obs = eval_env.reset()
        steps = 0

        with tqdm(total=n_episodes, desc="Evaluating", leave=False) as pbar:
            while len(episode_returns) < n_episodes:
                # policy mutates obs in-place and writes 'action'
                inference_policy(obs)

                step_input = TensorDict({"action": obs["action"]}, batch_size=eval_env.batch_size, device=policy_dev)
                step_td = eval_env.step(step_input)
                next_td = get_next_td(step_td)

                reward = next_td["reward"]
                if reward.ndim > 1:
                    reward = reward.view(eval_env.env_batch_size, -1).sum(dim=1)
                cum_returns += reward.to(policy_dev) * (~dones)
    
                done = next_td["done"]
                if done.ndim > 1:
                    done_mask = done.view(eval_env.env_batch_size, -1).any(dim=1).to(torch.bool).to(policy_dev)
                else:
                    done_mask = done.to(torch.bool).to(policy_dev)

                if done_mask.any():
                    idxs = done_mask.nonzero(as_tuple=True)[0]
                    for i in idxs.tolist():
                        episode_returns.append(float(cum_returns[i].item()))
                        cum_returns[i] = 0.0
                        if len(episode_returns) >= n_episodes:
                            break

                # partial reset for finished envs
                next_td.set("_reset", next_td.get("done"))
                obs = eval_env.reset(next_td)

                steps += 1
                pbar.update(1)
                if max_steps is not None and steps >= max_steps:
                    break

    avg_return = float(sum(episode_returns[:n_episodes]) / len(episode_returns[:n_episodes])) if episode_returns else 0.0
    print(f"Evaluation: avg_return={avg_return:.4f} episodes={len(episode_returns[:n_episodes])}")
    return {"avg_return": avg_return, "episode_returns": episode_returns[:n_episodes]}


def main(args, cfg, device) -> None:

    env = make_env(cfg, device=device)
    eval_env = make_env(cfg, device=device)
    bc_model = load_bc_model(cfg, device=device)

    split_layer = int(cfg.get("freeze_transformer_layers", 7))
    architecture = MPiFormerTorchRLArchitecture(
        bc_model=bc_model,
        pc_bounds=cfg["bc_checkpoint_parameters"]["pc_bounds"],
        split_layer=split_layer,
        deep_copy=True,
        log_std_init=float(cfg.get("log_std_init", -10.0)),
    )

    # build TorchRL TensorDictModules to wrap pytorch modules
    # use batch_first=True so modules accept and return [B, S, D] features compatible with replay buffer and gpu env
    backbone_module = architecture.build_backbone_module(
        in_keys=("point_cloud_labels", "point_cloud", "configuration"),
        out_key="features",
        batch_first=True,
    )

    action_low = -float(cfg.get("env", {}).get("max_delta_q", 1.0))
    action_high = float(cfg.get("env", {}).get("max_delta_q", 1.0))
    actor = architecture.build_probabilistic_actor(
        in_keys=("loc", "scale"),
        out_key="action",
        distribution_kwargs={"low": action_low, "high": action_high},
        return_log_prob=True,
        batch_first=True,
    )
    # Build critic modules (list with two TensorDictModules returning 'state_action_value') and create SACLoss
    critic_modules = architecture.build_critic_modules(features_key="features", action_key="action", batch_first=True)

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
    
    loss_module = SACLoss(
        actor_network=actor,
        qvalue_network=critic_modules,
        target_entropy=target_entropy,
        fixed_alpha=fixed_alpha,
        alpha_init=alpha_init,
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=gamma)

    # Build inference policy (backbone + actor) for deterministic evaluation
    try:
        inference_policy = architecture.build_actor_pipeline(action_spec=None, batch_first=True).to(device)
    except Exception:
        # Fallback: compose backbone + actor manually
        inference_policy = TensorDictSequential(backbone_module, actor)  # type: ignore[operator]
        inference_policy = inference_policy.to(device)

    # Initial evaluation (if configured)
    try:
        n_eval_cfg = int(cfg.get("n_eval_episodes", 0))
        if n_eval_cfg > 0:
            _res = evaluate_policy(inference_policy, eval_env, n_eval_cfg)
            print(f"Initial evaluation avg_return={_res['avg_return']:.4f}")
    except Exception as e:
        print(f"Warning: initial evaluation failed: {e}")
        # traceback
        import traceback
        traceback.print_exc()

    # lr and optimizers
    actor_lr = extract_start_lr(cfg.get("finetuning_actor_lr"), 3e-4)
    critic_lr = extract_start_lr(cfg.get("finetuning_critic_lr"), 3e-4)
    actor_optim = Adam(architecture.actor_head.parameters(), lr=actor_lr)
    qvalue_optim = Adam(list(architecture.critic_head_q1.parameters()) + list(architecture.critic_head_q2.parameters()), lr=critic_lr)
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
    eval_every = int(cfg.get("logger", {}).get("log_eval_freq", 0))
    n_eval_episodes = int(cfg.get("n_eval_episodes", 0))

    obs_td = env.reset()
    metrics = TrainMetrics()

    print(
        f"Starting training: steps={total_steps}, batch with {env.env_batch_size} envs, "
        f"learning_starts={learning_starts}, update_every={update_every}, gradient_steps={gradient_steps}"
    )

    # --- Training loop ---
    for step in tqdm(range(total_steps), desc="Training", unit="step", total=total_steps):
        with torch.no_grad():
            # Note: outputs added to tensordict in place
            backbone_module(obs_td) # predicts features and adds to obs_td in place
            features = obs_td["features"]
            actor(obs_td) # predicts action from features and adds to obs_td in place
            action = obs_td["action"]

        step_input = TensorDict({"action": action}, batch_size=env.batch_size, device=device)
        step_td = env.step(step_input) # apply action to env and get next observation, reward, done, etc in step_td
        next_td = get_next_td(step_td)

        with torch.no_grad():
            backbone_module(next_td) # predicts next features and adds to next_td in place

        transition = TensorDict(
            {
                "features": features.detach(), # state = observation features from backbone
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
            # periodic evaluation
            if eval_every > 0 and n_eval_episodes > 0 and (step + 1) % eval_every == 0:
                try:
                    _res = evaluate_policy(inference_policy, eval_env, n_eval_episodes)
                    print(f"Periodic evaluation step={step+1} avg_return={_res['avg_return']:.4f}")
                except Exception as e:
                    print(f"Warning: periodic evaluation failed: {e}")

    # --- Save final checkpoint ---
    # (actor/critic heads + optimizers, no frozen backbone, no replay buffer)
    try:
        save_dir = Path(cfg.get("save_dir", "checkpoints"))
        final_step = (step + 1) if "step" in locals() else total_steps
        save_checkpoint(save_dir, architecture, actor_optim, qvalue_optim, alpha_optim, loss_module, cfg, final_step)
    except Exception as e:
        print(f"Warning: failed to save checkpoint: {e}")

    env.close()
    print("Training complete.")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_cfg(args.cfg_path)

    device = resolve_device(args.device)
    set_seed(args.seed)
    print(f"Using device: {device}")

    main(args, cfg, device)
