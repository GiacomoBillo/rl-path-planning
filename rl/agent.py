"""
RL agent implemented with Stable Baselines3 (https://stable-baselines3.readthedocs.io/en/v2.4.1/)

features:
- SAC algorithm
- load policy from BC checkpoint from AvoidEverything (warm-start TODO)
- HER buffer (TODO)
- vectorized env for parallel rollouts on different CPU cores (TODO)

Usage:
    python3 rl/agent.py model_configs/rl_sac_cubbies.yaml
"""


import argparse
from typing import Any, Callable

import numpy as np
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from avoid_everything.data_loader import DataModule
from avoid_everything.pretraining import PretrainingMotionPolicyTransformer
from rl.environment import AvoidEverythingEnv
from rl.feature_extractor import MPiFormerExtractor
from rl.my_sac import SACDebug


def get_args_and_cfg() -> tuple[argparse.Namespace, dict]:
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to the RL config YAML file")
    parser.add_argument("--eval_bc", action="store_true", help="Whether to evaluate the pretrained BC policy before training")
    parser.add_argument(
        "--overfit",
        nargs="?",
        type=int,
        const=1,
        default=None,
        metavar="N",
        help=(
            "Overfit on N training scenes to verify the agent can learn. "
            "Pass without a value to use 1 scene, or specify a number of samples to use. "
            "Omit entirely to train on the full dataset."
        ),
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    return args, cfg


class DebugCallback(BaseCallback):
    """Unified callback for training and evaluation.

    Handles:
    - Progress bar (for evaluation)
    - Step logging (reward, done, info) with formatting
    - Rendering (safely restricted to single-env to avoid clutter)

    'locals' refers to the local variables of the scope where the callback is called.

    About 'infos':
    - In Stable Baselines3, environments are typically wrapped in a VecEnv.
    - 'infos' is a list of dictionaries, one per environment.
    - We log/render only the first environment (index 0) to avoid clutter.
    """

    def __init__(
        self,
        n_eval_episodes: int = 0,
        is_eval: bool = False,
        log_steps: bool = False,
        render: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.n_eval_episodes = n_eval_episodes
        self.is_eval = is_eval
        self.log_steps = log_steps
        self.render_enabled = render

        self.pbar = None
        self.last_done_count = 0

        if self.is_eval and self.n_eval_episodes > 0:
            self.pbar = tqdm(total=self.n_eval_episodes, desc="Evaluating", unit="ep")

    def _on_rollout_start(self):
        """Called by model.learn() at the beginning of every rollout.
        TODO: missing equivalent for evaluate_policy()
        """
        if self.log_steps:
            print(f"[{'eval' if self.is_eval else 'train'}-rollout-start] ")

        env = self.training_env
        if self.render_enabled and env is not None:
            # Check if it's a Stable Baselines3 VecEnv
            if hasattr(env, "env_method"):
                # Safely call render ONLY on the first environment (index 0)
                env.env_method("render", indices=0)
            else:
                # Fallback for standard, non-vectorized Gymnasium environments
                env.render()
        

    def _on_step(self) -> bool:
        """Called by model.learn() at every step."""
        return self._process_step(self.locals, self.training_env)

    def __call__(self, locals_: dict, globals_: dict) -> None:
        """Called by evaluate_policy() at every step."""
        # evaluate_policy passes 'env' in locals_
        env = locals_.get("env")
        self._process_step(locals_, env)

    def _process_step(self, locals_: dict, env: Any) -> bool:
        # 1. Normalize variable names
        # Check keys explicitly to avoid False-y issues with boolean/numpy arrays
        infos = locals_.get("infos", locals_.get("info", []) )
        dones = locals_.get("dones", locals_.get("done", []) )
        rewards = locals_.get("rewards", locals_.get("reward", []) )

        # 2. Select first env (index 0) in case of parallel envs
        info = infos[0] if (isinstance(infos, list) and len(infos) > 0) else infos
        done_val = dones[0] if (isinstance(dones, (list, np.ndarray)) and len(dones) > 0) else dones
        reward_val = rewards[0] if (isinstance(rewards, (list, np.ndarray)) and len(rewards) > 0) else rewards

        # 3. Log steps
        if self.log_steps:
            # Approximate numbers for cleaner logs
            reward_str = f"{float(reward_val):.3f}"
            info_str = self._format_info(info)
            collision = info.get("collision", None) if isinstance(info, dict) else None
            
            phase = "eval" if self.is_eval else "train"
            print(f"[{phase}-step] reward={reward_str} terminated={done_val} collision={collision} info={info_str}")


        # 4. Render
        if self.render_enabled and env is not None:
            # Check if it's a Stable Baselines3 VecEnv
            if hasattr(env, "env_method"):
                # Safely call render ONLY on the first environment (index 0)
                env.env_method("render", indices=0)
            else:
                # Fallback for standard, non-vectorized Gymnasium environments
                env.render()

        # 5. Progress Bar (Eval only)
        if self.is_eval and self.pbar:
            # evaluate_policy accumulates 'episode_rewards' in locals
            episode_rewards = locals_.get("episode_rewards", [])
            n_done = len(episode_rewards)
            if n_done > self.last_done_count:
                self.pbar.update(n_done - self.last_done_count)
                self.last_done_count = n_done
                if n_done > 0:
                    self.pbar.set_postfix({"mean_r": f"{np.mean(episode_rewards):.2f}"})

            if self.last_done_count >= self.n_eval_episodes:
                self.pbar.close()
                self.pbar = None

        return True

    def _format_info(self, info: Any) -> Any:
        """Recursively round floats in the info dict."""
        if isinstance(info, float):
            return round(info, 3)
        elif isinstance(info, dict):
            return {k: self._format_info(v) for k, v in info.items()}
        elif isinstance(info, list):
            return [self._format_info(x) for x in info]
        return info


def get_dataloaders(cfg: dict, args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Create train and eval dataloaders from the AvoidEverything dataset."""

    # --- Data ---
    # DataModule handles all dataset construction (same pattern as run_validation_rollouts.py).
    # StateDataset is used (not TrajectoryDataset): each sample is a single state
    # (config, obstacles, target), which is exactly what the RL env's reset() needs.
    dm = DataModule(
        urdf_path=cfg["urdf_path"],
        data_dir=cfg["data_dir"],
        train_trajectory_key=cfg["train_trajectory_key"],
        val_trajectory_key=cfg["val_trajectory_key"],
        num_robot_points=cfg["num_robot_points"],
        num_obstacle_points=cfg["num_obstacle_points"],
        num_target_points=cfg["num_target_points"],
        action_chunk_length=cfg["action_chunk_length"],
        random_scale=0.0,  # No noise for RL (clean states)
        train_batch_size=1,  # RL env processes one problem at a time
        val_batch_size=1,
        num_workers=cfg["num_workers"],
    )
    dm.setup("fit")

    # Construct eval dataloader with shuffle=True so eval_env sees varied problems per episode
    # (dm.val_dataloader()[VAL_STATE] uses shuffle=False, unsuitable for RL reset()).
    if args.overfit is None:
        train_dl = dm.train_dataloader()  # StateDataset, shuffle=True
        eval_dl = DataLoader(
            dm.data_val_state, batch_size=1, shuffle=True, num_workers=cfg["num_workers"]
        )
        print(f"✓ Datasets: {len(dm.data_train)} train samples, {len(dm.data_val_state)} val samples")
    else:
        n = args.overfit
        overfit_dataset = Subset(dm.data_train, list(range(n)))
        train_dl = DataLoader(overfit_dataset, batch_size=1, shuffle=True, num_workers=cfg["num_workers"])
        eval_dl = DataLoader(overfit_dataset, batch_size=1, shuffle=True, num_workers=cfg["num_workers"])
        print(f"⚠  Overfitting mode: using {n} training scene(s) for both train and eval")

    return train_dl, eval_dl


def debug_actor_weights(model, stage):
    s = 0
    with torch.no_grad():
        for p in model.policy.actor.parameters():
            s += p.data.sum().item()
    print(f"[DEBUG] Actor weight sum at {stage}: {s:.4f}")

def warmup_critic(
        model: SAC, 
        critic_warmup_steps: int,
        train_callback: DebugCallback
    ) -> SAC:
    debug_actor_weights(model, "START_WARMUP")
    if critic_warmup_steps <= 0:
        print("✓ No critic warmup steps specified, skipping warmup.")
        return model
    
    print(f"\nWarming up critic for {critic_warmup_steps} steps (actor frozen)...")
    # Freeze actor components that should be trainable later
    # 1. Feature extractor (transformer part only; backbone remains frozen as per config)
    model.policy.actor.features_extractor.freeze_transformer()
    # 2. Policy heads (mu and log_std)
    for param in model.policy.actor.mu.parameters():
        param.requires_grad = False
    for param in model.policy.actor.log_std.parameters():
        param.requires_grad = False
    # set log_std to zero
    # with torch.no_grad():
    #     model.policy.actor.log_std.weight.fill_(0.0)
    #     model.policy.actor.log_std.bias.fill_(-20.0)

    # 3. Set to eval mode (disable Dropout/BatchNorm updates)
    model.policy.actor.set_training_mode(False) # TODO is it needed?
    # Prevent SB3 from switching actor back to train mode during learn()
    original_actor_set_mode = model.policy.actor.set_training_mode
    model.policy.actor.set_training_mode = lambda mode: None

    # Train (only critic updates because actor has no gradients)
    print("[AUDIT] Capturing actor state before warmup...")
    initial_params = {name: p.detach().clone() for name, p in model.policy.actor.named_parameters()}
    
    print("[AUDIT PRE-LEARN] Checking for unfrozen parameters...")
    unfrozen_count = 0
    for name, p in model.policy.actor.named_parameters():
        if p.requires_grad:
            print(f"  !!! UNFROZEN: {name}")
            unfrozen_count += 1
        else:
            print(f"  ✓ Frozen: {name}")
    if unfrozen_count == 0:
        print("  ✓ All actor parameters are frozen.")
    else:
        print(f"  ⚠ Found {unfrozen_count} unfrozen actor parameters!")

    model.learn(
        total_timesteps=critic_warmup_steps,
        progress_bar=True,
        callback=train_callback,
        )

    print("[AUDIT] Checking for changed parameters after warmup...")
    for name, p in model.policy.actor.named_parameters():
        if name in initial_params:
            diff = (p.data - initial_params[name]).abs().sum().item()
            if diff > 1e-6:
                print(f"  !!! CHANGED: {name} | Diff: {diff:.6f} | Grad: {p.requires_grad}")
            else:
                print(f"  ✓ Unchanged: {name} | Diff: {diff:.6f} | Grad: {p.requires_grad}")

    # Restore actor set_training_mode and Unfreeze actor components
    model.policy.actor.set_training_mode = original_actor_set_mode
    model.policy.actor.features_extractor.unfreeze_transformer()
    for param in model.policy.actor.mu.parameters():
        param.requires_grad = True
    for param in model.policy.actor.log_std.parameters():
        param.requires_grad = True
    model.policy.actor.set_training_mode(True)

    debug_actor_weights(model, "END_WARMUP")
    print("✓ Critic warmup complete. Unfroze actor.")



def bootstrap_agent(
        env: AvoidEverythingEnv,
        eval_env: AvoidEverythingEnv,
        cfg: dict,
        args: argparse.Namespace,
        eval_callback: DebugCallback,
) -> SAC:
    # --- Load BC checkpoint (must happen before SAC model creation for warm-start) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bc_model = PretrainingMotionPolicyTransformer.load_from_checkpoint(
        cfg["bc_checkpoint_path"],
        urdf_path=cfg["urdf_path"],
        num_robot_points=cfg["num_robot_points"],
        action_chunk_length=cfg["action_chunk_length"],
        **cfg["bc_checkpoint_parameters"],
    ).to(device)
    print(f"✓ BC checkpoint loaded from: {cfg['bc_checkpoint_path']}")

    # --- SAC model with split BC feature extractor ---
    # 1) Shared frozen backbone (PointNet++ + joint encoder) via MPiFormerBackbone
    # 2) Per-instance trainable transformer via MPiFormerTransformerExtractor
    # 3) Linear-only heads (net_arch={"pi": [], "qf": []})
    
    policy_kwargs = {
        "features_extractor_class": MPiFormerExtractor,
        "features_extractor_kwargs": {
            "bc_model": bc_model,
            "pc_bounds": cfg["bc_checkpoint_parameters"]["pc_bounds"],
            "freeze_perception": True,  # Freeze perception encoder (equivalent to old freeze_backbone)
            "freeze_transformer": False, # Transformer is trainable
            "deep_copy_perception": True,
            "deep_copy_transformer": True,
        },
        "net_arch": {
            "pi": [],  # Actor/Policy no hidden layers, just Linear(d_model, robot_dof)
            "qf": []   # Critic no hidden layers, just Linear(d_model + action_dim, 1)
        },
        "share_features_extractor": False,  # Each (actor, critic) gets separate transformer
        "log_std_init": -20.0,
    }
    model = SACDebug( # use deterministic actions for debug
        "MultiInputPolicy",
        env,
        deterministic_rollout=True,
        verbose=True,
        seed=cfg["seed"],
        buffer_size=cfg["buffer_size"],
        policy_kwargs=policy_kwargs,
        batch_size=1,  # for fast iteration in prototyping and avoid OoD locally
        learning_starts=0,  # Start training immediately with BC policy, not random exploration
        train_freq=(1, "episode"), # train at the end of every episode
        ent_coef= 0, #'auto_0.1' # reduce entropy as the policy is pretrained
    )
    # model = SAC(
        # "MultiInputPolicy",
    #     env,
    #     verbose=1,
    #     seed=cfg["seed"],
    #     buffer_size=cfg["buffer_size"],
    #     policy_kwargs=policy_kwargs,
    #     batch_size=1,  # for fast iteration in prototyping and avoid OoD locally
    #     learning_starts=0,  # Start training immediately with BC policy, not random exploration
    #     train_freq=(1, "episode"), # train at the end of every episode
    #     ent_coef= 0, #'auto_0.1' # reduce entropy as the policy is pretrained
    # )
    vec_env = model.get_env()
    print(f"Train will run with n_envs={vec_env.num_envs if vec_env is not None else 'unknown'}")
    # TODO: HER (requires adding achieved_goal/desired_goal keys to observation space)

    # Warm-start actor mean head from BC action_decoder (same Linear(512, 7) shape with net_arch={"pi": []})
    with torch.no_grad():
        model.policy.actor.mu.weight.copy_(bc_model.action_decoder.weight)
        model.policy.actor.mu.bias.copy_(bc_model.action_decoder.bias)
    print("✓ Warm-started actor mu from BC action_decoder")

    # --- Evaluate BC-warm-started policy (pre-training baseline) ---
    if args.eval_bc:
        print("\nEvaluating BC-warm-started policy...")
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=cfg["n_eval_episodes"],
            deterministic=True,
            callback=eval_callback,
        )
        print(f"BC-warm-started agent: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    return model



if __name__ == "__main__":
    args, cfg = get_args_and_cfg()

    # --- Data ---
    train_dl, eval_dl = get_dataloaders(cfg, args)

    # --- Environments ---
    # Wrap with Monitor explicitly so SB3 doesn't auto-wrap and episode stats are tracked consistently.
    # Order: Monitor (outermost) → TimeLimit (inside AvoidEverythingEnv) → _AvoidEverythingEnv (core).
    render_mode = "human" if args.render else None
    render_backend = "ros"
    print(f"Render mode: {render_mode}")
    env = Monitor(AvoidEverythingEnv(dataloader=train_dl, render_mode=render_mode, render_backend=render_backend))
    eval_env = Monitor(AvoidEverythingEnv(dataloader=eval_dl, render_mode=render_mode, render_backend=render_backend))
    # wrap vec
    env = DummyVecEnv([lambda: env])
    eval_env = DummyVecEnv([lambda: eval_env])

    # callbacks to debug train and eval
    eval_callback = DebugCallback(
            n_eval_episodes=cfg["n_eval_episodes"],
            is_eval=True,
            log_steps=args.verbose,
            render=args.render,
        )
    train_callback = DebugCallback(
        log_steps=args.verbose,
        render=args.render,
    )


    # --- Bootstrap RL agent from BC pretrained policy ---
    model: SAC = bootstrap_agent(env, eval_env, cfg, args, eval_callback)
    debug_actor_weights(model, "AFTER_BOOTSTRAP")

    
    # Warm-up critic with fixed/frozen actor
    warmup_critic(model, cfg["critic_warmup_steps"], train_callback)


    # --- Train ---
    print("\nTraining SAC agent...")
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        progress_bar=True,
        callback=train_callback,
    )

    # --- Evaluate trained model ---
    print("\nEvaluating trained SAC agent...")
    debug_actor_weights(model, "BEFORE_EVAL")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=cfg["n_eval_episodes"],
        deterministic=True,
        callback=eval_callback,
    )
    print(f"Trained {model.__class__.__name__} agent: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # --- Save ---
    model.save(cfg["save_path"])
    print(f"✓ Model saved to: {cfg['save_path']}")
