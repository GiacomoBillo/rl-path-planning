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

import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from torch.utils.data import DataLoader

from avoid_everything.data_loader import DataModule
from avoid_everything.pretraining import PretrainingMotionPolicyTransformer
from rl.environment import AvoidEverythingEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg_path", help="Path to the RL config YAML file")
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

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
    print(f"✓ Datasets: {len(dm.data_train)} train samples, {len(dm.data_val_state)} val samples")

    train_dl = dm.train_dataloader()  # StateDataset, shuffle=True
    # Construct eval dataloader with shuffle=True so eval_env sees varied problems per episode
    # (dm.val_dataloader()[VAL_STATE] uses shuffle=False, unsuitable for RL reset()).
    eval_dl = DataLoader(
        dm.data_val_state, batch_size=1, shuffle=True, num_workers=cfg["num_workers"]
    )

    # --- Environments ---
    # Wrap with Monitor explicitly so SB3 doesn't auto-wrap and episode stats are tracked consistently.
    # Order: Monitor (outermost) → TimeLimit (inside AvoidEverythingEnv) → _AvoidEverythingEnv (core).
    env = Monitor(AvoidEverythingEnv(dataloader=train_dl, render_mode=None))
    eval_env = Monitor(AvoidEverythingEnv(dataloader=eval_dl, render_mode=None))

    # --- SAC model ---
    model = SAC("MultiInputPolicy", env, verbose=1, seed=cfg["seed"], buffer_size=cfg["buffer_size"])
    # TODO: HER (requires adding achieved_goal/desired_goal keys to observation space)

    # --- Evaluate random policy (pre-training baseline) ---
    print("\nEvaluating random policy...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=cfg["n_eval_episodes"], deterministic=True
    )
    print(f"Random agent: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # --- Load BC checkpoint (pre-trained AvoidEverything policy) ---
    # TODO: wire bc_model weights into the SAC policy via a custom SB3 feature extractor

    # -- Trasfer learning: bootstrap RL policy with BC pre-trained weights to avoid cold start ---
    # TODO

    # --- Train ---
    print("\nTraining SAC agent...")
    model.learn(total_timesteps=cfg["total_timesteps"], progress_bar=True)

    # --- Evaluate trained model ---
    print("\nEvaluating trained SAC agent...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=cfg["n_eval_episodes"], deterministic=True
    )
    print(f"Trained {model.__class__.__name__} agent: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # --- Save ---
    model.save(cfg["save_path"])
    print(f"✓ Model saved to: {cfg['save_path']}")
