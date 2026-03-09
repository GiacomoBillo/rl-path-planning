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
from typing import Callable

import numpy as np
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from avoid_everything.data_loader import DataModule
from avoid_everything.pretraining import PretrainingMotionPolicyTransformer
from rl.environment import AvoidEverythingEnv
from rl.feature_extractor import MPiFormerExtractor


def make_eval_progress_callback(n_eval_episodes: int) -> Callable:
    """Return a callback for evaluate_policy that shows a tqdm progress bar.

    The callback is called at every env step. It watches ``episode_rewards``
    (a list in evaluate_policy's locals that grows by one per completed episode)
    to drive the bar and display a live mean reward.
    """
    pbar = tqdm(total=n_eval_episodes, desc="Evaluating", unit="ep")
    last_count = [0]  # list so the closure can mutate it

    def callback(locals_: dict, _globals: dict) -> None:
        n_done = len(locals_["episode_rewards"])
        if n_done > last_count[0]:
            pbar.update(n_done - last_count[0])
            last_count[0] = n_done
            pbar.set_postfix({"mean_r": f"{np.mean(locals_['episode_rewards']):.2f}"})
        if last_count[0] >= n_eval_episodes:
            pbar.close()

    return callback

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

    # --- SAC model with BC feature extractor ---
    # Empty actor net (net_arch={"pi": []}) means mu = Linear(512, 7) — same shape as BC action_decoder — so we can copy its weights for a full warm-start.
    # Standard critic net (net_arch={"qf": [256, 256]}) keeps a proper Q-value MLP.
    policy_kwargs = {
        "features_extractor_class": MPiFormerExtractor,
        "features_extractor_kwargs": {
            "bc_model": bc_model,
            "pc_bounds": cfg["bc_checkpoint_parameters"]["pc_bounds"],
            "freeze": cfg["freeze_bc_encoder"],
        },
        "net_arch": {
            "pi": [], # Actor/Policy no hidden layers, just Linear(512, 7) from encoded_features to action space, 
                      # to copy weights from pretrained BC action_decoder 
            "qf": []}, # Critic TODO: might need some hidden layers here for good Q-estimation (e.g. [256, 256])
        "share_features_extractor": True,
    }
    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=cfg["seed"],
        buffer_size=cfg["buffer_size"],
        policy_kwargs=policy_kwargs,
    )
    # TODO: HER (requires adding achieved_goal/desired_goal keys to observation space)

    # Warm-start actor mean head from BC action_decoder (same Linear(512, 7) shape with net_arch={"pi": []})
    with torch.no_grad():
        model.policy.actor.mu.weight.copy_(bc_model.action_decoder.weight)
        model.policy.actor.mu.bias.copy_(bc_model.action_decoder.bias)
    print("✓ Warm-started actor mu from BC action_decoder")

    # --- Evaluate BC-warm-started policy (pre-training baseline) ---
    print("\nEvaluating BC-warm-started policy...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=cfg["n_eval_episodes"], deterministic=True,
        callback=make_eval_progress_callback(cfg["n_eval_episodes"]),
    )
    print(f"BC-warm-started agent: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # --- Train ---
    print("\nTraining SAC agent...")
    model.learn(total_timesteps=cfg["total_timesteps"], progress_bar=True)

    # --- Evaluate trained model ---
    print("\nEvaluating trained SAC agent...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=cfg["n_eval_episodes"], deterministic=True,
        callback=make_eval_progress_callback(cfg["n_eval_episodes"]),
    )
    print(f"Trained {model.__class__.__name__} agent: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # --- Save ---
    model.save(cfg["save_path"])
    print(f"✓ Model saved to: {cfg['save_path']}")
