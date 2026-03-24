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
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.data import DataLoader, Subset

from avoid_everything.data_loader import DataModule
from avoid_everything.pretraining import PretrainingMotionPolicyTransformer
from rl.environment import AvoidEverythingEnv
from rl.feature_extractor import MPiFormerExtractor
from rl.my_sac import MySAC
from rl.callbacks import DebugCallback


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
    parser.add_argument("--verbose", type=int, default=1, help="verbose parameter of SAC")
    parser.add_argument("--debug", type=int, default=1, help="0: None, 1: Summary, 2: Component Summary, 3: Detailed")
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    return args, cfg


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


def bootstrap_agent(
        env: AvoidEverythingEnv,
        eval_env: AvoidEverythingEnv,
        cfg: dict,
        args: argparse.Namespace,
        eval_callback: DebugCallback,
) -> MySAC:
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
        "log_std_init": cfg["log_std_init"],  # does NOT work if no sde! Must initialize log_std explicitly after model creation.
    }
    model = MySAC( # use deterministic actions for debug
        "MultiInputPolicy",
        env,
        force_deterministic=True,
        debug_verbose=args.debug,
        verbose=args.verbose,
        seed=cfg["seed"],
        buffer_size=cfg["buffer_size"],
        learning_starts=0,  # Start training immediately with BC policy, not random exploration
        batch_size=1,  # for fast iteration in prototyping and avoid OoD locally
        ent_coef=cfg["entropy_coef"],  # reduce entropy as the policy is pretrained
        policy_kwargs=policy_kwargs,
        train_freq=(1, "episode"), # train at the end of every episode
    )
    vec_env = model.get_env()
    print(f"Train will run with n_envs={vec_env.num_envs if vec_env is not None else 'unknown'}")
    # TODO: HER (requires adding achieved_goal/desired_goal keys to observation space)
    # TODO: BC buffer pre-filling (collect BC rollouts before learn() to replace random exploration phase)

    # Warm-start actor mean head from BC action_decoder (same Linear(512, 7) shape with net_arch={"pi": []})
    with torch.no_grad():
        model.policy.actor.mu.weight.copy_(bc_model.action_decoder.weight)
        model.policy.actor.mu.bias.copy_(bc_model.action_decoder.bias)
    print("✓ Warm-started actor mu from BC action_decoder")
    
    # Explicit log_std initialization because SB3 only initialize it when using SDE
    log_std_init_value = policy_kwargs.get("log_std_init", -20.0)
    model.initialize_log_std(log_std_value=log_std_init_value, state_independent_start=True)

    # Cleanup to save memory: remove bc_model reference from policy_kwargs
    # SB3 stores policy_kwargs in self.policy_kwargs. We remove the heavy bc_model
    # since it has already been copied into the Actor/Critic networks.
    if "bc_model" in model.policy_kwargs.get("features_extractor_kwargs", {}):
        del model.policy_kwargs["features_extractor_kwargs"]["bc_model"]
    # Also delete local variable and empty cache
    del bc_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ Freed original BC model memory")


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
        eval_callback.close_progress_bar()
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

    # callbacks to debug train, eval, and prefill
    eval_callback = DebugCallback(
        description="eval",
        log_steps=(args.debug >= 3),
        render=args.render,
        progress_bar=True,
        total=cfg["n_eval_episodes"],
        verbose=args.verbose,
    )
    train_callback = DebugCallback(
        description="train",
        log_steps=(args.debug >= 3),
        render=args.render,
        verbose=args.verbose,
    )
    prefill_callback = DebugCallback(
        description="prefill-buffer",
        log_steps=(args.debug >= 3),
        render=(args.render and bool(cfg.get("prefill_render", False))),
        progress_bar=True,
        verbose=args.verbose,
    )


    # --- Bootstrap RL agent from BC pretrained policy ---
    model: MySAC = bootstrap_agent(env, eval_env, cfg, args, eval_callback)
    model.monitor_agent("AFTER BOOTSTRAP")


    # Pre-fill replay buffer with BC policy rollouts
    model.prefill_replay_buffer(cfg, callback=prefill_callback)
    
    
    # Warm-up critic with fixed/frozen actor
    model.warmup_critic(cfg["critic_warmup_steps"], train_callback)


    # --- Train ---
    print("\nTraining SAC agent...")
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        progress_bar=True,
        callback=train_callback,
        reset_num_timesteps=False,  # Keep counting timesteps across multiple learn() calls (e.g. critic warmup + main training)
        log_interval=1,  # Log every episode
    )
    model.monitor_agent("AFTER TRAINING")

    # --- Evaluate trained model ---
    print("\nEvaluating trained SAC agent...")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=cfg["n_eval_episodes"],
        deterministic=True,
        callback=eval_callback,
    )
    eval_callback.close_progress_bar()
    print(f"Trained {model.__class__.__name__} agent: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # --- Save ---
    model.save(cfg["save_path"])
    print(f"✓ Model saved to: {cfg['save_path']}")
