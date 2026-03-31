"""
RL agent implemented with Stable Baselines3 (https://stable-baselines3.readthedocs.io/en/v2.4.1/)

features:
- SAC algorithm with warm-start from BC policy bootstrapping
- Multi-format logging (stdout, TensorBoard, CSV, WandB)
- Custom debug callbacks 
- Replay buffer prefill with BC rollouts
- Critic warm-up phase before full training
- Configurable learning rate schedules for warmup and finetuning phases
- HER buffer (TODO)
- vectorized env for parallel rollouts on different CPU cores (TODO)

Logging:
    The agent automatically logs training metrics to multiple outputs:
    - stdout: Console output for real-time monitoring
    - TensorBoard: Visualizations at ./logs/{run_name}-{timestamp}/
    - WandB: Cloud tracking
    
    Configure logging in the YAML config under the 'logger' section
    
    Metrics logged automatically by SB3:
    - rollout/ep_rew_mean, rollout/ep_len_mean
    - train/actor_loss, train/critic_loss, train/ent_coef
    - time/fps, time/total_timesteps

Usage:
    python3 rl/agent.py model_configs/rl_sac_cubbies.yaml
    
    # View TensorBoard logs:
    tensorboard --logdir ./logs/
"""


import argparse
import os
import shutil
import sys
from datetime import datetime
import traceback

import torch
import yaml
from dotenv import load_dotenv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb

from avoid_everything.pretraining import PretrainingMotionPolicyTransformer
from avoid_everything.type_defs import DatasetType
from rl.environment import AvoidEverythingEnv
from rl.feature_extractor import MPiFormerExtractor
from rl.my_sac import MySAC
from rl.callbacks import DebugCallback
from rl.lr_schedules import build_lr_schedule


class Tee:
    """
    Write to both a file and the original stream (like Unix tee command).
    Used to redirect stdout/stderr to a log file while maintaining console output.
    """
    def __init__(self, file_handle, original_stream):
        self.file = file_handle
        self.stream = original_stream
    
    def write(self, message):
        self.stream.write(message)
        self.stream.flush()
        self.file.write(message)
        self.file.flush()
    
    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False


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


def setup_run_directory(args: argparse.Namespace, cfg: dict) -> dict:
    # Create run directory and save command, args, and config for reproducibility
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = cfg["logger"].get("run_name", "run")  # default to "run" if not specified
    run_id = run_name + "_" + timestamp
    run_dir = os.path.join(cfg["logger"]["log_dir"], run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Copy original config file to preserve comments and formatting
    config_dest = os.path.join(run_dir, "config.yaml")
    shutil.copy2(args.cfg_path, config_dest)
    print(f"✓ Config copied to: {config_dest}")
    
    # Save command to text file for reproducibility
    command = " ".join(sys.argv)
    command_file = os.path.join(run_dir, "command.txt")
    with open(command_file, "w") as f:
        f.write(command + "\n")
    print(f"✓ Command saved to: {command_file}")
    
    # Alternative: Save command + args to YAML (currently disabled)
    # run_info = {
    #     "command": command,
    #     "args": vars(args),
    #     "timestamp": timestamp,
    # }
    # run_info_path = os.path.join(run_dir, "run_info.yaml")
    # with open(run_info_path, "w") as f:
    #     yaml.dump(run_info, f, default_flow_style=False)
    run_info = {
        "run_id": run_id,
        "timestamp": timestamp,
        "run_name": run_name,
        "run_dir": run_dir,
    }
    return run_info



def bootstrap_agent(
        env: AvoidEverythingEnv,
        eval_env: AvoidEverythingEnv,
        cfg: dict,
        args: argparse.Namespace,
        eval_callback: DebugCallback,
        run: dict,
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
        tensorboard_log=run["run_dir"],  # Enable TensorBoard logging
        force_deterministic=True,
        debug_verbose=args.debug,
        verbose=args.verbose,
        seed=cfg["seed"],
        buffer_size=cfg["buffer_size"],
        learning_starts=0,  # Start training immediately with BC policy, not random exploration
        batch_size=cfg["batch_size"],
        ent_coef=cfg["entropy_coef"],  # reduce entropy as the policy is pretrained
        policy_kwargs=policy_kwargs,
        train_freq= (1,"step"), #tuple(cfg["train_freq"]) if isinstance(cfg["train_freq"], list) else cfg["train_freq"]
    )
    vec_env = model.get_env()
    print(f"Train will run with n_envs={vec_env.num_envs if vec_env is not None else 'unknown'}")
    # TODO: HER (requires adding achieved_goal/desired_goal keys to observation space)

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


def main(args: argparse.Namespace, cfg: dict, run: dict) -> None:
    # --- Environments ---
    # Create environments first (they create their own Robot instances)
    # Wrap with Monitor explicitly so SB3 doesn't auto-wrap and episode stats are tracked consistently.
    # Order: Monitor (outermost) → TimeLimit (inside AvoidEverythingEnv) → _AvoidEverythingEnv (core).
    render_mode = "human" if args.render else None
    render_backend = "ros"
    print(f"Render mode: {render_mode}")
    
    # Create environments without dataloaders initially
    # Monitor with info_keywords to capture episode-level metrics from final step
    info_keywords = ("target_reached", "collision", "position_error", "orientation_error", 
                     "episode_num_collisions", "episode_num_steps", "episode_return", "TimeLimit.truncated")
    env = Monitor(AvoidEverythingEnv(render_mode=render_mode, render_backend=render_backend),
                  info_keywords=info_keywords)
    eval_env = Monitor(AvoidEverythingEnv(render_mode=render_mode, render_backend=render_backend),
                       info_keywords=info_keywords)
    
    # --- Data ---
    # Set up TrajectoryDataset for each environment using its internal robot.
    # TrajectoryDataset ensures episodes start from trajectory initial states (q0),
    # matching the expert demonstration start distribution.
    overfit_idx = args.overfit if args.overfit is not None else None
    if overfit_idx is not None:
        print(f"⚠  Overfitting mode: using trajectory index {overfit_idx} for both train and eval")
    # Configure train environment dataset
    env.unwrapped.set_scene_generation_from_dataset(
        data_dir=cfg["data_dir"],
        trajectory_key=cfg["train_trajectory_key"],
        dataset_type=DatasetType.TRAIN,
        num_workers=cfg["num_workers"],
        random_scale=0.0,  # No noise for RL (clean states)
        overfit_idx=overfit_idx,
    )
    # Configure eval environment dataset
    eval_env.unwrapped.set_scene_generation_from_dataset(
        data_dir=cfg["data_dir"],
        trajectory_key=cfg["val_trajectory_key"],
        dataset_type=DatasetType.VAL,
        num_workers=cfg["num_workers"],
        random_scale=0.0,
        overfit_idx=overfit_idx,
    )
    
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
    model: MySAC = bootstrap_agent(env, eval_env, cfg, args, eval_callback, run)
    model.monitor_agent("AFTER BOOTSTRAP")

    
    # Combine callbacks: train_callback for debugging, wandb_callback for metrics
    train_callbacks = [train_callback]
    # if wandb_callback is not None:
    #     train_callbacks.append(wandb_callback)

    # Pre-fill replay buffer with BC policy rollouts
    model.prefill_replay_buffer(cfg,
                                callback=prefill_callback)
    
    # --- Calculate phase boundaries for phase-aware LR schedules ---
    
    # --- Update LRs for critic warmup phase ---
    warmup_steps = cfg["critic_warmup_steps"]
    warmup_lr_schedule = build_lr_schedule(
        cfg["critic_warmup_lr"],
        phase_start_step=0,
        phase_total_steps=warmup_steps,
    )
    print(f"✓ Critic warmup LR config: {cfg['critic_warmup_lr']}")
    model.update_learning_rates(critic_schedule=warmup_lr_schedule)
    
    # --- Warm-up critic with fixed/frozen actor ---
    model.warmup_critic(cfg["critic_warmup_steps"],
                        train_callbacks)
    # save warmuped-up model
    save_path = os.path.join(cfg["save_path"], run["run_name"] + "_warmup_" + run["timestamp"])
    model.save(save_path)
    print(f"✓ Warmed-up model saved to: {save_path}")

    # --- Update LRs for RL finetuning phase ---
    current_step = model.num_timesteps
    finetuning_steps = cfg["total_timesteps"]
    actor_lr_schedule = build_lr_schedule(
        cfg["finetuning_actor_lr"],
        phase_start_step=current_step,
        phase_total_steps=finetuning_steps,
    )
    critic_lr_schedule = build_lr_schedule(
        cfg["finetuning_critic_lr"],
        phase_start_step=current_step,
        phase_total_steps=finetuning_steps,
    )
    
    print(f"✓ Finetuning actor LR config: {cfg['finetuning_actor_lr']}")
    print(f"✓ Finetuning critic LR config: {cfg['finetuning_critic_lr']}")
    model.update_learning_rates(
        actor_schedule=actor_lr_schedule,
        critic_schedule=critic_lr_schedule
    )

    # --- Train RL fine-tuning ---
    print(f"\nRL finetuning for {finetuning_steps} steps")
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        progress_bar=True,
        callback=train_callbacks,
        reset_num_timesteps=False,  # Keep counting timesteps across multiple learn() calls (e.g. critic warmup + main training)
        log_interval=1,  # Log every episode
        tb_log_name="rl_finetuning",  # TensorBoard subdirectory for this training phase
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
    save_path = os.path.join(cfg["save_path"], run["run_id"])
    model.save(save_path)
    print(f"✓ Model saved to: {save_path}")


if __name__ == "__main__":
    args, cfg = get_args_and_cfg()
    run = setup_run_directory(args, cfg)
    
    # Initialize WandB if enabled in config
    logger_config = cfg.get("logger", {})
    formats = logger_config.get("formats", [])
    
    if "wandb" in formats:
        # Load .env for API key and entity
        load_dotenv()
        
        # Login with API key if available
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)
        
        # Get project and entity from config or .env
        wandb_project = logger_config.get("wandb_project")
        if not wandb_project:
            print("WARNING: wandb_project not found in config, skipping WandB initialization")
        else:
            wandb_entity = logger_config.get("wandb_entity") or os.getenv("WANDB_ENTITY")
            
            wandb.tensorboard.patch(root_logdir=run["run_dir"]) # for several events subdirectories
            # Initialize WandB run
            wandb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                id=run["run_id"],
                # dir=run["run_dir"],  # log WandB in logs directory
                name=run["run_id"], # Use run_id as the WandB run name
                config=cfg,
                sync_tensorboard=True,
                reinit="finish_previous",
            )
            print(f"✓ WandB initialized: project={wandb_project}, entity={wandb_entity}, run_id={wandb_run.id}")
    
    try:
        # sys.stdout = Tee(log_file, original_stdout)
        # sys.stderr = Tee(log_file, original_stderr)
        # print(f"✓ Output logging to: {log_file_path}")
  
        main(args, cfg, run)

    except Exception as e:
        # Log the error details while Tee is still active (captures to file!)
        print(f"\n{'='*70}")
        print(f"FATAL ERROR: Training crashed")
        print(f"Exception: {type(e).__name__}: {e}")
        print(f"{'='*70}\n")
        traceback.print_exc()  # Full traceback saved to both console and file
        raise  # Re-raise to maintain proper error propagation
  
    finally:
        # Finish WandB run if it was initialized
        if wandb_run is not None:
            wandb.finish()
