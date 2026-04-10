"""
RL Finetuning Script

This script handles the second phase of RL training:
1. Load warmed checkpoint (actor + critic + replay buffer)
2. Set up RL finetuning with proper learning rate schedules
3. Run SAC training with actor and critic both trainable
4. Save trained checkpoints

The warmed checkpoint should be created by warmup.py first.

Usage:
    # RL finetuning from checkpoint
    python rl/train.py [config_path] --checkpoint [checkpoint_path]
    
    Optional arguments:
    --eval_after            Evaluate the trained policy after finetuning (default: True)
    --overfit N             Overfit on training scene with idx N for debugging (default: 1 if flag is used)
    --render                Render the environment during training
    --eval_freq N           Periodic evaluation frequency in training steps (default: 1000, <=0 disables periodic eval)

Output:
    - Trained checkpoint saved to: {save_path}/{run_name}_finetuning_{timestamp}.zip
    - TensorBoard logs: {log_dir}/{run_name}_finetuning_{timestamp}/
    - WandB run: {run_name}_finetuning_{timestamp} (if enabled)
"""

import argparse
import os
import sys
import traceback

import yaml
import wandb
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from avoid_everything.type_defs import DatasetType
from rl.callbacks import DebugCallback
from rl.common import (
    setup_run_directory,
    create_env,
    load_checkpoint_with_metadata,
    setup_wandb,
    save_checkpoint_with_metadata,
    get_save_path,
)
from rl.lr_schedules import build_lr_schedule


def get_args_and_cfg():
    """Parse command-line arguments and load configuration."""
    parser = argparse.ArgumentParser(
        description="RL Finetuning from Warmed Checkpoint"
    )
    parser.add_argument("cfg_path", help="Path to the RL config YAML file")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to warmed checkpoint from warmup.py (with or without .zip extension)"
    )
    parser.add_argument(
        "--eval_after",
        action="store_true",
        default=True,
        help="Evaluate the trained policy after finetuning"
    )
    parser.add_argument(
        "--overfit",
        nargs="?",
        type=int,
        const=1,
        default=None,
        metavar="N",
        help="Overfit on N training scenes for debugging (default: 1 if flag is used)"
    )
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--verbose", type=int, default=1, help="SB3 verbosity level")
    parser.add_argument(
        "--debug",
        type=int,
        default=1,
        help="Debug level: 0=None, 1=Summary, 2=Component, 3=Detailed"
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1000,
        help="Periodic evaluation frequency in training steps (default: 1000, <=0 disables periodic eval)"
    )
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    return args, cfg


def main(args, cfg):
    """Main training workflow."""
    # Setup run directory with 'finetune' phase
    run_info = setup_run_directory(args.cfg_path, cfg, phase="finetune")
    print(f"\n=== RL Finetuning ===")
    print(f"Run ID: {run_info['run_id']}")
    print(f"Run directory: {run_info['run_dir']}")
    print(f"Loading checkpoint: {args.checkpoint}\n")

    # Initialize WandB if enabled
    wandb_run = setup_wandb(cfg, run_info, tags=["rl-finetuning"])

    try:
        # --- Create Environments ---
        print("Creating environments...")
        overfit_idx = args.overfit if args.overfit is not None else None
        if overfit_idx is not None:
            print(f"⚠  Overfitting mode: using trajectory index {overfit_idx}")

        env = create_env(
            cfg,
            render=args.render,
            dataset_type=DatasetType.TRAIN,
            overfit_idx=overfit_idx,
        )
        eval_env = create_env(
            cfg,
            render=args.render,
            dataset_type=DatasetType.VAL,
            overfit_idx=overfit_idx,
        )
        print("✓ Environments created")

        # --- Load Warmed Checkpoint ---
        print("\nLoading warmed checkpoint...")
        model, metadata = load_checkpoint_with_metadata(
            args.checkpoint,
            env=env,
            load_replay_buffer=True,
            cfg=cfg,  # Pass config for BC model loading if needed
        )
        print(f"✓ Checkpoint loaded")

        if metadata:
            print(f"  Checkpoint metadata:")
            for key, val in metadata.items():
                print(f"    {key}: {val}")

        # Update model's tensorboard log directory
        model.tensorboard_log = run_info["run_dir"]

        # Setup loggers for training phase
        model.setup_logger(
            logger_config=cfg["logger"],
            run_dir=run_info["run_dir"],
        )

        model.monitor_agent("AFTER LOADING CHECKPOINT")


        # --- Pre-fill Replay Buffer ---
        print("\n=== Pre-filling replay buffer ===")
        prefill_callback = DebugCallback(
            description="prefill-buffer",
            log_steps=(args.debug >= 3),
            render=(args.render and cfg.get("prefill_render", False)),
            progress_bar=True,
            verbose=args.verbose,
        )
        model.prefill_replay_buffer(cfg, callback=prefill_callback)
        print(f"✓ Replay buffer prefilled with {model.replay_buffer.size()} transitions\n")


        # --- Set Up RL Finetuning Learning Rates ---
        print("\n=== Setting up RL finetuning learning rates ===")
        current_step = model.num_timesteps
        finetuning_steps = cfg["total_timesteps"]
        print(f"Current timesteps: {current_step}")
        print(f"Finetuning steps: {finetuning_steps}")

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

        # --- RL Finetuning ---
        print(f"\n=== RL finetuning for {finetuning_steps} steps ===")

        # Create training callback
        train_callback = DebugCallback(
            description=run_info["phase"],
            log_steps=(args.debug >= 3),
            render=args.render,
            verbose=args.verbose,
        )

        periodic_eval_callback = None
        effective_eval_freq = 0
        if args.eval_freq > 0:
            n_envs = env.num_envs if hasattr(env, "num_envs") else 1
            effective_eval_freq = max(args.eval_freq // max(n_envs, 1), 1)
            if effective_eval_freq != args.eval_freq:
                print(f"✓ Periodic eval frequency scaled for VecEnv: {effective_eval_freq} calls")

            periodic_eval_callback = EvalCallback(
                eval_env=eval_env,
                best_model_save_path=None,
                log_path=run_info["run_dir"],
                eval_freq=effective_eval_freq,
                n_eval_episodes=cfg["n_eval_episodes"],
                deterministic=True,
                render=False,
                verbose=args.verbose,
            )
            print(
                f"✓ Periodic eval enabled: every {args.eval_freq} training steps "
                f"({cfg['n_eval_episodes']} episodes)"
            )

        callbacks = [train_callback]
        if periodic_eval_callback is not None:
            callbacks.append(periodic_eval_callback)

        # Run training
        model.learn(
            total_timesteps=finetuning_steps,
            progress_bar=True,
            callback=callbacks,
            reset_num_timesteps=False,  # Continue counting from loaded checkpoint
            log_interval=1,  # Log every episode
        )
        print(f"✓ RL finetuning completed ({finetuning_steps} steps)\n")

        model.monitor_agent("AFTER TRAINING")

        # --- Optional: Evaluate Trained Policy ---
        if args.eval_after:
            print("\n=== Evaluating trained policy ===")
            eval_callback = DebugCallback(
                description="eval-trained",
                log_steps=(args.debug >= 3),
                render=args.render,
                progress_bar=True,
                total=cfg["n_eval_episodes"],
                verbose=args.verbose,
            )

            mean_reward, std_reward = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=cfg["n_eval_episodes"],
                deterministic=True,
                callback=eval_callback,
            )
            eval_callback.close_progress_bar()
            print(f"Trained policy: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}\n")

            # Log to WandB
            if wandb_run is not None:
                wandb.log({
                    "eval/mean_reward": mean_reward,
                    "eval/std_reward": std_reward,
                })

        # --- Save Trained Checkpoint ---
        print("\n=== Saving trained checkpoint ===")
        save_path = get_save_path(cfg, run_info)

        # Save with metadata
        training_metadata = {
            **run_info,
            "total_steps": model.num_timesteps,
            "finetuning_steps": finetuning_steps,
            "loaded_from_checkpoint": args.checkpoint,
            "saved_checkpoint_path": save_path+".zip",
        }

        save_checkpoint_with_metadata(model, save_path, training_metadata)
        print(f"\nTrained checkpoint saved to: {save_path}.zip")
        print(f"Load this checkpoint with eval.py to evaluate or train.py to continue training.\n")

        print("=== Training phase completed successfully ===\n")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"FATAL ERROR: Training failed")
        print(f"Exception: {type(e).__name__}: {e}")
        print(f"{'='*70}\n")
        traceback.print_exc()
        raise

    finally:
        # Finish WandB run
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    args, cfg = get_args_and_cfg()
    main(args, cfg)
