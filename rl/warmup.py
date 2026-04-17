"""
BC Loading and Critic Warmup Script

This script handles the first phase of RL training:
1. Load pretrained BC policy
2. Bootstrap SAC agent with BC warmup (frozen perception, trainable transformer)
3. Optionally evaluate BC policy before warmup
4. Warm up critic with frozen actor
5. Save warmed checkpoint (actor + critic + replay buffer)

The saved checkpoint can then be loaded by train.py for RL finetuning.

Usage:
    # Basic warmup
    python3 rl/warmup.py model_configs/rl_sac_cubbies.yaml
    
    # With BC evaluation before warmup
    python3 rl/warmup.py model_configs/rl_sac_cubbies.yaml --eval_bc
    
    # Overfit mode for debugging
    python3 rl/warmup.py model_configs/rl_sac_cubbies.yaml --overfit 1
    
    # With rendering
    python3 rl/warmup.py model_configs/rl_sac_cubbies.yaml --render

Output:
    - Warmed checkpoint saved to: {save_path}/{run_name}_warmup_{timestamp}.zip
    - TensorBoard logs: {log_dir}/{run_name}_warmup_{timestamp}/
    - WandB run: {run_name}_warmup_{timestamp} (if enabled)
"""

import argparse
import os
import sys
import traceback

import yaml
import wandb

from avoid_everything.type_defs import DatasetType
from rl.callbacks import DebugCallback, evaluate_policy_with_metrics
from rl.common import (
    setup_run_directory,
    create_env,
    load_bc_checkpoint,
    bootstrap_sac_from_bc,
    setup_wandb,
    save_checkpoint_with_metadata,
    get_save_path,
)
from rl.lr_schedules import build_lr_schedule


def get_args_and_cfg():
    """Parse command-line arguments and load configuration."""
    parser = argparse.ArgumentParser(
        description="BC Loading and Critic Warmup for RL Training"
    )
    parser.add_argument("cfg_path", help="Path to the RL config YAML file")
    parser.add_argument(
        "--eval_bc",
        action="store_true",
        help="Evaluate the BC-warmed policy before critic warmup"
    )
    parser.add_argument(
        "--overfit",
        nargs="?",
        type=int,
        const=0,
        default=None,
        metavar="N",
        help="Overfit on training scene with specific index for debugging (default: 0 if flag is used)"
    )
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--verbose", type=int, default=1, help="SB3 verbosity level")
    parser.add_argument(
        "--debug",
        type=int,
        default=1,
        help="Debug level: 0=None, 1=Summary, 2=Component, 3=Detailed"
    )
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    return args, cfg


def main(args, cfg):
    """Main warmup workflow."""
    # Setup run directory with 'warmup' phase
    run_info = setup_run_directory(args.cfg_path, cfg, phase="warmup")
    print(f"\n=== BC Loading and Critic Warmup ===")
    print(f"Run ID: {run_info['run_id']}")
    print(f"Run directory: {run_info['run_dir']}\n")

    # Initialize WandB if enabled (job_type auto-derived from phase)
    wandb_run = setup_wandb(cfg, run_info, tags=["critic-warmup"])

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
            dataset_type=DatasetType.TRAIN,
            overfit_idx=overfit_idx,
            eval_env=True,
        )
        print("✓ Environments created")

        # --- Load BC Checkpoint ---
        print("\nLoading BC checkpoint...")
        bc_model = load_bc_checkpoint(cfg)

        # --- Bootstrap SAC Agent ---
        print("\nBootstrapping SAC agent from BC...")
        model = bootstrap_sac_from_bc(
            env=env,
            cfg=cfg,
            bc_model=bc_model,
            run_dir=run_info["run_dir"],
            verbose=args.verbose,
            debug=args.debug,
        )

        # Setup loggers
        model.setup_logger(
            logger_config=cfg["logger"],
            run_dir=run_info["run_dir"],
        )

        model.monitor_agent("AFTER BOOTSTRAP")

        # --- Optional: Evaluate BC-Warmed Policy ---
        if args.eval_bc:
            print("\n=== Evaluating BC-warmed policy (before critic warmup) ===")
            eval_callback = DebugCallback(
                description="eval-bc",
                log_steps=(args.debug >= 3),
                render=args.render,
                progress_bar=True,
                total=cfg["n_eval_episodes"],
                verbose=args.verbose,
            )

            eval_metrics = evaluate_policy_with_metrics(
                model=model,
                eval_env=eval_env,
                n_eval_episodes=cfg["n_eval_episodes"],
                deterministic=True,
                debug_callback=eval_callback,
            )
            eval_callback.close_progress_bar()
            print("BC-warm-started policy: ")
            for key, value in eval_metrics.items():
                print(f"  {key}: {value:.4f}")
                model.logger.record(f"eval_bc/{key}", value)
            model.logger.dump(step=model.num_timesteps)


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

        # --- Critic Warmup Phase ---
        print(f"\n=== Warming up critic (frozen actor) ===")
        warmup_steps = cfg["critic_warmup_steps"]
        print(f"Warmup steps: {warmup_steps}")

        # Set up warmup learning rate schedule
        warmup_lr_schedule = build_lr_schedule(
            cfg["critic_warmup_lr"],
            phase_start_step=0,
            phase_total_steps=warmup_steps,
        )
        print(f"✓ Critic warmup LR config: {cfg['critic_warmup_lr']}")
        model.update_learning_rates(critic_schedule=warmup_lr_schedule)

        # Create training callback
        train_callback = DebugCallback(
            description=run_info["phase"],
            log_steps=(args.debug >= 3),
            render=args.render,
            verbose=args.verbose,
        )

        # Run critic warmup
        model.warmup_critic(warmup_steps, [train_callback])
        print(f"✓ Critic warmup completed ({warmup_steps} steps)\n")

        model.monitor_agent("AFTER WARMUP")

        # --- Save Warmed Checkpoint ---
        print("\n=== Saving warmed checkpoint ===")
        save_path = get_save_path(cfg, run_info)

        # Save with metadata
        metadata = {
            **run_info,
            "total_steps": model.num_timesteps,
            "warmup_steps": warmup_steps,
            "loaded_from_checkpoint": cfg["bc_checkpoint_path"],
            "saved_checkpoint_path": save_path+".zip",
        }

        save_checkpoint_with_metadata(model, save_path, metadata)
        print(f"\nWarmed checkpoint saved to: {save_path}.zip")
        print(f"Load this checkpoint with train.py for RL finetuning.\n")

        print("=== Warmup phase completed successfully ===\n")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"FATAL ERROR: Warmup failed")
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
