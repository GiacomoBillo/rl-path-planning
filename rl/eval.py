"""
Policy Evaluation Script

This script evaluates any RL checkpoint:
1. Load checkpoint (BC-warmed, RL finetuned)
2. Run deterministic evaluation
3. Log detailed metrics to console (and optionally to WandB)

Can evaluate checkpoints from:
- warmup.py (BC-warmed policy with warmed critic)
- train.py (trained RL policy)
- agent.py (any saved checkpoint)

Usage:
    # Evaluate warmed checkpoint
    python3 rl/eval.py [config_path] --checkpoint [checkpoint_path]

    optional arguments:
    --n_episodes N          Number of evaluation episodes (overrides config files)
    --render                Render the environment during evaluation
    --overfit N             Evaluate on specific trajectory index (for debugging)
    --wandb                 Enable WandB logging for this eval run
    
Output:
    - Evaluation metrics printed to console
    - TensorBoard logs: {log_dir}/{experiment_group}/{timestamp}_eval/
    - WandB run: {experiment_group}/{timestamp}_eval (if --wandb is used)
"""

import argparse
import sys
import traceback

import yaml
import wandb

from avoid_everything.type_defs import DatasetType
from rl.callbacks import DebugCallback, evaluate_policy_with_metrics
from rl.common import (
    setup_run_directory,
    create_env,
    load_checkpoint_with_metadata,
    setup_wandb,
)


def get_args_and_cfg():
    """Parse command-line arguments and load configuration."""
    parser = argparse.ArgumentParser(
        description="Evaluate RL Policy from Checkpoint"
    )
    parser.add_argument("cfg_path", help="Path to the RL config YAML file")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint to evaluate (with or without .zip extension)"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes (overrides config)"
    )
    parser.add_argument(
        "--overfit",
        nargs="?",
        type=int,
        const=1,
        default=None,
        metavar="N",
        help="Evaluate on specific trajectory index (for debugging)"
    )
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument(
        "--debug",
        type=int,
        default=1,
        help="Debug level: 0=None, 1=Summary, 2=Component, 3=Detailed"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions (default: True)"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB for this eval run (disabled by default)"
    )
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        cfg = yaml.safe_load(f)

    return args, cfg


def main(args, cfg):
    """Main evaluation workflow."""
    # Setup run directory with 'eval' phase
    run_info = setup_run_directory(args.cfg_path, cfg, phase="eval")
    print(f"\n=== Policy Evaluation ===")
    print(f"Run ID: {run_info['run_id']}")
    print(f"Run directory: {run_info['run_dir']}")
    print(f"Evaluating checkpoint: {args.checkpoint}\n")

    # Initialize WandB only when explicitly enabled for eval.
    wandb_run = None
    if args.wandb:
        wandb_run = setup_wandb(cfg, run_info, tags=["evaluation"])
    else:
        print("WandB disabled for this eval run (pass --wandb to enable).")

    try:
        # Override n_episodes if provided
        if args.n_episodes is not None:
            cfg["n_eval_episodes"] = args.n_episodes
            print(f"✓ Overriding n_eval_episodes to {args.n_episodes}")

        # --- Create Evaluation Environment ---
        print("\nCreating evaluation environment...")
        overfit_idx = args.overfit if args.overfit is not None else None
        if overfit_idx is not None:
            print(f"⚠  Evaluating on trajectory index {overfit_idx}")

        eval_env = create_env(
            cfg,
            render=args.render,
            dataset_type=DatasetType.TRAIN,
            overfit_idx=overfit_idx,
            env_role="eval",
            env_cfg=cfg["env"],
        )
        print("✓ Evaluation environment created")

        # --- Load Checkpoint ---
        print("\nLoading checkpoint...")
        model, checkpoint_metadata = load_checkpoint_with_metadata(
            args.checkpoint,
            cfg,
            env=eval_env,
            load_replay_buffer=False,
        )
        print("✓ Checkpoint loaded")
        if checkpoint_metadata:
            print(f"  Checkpoint metadata:")
            for key, val in checkpoint_metadata.items():
                print(f"    {key}: {val}")

        model.setup_logger(
            logger_config=cfg["logger"],
            run_dir=run_info["run_dir"],
        )

        # --- Run Evaluation ---
        print(f"\n=== Running evaluation ({cfg['n_eval_episodes']} episodes) ===")
        
        eval_debug_callback = DebugCallback(
            description="eval",
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
            deterministic=args.deterministic,
            debug_callback=eval_debug_callback,
        )
        eval_debug_callback.close_progress_bar()

        for key, value in eval_metrics.items():
            model.logger.record(f"eval/{key}", value)
        model.logger.dump(step=model.num_timesteps)

        # --- Print Results ---
        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Episodes: {cfg['n_eval_episodes']}")
        print(f"Mean Reward: {eval_metrics['mean_reward']:.4f}")
        print(f"Std Reward: {eval_metrics['std_reward']:.4f}")
        print(f"Mean Episode Length: {eval_metrics['mean_ep_length']:.4f}")
        print(f"Std Episode Length: {eval_metrics['std_ep_length']:.4f}")
        print(f"Target Reached Rate: {eval_metrics['target_reached_rate']:.4f}")
        print(f"Mean Num Collisions: {eval_metrics['mean_num_collisions']:.4f}")
        print(f"Mean Position Error: {eval_metrics['mean_position_error']:.4f}")
        print(f"Mean Orientation Error: {eval_metrics['mean_orientation_error']:.4f}")
        print(f"Mean Action Abs (|delta q|): {eval_metrics['mean_episode_action_abs']:.4f}")
        print(f"{'='*70}\n")

        # --- Log to WandB ---
        if wandb_run is not None:
            wandb.log({
                f"eval/{key}": value for key, value in eval_metrics.items()
            })
        print("=== Evaluation completed successfully ===\n")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"FATAL ERROR: Evaluation failed")
        print(f"Exception: {type(e).__name__}: {e}")
        print(f"{'='*70}\n")
        traceback.print_exc()
        raise

    finally:
        # Finish WandB run
        if 'wandb_run' in locals() and wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    args, cfg = get_args_and_cfg()
    main(args, cfg)
