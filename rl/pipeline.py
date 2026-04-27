"""
Full RL training pipeline:
1. **Environment Creation**: Initialize train and eval environments once
2. **Bootstrap**: Load BC policy, initialize SAC model, setup loggers
3. **Prefill**: Populate replay buffer with BC rollouts (once for both phases)
4. **Warmup**: Critic warmup with frozen actor
5. **Finetune**: Initial eval → Actor and critic training with periodic evals → Final eval
6. **Checkpoint**: Save trained model

Key Design:
- Single timestamp for both phases (MMDD_HHMM format)
- Single WandB run spanning both phases (phase-prefixed metrics)
- Model held in memory between phases (no intermediate saves)
- Replay buffer prefilled once, reused through both phases
- Environments created once, reused for both phases
- All evals (pre/periodic/post) logged under finetune/eval/* namespace

Logging:
    Metrics logged with phase prefix:
    - warmup/: critic_loss, etc.
    - finetune/eval/: mean_reward, std_reward, mean_ep_length, etc. (pre, periodic, post)
    - finetune/: actor_loss, critic_loss, etc.
    
    Directories: logs/{group}/{timestamp}_warmup, logs/{group}/{timestamp}_finetune
    WandB: Single run with group, shared timestamp, phase-prefixed metrics

Usage:
    python3 rl/pipeline.py model_configs/rl_sac_cubbies.yaml [--eval_bc] [--overfit 0] [--render]
    tensorboard --logdir ./logs/
"""

import argparse
import os
import shutil
import sys
import traceback
from datetime import datetime
from typing import Optional, Tuple

import torch
import yaml
from dotenv import load_dotenv
import wandb

from avoid_everything.type_defs import DatasetType
from rl.callbacks import DebugCallback, PeriodicEvalMetricsCallback, evaluate_policy_with_metrics
from rl.common import (
    setup_run_directory,
    setup_wandb,
    create_env,
    load_bc_checkpoint,
    bootstrap_sac_from_bc,
    save_checkpoint_with_metadata,
    get_save_path,
)
from rl.lr_schedules import build_lr_schedule
from rl.my_sac import MySAC


def get_args_and_cfg() -> Tuple[argparse.Namespace, dict]:
    """Parse command-line arguments and load configuration."""
    parser = argparse.ArgumentParser(
        description="End-to-End RL Pipeline: Warmup + Finetune + Eval"
    )
    parser.add_argument("cfg_path", help="Path to RL config YAML file")
    parser.add_argument(
        "--eval_bc",
        action="store_true",
        help="Evaluate BC-warmed policy before critic warmup"
    )
    parser.add_argument(
        "--overfit",
        nargs="?",
        type=int,
        const=0,
        default=None,
        metavar="N",
        help="Overfit on trajectory index N (default: 0 if flag used)"
    )
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--verbose", type=int, default=1, help="SB3 verbosity")
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


def run_env_creation(
    args: argparse.Namespace,
    cfg: dict,
) -> Tuple:
    """Create training and evaluation environments (once, reused for both phases)."""
    print("Creating environments...")
    overfit_idx = args.overfit if args.overfit is not None else None
    if overfit_idx is not None:
        print(f"⚠  Overfitting mode: trajectory index {overfit_idx}")

    env = create_env(
        cfg,
        render=args.render,
        dataset_type=DatasetType.TRAIN,
        overfit_idx=overfit_idx,
        env_role="train",
        env_cfg=cfg["env"],
    )
    eval_env = create_env(
        cfg,
        render=args.render,
        dataset_type=DatasetType.TRAIN,
        overfit_idx=overfit_idx,
        env_role="eval",
        env_cfg=cfg["env"],
    )
    return env, eval_env


def run_bootstrap(
    args: argparse.Namespace,
    cfg: dict,
    run_info: dict,
    env,
) -> MySAC:
    """Load BC checkpoint and bootstrap SAC model."""
    print("Loading BC checkpoint...")
    bc_model = load_bc_checkpoint(cfg)

    print("\nBootstrapping SAC from BC...")
    model = bootstrap_sac_from_bc(
        env=env,
        cfg=cfg,
        bc_model=bc_model,
        run_dir=run_info["run_dir"],
        verbose=args.verbose,
        debug=args.debug,
    )

    model.setup_logger(
        logger_config=cfg["logger"],
        run_dir=run_info["run_dir"],
    )
    model.monitor_agent("AFTER BOOTSTRAP")
    
    return model


def run_prefill(
    args: argparse.Namespace,
    cfg: dict,
    model: MySAC,
) -> None:
    """Prefill replay buffer (once, before warmup)."""
    print("\n=== Pre-filling replay buffer (once for both phases) ===")
    prefill_callback = DebugCallback(
        description="prefill-buffer",
        log_steps=(args.debug >= 3),
        render=(args.render and cfg.get("prefill_render", False)),
        progress_bar=True,
        verbose=args.verbose,
    )
    model.prefill_replay_buffer(cfg, callback=prefill_callback)
    print(f"✓ Replay buffer prefilled with {model.replay_buffer.size()} transitions\n")


def run_warmup(
    args: argparse.Namespace,
    cfg: dict,
    run_info: dict,
    model: MySAC,
    eval_env,
) -> None:
    """Run warmup phase: optional BC eval, then critic warmup."""

    # Optional BC eval
    if args.eval_bc:
        print("=== Evaluating BC-warmed policy ===")
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
        print("BC-warmed policy results:")
        for key, value in eval_metrics.items():
            print(f"  {key}: {value:.4f}")
            model.logger.record(f"warmup/eval_bc/{key}", value)
        model.logger.dump(step=model.num_timesteps)

    # Critic warmup
    print(f"\n=== Warming up critic (frozen actor) ===")
    warmup_steps = cfg["critic_warmup_steps"]

    warmup_lr_schedule = build_lr_schedule(
        cfg["critic_warmup_lr"],
        phase_start_step=0,
        phase_total_steps=warmup_steps,
    )
    print(f"✓ Critic warmup LR config: {cfg['critic_warmup_lr']}")
    model.update_learning_rates(critic_schedule=warmup_lr_schedule)

    train_callback = DebugCallback(
        description="warmup",
        log_steps=(args.debug >= 3),
        render=args.render,
        verbose=args.verbose,
    )
    logger_cfg = cfg.get("logger", {})
    log_ep_freq = int(logger_cfg.get("log_ep_freq", logger_cfg.get("log_interval", 1)))
    if log_ep_freq <= 0:
        raise ValueError(f"logger.log_ep_freq must be > 0, got {log_ep_freq}")

    model.set_log_phase_prefix("warmup")
    model.warmup_critic(warmup_steps, [train_callback], log_interval=log_ep_freq)




def run_finetune(
    args: argparse.Namespace,
    cfg: dict,
    run_info: dict,
    model: MySAC,
    eval_env,
) -> None:
    """Run finetune phase: Initial eval → Training → Final eval, all under finetune/eval."""
    print(f"\n{'='*70}")
    print(f"FINETUNE PHASE")
    print(f"{'='*70}")
    print(f"Run ID: {run_info['run_id']}\n")

    # Setup LR schedules for finetune
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
    print(f"Target entropy: {model.target_entropy}")

    model.update_learning_rates(
        actor_schedule=actor_lr_schedule,
        critic_schedule=critic_lr_schedule
    )

    # Setup eval callback (will evaluate at training start, periodically, and at end)
    logger_cfg = cfg.get("logger", {})
    log_eval_freq = int(logger_cfg.get("log_eval_freq", 0))
    if log_eval_freq < 0:
        raise ValueError(f"logger.log_eval_freq must be >= 0, got {log_eval_freq}")

    periodic_eval_callback = None
    if log_eval_freq > 0:
        env = model.get_env()
        n_envs = env.num_envs if hasattr(env, "num_envs") else 1
        effective_eval_freq = max(log_eval_freq // max(n_envs, 1), 1)
        if effective_eval_freq != log_eval_freq:
            print(f"✓ Periodic eval frequency scaled: {effective_eval_freq} calls")

        periodic_eval_callback = PeriodicEvalMetricsCallback(
            eval_env=eval_env,
            eval_freq=effective_eval_freq,
            n_eval_episodes=cfg["n_eval_episodes"],
            deterministic=True,
            verbose=args.verbose,
            logger_prefix="finetune/eval",
            print_prefix="Policy eval",
        )
        print(f"✓ Periodic eval: every {log_eval_freq} steps ({cfg['n_eval_episodes']} eps)")

    # Training (with evals at start, periodic during, and at end via callbacks)
    print(f"\n=== RL finetuning for {finetuning_steps} steps ===")

    train_callback = DebugCallback(
        description="finetune",
        log_steps=(args.debug >= 3),
        render=args.render,
        verbose=args.verbose,
    )

    callbacks = [train_callback]
    if periodic_eval_callback is not None:
        callbacks.append(periodic_eval_callback)

    log_ep_freq = int(logger_cfg.get("log_ep_freq", logger_cfg.get("log_interval", 1)))
    if log_ep_freq <= 0:
        raise ValueError(f"logger.log_ep_freq must be > 0, got {log_ep_freq}")

    model.set_log_phase_prefix("finetune")
    model.learn(
        total_timesteps=finetuning_steps,
        progress_bar=True,
        callback=callbacks,
        reset_num_timesteps=False,
        log_interval=log_ep_freq,
    )
    print(f"✓ RL finetuning completed ({finetuning_steps} steps)")
    model.monitor_agent("AFTER TRAINING")
    print("=== Finetune phase completed ===\n")


def save_final_checkpoint(
    cfg: dict,
    run_info: dict,
    model: MySAC,
) -> None:
    """Save trained checkpoint (only after finetune)."""
    save_path = get_save_path(cfg, run_info)
    model.save(save_path)
    print(f"✓ Model saved to: {save_path}.zip")



def main(args: argparse.Namespace, cfg: dict) -> None:
    """Main orchestration: env creation → bootstrap → prefill → warmup → finetune → eval."""
    # Setup unified run info (phase="" for single run across both phases)
    run_info = setup_run_directory(args.cfg_path, cfg, phase="")
    
    print(f"Experiment: {run_info['group']}")
    print(f"Timestamp: {run_info['timestamp']}")
    print(f"Run dir: {run_info['run_dir']}\n")

    # Create environments once
    env, eval_env = run_env_creation(args, cfg)

    # Initialize WandB (single run for both phases with phase-prefixed metrics)
    wandb_run = setup_wandb(cfg, run_info, tags=["full-pipeline"])


    try:
        # Sequential phases
        model = run_bootstrap(args, cfg, run_info, env)
        run_prefill(args, cfg, model)
        run_warmup(args, cfg, run_info, model, eval_env)
        run_finetune(args, cfg, run_info, model, eval_env)
        save_final_checkpoint(cfg, run_info, model)
        
        print(f"\n{'='*70}")
        print(f"✓ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"FATAL ERROR: Pipeline failed")
        print(f"Exception: {type(e).__name__}: {e}")
        print(f"{'='*70}\n")
        traceback.print_exc()
        raise

    finally:
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    args, cfg = get_args_and_cfg()
    main(args, cfg)
