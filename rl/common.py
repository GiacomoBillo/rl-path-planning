"""
Shared utilities for RL training pipeline.

This module contains reusable functions used across different training phases:
- warmup.py: BC loading + critic warmup
- train.py: RL finetuning
- eval.py: Policy evaluation
- agent.py: End-to-end training (backward compatibility)

Key functions:
- setup_run_directory(): Create timestamped run directory
- create_env(): Create training or evaluation environment
- load_bc_checkpoint(): Load pretrained BC policy
- bootstrap_sac_from_bc(): Initialize SAC agent with BC warmup
- setup_wandb(): Initialize WandB logging
- save_checkpoint(): Save model with metadata
- load_checkpoint(): Load model from checkpoint
"""

import os
import shutil
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import torch
import yaml
from dotenv import load_dotenv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb

from avoid_everything.pretraining import PretrainingMotionPolicyTransformer
from avoid_everything.type_defs import DatasetType
from rl.environment import AvoidEverythingEnv
from rl.feature_extractor import MPiFormerExtractor
from rl.my_sac import MySAC


def setup_run_directory(cfg_path: str, cfg: dict, phase: str = "") -> dict:
    """
    Create run directory and save command, args, and config for reproducibility.
    
    Uses experiment_group for unified organization across WandB and filesystem.
    Directory structure: logs/{experiment_group}/{timestamp}_{phase}/
    
    Args:
        cfg_path: Path to the config YAML file
        cfg: Loaded config dictionary
        phase: Phase name (e.g., 'warmup', 'finetune', 'eval')
        
    Returns:
        Dictionary with run metadata:
        - group: Experiment group name
        - phase: Phase name
        - timestamp: Short timestamp string (MMDD_HHMM)
        - run_name: Name for this specific run ({timestamp}_{phase})
        - run_id: Globally unique ID ({group}_{timestamp}_{phase})
        - run_dir: Path to the run directory
    """
    # Short timestamp format: MMDD_HHMM
    timestamp = datetime.now().strftime("%m%d_%H%M")
    
    # Get experiment group from config (required)
    experiment_group = cfg["logger"].get("experiment_group")
    if not experiment_group:
        raise ValueError("Config must specify logger.experiment_group")
    
    # Build run name: {timestamp}_{phase}
    if phase:
        run_name = f"{timestamp}_{phase}"
    else:
        run_name = timestamp
    
    # Build globally unique run_id: {group}_{timestamp}_{phase}
    run_id = f"{experiment_group}_{run_name}"
    
    # Create directory: logs/{group}/{timestamp}_{phase}/
    log_dir = cfg["logger"]["log_dir"]
    group_dir = os.path.join(log_dir, experiment_group)
    run_dir = os.path.join(group_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Copy original config file to preserve comments and formatting
    config_dest = os.path.join(run_dir, "config.yaml")
    shutil.copy2(cfg_path, config_dest)
    print(f"✓ Config copied to: {config_dest}")
    
    # Save command to text file for reproducibility
    command = " ".join(sys.argv)
    command_file = os.path.join(run_dir, "command.txt")
    with open(command_file, "w") as f:
        f.write(command + "\n")
    print(f"✓ Command saved to: {command_file}")
    
    run_info = {
        "group": experiment_group,
        "phase": phase,
        "timestamp": timestamp,
        "run_name": run_name,
        "run_id": run_id,
        "run_dir": run_dir,
    }
    return run_info


def create_env(
    cfg: dict,
    render: bool = False,
    dataset_type: DatasetType = DatasetType.TRAIN,
    trajectory_key: str = None,
    overfit_idx: Optional[int] = None,
) -> DummyVecEnv:
    """
    Create and configure an AvoidEverything environment.
    
    Args:
        cfg: Configuration dictionary
        render: Whether to render the environment
        dataset_type: Type of dataset (TRAIN or VAL)
        trajectory_key: Key for trajectory in dataset (defaults to cfg['train_trajectory_key'] or cfg['val_trajectory_key'])
        overfit_idx: Optional index for overfitting mode
        
    Returns:
        Vectorized environment wrapped with Monitor and DummyVecEnv
    """
    render_mode = "human" if render else None
    render_backend = "ros"
    
    # Monitor with info_keywords to capture episode-level metrics
    info_keywords = (
        "target_reached", "collision", "position_error", "orientation_error",
        "episode_num_collisions", "episode_num_steps", "episode_return",
        "TimeLimit.truncated", "episode_limit_violation_sum"
    )
    
    env = Monitor(
        AvoidEverythingEnv(render_mode=render_mode, render_backend=render_backend),
        info_keywords=info_keywords
    )
    
    # Set up dataset
    if trajectory_key is None:
        trajectory_key = cfg["train_trajectory_key"] if dataset_type == DatasetType.TRAIN else cfg["val_trajectory_key"]
    
    env.unwrapped.set_scene_generation_from_dataset(
        data_dir=cfg["data_dir"],
        trajectory_key=trajectory_key,
        dataset_type=dataset_type,
        num_workers=cfg["num_workers"],
        random_scale=0.0,  # No noise for RL (clean states)
        overfit_idx=overfit_idx,
    )
    
    # Wrap with vectorized environment
    env = DummyVecEnv([lambda: env])
    
    return env


def load_bc_checkpoint(cfg: dict) -> PretrainingMotionPolicyTransformer:
    """
    Load pretrained BC policy from checkpoint.
    
    Args:
        cfg: Configuration dictionary containing BC checkpoint path and parameters
        
    Returns:
        Loaded BC model on appropriate device
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bc_model = PretrainingMotionPolicyTransformer.load_from_checkpoint(
        cfg["bc_checkpoint_path"],
        urdf_path=cfg["urdf_path"],
        num_robot_points=cfg["num_robot_points"],
        action_chunk_length=cfg["action_chunk_length"],
        **cfg["bc_checkpoint_parameters"],
    ).to(device)
    print(f"✓ BC checkpoint loaded from: {cfg['bc_checkpoint_path']}")
    return bc_model


def bootstrap_sac_from_bc(
    env: DummyVecEnv,
    cfg: dict,
    bc_model: PretrainingMotionPolicyTransformer,
    run_dir: str,
    verbose: int = 1,
    debug: int = 1,
) -> MySAC:
    """
    Bootstrap SAC agent with BC policy warm-start.
    
    Creates a MySAC agent with:
    - Frozen BC perception encoder
    - Trainable transformer encoder
    - Actor mu initialized from BC action decoder
    - Explicit log_std initialization
    
    Args:
        env: Vectorized training environment
        cfg: Configuration dictionary
        bc_model: Pretrained BC model
        run_dir: Directory for TensorBoard logs
        verbose: Verbosity level for SAC
        debug: Debug verbosity level for MySAC
        
    Returns:
        Initialized MySAC agent with BC warm-start
    """
    # SAC model with split BC feature extractor
    # 1) Shared frozen backbone (PointNet++ + joint encoder) via MPiFormerBackbone
    # 2) Per-instance trainable transformer via MPiFormerTransformerExtractor
    # 3) Linear-only heads (net_arch={"pi": [], "qf": []})
    
    policy_kwargs = {
        "features_extractor_class": MPiFormerExtractor,
        "features_extractor_kwargs": {
            "bc_model": bc_model,
            "pc_bounds": cfg["bc_checkpoint_parameters"]["pc_bounds"],
            "freeze_perception": True,  # Freeze perception encoder
            "freeze_transformer": False,  # Transformer is trainable
            "deep_copy_perception": True,
            "deep_copy_transformer": True,
        },
        "net_arch": {
            "pi": [],  # Actor: just Linear(d_model, robot_dof)
            "qf": []   # Critic: just Linear(d_model + action_dim, 1)
        },
        "share_features_extractor": False,  # Each (actor, critic) gets separate transformer
        "log_std_init": cfg["log_std_init"],
    }
    
    model = MySAC(
        "MultiInputPolicy",
        env,
        tensorboard_log=run_dir,
        force_deterministic=True,
        debug_verbose=debug,
        verbose=verbose,
        seed=cfg["seed"],
        buffer_size=cfg["buffer_size"],
        learning_starts=0,  # Start training immediately with BC policy
        batch_size=cfg["batch_size"],
        ent_coef=cfg["entropy_coef"],
        policy_kwargs=policy_kwargs,
        train_freq=(1, "step"),
    )
    
    vec_env = model.get_env()
    print(f"Train will run with n_envs={vec_env.num_envs if vec_env is not None else 'unknown'}")

    # Warm-start actor mean head from BC action_decoder
    with torch.no_grad():
        model.policy.actor.mu.weight.copy_(bc_model.action_decoder.weight)
        model.policy.actor.mu.bias.copy_(bc_model.action_decoder.bias)
    print("✓ Warm-started actor mu from BC action_decoder")
    
    # Explicit log_std initialization
    log_std_init_value = policy_kwargs.get("log_std_init", -20.0)
    model.initialize_log_std(log_std_value=log_std_init_value, state_independent_start=True)

    # Cleanup to save memory
    if "bc_model" in model.policy_kwargs.get("features_extractor_kwargs", {}):
        del model.policy_kwargs["features_extractor_kwargs"]["bc_model"]
    del bc_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ Freed original BC model memory")

    return model


def setup_wandb(cfg: dict, run_info: dict, tags: list = None) -> Optional[wandb.sdk.wandb_run.Run]:
    """
    Initialize WandB logging if enabled in config.
    
    Uses experiment_group as WandB group for organizing related runs.
    Uses phase as job_type automatically.
    
    Args:
        cfg: Configuration dictionary
        run_info: Run metadata from setup_run_directory()
        tags: Additional tags for the run
        
    Returns:
        WandB run object if initialized, None otherwise
    """
    logger_config = cfg.get("logger", {})
    formats = logger_config.get("formats", [])
    
    if "wandb" not in formats:
        return None
    
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
        return None
    
    wandb_entity = logger_config.get("wandb_entity") or os.getenv("WANDB_ENTITY")
    
    # Patch TensorBoard for automatic syncing
    wandb.tensorboard.patch(root_logdir=run_info["run_dir"])
    
    # Use phase as job_type
    job_type = run_info.get("phase", "")
    
    # Prepare tags
    run_tags = []
    if job_type:
        run_tags.append(job_type)
    if tags:
        run_tags.extend(tags)
    
    # Initialize WandB run
    # group: experiment_group for organizing related runs
    # job_type: phase (warmup, finetune, eval)
    # name: {timestamp}_{phase} for this specific run
    # id: globally unique {group}_{timestamp}_{phase}
    wandb_run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        group=run_info["group"],          # Experiment group
        job_type=job_type or None,        # Phase: warmup, train, eval, full
        id=run_info["run_id"],             # Globally unique ID
        name=run_info["run_name"],         # Display name
        config=cfg,
        sync_tensorboard=True,
        reinit="finish_previous",
        tags=run_tags,
    )
    print(f"✓ WandB initialized:")
    print(f"    project={wandb_project}, entity={wandb_entity}")
    print(f"    group={run_info['group']}, job_type={job_type}")
    print(f"    run_name={run_info['run_name']}, run_id={wandb_run.id}")
    print(f"    tags={run_tags}")
    
    return wandb_run


def get_save_path(cfg: dict, run_info: dict) -> str:
    """
    Get checkpoint save path based on experiment group and run info.
    
    Creates directory structure: {save_dir}/{group}/{timestamp}_{phase}
    
    Args:
        cfg: Configuration dictionary
        run_info: Run metadata from setup_run_directory()
        
    Returns:
        Full path for checkpoint (without .zip extension)
    """
    save_dir = cfg.get("save_dir", "./checkpoints")
    group = run_info["group"]
    run_name = run_info["run_name"]
    
    # Create group directory
    group_dir = os.path.join(save_dir, group)
    os.makedirs(group_dir, exist_ok=True)
    
    # Return full checkpoint path: checkpoints/{group}/{timestamp}_{phase}
    save_path = os.path.join(group_dir, run_name)
    return save_path


def save_checkpoint_with_metadata(
    model: MySAC,
    save_path: str,
    metadata: Dict[str, Any] = None,
) -> None:
    """
    Save SAC model checkpoint with additional metadata.
    
    Args:
        model: MySAC model to save
        save_path: Path to save checkpoint (without .zip extension)
        metadata: Optional metadata dictionary to save alongside checkpoint
    """
    # Save model using SB3's native save (creates .zip file)
    model.save(save_path)
    print(f"✓ Model saved to: {save_path}.zip")
    
    # Save additional metadata if provided
    if metadata:
        metadata_path = f"{save_path}_metadata.yaml"
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False)
        print(f"✓ Metadata saved to: {metadata_path}")


def load_checkpoint_with_metadata(
    checkpoint_path: str,
    env: Optional[DummyVecEnv] = None,
    load_replay_buffer: bool = False,
) -> Tuple[MySAC, Optional[Dict[str, Any]]]:
    """
    Load SAC model checkpoint with metadata.
    
    Args:
        checkpoint_path: Path to checkpoint (with or without .zip extension)
        env: Optional environment (if None, loads without environment)
        load_replay_buffer: Whether to load replay buffer from checkpoint
        
    Returns:
        Tuple of (loaded model, metadata dict or None)
    """
    # Remove .zip extension if provided
    if checkpoint_path.endswith(".zip"):
        checkpoint_path = checkpoint_path[:-4]
    
    # Load model
    custom_objects = {"learning_rate": 0.0}  # Placeholder, will be updated by LR schedule
    
    if env is not None:
        model = MySAC.load(
            checkpoint_path,
            env=env,
            custom_objects=custom_objects,
            load_replay_buffer=load_replay_buffer,
        )
    else:
        model = MySAC.load(
            checkpoint_path,
            custom_objects=custom_objects,
        )
    
    print(f"✓ Model loaded from: {checkpoint_path}.zip")
    print(f"Model on device: {model.device}")
    
    # Try to load metadata
    metadata = None
    metadata_path = f"{checkpoint_path}_metadata.yaml"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
        print(f"✓ Metadata loaded from: {metadata_path}")
    
    return model, metadata
