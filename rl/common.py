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
import zipfile
import json
import base64
import pickle
import math

import torch
import yaml
from dotenv import load_dotenv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
import wandb

from avoid_everything.pretraining import PretrainingMotionPolicyTransformer
from avoid_everything.type_defs import DatasetType
from rl.environment import AvoidEverythingEnv, validate_max_delta_q
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


def _create_single_env(
    cfg: dict,
    env_cfg: dict,
    *,
    render: bool = False,
    dataset_type: DatasetType = DatasetType.TRAIN,
    trajectory_key: Optional[str] = None,
    overfit_idx: Optional[int] = None,
    eval_env: bool = False,
    env_idx: int = 0,
    total_env_number: int = 1,
    max_delta_q: float = 1.0,
) -> Monitor:
    """Create one Monitor-wrapped environment instance."""
    render_mode = "human" if render else None
    render_backend = "ros"

    # Monitor with info_keywords to capture episode-level metrics
    info_keywords = (
        "target_reached", "collision", "position_error", "orientation_error",
        "episode_num_collisions", "episode_num_steps", "episode_return",
        "TimeLimit.truncated", 
        "episode_limit_violation_rate", "episode_action_clip_violation_rate",
        "episode_action_abs_mean",
    )

    env = Monitor(
        AvoidEverythingEnv(
            render_mode=render_mode,
            render_backend=render_backend,
            terminate_ep_on_collision=env_cfg.get("terminate_ep_on_collision", True),
            reward_config=env_cfg.get("reward"),
            action_delta_clip=env_cfg.get("action_delta_clip", max_delta_q),
            max_delta_q=max_delta_q,
        ),
        info_keywords=info_keywords
    )

    # Set up dataset
    if trajectory_key is None:
        trajectory_key = cfg["train_trajectory_key"] if dataset_type == DatasetType.TRAIN else cfg["val_trajectory_key"]

    env.unwrapped.set_scene_generation_from_dataset(
        data_dir=cfg["data_dir"],
        trajectory_key=trajectory_key,
        dataset_type=dataset_type,
        num_workers=env_cfg.get("num_workers", 0),
        random_scale=0.0,  # No noise for RL (clean states)
        overfit_idx=overfit_idx,
        n_eval_episodes=cfg["n_eval_episodes"] if eval_env else None,
        env_idx=env_idx,
        total_env_number=total_env_number,
    )

    return env


def create_env(
    cfg: dict,
    render: bool = False,
    dataset_type: DatasetType = DatasetType.TRAIN,
    trajectory_key: Optional[str] = None,
    overfit_idx: Optional[int] = None,
    env_role: str = "train",
    env_cfg: Optional[dict] = None,
) -> VecEnv:
    """
    Create and configure a vectorized AvoidEverything environment.

    Keeps Gymnasium API at the environment level and delegates VecEnv API handling
    to SB3 wrappers (DummyVecEnv/SubprocVecEnv).

    Args:
        cfg: Configuration dictionary
        render: Whether to render the environment
        dataset_type: Type of dataset (TRAIN or VAL)
        trajectory_key: Key for trajectory in dataset
        overfit_idx: Optional index for overfitting mode
        env_role: Role-specific config selector ("train" or "eval")

    Returns:
        SB3 VecEnv instance (DummyVecEnv or SubprocVecEnv)
    """
    env_cfg = env_cfg if env_cfg is not None else cfg.get("env", {})
    if not isinstance(env_cfg, dict):
        raise ValueError("env_cfg must be a dictionary.")

    n_envs = int(env_cfg.get(f"{env_role}_n_envs", env_cfg.get("n_envs", 1)))
    if n_envs < 1:
        raise ValueError(f"{env_role}_n_envs must be >= 1, got {n_envs}")

    vec_env_type = str(
        env_cfg.get(f"{env_role}_vec_env_type", env_cfg.get("vec_env_type", "dummy"))
    ).lower()
    if vec_env_type not in {"dummy", "subproc"}:
        raise ValueError(
            f"Unsupported vec_env_type={vec_env_type!r}. Expected 'dummy' or 'subproc'."
        )
    
    max_delta_q = validate_max_delta_q(env_cfg.get("max_delta_q", 1.0))
    print(f"✓ Environment with max_delta_q={max_delta_q}")

    # Rendering with subprocess workers is fragile for ROS/PyBullet visualization.
    if render and vec_env_type == "subproc":
        print(
            f"⚠  {env_role}: render=True is incompatible with SubprocVecEnv in this setup; "
            "falling back to DummyVecEnv."
        )
        vec_env_type = "dummy"

    dataset_num_workers = int(env_cfg.get("num_workers", 0))
    if vec_env_type == "subproc":
        # Avoid nested multiprocessing by default (SubprocVecEnv workers + DataLoader workers).
        dataset_num_workers = int(env_cfg.get("subproc_dataset_num_workers", 0))
        if dataset_num_workers == 0 and int(env_cfg.get("num_workers", 0)) != 0:
            print(
                f"⚠  {env_role}: using subproc_dataset_num_workers=0 to avoid nested multiprocessing."
            )

    def make_env_fn(env_idx: int):
        def _init():
            env = _create_single_env(
                cfg,
                {
                    **env_cfg,
                    "num_workers": dataset_num_workers,
                },
                render=render,
                dataset_type=dataset_type,
                trajectory_key=trajectory_key,
                overfit_idx=overfit_idx,
                env_idx=env_idx,
                total_env_number=n_envs,
                eval_env=(env_role == "eval"),
                max_delta_q=max_delta_q,
            )
            return env

        return _init

    env_fns = [make_env_fn(i) for i in range(n_envs)]
    if vec_env_type == "subproc":
        start_method = env_cfg.get("subproc_start_method", None)
        env: VecEnv = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        env = DummyVecEnv(env_fns)

    # Single source of truth for env seeding: top-level cfg["seed"].
    base_seed = cfg.get("seed", None)
    if base_seed is not None:
        # Note: env seed also set in SB3 BaseAlgorithm, but duplicated here for safety
        env.seed(int(base_seed))  

    print(
        f"✓ Created {env_role} VecEnv: type={vec_env_type}, n_envs={n_envs}, "
        f"dataset_num_workers={dataset_num_workers}"
    )
    return env


def load_bc_checkpoint(cfg: dict, verbose=True) -> PretrainingMotionPolicyTransformer:
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
    if verbose:
        print(f"✓ BC checkpoint loaded from: {cfg['bc_checkpoint_path']}")
    return bc_model


def scale_actor_for_action_bounds(model: MySAC, action_scale_factor: float) -> None:
    """
    Scale actor parameters to preserve physical action magnitude after action-space rescaling.

    Applies:
      - mu.weight and mu.bias multiplied by action_scale_factor
      - log_std bias/parameter shifted by log(action_scale_factor)
    """
    factor = float(action_scale_factor)
    if factor <= 0:
        raise ValueError(f"action_scale_factor must be > 0, got {factor}.")
    if math.isclose(factor, 1.0):
        return

    noise_offset = math.log(factor)
    actor = model.policy.actor
    with torch.no_grad():
        actor.mu.weight.mul_(factor)
        actor.mu.bias.mul_(factor)

        log_std_head = actor.log_std
        if hasattr(log_std_head, "bias") and log_std_head.bias is not None:
            log_std_head.bias.add_(noise_offset)
        elif isinstance(log_std_head, torch.nn.Parameter):
            log_std_head.add_(noise_offset)
        else:
            raise TypeError(
                f"Unsupported actor.log_std type for scaling: {type(log_std_head).__name__}"
            )


def bootstrap_sac_from_bc(
    env: VecEnv,
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
    
    features_extractor_kwargs = {
        "bc_model": bc_model,
        "pc_bounds": cfg["bc_checkpoint_parameters"]["pc_bounds"],
        "freeze_perception": cfg.get("freeze_perception", True),  # Freeze perception encoder
        "freeze_transformer": cfg.get("freeze_transformer", False),  # Transformer is trainable
        "freeze_transformer_layers": cfg.get("freeze_transformer_layers", None), 
        "deep_copy_perception": True,
        "deep_copy_transformer": True,
    }

    policy_kwargs = {
        "features_extractor_class": MPiFormerExtractor,
        "features_extractor_kwargs": features_extractor_kwargs,
        "net_arch": {
            "pi": [],  # Actor: just Linear(d_model, robot_dof)
            "qf": []   # Critic: just Linear(d_model + action_dim, 1)
        },
        "share_features_extractor": False,  # Each (actor, critic) gets separate transformer
        "log_std_init": cfg["log_std_init"],
    }

    raw_freq = cfg["train_freq"]
    parsed_train_freq = tuple(raw_freq) if isinstance(raw_freq, list) else raw_freq
    
    model = MySAC(
        "MultiInputPolicy",
        env,
        tensorboard_log=run_dir,
        debug_verbose=debug,
        verbose=verbose,
        seed=cfg["seed"],
        buffer_size=cfg["buffer_size"],
        learning_starts=0,  # Start training immediately with BC policy
        batch_size=cfg["batch_size"],
        gradient_steps=cfg["gradient_steps"],
        ent_coef=cfg["entropy_coef"],
        target_entropy=cfg["target_entropy"],
        policy_kwargs=policy_kwargs,
        train_freq=parsed_train_freq,
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

    # Scale actor outputs to match env.max_delta_q while keeping physical action magnitude.
    action_scale_factor = 1.0 / validate_max_delta_q(cfg.get("env", {}).get("max_delta_q", 1.0))
    if not math.isclose(action_scale_factor, 1.0):
        scale_actor_for_action_bounds(model, action_scale_factor)
        print(f"✓ Applied actor action_scale_factor={action_scale_factor:.6g}")

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
    run_tags = list(logger_config.get("wandb_tags", []))
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
    cfg: dict,
    env: Optional[VecEnv] = None,
    load_replay_buffer: bool = False,
) -> Tuple[MySAC, Optional[Dict[str, Any]]]:
    """
    Load SAC model checkpoint with metadata.
    
    Args:
        checkpoint_path: Path to checkpoint (with or without .zip extension)
        env: Optional environment (if None, loads without environment)
        load_replay_buffer: Whether to load replay buffer from checkpoint
        cfg: config dict (used to load BC model, old checkpoint, for creating the structure of the network)
        
    Returns:
        Tuple of (loaded model, metadata dict or None)
    """
    # Remove .zip extension if provided
    if checkpoint_path.endswith(".zip"):
        checkpoint_path = checkpoint_path[:-4]
    
    
    # Load BC model and prepare policy_kwargs with bc_model injected
    bc_model = load_bc_checkpoint(cfg, verbose=False)
    
    # Load existing policy_kwargs from checkpoint and inject bc_model    
    with zipfile.ZipFile(checkpoint_path + ".zip", 'r') as archive:
        data_bytes = archive.read('data')
        data = json.loads(data_bytes)
        
        # Get policy_kwargs (it's serialized)
        pk_json = data.get('policy_kwargs', {})
        if ':serialized:' in pk_json:
            serialized = base64.b64decode(pk_json[':serialized:'])
            policy_kwargs = pickle.loads(serialized)
        else:
            policy_kwargs = pk_json
    
    # Inject bc_model into features_extractor_kwargs
    if 'features_extractor_kwargs' not in policy_kwargs:
        policy_kwargs['features_extractor_kwargs'] = {}
    policy_kwargs['features_extractor_kwargs']['bc_model'] = bc_model
    
    # Pass the complete policy_kwargs in custom_objects to override the saved one
    custom_objects = {
        "policy_kwargs": policy_kwargs,
    }
    
    # Load model
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
    
    # Cleanup BC model after loading (same as bootstrap_sac_from_bc)
    if "bc_model" in model.policy_kwargs.get("features_extractor_kwargs", {}):
        del model.policy_kwargs["features_extractor_kwargs"]["bc_model"]
    del bc_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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
