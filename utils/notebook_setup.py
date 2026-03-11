"""
Initialization and utility functions for Jupyter notebooks.

This module provides reusable functions for setting up common components:
- Python path configuration for robofin package
- Configuration loading from YAML files
- Robot initialization from URDF
- Dataset loading (validation, training, test)
- Visualization client setup
- Import verification

Usage:
    from utils.notebook_setup import (
        setup_notebook_environment,
        load_configuration,
        load_robot,
        load_datasets,
        setup_viz_client,
        verify_imports
    )
    
    # In a notebook:
    setup_notebook_environment()
    verify_imports()
    config = load_configuration("/workspace/model_configs/evaluation.yaml")
    robot = load_robot(config['shared_parameters']['urdf_path'])
    val_dataset = load_datasets(robot, config, dataset_types=['val'])
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Standard library imports
from pathlib import Path
import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader

# IPython/Jupyter imports
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# Project imports
import viz_client
from robofin.robots import Robot
from avoid_everything.data_loader import TrajectoryDataset
from avoid_everything.type_defs import DatasetType
from utils import visualization


def setup_notebook_environment(verbose=True) -> None:
    """
    Configure Python path for robofin package and enable IPython features.
    
    This function:
    - Adds robofin to PYTHONPATH and sys.path
    - Enables IPython autoreload (if in Jupyter environment)
    
    Args:
        robofin_path: Path to robofin directory. If None, looks for robofin
                     relative to current working directory (default: os.getcwd()/robofin).
    """
    robofin_path = os.path.join(os.getcwd(), 'robofin')
    robofin_path = os.path.abspath(robofin_path)
    
    # Update PYTHONPATH environment variable for robofin
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if robofin_path not in current_pythonpath:
        os.environ['PYTHONPATH'] = (
            f"{robofin_path}:{current_pythonpath}" 
            if current_pythonpath 
            else robofin_path
        )
    
    # Add to sys.path for immediate effect
    if robofin_path not in sys.path:
        sys.path.insert(0, robofin_path)
    
    # Enable IPython autoreload if available
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic('load_ext', 'autoreload')
            ipython.run_line_magic('autoreload', '2')
    except (ImportError, AttributeError):
        pass  # Not in Jupyter/IPython environment

    if verbose:
        print(f"✓ Notebook environment setup complete.")


def get_common_imports():
    """
    Return commonly used modules for notebooks.
    
    Returns:
        dict: Dictionary containing all common imports with descriptive keys
        
    Use:
        >>> imports = get_common_imports()
        >>> global.update(imports)
    """    
    return {
        # Standard imports
        'torch': torch,
        'np': np,
        'Path': Path,
        'DataLoader': DataLoader,
        
        # IPython/Jupyter
        'widgets': widgets,
        'display': display,
        'clear_output': clear_output,
        'HTML': HTML,
        
        # Project imports
        'Robot': Robot,
        'TrajectoryDataset': TrajectoryDataset,
        'DatasetType': DatasetType,
        'visualization': visualization,
        'viz_client': viz_client,
    }


def load_configuration(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
    
    Returns:
        Dictionary containing the loaded configuration.
    
    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Configuration loaded from: {config_path}")
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML config at {config_path}: {e}")


def load_robot(urdf_path: str, device: str = 'cpu') -> Robot:
    """
    Initialize a Robot instance from a URDF file.
    
    Args:
        urdf_path: Path to the URDF file.
        device: Device to load robot on ('cpu' or 'cuda'). Default: 'cpu'.
    
    Returns:
        Initialized Robot instance.
    
    Raises:
        FileNotFoundError: If URDF file does not exist.
    """    
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    
    robot = Robot(str(urdf_path), device=device)
    print(f"✓ Robot loaded from URDF: {urdf_path}")
    
    return robot


def load_dataset(
    robot,
    config: Dict[str, Any],
    dataset_type: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load trajectory dataset based on configuration.
    
    Args:
        robot: Robot instance (from load_robot).
        config: Configuration dictionary (from load_configuration).
        dataset_types: List of dataset types to load. Options: 'train', 'val', 'test'.
                      If None, loads ['val'].
    
    Returns:
        Dictionary mapping dataset type names to TrajectoryDataset instances.
        E.g., {'val': TrajectoryDataset(...), 'train': TrajectoryDataset(...)}
    
    Raises:
        ImportError: If avoid_everything is not installed.
        FileNotFoundError: If data directory does not exist.
    """
    
    # Get parameters from config
    data_params = config.get('data_module_parameters', {})
    shared_params = config.get('shared_parameters', {})
    
    data_dir = data_params.get('data_dir')
    if not data_dir or not Path(data_dir).exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Map dataset type strings to DatasetType enum
    dataset_type_map = {
        'train': DatasetType.TRAIN,
        'val': DatasetType.VAL,
        'test': DatasetType.TEST,
    }
    
    if dataset_type not in dataset_type_map:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}. "
            f"Must be one of {list(dataset_type_map.keys())}"
        )
    
    enum_type = dataset_type_map[dataset_type]
    trajectory_key = data_params.get(f'{dataset_type}_trajectory_key', 'expert_trajectories')
    
    dataset = TrajectoryDataset.load_from_directory(
        robot=robot,
        directory=data_dir,
        dataset_type=enum_type,
        trajectory_key=trajectory_key,
        num_robot_points=shared_params.get('num_robot_points', 2048),
        num_obstacle_points=data_params.get('num_obstacle_points', 2048),
        num_target_points=data_params.get('num_target_points', 1024),
        random_scale=data_params.get('random_scale', random_scale),
    )
    
    print(f"✓ Loaded {dataset_type} dataset")
    
    return dataset


