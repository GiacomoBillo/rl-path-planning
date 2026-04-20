"""
RL environment for reach traget while avoiding obstacles.
Custom Gymnasium environment for training the RL agent.
"""

import random
import importlib
from typing import Literal, Any, Optional
from collections.abc import Sequence
import numpy as np
import gymnasium as gym
import numba as nb
import torch
import time

from torch.utils.data import DataLoader
from gymnasium.utils.env_checker import check_env
from gymnasium.wrappers import TimeLimit
from geometrout.primitive import Cuboid, CuboidArray, Cylinder, CylinderArray
from stable_baselines3.common.env_checker import check_env as check_env_sb3

from robofin.robots import Robot
from robofin.samplers import NumpyRobotSampler
from robofin.old.collision import FrankaCollisionSpheres
from robofin.collision import CollisionSpheres
from avoid_everything.geometry import TorchCuboids, TorchCylinders
from avoid_everything.data_loader import StateDataset, TrajectoryDataset
from avoid_everything.type_defs import DatasetType
from torch.utils.data import Subset
from pathlib import Path

import viz_client
from utils.visualization import visualize_problem

BYPASS_LAZY_ROS_RENDER = True # if True, always publish the full scene on each render call instead of only on the first call per episode. 



def build_reward_config(reward_config: dict[str, Any] | None) -> dict[str, float]:
    reward_config = reward_config or {}
    if not isinstance(reward_config, dict):
        raise ValueError("reward_config must be a dict when provided.")

    total_magnitude = float(reward_config.get("total_magnitude", 1.0))
    goal_weight = float(reward_config.get("goal_weight", 1.0))
    collision_weight = float(reward_config.get("collision_weight", -1.0))
    goal_distance_weight = float(reward_config.get("goal_distance_weight", 0.0))

    if total_magnitude <= 0:
        raise ValueError("reward.total_magnitude must be > 0.")
    if goal_weight < 0:
        raise ValueError("reward.goal_weight must be >= 0.")
    if collision_weight > 0:
        raise ValueError("reward.collision_weight must be <= 0.")
    if goal_distance_weight < 0:
        raise ValueError("reward.goal_distance_weight must be >= 0.")

    return {
        "total_magnitude": total_magnitude,
        "goal_weight": goal_weight,
        "collision_weight": collision_weight,
        "goal_distance_weight": goal_distance_weight,
    }


def build_action_delta_clip(action_delta_clip: float | Sequence[float], dof: int) -> np.ndarray:
    """Build per-joint hard-clip bounds from scalar or sequence config."""
    if isinstance(action_delta_clip, (int, float)):
        clip = float(action_delta_clip)
        if clip <= 0:
            raise ValueError("action_delta_clip must be > 0.")
        return np.full((dof,), clip, dtype=np.float32)

    clip_array = np.asarray(action_delta_clip, dtype=np.float32)
    if clip_array.shape != (dof,):
        raise ValueError(f"action_delta_clip sequence must have shape {(dof,)}, got {clip_array.shape}.")
    if np.any(clip_array <= 0):
        raise ValueError("All action_delta_clip values must be > 0.")
    return clip_array


class AvoidEverythingEnv(TimeLimit):
    """
    Public environment: wraps `_AvoidEverythingEnv` with Gymnasium's `TimeLimit`.

    Accepts all `_AvoidEverythingEnv` constructor arguments plus `max_episode_steps`.
    Episodes are truncated (not terminated) once the step limit is reached.

    Example::

        env = AvoidEverythingEnv(dataloader=dl, urdf_path="assets/panda/panda.urdf",
                                  max_episode_steps=200)
    """

    def __init__(
        self,
        dataloader: DataLoader = None,
        urdf_path: str = "assets/panda/panda.urdf",
        num_robot_points: int = 2048,
        num_obstacle_points: int = 4096,
        num_target_points: int = 128,
        collision_mode: Literal["franka", "spheres", "torch"] = "franka",
        scene_buffer: float = 0.0,
        position_threshold: float = 0.01, # meters
        orientation_threshold: float = 15.0, # degrees
        render_mode: Literal["human", "rgb_array"] | None = None,
        render_backend: Literal["ros", "pybullet"] | None = None,
        render_fps: float | None = None,
        max_episode_steps: int = 100,
        highlight_collisions: bool = True,
        terminate_ep_on_collision: bool = True,
        reward_config: dict | None = None,
        action_delta_clip: float | Sequence[float] = 0.2,
    ):
        """Initialize environment with an automatic TimeLimit wrapper.

        Args:
            dataloader: DataLoader providing batches of problems (start config, obstacles, target)
            urdf_path: path to robot URDF file
            num_robot_points: number of points to sample on the robot for the point cloud observation
            num_obstacle_points: number of points to sample on the obstacles for the point cloud observation
            num_target_points: number of points to sample on the target for the point cloud observation
            collision_mode: which collision checker to use ("franka", "spheres", or "torch")
            scene_buffer: additional buffer distance for collision checking (in meters)
            position_threshold: max end-effector position error to consider target reached (meters, default 0.01)
            orientation_threshold: max end-effector orientation error to consider target reached (degrees, default 15)
            render_mode: Gymnasium render mode ("human", "rgb_array", or None). None disables rendering.
            render_backend: rendering engine ("ros" or "pybullet"). None auto-selects.
            render_fps: target frame rate for rendering (None = as fast as possible)
            max_episode_steps: maximum number of steps per episode before truncation (default 50)
            highlight_collisions: whether to highlight the robot in red when in collision (only applies to render_backend='ros')
            terminate_ep_on_collision: whether collisions terminate the episode (default True)
            reward_config: optional reward configuration block.
            action_delta_clip: hard clip bound(s) for joint deltas.
        """
        # instantiate the base env
        env = _AvoidEverythingEnv(
            dataloader=dataloader,
            urdf_path=urdf_path,
            num_robot_points=num_robot_points,
            num_obstacle_points=num_obstacle_points,
            num_target_points=num_target_points,
            collision_mode=collision_mode,
            scene_buffer=scene_buffer,
            position_threshold=position_threshold,
            orientation_threshold=orientation_threshold,
            render_mode=render_mode,
            render_backend=render_backend,
            render_fps=render_fps,
            highlight_collisions=highlight_collisions,
            terminate_ep_on_collision=terminate_ep_on_collision,
            reward_config=reward_config,
            action_delta_clip=action_delta_clip,
        )
        # apply time limit wrapper
        super().__init__(env, max_episode_steps=max_episode_steps)
    
    def step(self, action):
        """Override step to add TimeLimit.truncated key for SB3 Monitor compatibility."""
        observation, reward, terminated, truncated, info = super().step(action)
        # Add TimeLimit.truncated key that SB3's Monitor expects (Gymnasium 1.0+ doesn't add this automatically)
        info["TimeLimit.truncated"] = truncated and not terminated
        return observation, reward, terminated, truncated, info


@nb.jit(nopython=True)
def _seed_numba_rng(seed: int):
    """Seed Numba's internal numpy RNG (separate from Python's np.random)."""
    np.random.seed(seed)


def _compute_target_errors(
    eef_pose: torch.Tensor,
    target_position: np.ndarray,
    target_orientation: np.ndarray,
) -> tuple[float, float]:
    """Compute position and orientation errors between an EEF pose and the target.

    Mirrors the math in rope.py::check_reaching_success().

    Args:
        eef_pose: (1, 4, 4) tensor — end-effector pose from robot.fk_torch()
        target_position: (3,) array — target EEF position in metres
        target_orientation: (3, 3) array — target EEF rotation matrix

    Returns:
        position_error: float, L2 distance in metres
        orientation_error: float, geodesic angle in degrees
    """
    target_pos_t = torch.tensor(target_position, dtype=torch.float32)          # (3,)
    target_rot_t = torch.tensor(target_orientation, dtype=torch.float32)        # (3, 3)

    position_error: float = torch.linalg.vector_norm(eef_pose[0, :3, -1] - target_pos_t).item()

    R = torch.matmul(eef_pose[0, :3, :3], target_rot_t.T)                      # (3, 3)
    trace = R.diagonal().sum()
    cos_value = torch.clamp((trace - 1) / 2, -1.0, 1.0)
    orientation_error: float = torch.abs(torch.rad2deg(torch.acos(cos_value))).item()

    return position_error, orientation_error


class _AvoidEverythingEnv(gym.Env):
    """
    Gymnasium environment: the robot must reach a target pose while avoiding obstacles.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": None}

    def __init__(
        self,
        dataloader: DataLoader = None,  # Dataset object
        urdf_path: str = "assets/panda/panda.urdf", # URDF path for the robot
        num_robot_points: int = 2048,
        num_obstacle_points: int = 4096,
        num_target_points: int = 128,
        collision_mode: Literal["franka", "spheres", "torch"] = "franka",
        scene_buffer: float = 0.0,
        position_threshold: float = 0.01,
        orientation_threshold: float = 15.0,
        render_mode: Literal["human", "rgb_array"] | None = None,
        render_backend: Literal["ros", "pybullet"] | None = None,
        render_fps: float | None = None,
        highlight_collisions: bool = True,
        terminate_ep_on_collision: bool = True,
        reward_config: dict | None = None,
        action_delta_clip: float | Sequence[float] = 0.2,
    ):
        """Initialize environment.
        Args:
            dataloader: DataLoader providing batches of problems (start config, obstacles, target)
            urdf_path: path to robot URDF file
            num_robot_points: number of points to sample on the robot for the point cloud observation
            num_obstacle_points: number of points to sample on the obstacles for the point cloud observation
            num_target_points: number of points to sample on the target for the point cloud observation
            collision_mode: which collision checker to use ("franka", "spheres", or "torch")
            scene_buffer: additional buffer distance for collision checking (in meters)
            position_threshold: max end-effector position error to consider target reached (meters, default 0.01)
            orientation_threshold: max end-effector orientation error to consider target reached (degrees, default 15)
            render_mode: Gymnasium render mode ("human", "rgb_array", or None). None disables rendering.
            render_backend: rendering engine ("ros" or "pybullet"). None auto-selects:
                "ros" for render_mode="human", "pybullet" for render_mode="rgb_array".
                render_backend="ros" is incompatible with render_mode="rgb_array".
            highlight_collisions: whether to highlight the robot in red when in collision (only applies to render_backend='ros')
            terminate_ep_on_collision: whether collisions terminate the episode (default True)
            reward_config: optional reward configuration block.
            action_delta_clip: hard clip bound(s) for joint deltas.
        """
        super().__init__()

        # Store dataset reference
        self.dataloader: DataLoader = dataloader
        self._dataloader_iter = iter(dataloader) if dataloader is not None else None

        # Store URDF path
        self.urdf_path = urdf_path

        # Initialize robot
        self.robot = Robot(urdf_path)

        # Point cloud parameters
        self.num_robot_points = num_robot_points
        self.num_obstacle_points = num_obstacle_points
        self.num_target_points = num_target_points

        # Initialize sampler for point cloud generation
        # NumpyRobotSampler() as in avoid_everything/data_loader.py
        self.robot_sampler = NumpyRobotSampler(
            self.robot,
            num_robot_points=num_robot_points,
            num_eef_points=num_target_points,
            use_cache=True, # TODO: why cache?
            with_base_link=True,
        )

        # Collision checking
        self.collision_mode = collision_mode
        self.scene_buffer = scene_buffer
        if collision_mode == "franka":
            self._collision_checker = FrankaCollisionSpheres()
        elif collision_mode == "spheres":
            self._collision_checker = CollisionSpheres(self.robot)
        elif collision_mode == "torch":
            self._collision_checker = None  # built per-episode in reset()
        else:
            raise ValueError(f"Unknown collision_mode: {collision_mode!r}. Choose 'franka', 'spheres', or 'torch'.")
        # Cached per-episode scene primitives (populated in reset())
        self._scene_primitives = None

        # Target-reaching thresholds (AvoidEverything default values)
        self.position_threshold = position_threshold       # metres
        self.orientation_threshold = orientation_threshold  # degrees

        # State space, state variables
        self.robot_config = None
        self.target_position = None
        self.target_orientation = None
        self.obstacles = None  

        # Observation space: point clouds + point labels + robot current configuration
        total_points = num_robot_points + num_obstacle_points + num_target_points
        self.observation_space = gym.spaces.Dict(
            { 
                #   pc_bounds: [[-1.5, -1.5, -0.1], [1.5, 1.5, 1.5]] from training config
                "point_cloud": gym.spaces.Box(
                    low=np.broadcast_to(np.array([-1.5, -1.5, -0.1], dtype=np.float32), (total_points, 3)), 
                    high=np.broadcast_to(np.array([1.5, 1.5, 1.5], dtype=np.float32), (total_points, 3)), 
                    dtype=np.float32
                ),
                "point_cloud_labels": gym.spaces.Box(0, 2, shape=(total_points, 1), dtype=np.int32),
                "configuration": gym.spaces.Box(-1.0, 1.0, shape=(self.robot.MAIN_DOF,), dtype=np.float32),
                # "target_position": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )

        # Action space: normalized joint deltas [-1, 1] for main DOF
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.robot.MAIN_DOF,), dtype=np.float32,)
        self.action_delta_clip = build_action_delta_clip(action_delta_clip, self.robot.MAIN_DOF)
        self.reward_config = build_reward_config(reward_config)
        self.terminate_ep_on_collision = terminate_ep_on_collision

        # Render setup
        self._render_setup(render_mode, render_backend, render_fps)
        self.highlight_collisions = highlight_collisions

    def reset(self, seed=None, options=None):
        """Reset environment
        sample random initial robot config, obstacle config and target config
        If a dataset is provided: load random problem from dataset."""

        # fix random seed
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            _seed_numba_rng(seed)

        # Sample random problem from dataloader
        if self.dataloader is not None:
            if self._dataloader_iter is None:
                self._dataloader_iter = iter(self.dataloader)
            try:
                batch = next(self._dataloader_iter)
            except StopIteration:
                self._dataloader_iter = iter(self.dataloader)
                batch = next(self._dataloader_iter)
            # Data loader returns a batch dict with tensors of shape (batch_size, ...)
            # Extract and squeeze single sample from batch
            self.problem = {key: val.squeeze(0).numpy() if hasattr(val, 'numpy') else val for key, val in batch.items()}

            self.robot_config = self.problem["configuration"]
            self.target_position = self.problem["target_position"]
            self.target_orientation = self.problem["target_orientation"]

            # Cache scene primitives for collision checking
            self._scene_primitives = self._build_scene_primitives()
            # Mark static scene as not yet published so render() will publish obstacles + ghost EEF once
            self._static_scene_published = False
        
        # Generate random problem
        else:
            raise NotImplementedError("Random problem generation not implemented yet. Please provide a dataloader with problems.")

        obs = self._get_obs()
        info = {} # TODO: add info if needed
        self.collision = False
        self.episode_num_steps = 0
        self.episode_num_collisions = 0
        self.episode_return = 0.0  # cumulative reward
        self.episode_limit_violation_count = 0  # number of clipped joint-components due to config limits
        self.episode_action_clip_violation_count = 0  # number of clipped joint-components due to action delta clip
        self.episode_action_abs_sum = np.zeros(self.robot.MAIN_DOF, dtype=np.float32)  # cumulative abs(delta q) per joint
        return obs, info

    def step(self, action):
        """Execute action: update config, check collision, compute reward.

        Args:
            action (np.array): array of shape (MAIN_DOF,) with values in [-1, 1], representing normalized joint deltas.
                Executed deltas are clipped to +/- action_delta_clip.

        Returns:
            obs (dict): dict with keys "point_cloud", "point_cloud_labels", "configuration"
            reward (float): scalar reward for this step
            terminated (bool): whether episode is done
            truncated (bool): whether episode is truncated
            info (dict): dict with additional info
        """
        action = np.asarray(action, dtype=np.float32)
        # check action dimension
        assert action.shape == (self.robot.MAIN_DOF,), f"Expected action shape {(self.robot.MAIN_DOF,)}, got {action.shape}"

        # Clip per-step deltas before applying them
        clipped_action = np.clip(action, -self.action_delta_clip, self.action_delta_clip)
        action_clip_violations = np.abs(action - clipped_action)
        self.episode_action_clip_violation_count += int(np.count_nonzero(action_clip_violations > 0))

        # Apply clipped action to configuration
        upclipped_config = self.robot_config + clipped_action
        # Clip action to make robot config in valid bounds
        clipped_config = np.clip(upclipped_config, -1.0, 1.0)
        self.robot_config = clipped_config
        self.episode_num_steps += 1
        denom = float(self.episode_num_steps * self.robot.MAIN_DOF)

        # Action magnitude tracking (per-joint abs(delta q))
        action_abs = np.abs(clipped_action)
        self.episode_action_abs_sum += action_abs
        episode_action_abs_mean = self.episode_action_abs_sum / self.episode_num_steps
        episode_action_clip_violation_rate = self.episode_action_clip_violation_count / denom

        # joint limit violation
        limit_violations = np.abs(clipped_config - upclipped_config)  # How much was the action clipped?
        # Accumulate the total magnitude of violations for this episode
        self.episode_limit_violation_count += int(np.count_nonzero(limit_violations > 0))
        episode_limit_violation_rate = self.episode_limit_violation_count / denom
        # TODO: penalize if action push joint config out of bounds? reward

        # Check for collision and target reaching
        collision = self._check_collision()
        self.collision = collision
        self.episode_num_collisions += int(collision)
        target_reached, pos_err, orien_err = self._check_target_reached()
        
        # Get new observation, compute reward, check termination and truncation
        obs = self._get_obs()
        reward, reward_terms = self._compute_reward(collision, target_reached, pos_err, orien_err)
        self.episode_return += reward
        terminated = self._check_terminated_ep(collision, target_reached)
        truncated = False # truncation applyed via TimeLimit wrapper, not handled here
        info = {
                # single step metrics
                "collision": collision,
                "target_reached": target_reached, 
                "position_error": pos_err, 
                "orientation_error": orien_err, 
                "action_clip_violations": action_clip_violations,
                "limit_violations": limit_violations,
                # episode metrics
                "episode_num_collisions": self.episode_num_collisions,
                "episode_num_steps": self.episode_num_steps,
                "episode_return": self.episode_return,
                "episode_action_clip_violation_rate": episode_action_clip_violation_rate,
                "episode_limit_violation_rate": episode_limit_violation_rate,
                "episode_action_abs_sum": self.episode_action_abs_sum,
                "episode_action_abs_mean": episode_action_abs_mean,
                "reward": reward,
                "reward_terms": reward_terms,
                }

        return obs, reward, terminated, truncated, info

    def _check_terminated_ep(self, collision: bool, target_reached: bool) -> bool:
        """Return whether the current episode should terminate."""
        if target_reached:
            return True
        if self.terminate_ep_on_collision and collision:
            return True
        return False

    def _get_obs(self):
        """Construct point cloud observation."""
        # Sample robot points at current configuration
        new_robot_points = self.robot_sampler.sample(self.robot.unnormalize_joints(self.robot_config))[:,:3] # (num_robot_points, 3)
        assert new_robot_points.shape == (self.num_robot_points, 3)

        # Keep obstacle points and target points fixed across the episode problem
        # Overwrite new robot cloud point
        point_cloud = self.problem["point_cloud"] # (num_robot_points + num_obstacle_points + num_target_points, 3)
        point_cloud[:self.num_robot_points] = new_robot_points
        point_cloud_labels = self.problem["point_cloud_labels"].astype(np.int32) # (num_robot_points + num_obstacle_points + num_target_points, 1)

        obs = {
            "point_cloud": point_cloud,
            "point_cloud_labels": point_cloud_labels,
            "configuration": self.robot_config,
        }
        return obs

    def _build_scene_primitives(self):
        """Build and cache scene primitives from self.problem for collision checking.

        Returns a mode-specific structure:
        - "franka": list of [CuboidArray, CylinderArray] (fast Numba scene_sdf)
        - "spheres": list of geometrout Cuboid and Cylinder objects (generic sdf)
        - "torch": dict with TorchCuboids and TorchCylinders tensors
        """
        cuboid_centers = self.problem["cuboid_centers"]    # (M1, 3)
        cuboid_dims    = self.problem["cuboid_dims"]        # (M1, 3)
        cuboid_quats   = self.problem["cuboid_quats"]       # (M1, 4) w,x,y,z

        cylinder_centers = self.problem["cylinder_centers"]  # (M2, 3)
        cylinder_radii   = self.problem["cylinder_radii"]    # (M2, 1)
        cylinder_heights = self.problem["cylinder_heights"]  # (M2, 1)
        cylinder_quats   = self.problem["cylinder_quats"]    # (M2, 4)

        if self.collision_mode in ("franka", "spheres"):
            cuboids = [
                Cuboid(cuboid_centers[i], cuboid_dims[i], cuboid_quats[i])
                for i in range(len(cuboid_centers))
                if np.all(cuboid_dims[i] > 0)  # skip zero-volume cuboids
            ]
            cylinders = [
                Cylinder(cylinder_centers[i], cylinder_radii[i, 0], cylinder_heights[i, 0], cylinder_quats[i])
                for i in range(len(cylinder_centers))
                if cylinder_radii[i, 0] > 0 and cylinder_heights[i, 0] > 0
            ]
            if self.collision_mode == "franka":
                primitives = []
                if cuboids:
                    primitives.append(CuboidArray(cuboids))
                if cylinders:
                    primitives.append(CylinderArray(cylinders))
                return primitives  # list of CuboidArray / CylinderArray
            else:  # "spheres"
                return cuboids + cylinders  # list of Cuboid / Cylinder

        else:  # "torch"
            return {
                "cuboids": TorchCuboids(
                    torch.tensor(cuboid_centers, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(cuboid_dims,    dtype=torch.float32).unsqueeze(0),
                    torch.tensor(cuboid_quats,   dtype=torch.float32).unsqueeze(0),
                ),
                "cylinders": TorchCylinders(
                    torch.tensor(cylinder_centers,              dtype=torch.float32).unsqueeze(0),
                    torch.tensor(cylinder_radii,                dtype=torch.float32).unsqueeze(0),
                    torch.tensor(cylinder_heights,              dtype=torch.float32).unsqueeze(0),
                    torch.tensor(cylinder_quats,                dtype=torch.float32).unsqueeze(0),
                ),
            }

    def _check_collision(self):
        """Check if current config is in collision with obstacles or itself.

        Uses self.collision_mode to select the checker:
          "franka"  — FrankaCollisionSpheres from original/old robofin (hard-coded Franka spheres, Numba-jit SDF, self-collision)
          "spheres" — CollisionSpheres from custom/generalized robofin (URDF-based generic spheres, self-collision)
          "torch"   — robot.compute_spheres + TorchCuboids/TorchCylinders (consistent with training metric, no self-collision)
        """
        if self._scene_primitives is None:
            return False

        config = self.robot.unnormalize_joints(self.robot_config)

        if self.collision_mode == "franka":
            # Use the finger opening value from robot_config.yaml via auxiliary_joint_defaults.
            # FrankaCollisionSpheres takes a single float for both fingers (they are mirrored).
            prismatic_joint = self.robot.auxiliary_joint_defaults.get("panda_finger_joint1", 0.04)
            return self._collision_checker.franka_arm_collides_fast(
                config,
                prismatic_joint,
                self._scene_primitives,  # list of CuboidArray / CylinderArray
                scene_buffer=self.scene_buffer,
                check_self=True,
            )

        elif self.collision_mode == "spheres":
            return self._collision_checker.robot_collides(
                config,
                auxiliary_joint_values=None,
                primitives=self._scene_primitives,  # list of Cuboid / Cylinder
                scene_buffer=self.scene_buffer,
                check_self=True,
            )

        else:  # "torch"
            # TODO: self-collision missing
            config_tensor = torch.tensor(config, dtype=torch.float32).unsqueeze(0)  # (1, MAIN_DOF)
            cuboids   = self._scene_primitives["cuboids"]
            cylinders = self._scene_primitives["cylinders"]
            collision_spheres = self.robot.compute_spheres(config_tensor)
            for radii, sphere_centers in collision_spheres:
                sdf_values = torch.minimum(
                    cuboids.sdf(sphere_centers),
                    cylinders.sdf(sphere_centers),
                )
                if torch.any(sdf_values <= radii + self.scene_buffer):
                    return True
            return False


    def _check_target_reached(self):
        """Check if current end-effector pose is within position and orientation thresholds of the target."""
        config = torch.tensor(
            self.robot.unnormalize_joints(self.robot_config), dtype=torch.float32
        ).unsqueeze(0)  # (1, MAIN_DOF)
        eef_pose = self.robot.fk_torch(config, link_name=self.robot.tcp_link_name)
        assert isinstance(eef_pose, torch.Tensor)
        pos_err, orien_err = _compute_target_errors(
            eef_pose, self.target_position, self.target_orientation
        )
        target_reached = pos_err < self.position_threshold and orien_err < self.orientation_threshold
        return target_reached, pos_err, orien_err


    def _compute_reward(
        self,
        collision: bool,
        target_reached: bool,
        position_error: float,
        orientation_error: float,
    ) -> tuple[float, dict[str, float]]:
        
        goal_term = self.reward_config["goal_weight"] if target_reached and not collision else 0.0
        collision_term = self.reward_config["collision_weight"] if collision else 0.0

        # gaussian distance reward
        # =0 when error in inf, =1 when error is 0, with gaussian smooth decay in between
        # with position and orientation errors normalized by thresholds
        goal_distance_term = \
            self.reward_config["goal_distance_weight"] \
            * np.exp(-(position_error / self.position_threshold)**2) \
            * np.exp(-(orientation_error / self.orientation_threshold)**2)

        reward_terms = {
            "goal": goal_term,
            "collision": collision_term,
            "goal_distance": goal_distance_term,
        }
        reward = self.reward_config["total_magnitude"] * sum(reward_terms.values())
        return reward, reward_terms

    def _render_setup(
        self,
        render_mode: str | None,
        render_backend: Literal["ros", "pybullet"] | None,
        render_fps: float | None,
    ) -> None:
        """Resolve and store render configuration; initialize lazy backend handles."""
        self.render_mode = render_mode
        if render_mode is None:
            self._render_backend = None
        elif render_backend is not None:
            if render_mode == "rgb_array" and render_backend == "ros":
                raise ValueError("render_backend='ros' does not support render_mode='rgb_array'.")
            self._render_backend = render_backend
        elif render_mode == "human":
            self._render_backend = "ros"
        elif render_mode == "rgb_array":
            self._render_backend = "pybullet"
        else:
            raise ValueError(f"Unknown render_mode: {render_mode!r}. Choose 'human', 'rgb_array', or None.")

        # backend-agnostic flag: static scene (obstacles + target marker) published once per episode
        self._static_scene_published: bool = False
        # PyBullet lazy handles — initialised on first render() call
        self._pybullet_module = None
        self._pb_client = None
        self._pb_robot_id = None
        self._pb_joint_map: dict = {}
        self._pb_obstacle_ids: list = []

        self.metadata["render_fps"] = render_fps
        self._last_render_time = None
        self._render_period = 1.0 / render_fps if render_fps is not None else None

        # connect to viz server
        if self._render_backend == "ros":
            viz_client.shutdown()
            viz_client.connect(str(self.robot.urdf_path))
            viz_client.clear_ghost_robot()
            viz_client.clear_obstacles()
            viz_client.clear_ghost_end_effector()

    def _get_pybullet_module(self):
        """Lazy-load and cache PyBullet only when the pybullet backend is actually used."""
        if self._pybullet_module is None:
            try:
                self._pybullet_module = importlib.import_module("pybullet")
            except ImportError as exc:
                raise ImportError(
                    "PyBullet rendering was requested, but `pybullet` is not installed. "
                    "Install it with `pip install pybullet` or switch to render_backend='ros'."
                ) from exc
        return self._pybullet_module


    def render(self):
        """Render the environment according to render_mode and render_backend.

        If render_fps was set at initialization, this method will block as needed to
        enforce the target frames-per-second (blocking throttle). This is simple and
        suitable for manual visualization/debugging, but will slow down any caller
        that calls env.render() frequently (e.g., training loops).
        """
        if self.render_mode is None:
            return None

        # Blocking to match render FPS 
        if self.metadata["render_fps"] is not None:
            now = time.perf_counter()
            if self._last_render_time is not None:
                elapsed = now - self._last_render_time
                to_sleep = self._render_period - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)

        if self._render_backend == "ros":
            result = self._render_ros()
        elif self._render_backend == "pybullet":
            result = self._render_pybullet()

        # Update last render timestamp after performing render
        self._last_render_time = time.perf_counter()
        return result

    def _render_ros(self):
        """Render via viz_client → RViz / Foxglove (render_mode='human', render_backend='ros').

        Uses visualize_problem() from utils/visualization.py to publish the full scene
        (obstacles + ghost EEF + robot config) once per episode, then only updates the
        robot config on each subsequent call (obstacles and target are static per episode).
        """

        # highlight collision
        if self.highlight_collisions and self.collision:
            viz_client.publish_ghost_robot(self.robot.unnormalize_joints(self.robot_config), color=[0.9, 0.1, 0.1], alpha=0.6)
        else:
            viz_client.clear_ghost_robot()

        # if not self._static_scene_published:
        if BYPASS_LAZY_ROS_RENDER or not self._static_scene_published: # if pybass, always publish full scene
            # Publish full scene: ghost EEF at target, obstacles, robot at current config
            problem_with_current = {**self.problem, "configuration": self.robot_config}
            visualize_problem(self.robot, problem_with_current)
            self._static_scene_published = True
        else:
            # Only the robot moved — cheaply update its configuration
            viz_client.publish_config(self.robot.unnormalize_joints(self.robot_config))

        return None

    def _render_pybullet(self):
        """Render via PyBullet (render_mode='human'→GUI window, 'rgb_array'→ndarray)."""
        _pybullet = self._get_pybullet_module()
        # Lazy initialisation
        if self._pb_client is None:
            mode = _pybullet.GUI if self.render_mode == "human" else _pybullet.DIRECT
            self._pb_client = _pybullet.connect(mode)
            _pybullet.setGravity(0, 0, 0, physicsClientId=self._pb_client)
            self._pb_robot_id = _pybullet.loadURDF(
                self.urdf_path,
                useFixedBase=True,
                physicsClientId=self._pb_client,
            )
            # Build joint-name → joint-index map
            for i in range(_pybullet.getNumJoints(self._pb_robot_id, physicsClientId=self._pb_client)):
                info = _pybullet.getJointInfo(self._pb_robot_id, i, physicsClientId=self._pb_client)
                self._pb_joint_map[info[1].decode()] = i

        # Update robot joint states
        unnorm_config = self.robot.unnormalize_joints(self.robot_config)
        for joint_name, angle in zip(self.robot.main_joint_names, unnorm_config):
            if joint_name in self._pb_joint_map:
                _pybullet.resetJointState(
                    self._pb_robot_id, self._pb_joint_map[joint_name], float(angle),
                    physicsClientId=self._pb_client,
                )

        # Create static obstacles and target marker once per episode
        if not self._static_scene_published:
            obstacle_rgba = [0.8, 0.5, 0.2, 0.8]
            for center, dims, quat_wxyz in zip(
                self.problem["cuboid_centers"],
                self.problem["cuboid_dims"],
                self.problem["cuboid_quats"],
            ):
                if not np.all(dims > 0):
                    continue
                quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
                vis = _pybullet.createVisualShape(
                    _pybullet.GEOM_BOX, halfExtents=(dims / 2).tolist(),
                    rgbaColor=obstacle_rgba, physicsClientId=self._pb_client,
                )
                body = _pybullet.createMultiBody(
                    baseVisualShapeIndex=vis,
                    basePosition=center.tolist(), baseOrientation=quat_xyzw,
                    physicsClientId=self._pb_client,
                )
                self._pb_obstacle_ids.append(body)

            for center, radius, height, quat_wxyz in zip(
                self.problem["cylinder_centers"],
                self.problem["cylinder_radii"],
                self.problem["cylinder_heights"],
                self.problem["cylinder_quats"],
            ):
                r, h = float(radius), float(height)
                if r <= 0 or h <= 0:
                    continue
                quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
                vis = _pybullet.createVisualShape(
                    _pybullet.GEOM_CYLINDER, radius=r, length=h,
                    rgbaColor=obstacle_rgba, physicsClientId=self._pb_client,
                )
                body = _pybullet.createMultiBody(
                    baseVisualShapeIndex=vis,
                    basePosition=center.tolist(), baseOrientation=quat_xyzw,
                    physicsClientId=self._pb_client,
                )
                self._pb_obstacle_ids.append(body)

            # target marker: small sphere at target_position
            try:
                tpos = np.array(self.target_position, dtype=np.float32)
                tvis = _pybullet.createVisualShape(_pybullet.GEOM_SPHERE, radius=0.02, rgbaColor=[0.2, 0.8, 0.2, 0.9], physicsClientId=self._pb_client)
                tbody = _pybullet.createMultiBody(baseVisualShapeIndex=tvis, basePosition=tpos.tolist(), physicsClientId=self._pb_client)
                self._pb_obstacle_ids.append(tbody)
            except Exception:
                # If target_position missing or invalid, skip target marker
                pass

            self._static_scene_published = True

        if self.render_mode == "rgb_array":
            width, height_px = 640, 480
            view_matrix = _pybullet.computeViewMatrix(
                cameraEyePosition=[1.5, 1.5, 1.5],
                cameraTargetPosition=[0.0, 0.0, 0.5],
                cameraUpVector=[0.0, 0.0, 1.0],
                physicsClientId=self._pb_client,
            )
            proj_matrix = _pybullet.computeProjectionMatrixFOV(
                fov=60, aspect=width / height_px, nearVal=0.1, farVal=10.0,
                physicsClientId=self._pb_client,
            )
            _, _, rgb, _, _ = _pybullet.getCameraImage(
                width, height_px, view_matrix, proj_matrix,
                physicsClientId=self._pb_client,
            )
            return np.array(rgb, dtype=np.uint8)[:, :, :3]

        return None

    def close(self):
        """Clean up render backends and call super().close()."""
        if self._render_backend == "ros" and viz_client is not None and viz_client.is_connected():
            viz_client.shutdown()

        if self._pb_client is not None:
            try:
                pybullet_module = self._pybullet_module
                if pybullet_module is not None:
                    pybullet_module.disconnect(self._pb_client)
            except Exception:
                pass
            self._pb_client = None
            self._pb_robot_id = None
            self._pb_joint_map = {}
            self._pb_obstacle_ids = []

        super().close()

    def set_scene_generation_from_dataset(
        self,
        data_dir: str,
        trajectory_key: str,
        dataset_type: DatasetType,
        num_workers: int = 0,
        random_scale: float = 0.0,
        overfit_idx: Optional[int] = None,
        n_eval_episodes: Optional[int] | None = None,
        shuffle: bool = True,
        env_idx: int = 0,
        total_env_number: int = 1,
    ):
        """Set up scene generation from a TrajectoryDataset.

        Creates a TrajectoryDataset using the environment's robot instance and sets it
        as the dataloader. This ensures RL episodes start from trajectory initial states (q0),
        matching the expert demonstration start distribution.

        Args:
            data_dir: Path to the dataset directory
            trajectory_key: Key for the trajectory data (e.g., "global_solutions")
            dataset_type: Type of dataset (DatasetType.TRAIN, DatasetType.VAL, etc.)
            num_workers: Number of workers for the DataLoader (default: 0)
            random_scale: Standard deviation of random noise to add to joints (default: 0.0)
            overfit_idx: If provided, only use this specific trajectory index (for overfitting mode)
            n_eval_episodes: Number of episodes to evaluate (for evaluation environments) to always evaluate the same scenes
            shuffle: Whether to shuffle samples in DataLoader
            env_idx: Current environment index for deterministic dataset splitting
            total_env_number: Total number of environments used in parallel
        """
        # Create TrajectoryDataset using environment's robot instance
        dataset = TrajectoryDataset.load_from_directory(
            robot=self.robot,
            directory=Path(data_dir),
            trajectory_key=trajectory_key,
            num_robot_points=self.num_robot_points,
            num_obstacle_points=self.num_obstacle_points,
            num_target_points=self.num_target_points,
            dataset_type=dataset_type,
            random_scale=random_scale,
        )

        shuffle = True
        # Handle overfit mode: create subset with single trajectory
        if overfit_idx is not None:
            dataset = Subset(dataset, [overfit_idx])
        elif n_eval_episodes:
            dataset = Subset(dataset, list(range(n_eval_episodes)))
            shuffle = False

        # split dataset across parallel envs
        if overfit_idx is None and total_env_number > 1:
            if env_idx < 0 or env_idx >= total_env_number:
                raise ValueError(
                    f"env_idx must be in [0, {total_env_number - 1}], got {env_idx}"
                )
            total_len = len(dataset)
            len_split = (total_len + total_env_number - 1) // total_env_number
            start = env_idx * len_split
            end = min(start + len_split, len(dataset))
            if start >= total_len:
                raise ValueError(
                    f"Dataset split for env_idx={env_idx} is empty "
                    f"(total_len={total_len}, total_env_number={total_env_number})."
                )
            indices_split = list(range(start, end))
            dataset = Subset(dataset, indices_split)
            print(
                f"✓ Environment {env_idx}/{total_env_number} using split indices "
                f"[{start}:{end}] out of {total_len} total samples"
            )

        # Create DataLoader for episode resets.
        self.dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        self._dataloader_iter = iter(self.dataloader)

    def get_num_split_samples(self) -> int:
        """Return number of samples in this env split."""
        if self.dataloader is None:
            return 0
        return len(self.dataloader.dataset)



if __name__ == "__main__":

    # Create robot
    robot = Robot("assets/panda/panda.urdf")
    
    # Create TrajectoryDataset (trajectory starts per sample, best for RL)
    # MOTIVATION: TrajectoryDataset vs StateDataset
    # - TrajectoryDataset: Each sample is a trajectory start state (q0)
    #   > Perfect for RL: episodes start from the same initial distribution as expert demos
    # - StateDataset: Each sample is any timestep from trajectories 
    #   > Used for BC training where we learn from all timesteps, not just starts
    dataset = TrajectoryDataset.load_from_directory(
        robot=robot,
        directory="datasets/ae_aristotle1_5mm_cubbies",
        dataset_type=DatasetType.TRAIN,
        trajectory_key="global_solutions",
        num_robot_points=2048,
        num_obstacle_points=4096,
        num_target_points=128,
        random_scale=0.0,  # No random noise for RL (we want clean states)
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    # Create DataLoader with batch_size=1 (environment processes one problem at a time)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print("✓ DataLoader created")

    # Create environment (TimeLimit wrapper, episodes truncated at 50 steps)
    env = AvoidEverythingEnv(dataloader=dataloader, urdf_path="assets/panda/panda.urdf", max_episode_steps=50)
    print("✓ Environment created")

    # Test reset
    obs, info = env.reset()
    print("✓ Initial observation retrieved:")
    print(f"  - point_cloud: {obs['point_cloud'].shape}, {obs['point_cloud'].dtype}")
    print(f"  - point_cloud_labels: {obs['point_cloud_labels'].shape}, {obs['point_cloud_labels'].dtype}")
    print(f"  - configuration : {obs['configuration'].shape}, {obs['configuration'].dtype}")

    # Test step
    action = np.zeros(env.action_space.shape, dtype=np.float32)  # No movement
    obs, reward, terminated, truncated, info = env.step(action)
    print("✓ Step executed:")
    print("  - Reward:", reward)
    print("  - Terminated:", terminated)
    print("  - Info:", info)

    # Check Environment Validity
    # from https://gymnasium.farama.org/introduction/create_custom_env/#debugging-your-environment
    try:
        check_env(env.unwrapped)
        print("✓ Environment passes all Gymnasium checks!")
        check_env_sb3(env)
        print("✓ Environment passes all SB3 checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")
