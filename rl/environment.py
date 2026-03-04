"""
RL environment for reach traget while avoiding obstacles.
Custom Gymnasium environment for training the RL agent.
"""

import numpy as np
import random
import gymnasium as gym
import numba as nb
import torch

from typing import Literal
from torch.utils.data import Dataset, DataLoader
from gymnasium.utils.env_checker import check_env
from geometrout.primitive import Cuboid, CuboidArray, Cylinder, CylinderArray

from robofin.robots import Robot
from robofin.samplers import NumpyRobotSampler
from robofin.geometry import construct_mixed_point_cloud
from robofin.old.collision import FrankaCollisionSpheres
from robofin.collision import CollisionSpheres
from avoid_everything.geometry import TorchCuboids, TorchCylinders
from avoid_everything.data_loader import StateDataset
from avoid_everything.type_defs import DatasetType


@nb.jit(nopython=True)
def _seed_numba_rng(seed: int):
    """Seed Numba's internal numpy RNG (separate from Python's np.random)."""
    np.random.seed(seed)


class AvoidEverythingEnv(gym.Env):
    """
    Gymnasium environment: the robot must reach a target pose while avoiding obstacles.
    """

    metadata = {"render_modes": []}  # No rendering for now

    def __init__(
        self,
        dataloader: DataLoader,  # Dataset object
        urdf_path: str = "assets/panda/panda.urdf", # URDF path for the robot
        num_robot_points: int = 2048,
        num_obstacle_points: int = 4096,
        num_target_points: int = 128,
        collision_mode: Literal["franka", "spheres", "torch"] = "franka",
        scene_buffer: float = 0.0,
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
        """
        super().__init__()

        # Store dataset reference
        self.dataloader: DataLoader = dataloader

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

        # State space, state variables
        self.robot_config = None
        self.target_position = None
        self.target_orientation = None
        self.obstacles = None  

        # Observation space: point clouds + point labels + robot current configuration
        total_points = num_robot_points + num_obstacle_points + num_target_points
        self.observation_space = gym.spaces.Dict(
            { 
                # TODO: raise warning, might need to set limited bounds 
                "point_cloud": gym.spaces.Box(-np.inf, np.inf, shape=(total_points, 3), dtype=np.float32),  
                "point_cloud_labels": gym.spaces.Box(0, 2, shape=(total_points, 1), dtype=np.int32),
                "configuration": gym.spaces.Box(-1.0, 1.0, shape=(self.robot.MAIN_DOF,), dtype=np.float32),
                # "target_position": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )

        # Action space: normalized joint deltas [-1, 1] for main DOF
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.robot.MAIN_DOF,), dtype=np.float32,)

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
            batch = next(iter(self.dataloader))
            # Data loader returns a batch dict with tensors of shape (batch_size, ...)
            # Extract and squeeze single sample from batch
            self.problem = {key: val.squeeze(0).numpy() if hasattr(val, 'numpy') else val for key, val in batch.items()}

            self.robot_config = self.problem["configuration"]
            self.target_position = self.problem["target_position"]
            self.target_orientation = self.problem["target_orientation"]

            # Cache scene primitives for collision checking
            self._scene_primitives = self._build_scene_primitives()
        
        # Generate random problem
        else:
            raise NotImplementedError("Random problem generation not implemented yet. Please provide a dataloader with problems.")

        obs = self._get_obs()
        info = {} # TODO: add info if needed
        return obs, info

    def step(self, action):
        """Execute action: update config, check collision, compute reward.

        Args:
            action (np.array): array of shape (MAIN_DOF,) with values in [-1, 1], representing normalized joint deltas.

        Returns:
            obs (dict): dict with keys "point_cloud", "point_cloud_labels", "configuration"
            reward (float): scalar reward for this step
            terminated (bool): whether episode is done 
            truncated (bool): whether episode is truncated
            info (dict): dict with additional info
        """
        # Apply action to configuration
        upclipped_config = self.robot_config + action
        # Clip action to make robot config in valid bounds
        clipped_config = np.clip(upclipped_config, -1.0, 1.0)
        # TODO: penalize if action is out of bounds? reward
        clipped_action = np.abs(clipped_config - upclipped_config)  # How much was the action clipped?
        self.robot_config = clipped_config

        # Check for collision and target reaching
        collision = self._check_collision()
        target_reached = self._check_target_reached()

        # Get new observation, compute reward, check termination and truncation
        obs = self._get_obs()
        reward = self._compute_reward(collision, target_reached)
        terminated = collision or target_reached # TODO: also terminate if max steps exceeded (with wrapper?)
        truncated = False  # TODO: No truncation for now, but could be used for max episode length, max steps exceeded with wrapper
        info = {"collision": collision}

        return obs, reward, terminated, truncated, info

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
        """Check if current end-effector pose is close enough to target using position and orientation thresholds 
        and task space position and orientation errors."""
        # TODO
        return False

    def _compute_reward(self, collision, target_reached):
        """Compute reward"""
        # simple sparse reward for now, TODO: design better reward function
        reward = -100 if collision else 100 if target_reached else -0.1  
        return reward

    # TODO: add render() method if needed: pybullet or viz_client/server   



if __name__ == "__main__":

    # Create robot
    robot = Robot("assets/panda/panda.urdf")
    
    # Create StateDataset (single timestep per sample, best for RL)
    # MOTIVATION: StateDataset vs TrajectoryDataset
    # - StateDataset: Each sample is a single state (config, obstacles, target)
    #   > Perfect for RL because reset() samples a single scenario, then agent explores via step()
    # - TrajectoryDataset: Each sample is a full trajectory 
    #   > Not needed here since the RL env manages its own rollouts
    dataset = StateDataset.load_from_directory(
        robot=robot,
        directory="datasets/ae_aristotle1_5mm_cubbies",
        dataset_type=DatasetType.TRAIN,
        trajectory_key="global_solutions",
        num_robot_points=2048,
        num_obstacle_points=4096,
        num_target_points=128,
        random_scale=0.0,  # No random noise for RL (we want clean states)
        action_chunk_length=1,  # Single step ahead supervision
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    # Create DataLoader with batch_size=1 (environment processes one problem at a time)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"✓ DataLoader created")


    # Create environment
    env = AvoidEverythingEnv(dataloader=dataloader, urdf_path="assets/panda/panda.urdf")
    print(f"✓ Environment created")

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
    print(f"  - Reward: {reward}")
    print(f"  - Terminated: {terminated}")
    print(f"  - Info: {info}")

    # Check Environment Validity
    # from https://gymnasium.farama.org/introduction/create_custom_env/#debugging-your-environment
    try:
        check_env(env)
        print("✓ Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")
