"""
TorchRL environment for reaching a target while avoiding obstacles.

- AvoidEverythingEnv now inherits from EnvBase
- reset/step are TensorDict-native and batched
- environment state remains on a single torch device
- uses old robofin point/collision utilities
- uses dense reward (target bonus and collision penalty)
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torch.utils.data import DataLoader, Subset
from torchrl.data.tensor_specs import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase

from avoid_everything.data_loader import TrajectoryDataset
from avoid_everything.geometry import TorchCuboids, TorchCylinders
from avoid_everything.type_defs import DatasetType
from robofin.old.collision import TorchFrankaCollisionSpheres
from robofin.old.samplers import TorchFrankaSampler
from robofin.robots import Robot


def build_action_delta_clip(
    action_delta_clip: float | Sequence[float], dof: int, device: torch.device
) -> torch.Tensor:
    """Build per-joint hard-clip bounds from scalar or sequence config."""
    if isinstance(action_delta_clip, (int, float)):
        clip = float(action_delta_clip)
        if clip <= 0:
            raise ValueError("action_delta_clip must be > 0.")
        return torch.full((dof,), clip, dtype=torch.float32, device=device)

    clip_array = np.asarray(action_delta_clip, dtype=np.float32)
    if clip_array.shape != (dof,):
        raise ValueError(
            f"action_delta_clip sequence must have shape {(dof,)}, got {clip_array.shape}."
        )
    if np.any(clip_array <= 0):
        raise ValueError("All action_delta_clip values must be > 0.")
    return torch.as_tensor(clip_array, dtype=torch.float32, device=device)


def validate_max_delta_q(max_delta_q: float) -> float:
    """Validate and normalize max_delta_q config value."""
    value = float(max_delta_q)
    if value <= 0:
        raise ValueError("max_delta_q must be > 0.")
    return value


def _compute_target_errors_batched(
    eef_pose: torch.Tensor,
    target_position: torch.Tensor,
    target_orientation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute batched position/orientation errors."""
    # eef_pose: [B, 4, 4], target_position: [B, 3], target_orientation: [B, 3, 3]
    position_error = torch.linalg.vector_norm(
        eef_pose[:, :3, -1] - target_position, dim=1
    )
    rel_rot = torch.matmul(eef_pose[:, :3, :3], target_orientation.transpose(1, 2))
    trace = rel_rot.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_value = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    orientation_error = torch.abs(torch.rad2deg(torch.acos(cos_value)))
    return position_error, orientation_error


class AvoidEverythingEnv(EnvBase):
    """
    Batched EnvBase implementation of AvoidEverything.

    Notes:
    - reset/step operate on TensorDict and batched tensors
    - state is maintained on self.device
    - reward is dense with collision and target terms
    """

    def __init__(
        self,
        dataloader: DataLoader | None = None,
        urdf_path: str = "assets/panda/panda.urdf",
        num_robot_points: int = 2048,
        num_obstacle_points: int = 4096,
        num_target_points: int = 128,
        collision_mode: Literal["torch_franka", "torch"] = "torch_franka",
        scene_buffer: float = 0.0,
        position_threshold: float = 0.01,
        orientation_threshold: float = 15.0,
        max_episode_steps: int = 100,
        terminate_ep_on_collision: bool = True,
        action_delta_clip: float | Sequence[float] = 0.2,
        max_delta_q: float = 1.0,
        batch_size: int = 64,
        device: str | torch.device | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

        resolved_device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = resolved_device
        self.env_batch_size = batch_size
        super().__init__(device=self.device, batch_size=torch.Size([batch_size]))

        self.urdf_path = urdf_path
        self.robot = Robot(urdf_path, device=self.device)
        self.robot_sampler = TorchFrankaSampler(
            num_robot_points=num_robot_points,
            num_eef_points=num_target_points,
            use_cache=True,
            with_base_link=True,
            device=str(self.device),
        )

        self.num_robot_points = num_robot_points
        self.num_obstacle_points = num_obstacle_points
        self.num_target_points = num_target_points
        self.total_points = (
            self.num_robot_points + self.num_obstacle_points + self.num_target_points
        )
        self.max_delta_q = validate_max_delta_q(max_delta_q)
        self.action_delta_clip = build_action_delta_clip(
            action_delta_clip, self.robot.MAIN_DOF, self.device
        )

        self.collision_mode = collision_mode
        self.scene_buffer = float(scene_buffer)
        self.position_threshold = float(position_threshold)
        self.orientation_threshold = float(orientation_threshold)
        self.max_episode_steps = int(max_episode_steps)
        self.terminate_ep_on_collision = bool(terminate_ep_on_collision)
        self._prismatic_joint = float(
            self.robot.auxiliary_joint_defaults.get("panda_finger_joint1", 0.04)
        )

        self._collision_checker = TorchFrankaCollisionSpheres(device=str(self.device))

        # Dataset state
        self.dataloader: DataLoader | None = dataloader
        self._dataloader_iter = iter(dataloader) if dataloader is not None else None
        self._num_split_samples = len(dataloader.dataset) if dataloader is not None else 0

        # Episode state tensors (allocated at reset)
        self.robot_config: torch.Tensor | None = None          # [B, dof], normalized
        self.target_position: torch.Tensor | None = None       # [B, 3]
        self.target_orientation: torch.Tensor | None = None    # [B, 3, 3]
        self._base_point_cloud: torch.Tensor | None = None     # [B, total_points, 3]
        self._point_cloud_labels: torch.Tensor | None = None   # [B, total_points, 1]
        self._scene_primitives: list[TorchCuboids | TorchCylinders] | None = None
        self._episode_step_count: torch.Tensor | None = None   # [B]

        self._build_specs()

    def _build_specs(self) -> None:
        # Specs must include batch dimensions to match env batch_size
        # Point cloud bounds (single environment shape: [total_points, 3])
        pc_low = torch.tensor([-1.5, -1.5, -0.1], device=self.device)
        pc_high = torch.tensor([1.5, 1.5, 1.5], device=self.device)
        
        # For batched specs, expand to [B, *shape]
        pc_low_batched = pc_low.unsqueeze(0).unsqueeze(0).expand(self.env_batch_size, self.total_points, 3)
        pc_high_batched = pc_high.unsqueeze(0).unsqueeze(0).expand(self.env_batch_size, self.total_points, 3)
        
        self.observation_spec = Composite(
            point_cloud=Bounded(
                low=pc_low_batched,
                high=pc_high_batched,
                shape=torch.Size([self.env_batch_size, self.total_points, 3]),
                dtype=torch.float32,
                device=self.device,
            ),
            point_cloud_labels=Bounded(
                low=0,
                high=2,
                shape=torch.Size([self.env_batch_size, self.total_points, 1]),
                dtype=torch.int32,
                device=self.device,
            ),
            configuration=Bounded(
                low=-1.0,
                high=1.0,
                shape=torch.Size([self.env_batch_size, self.robot.MAIN_DOF]),
                dtype=torch.float32,
                device=self.device,
            ),
            shape=self.batch_size,
            device=self.device,
        )
        
        self.action_spec = Bounded(
            low=-self.max_delta_q,
            high=self.max_delta_q,
            shape=torch.Size([self.env_batch_size, self.robot.MAIN_DOF]),
            dtype=torch.float32,
            device=self.device,
        )
        
        self.reward_spec = Unbounded(
            shape=torch.Size([self.env_batch_size, 1]),
            dtype=torch.float32,
            device=self.device,
        )
        
        self.done_spec = Composite(
            done=Bounded(
                low=False,
                high=True,
                shape=torch.Size([self.env_batch_size, 1]),
                dtype=torch.bool,
                device=self.device,
            ),
            terminated=Bounded(
                low=False,
                high=True,
                shape=torch.Size([self.env_batch_size, 1]),
                dtype=torch.bool,
                device=self.device,
            ),
            truncated=Bounded(
                low=False,
                high=True,
                shape=torch.Size([self.env_batch_size, 1]),
                dtype=torch.bool,
                device=self.device,
            ),
            shape=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _next_problem_batch(self) -> dict[str, torch.Tensor]:
        if self.dataloader is None:
            raise RuntimeError(
                "No dataloader configured. Call set_scene_generation_from_dataset(...) first."
            )
        if self._dataloader_iter is None:
            self._dataloader_iter = iter(self.dataloader)

        try:
            batch = next(self._dataloader_iter)
        except StopIteration:
            self._dataloader_iter = iter(self.dataloader)
            batch = next(self._dataloader_iter)

        if not isinstance(batch, dict):
            raise TypeError("Expected dataloader to yield dict batches.")

        out: dict[str, torch.Tensor] = {}
        for key, val in batch.items():
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val)
            out[key] = val.to(self.device, non_blocking=True)

        if out["configuration"].shape[0] != self.env_batch_size:
            raise RuntimeError(
                f"Expected dataloader batch dimension {self.env_batch_size}, "
                f"got {out['configuration'].shape[0]}."
            )
        return out

    def _build_scene_primitives(
        self, batch: dict[str, torch.Tensor]
    ) -> list[TorchCuboids | TorchCylinders]:
        return [
            TorchCuboids(
                batch["cuboid_centers"].to(torch.float32),
                batch["cuboid_dims"].to(torch.float32),
                batch["cuboid_quats"].to(torch.float32),
            ),
            TorchCylinders(
                batch["cylinder_centers"].to(torch.float32),
                batch["cylinder_radii"].to(torch.float32),
                batch["cylinder_heights"].to(torch.float32),
                batch["cylinder_quats"].to(torch.float32),
            ),
        ]

    def _get_obs(self) -> dict[str, torch.Tensor]:
        if self.robot_config is None or self._base_point_cloud is None:
            raise RuntimeError("Environment state is not initialized. Call reset() first.")

        unnormalized_cfg = self.robot.unnormalize_joints(self.robot_config)
        sampled_robot_points = self.robot_sampler.sample(
            unnormalized_cfg,
            self._prismatic_joint,
            num_points=self.num_robot_points,
        )[:, :, :3]
        point_cloud = self._base_point_cloud.clone()
        point_cloud[:, : self.num_robot_points, :] = sampled_robot_points
        return {
            "point_cloud": point_cloud,
            "point_cloud_labels": self._point_cloud_labels,
            "configuration": self.robot_config,
        }

    def _compute_collision(self) -> torch.Tensor:
        if self.robot_unnorm_q is None or self._scene_primitives is None:
            raise RuntimeError("Collision state is not initialized.")
    
        collides = self._collision_checker.franka_arm_collides(
            q=self.robot_unnorm_q,
            prismatic_joint=self._prismatic_joint,
            primitives=self._scene_primitives,
            scene_buffer=self.scene_buffer,
            check_self=True,
        )
        return collides


    def _compute_target_reached(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.robot_unnorm_q is None:
            raise RuntimeError("Robot state is not initialized.")
        if self.target_position is None or self.target_orientation is None:
            raise RuntimeError("Target state is not initialized.")

        eef_pose = self.robot.fk_torch(self.robot_unnorm_q, link_name=self.robot.tcp_link_name)
        assert isinstance(eef_pose, torch.Tensor)
        pos_err, orient_err = _compute_target_errors_batched(
            eef_pose,
            self.target_position,
            self.target_orientation,
        )
        target_reached = torch.logical_and(
            pos_err < self.position_threshold,
            orient_err < self.orientation_threshold,
        )
        return target_reached, pos_err, orient_err

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs) -> TensorDictBase:
        """Reset environment state.

        Supports partial reset when tensordict contains an "_reset" boolean mask (shape [B,1]).
        """
        # Full reset when no mask is provided
        if tensordict is None or "_reset" not in getattr(tensordict, "keys", lambda: [])():
            batch = self._next_problem_batch()
            self.robot_config = batch["configuration"].to(torch.float32).clone()
            self.target_position = batch["target_position"].to(torch.float32).clone()
            self.target_orientation = batch["target_orientation"].to(torch.float32).clone()
            self._base_point_cloud = batch["point_cloud"].to(torch.float32).clone()
            self._point_cloud_labels = batch["point_cloud_labels"].to(torch.int32).clone()
            self._scene_primitives = self._build_scene_primitives(batch)
            self._episode_step_count = torch.zeros(
                self.env_batch_size, dtype=torch.int64, device=self.device
            )

            obs = self._get_obs()
            false_flag = torch.zeros((self.env_batch_size, 1), dtype=torch.bool, device=self.device)
            out = TensorDict(
                {
                    "point_cloud": obs["point_cloud"],
                    "point_cloud_labels": obs["point_cloud_labels"],
                    "configuration": obs["configuration"],
                    "done": false_flag.clone(),
                    "terminated": false_flag.clone(),
                    "truncated": false_flag.clone(),
                },
                batch_size=self.batch_size,
                device=self.device,
            )
            return out

        # Partial reset: expect an explicit mask under "_reset"
        mask_td = tensordict.get("_reset")
        if mask_td is None:
            return self._reset(None)

        mask = mask_td.squeeze(-1).to(torch.bool)
        # nothing to reset
        if not mask.any():
            obs = self._get_obs()
            false_flag = torch.zeros((self.env_batch_size, 1), dtype=torch.bool, device=self.device)
            out = TensorDict(
                {
                    "point_cloud": obs["point_cloud"],
                    "point_cloud_labels": obs["point_cloud_labels"],
                    "configuration": obs["configuration"],
                    "done": false_flag.clone(),
                    "terminated": false_flag.clone(),
                    "truncated": false_flag.clone(),
                },
                batch_size=self.batch_size,
                device=self.device,
            )
            return out

        # full replace when all True
        if mask.all():
            return self._reset(None)

        # perform partial replacement for masked indices
        idxs = mask.nonzero(as_tuple=True)[0]
        k = int(idxs.numel())
        new_batch = self._next_problem_batch()

        # update scalar/batched tensors
        self.robot_config[idxs] = new_batch["configuration"][:k].to(torch.float32).clone()
        self.target_position[idxs] = new_batch["target_position"][:k].to(torch.float32).clone()
        self.target_orientation[idxs] = new_batch["target_orientation"][:k].to(torch.float32).clone()
        self._base_point_cloud[idxs] = new_batch["point_cloud"][:k].to(torch.float32).clone()
        self._point_cloud_labels[idxs] = new_batch["point_cloud_labels"][:k].to(torch.int32).clone()

        # update scene primitives in-place
        new_prims = self._build_scene_primitives(new_batch)
        if self._scene_primitives is None:
            self._scene_primitives = new_prims
        else:
            old_cuboids, old_cylinders = self._scene_primitives
            new_cuboids, new_cylinders = new_prims

            old_cuboids.centers[idxs] = new_cuboids.centers[:k]
            old_cuboids.dims[idxs] = new_cuboids.dims[:k]
            old_cuboids.quats[idxs] = new_cuboids.quats[:k]
            old_cuboids.inv_frames[idxs] = new_cuboids.inv_frames[:k]
            old_cuboids.mask[idxs] = new_cuboids.mask[:k]

            old_cylinders.centers[idxs] = new_cylinders.centers[:k]
            old_cylinders.radii[idxs] = new_cylinders.radii[:k]
            old_cylinders.heights[idxs] = new_cylinders.heights[:k]
            old_cylinders.quats[idxs] = new_cylinders.quats[:k]
            old_cylinders.inv_frames[idxs] = new_cylinders.inv_frames[:k]
            old_cylinders.mask[idxs] = new_cylinders.mask[:k]

        # reset episode counters for masked indices
        self._episode_step_count[idxs] = 0

        obs = self._get_obs()
        false_flag = torch.zeros((self.env_batch_size, 1), dtype=torch.bool, device=self.device)
        out = TensorDict(
            {
                "point_cloud": obs["point_cloud"],
                "point_cloud_labels": obs["point_cloud_labels"],
                "configuration": obs["configuration"],
                "done": false_flag.clone(),
                "terminated": false_flag.clone(),
                "truncated": false_flag.clone(),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return out

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.robot_config is None or self._episode_step_count is None:
            raise RuntimeError("Environment state is not initialized. Call reset() first.")

        action = tensordict.get("action", None)
        if action is None:
            raise KeyError("Input TensorDict must contain key 'action'.")
        action = action.to(self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape != (self.env_batch_size, self.robot.MAIN_DOF):
            raise ValueError(
                f"Expected action shape {(self.env_batch_size, self.robot.MAIN_DOF)}, got {tuple(action.shape)}."
            )

        clipped_action = torch.clamp(action, -self.action_delta_clip, self.action_delta_clip)
        self.robot_config = torch.clamp(self.robot_config + clipped_action, -1.0, 1.0)
        self.robot_unnorm_q = self.robot.unnormalize_joints(self.robot_config)
        self._episode_step_count = self._episode_step_count + 1

        collision = self._compute_collision()
        target_reached, pos_err, orient_err = self._compute_target_reached()

        if self.terminate_ep_on_collision:
            terminated = torch.logical_or(target_reached, collision)
        else:
            terminated = target_reached
        truncated = self._episode_step_count >= self.max_episode_steps
        done = torch.logical_or(terminated, truncated)

        reward = self._compute_reward(collision, target_reached).unsqueeze(-1)

        obs = self._get_obs()
        out = TensorDict(
            {
                # state, reward, done, terminated, truncated
                "point_cloud": obs["point_cloud"],
                "point_cloud_labels": obs["point_cloud_labels"],
                "configuration": obs["configuration"],
                "reward": reward,
                "done": done.unsqueeze(-1),
                "terminated": terminated.unsqueeze(-1),
                "truncated": truncated.unsqueeze(-1),

                # additional info
                "collision": collision.unsqueeze(-1),
                "target_reached": target_reached.unsqueeze(-1),
                "position_error": pos_err.unsqueeze(-1),
                "orientation_error": orient_err.unsqueeze(-1),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return out
    

    def _compute_reward(self, collision: torch.Tensor, target_reached: torch.Tensor) -> torch.Tensor:
        # torch.where
        reward = torch.where(collision,
                             torch.tensor(-1.0, device=self.device), # collision penalty
                             torch.where(target_reached, 
                                         torch.tensor(1.0, device=self.device), # target bonus
                                         torch.tensor(0.0, device=self.device))) # no reward
        return reward

    def close(self) -> None:
        self._dataloader_iter = None

    def set_scene_generation_from_dataset(
        self,
        data_dir: str,
        trajectory_key: str,
        dataset_type: DatasetType,
        num_workers: int = 0,
        random_scale: float = 0.0,
        n_eval_episodes: Optional[int] | None = None,
        shuffle: bool = True,
    ) -> None:
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

        if n_eval_episodes:
            dataset = Subset(dataset, range(n_eval_episodes))
            shuffle = False

        

        # self._num_split_samples = len(dataset)
        if len(dataset) < self.env_batch_size:
            raise ValueError(
                f"Dataset has {len(dataset)} samples but batch_size={self.env_batch_size}. "
                "Increase dataset size or reduce batch_size."
            )

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.env_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True,
        )
        self._dataloader_iter = iter(self.dataloader)

    def get_num_split_samples(self) -> int:
        return self._num_split_samples
