# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from pathlib import Path
from typing import Dict, Optional, Union
from functools import lru_cache

from termcolor import cprint
import torch
from torch.utils.data import DataLoader, Dataset

from robofin.robots import Robot
from robofin.samplers import NumpyRobotSampler

from avoid_everything_except_exploration.dataset import Dataset as MPNDataset
from avoid_everything_except_exploration.geometry import construct_mixed_point_cloud
from avoid_everything_except_exploration.type_defs import DatasetType

class Base(Dataset):
    """
    This base class should never be used directly, but it handles the filesystem
    management and the basic indexing. When using these dataloaders, the directory
    holding the data should look like so:
        directory/
          train/
             train.hdf5
          val/
             val.hdf5
          test/
             test.hdf5
    Note that only the relevant subdirectory is required, i.e. when creating a
    dataset for training, this class will not check for (and will not use) the val/
    and test/ subdirectories.
    """

    def __init__(
        self,
        robot: Robot,
        data_path: Union[Path, str],
        dataset_type: DatasetType,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        random_scale: float,
        goal_reward: Optional[float] = None,
        collision_reward: Optional[float] = None,
        step_reward: Optional[float] = None,
    ):
        """
        :param robot (Robot): Robot object
        :param directory Path: The path to the root of the data directory
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param num_target_points int: The number of points to sample from the target
                                      robot end effector
        :param dataset_type DatasetType: What type of dataset this is
        :param random_scale float: The standard deviation of the random normal
                                   noise to apply to the joints during training.
                                   This is only used for train datasets.
        """
        self.robot = robot
        self._database = Path(data_path)
        self.trajectory_key = trajectory_key
        self.train = dataset_type == DatasetType.TRAIN
        self.normalization_params = None
        if not self.file_exists:
            self.state_count = 0
            self.problem_count = 0
        else:
            with MPNDataset(self.robot, self._database) as f:
                self.state_count = len(f[self.trajectory_key])
                self.problem_count = len(f)

        self.num_obstacle_points = num_obstacle_points
        self.num_robot_points = num_robot_points
        self.num_target_points = num_target_points
        self.random_scale = random_scale
        self.robot_sampler = NumpyRobotSampler(
            self.robot,
            num_robot_points=self.num_robot_points,
            num_eef_points=self.num_target_points,
            use_cache=True,
            with_base_link=True,
        )
        self.goal_reward = goal_reward if goal_reward is not None else 0.0
        self.collision_reward = collision_reward if collision_reward is not None else 0.0
        self.step_reward = step_reward if step_reward is not None else 0.0

    def batch_scenes_by_idx(self, idxs: torch.Tensor, *, pin: bool = True) -> dict[str, torch.Tensor]:
        """
        Returns a dict of stacked CPU tensors for all unique scene indices in `idxs`.
        Shapes are [U, ...] where U = len(idxs).
        NOTE: assumes per-scene shapes are consistent across the dataset (already padded).
        """
        idxs = idxs.to('cpu', non_blocking=False).to(torch.int64).view(-1)
        # Reuse cached single-scene loads to avoid expensive recomputation
        items = [self.scene_by_idx(int(i)) for i in idxs.tolist()]
        assert len(items) > 0, "No indices passed to batch_scenes_by_idx"

        keys = items[0].keys()
        out: dict[str, torch.Tensor] = {}
        for k in keys:
            v = torch.stack([it[k] for it in items], dim=0)  # [U, ...], all CPU
            out[k] = v.pin_memory() if pin else v
        return out

    @lru_cache(maxsize=4096)
    def scene_by_idx(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns dictionary with static scene info for a given idx (CPU tensors)."""
        item = {}
        with MPNDataset(self.robot, self._database, "r") as f:
            pidx = f[self.trajectory_key].lookup_pidx(int(idx))
            problem = f[self.trajectory_key].problem(pidx)
            flobs = f[self.trajectory_key].flattened_obstacles(pidx)

            # target pose (CPU)
            target_pose = self.robot.fk(problem.target)[self.robot.tcp_link_name].squeeze()
            target_points = torch.as_tensor(
                self.robot_sampler.sample_end_effector(target_pose)[..., :3]
            ).float()  # [N_target, 3]

            # scene obstacle points (CPU)
            scene_points = torch.as_tensor(
                construct_mixed_point_cloud(problem.obstacles, self.num_obstacle_points)[..., :3]
            ).float()  # [N_scene, 3]

            item["cuboid_centers"] = torch.as_tensor(flobs.cuboid_centers).float()
            item["cuboid_dims"] =    torch.as_tensor(flobs.cuboid_dims).float()
            item["cuboid_quats"] =   torch.as_tensor(flobs.cuboid_quaternions).float()
            item["cylinder_centers"] = torch.as_tensor(flobs.cylinder_centers).float()
            item["cylinder_radii"] =   torch.as_tensor(flobs.cylinder_radii).float()
            item["cylinder_heights"] = torch.as_tensor(flobs.cylinder_heights).float()
            item["cylinder_quats"] =   torch.as_tensor(flobs.cylinder_quaternions).float()
            item["target_position"] =  torch.as_tensor(target_pose[:3, 3]).float()
            item["target_orientation"] = torch.as_tensor(target_pose[:3, :3]).float()
            item["scene_points"] = scene_points
            item["target_points"] = target_points
        return item

    @property
    def file_exists(self) -> bool:
        return self._database.exists()

    @property
    def md5_checksum(self):
        with MPNDataset(self.robot, self._database) as f:
            return f.md5_checksum

    @classmethod
    def load_from_directory(
        cls,
        robot: Robot,
        directory: Union[Path, str],
        dataset_type: DatasetType,
        *args,
        **kwargs,
    ):
        directory = Path(directory)
        if dataset_type in (DatasetType.TRAIN, "train"):
            enclosing_path = directory / "train"
            data_path = enclosing_path / "train.hdf5"
        elif dataset_type in (DatasetType.COL, "train"):
            enclosing_path = directory / "train"
            data_path = enclosing_path / "train.hdf5"
        elif dataset_type in (DatasetType.VAL_STATE, "val"):
            enclosing_path = directory / "val"
            data_path = enclosing_path / "val.hdf5"
        elif dataset_type in (DatasetType.VAL, "val"):
            enclosing_path = directory / "val"
            data_path = enclosing_path / "val.hdf5"
        elif dataset_type in (DatasetType.MINI_TRAIN, "mini_train"):
            enclosing_path = directory / "val"
            data_path = enclosing_path / "mini_train.hdf5"
        elif dataset_type in (DatasetType.VAL_PRETRAIN, "val_pretrain"):
            enclosing_path = directory / "val"
            data_path = enclosing_path / "val_pretrain.hdf5"
        elif dataset_type in (DatasetType.TEST, "test"):
            enclosing_path = directory / "test"
            data_path = enclosing_path / "test.hdf5"
        else:
            raise Exception(f"Invalid dataset type: {dataset_type}")
        return cls(
            robot,
            data_path,
            dataset_type,
            *args,
            **kwargs,
        )

    def clamp_and_normalize(self, configuration_tensor: torch.Tensor):
        """
        Normalizes the joints between -1 and 1 according the the joint limits

        :param configuration_tensor (torch.Tensor): The input tensor. 
            Has dim [self.robot.MAIN_DOF]
        """
        # NOTE: self.robot.main_joint_limits based on URDF is different from the
        # original implementation's RealFrankaConstanst.JOINT_LIMITS for joint 6
        limits = torch.as_tensor(self.robot.main_joint_limits).float()
        configuration_tensor = torch.minimum(
            torch.maximum(configuration_tensor, limits[:, 0]), limits[:, 1]
        )
        return self.robot.normalize_joints(configuration_tensor)

    def get_inputs(self, problem, flobs) -> Dict[str, torch.Tensor]:
        """
        Loads all the relevant data and puts it in a dictionary. This includes
        normalizing all configurations and constructing the pointcloud.
        If a training dataset, applies some randomness to joints (before
        sampling the pointcloud).

        :param trajectory_idx int: The index of the trajectory in the hdf5 file
        :param timestep int: The timestep within that trajectory
        :rtype Dict[str, torch.Tensor]: The data used aggregated by the dataloader
                                        and used for training
        """
        item = {}
        target_pose = self.robot.fk(problem.target)[self.robot.tcp_link_name].squeeze()
        target_points = torch.as_tensor(
            self.robot_sampler.sample_end_effector(
                target_pose,
            )[..., :3]
        ).float()
        item["target_position"] = torch.as_tensor(target_pose[:3, 3]).float()
        item["target_orientation"] = torch.as_tensor(target_pose[:3, :3]).float()

        item["cuboid_dims"] = torch.as_tensor(flobs.cuboid_dims).float()
        item["cuboid_centers"] = torch.as_tensor(flobs.cuboid_centers).float()
        item["cuboid_quats"] = torch.as_tensor(flobs.cuboid_quaternions).float()

        item["cylinder_radii"] = torch.as_tensor(flobs.cylinder_radii).float()
        item["cylinder_heights"] = torch.as_tensor(flobs.cylinder_heights).float()
        item["cylinder_centers"] = torch.as_tensor(flobs.cylinder_centers).float()
        item["cylinder_quats"] = torch.as_tensor(flobs.cylinder_quaternions).float()

        scene_points = torch.as_tensor(
            construct_mixed_point_cloud(problem.obstacles, self.num_obstacle_points)[
                ..., :3
            ]
        ).float()
        item["point_cloud"] = torch.cat((scene_points, target_points), dim=0)
        item["point_cloud_labels"] = torch.cat(
            (
                torch.ones(len(scene_points), 1),
                2 * torch.ones(len(target_points), 1),
            )
        )

        return item


class TrajectoryDataset(Base):
    """
    This dataset is used exclusively for validating. Each element in the dataset
    represents a trajectory start and scene. There is no supervision because
    this is used to produce an entire rollout and check for success. When doing
    validation, we care more about success than we care about matching the
    expert's behavior (which is a key difference from training).
    """

    def __init__(
        self,
        robot: Robot,
        data_path: Union[Path, str],
        dataset_type: DatasetType,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        random_scale: float = 0.0,
    ):
        """
        :param directory Path: The path to the root of the data directory
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param num_target_points int: The number of points to sample from the target
                                      robot end effector
        :param dataset_type DatasetType: What type of dataset this is
        """
        super().__init__(
            robot,
            data_path,
            dataset_type,
            trajectory_key,
            num_robot_points,
            num_obstacle_points,
            num_target_points,
            random_scale,
        )

    @classmethod
    def load_from_directory(
        cls,
        robot: Robot,
        directory: Path,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        dataset_type: DatasetType,
        random_scale: float,
    ):
        return super().load_from_directory(
            robot,
            directory,
            dataset_type,
            trajectory_key,
            num_robot_points,
            num_obstacle_points,
            num_target_points,
            random_scale,
        )

    def __len__(self):
        """
        Necessary for Pytorch. For this dataset, the length is the total number
        of problems
        """
        return self.problem_count

    def unpadded_expert(self, pidx: int):
        with MPNDataset(self.robot, self._database, "r") as f:
            return torch.as_tensor(f[self.trajectory_key].expert(pidx))

    def __getitem__(self, pidx: int) -> Dict[str, torch.Tensor]:
        """
        Required by Pytorch. Queries for data at a particular index. Note that
        in this dataset, the index always corresponds to the trajectory index.

        :param pidx int: The problem index
        :rtype Dict[str, torch.Tensor]: Returns a dictionary that can be assembled
            by the data loader before using in training.
        """
        with MPNDataset(self.robot, self._database, "r") as f:
            problem = f[self.trajectory_key].problem(pidx)
            flobs = f[self.trajectory_key].flattened_obstacles(pidx)
            item = self.get_inputs(problem, flobs)
            config = f[self.trajectory_key].problem(pidx).q0
            config_tensor = torch.as_tensor(config).float()

            if self.train:
                # Add slight random noise to the joints (in non-normalized joint space)
                randomized = (
                    self.random_scale * torch.randn(config_tensor.shape) + config_tensor
                )

                item["configuration"] = self.clamp_and_normalize(randomized)
                robot_points = self.robot_sampler.sample(
                    randomized.numpy()
                )[:, :3]
            else:
                item["configuration"] = self.clamp_and_normalize(config_tensor)
                robot_points = self.robot_sampler.sample(
                    config_tensor.numpy(),
                )[:, :3]
            robot_points = torch.as_tensor(robot_points).float()

            item["point_cloud"] = torch.cat((robot_points, item["point_cloud"]), dim=0)
            item["point_cloud_labels"] = torch.cat(
                (
                    torch.zeros(len(robot_points), 1),
                    item["point_cloud_labels"],
                )
            )
            item["expert"] = torch.as_tensor(f[self.trajectory_key].padded_expert(pidx))
        item["pidx"] = torch.as_tensor(pidx)

        return item


class StateDataset(Base):
    """
    This is the dataset used primarily for training. Each element in the dataset
    represents the robot and scene at a particular time $t$. Likewise, the
    supervision is the robot's configuration at q_{t+1}.
    """

    def __init__(
        self,
        robot: Robot,
        data_path: Union[Path, str],
        dataset_type: DatasetType,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        random_scale: float,
    ):
        """
        :param directory Path: The path to the root of the data directory
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param num_target_points int: The number of points to sample from the target
                                      robot end effector
        :param dataset_type DatasetType: What type of dataset this is
        :param random_scale float: The standard deviation of the random normal
                                   noise to apply to the joints during training.
                                   This is only used for train datasets.
        """
        super().__init__(
            robot,
            data_path,
            dataset_type,
            trajectory_key,
            num_robot_points,
            num_obstacle_points,
            num_target_points,
            random_scale,
        )

    @classmethod
    def load_from_directory(
        cls,
        robot: Robot,
        directory: Path,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        dataset_type: DatasetType,
        random_scale: float,
    ):
        return super().load_from_directory(
            robot,
            directory,
            dataset_type,
            trajectory_key,
            num_robot_points,
            num_obstacle_points,
            num_target_points,
            random_scale,
        )

    def __len__(self):
        """
        Returns the total number of start configurations in the dataset (i.e.
        the length of the trajectories times the number of trajectories)

        """
        return self.state_count

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a training datapoint representing a single configuration in a
        single scene with the configuration at the next timestep as supervision

        :param idx int: Index represents the timestep within the trajectory
        :rtype Dict[str, torch.Tensor]: The data used for training
        """
        with MPNDataset(self.robot, self._database, "r") as f:
            pidx = f[self.trajectory_key].lookup_pidx(idx)
            problem = f[self.trajectory_key].problem(pidx)
            flobs = f[self.trajectory_key].flattened_obstacles(pidx)
            item = self.get_inputs(problem, flobs)
            # Retrieve current and next configuration (single-step supervision)
            configs = f[self.trajectory_key].state_range(idx, lookahead=2)
            config = configs[0]
            supervision = configs[1]
            config_tensor = torch.as_tensor(config).float()

            if self.train:
                # Add slight random noise to the joints (in non-normalized joint space)
                randomized = (
                    self.random_scale * torch.randn(config_tensor.shape) + config_tensor
                )

                item["configuration"] = self.clamp_and_normalize(randomized)
                robot_points = self.robot_sampler.sample(
                    randomized.numpy()
                )[:, :3]
            else:
                item["configuration"] = self.clamp_and_normalize(config_tensor)
                robot_points = self.robot_sampler.sample(
                    config_tensor.numpy()
                )[:, :3]
            robot_points = torch.as_tensor(robot_points).float()
            item["point_cloud"] = torch.cat((robot_points, item["point_cloud"]), dim=0)
            item["point_cloud_labels"] = torch.cat(
                (
                    torch.zeros(len(robot_points), 1),
                    item["point_cloud_labels"],
                )
            )

            item["idx"] = torch.as_tensor(idx)
            supervision_tensor = torch.as_tensor(supervision).float()
            item["supervision"] = self.clamp_and_normalize(supervision_tensor)

        return item

class StateRewardDataset(Base):
    """
    This is the dataset used for training with Cycle of Learning (CoL). Each 
    element in the dataset represents the robot and scene at a particular time 
    $t$, robot's configuration at $t+1$, the action for the transition and the 
    reward for the transition.
    """

    def __init__(
        self,
        robot: Robot,
        data_path: Union[Path, str],
        dataset_type: DatasetType,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        random_scale: float,
        goal_reward: float,
        collision_reward: float,
        step_reward: float,
    ):
        """
        :param directory Path: The path to the root of the data directory
        :param num_robot_points int: The number of points to sample from the robot
        :param num_obstacle_points int: The number of points to sample from the obstacles
        :param num_target_points int: The number of points to sample from the target
                                      robot end effector
        :param dataset_type DatasetType: What type of dataset this is
        :param random_scale float: The standard deviation of the random normal
                                   noise to apply to the joints during training.
                                   This is only used for train datasets.
        """
        super().__init__(
            robot,
            data_path,
            dataset_type,
            trajectory_key,
            num_robot_points,
            num_obstacle_points,
            num_target_points,
            random_scale,
            goal_reward,
            collision_reward,
            step_reward,
        )

    @classmethod
    def load_from_directory(
        cls,
        robot: Robot,
        directory: Path,
        trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        dataset_type: DatasetType,
        random_scale: float,
        goal_reward: float,
        collision_reward: float,
        step_reward: float,
    ):
        return super().load_from_directory(
            robot,
            directory,
            dataset_type,
            trajectory_key,
            num_robot_points,
            num_obstacle_points,
            num_target_points,
            random_scale,
            goal_reward,
            collision_reward,
            step_reward,
        )
        
    def __len__(self):
        """
        Returns the total number of start configurations in the dataset (i.e.
        the length of the trajectories times the number of trajectories)

        """
        return self.state_count

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a training datapoint representing a single configuration in a
        single scene with the configuration at the next timestep and the reward 
        for the transition.

        :param idx int: Index represents the timestep within the trajectory
        :rtype Dict[str, torch.Tensor]: The data used for training
        """
        with MPNDataset(self.robot, self._database, "r") as f:
            pidx = f[self.trajectory_key].lookup_pidx(idx)
            problem = f[self.trajectory_key].problem(pidx)
            flobs = f[self.trajectory_key].flattened_obstacles(pidx)
            item = self.get_inputs(problem, flobs)

            configs = f[self.trajectory_key].state_range(idx, lookahead=2)
            config = configs[0]
            next_config = configs[1]
            config_tensor = torch.as_tensor(config).float()

            if self.train:
                # Add slight random noise to the joints (in non-normalized joint space)
                randomized = (
                    self.random_scale * torch.randn(config_tensor.shape) + config_tensor
                )

                item["configuration"] = self.clamp_and_normalize(randomized)
                robot_points = self.robot_sampler.sample(
                    randomized.numpy()
                )[:, :3]
            else:
                item["configuration"] = self.clamp_and_normalize(config_tensor)
                robot_points = self.robot_sampler.sample(
                    config_tensor.numpy()
                )[:, :3]
            robot_points = torch.as_tensor(robot_points).float()
            item["point_cloud"] = torch.cat((robot_points, item["point_cloud"]), dim=0)
            item["point_cloud_labels"] = torch.cat(
                (
                    torch.zeros(len(robot_points), 1),
                    item["point_cloud_labels"],
                )
            )

            item["idx"] = torch.as_tensor(idx)
            next_config_tensor = torch.as_tensor(next_config).float()
            item["next_configuration"] = self.clamp_and_normalize(next_config_tensor)
            item["action"] = item["next_configuration"] - item["configuration"]
            
            # Reward is goal if the next state equals the last expert state;
            # otherwise it's the step reward. We compare normalized states to
            # match item["next_configuration"].
            keyed = f[self.trajectory_key]
            expert_len = int(keyed.expert_length(pidx))
            # Get the last state in the expert trajectory for this problem index
            last_state_np = keyed.file[keyed.key][pidx, expert_len - 1]
            last_state_norm = self.clamp_and_normalize(torch.as_tensor(last_state_np).float())
            is_goal_transition = torch.allclose(
                item["next_configuration"], last_state_norm, rtol=1e-6, atol=1e-6
            )
            item["reward"] = (
                torch.as_tensor(self.goal_reward).float().unsqueeze(0)
                if is_goal_transition
                else torch.as_tensor(self.step_reward).float().unsqueeze(0)
            )
            item["done"] = torch.as_tensor(is_goal_transition).float().unsqueeze(0)
            item["is_expert"] = torch.ones(1, dtype=torch.float32)

        return item



class DataModule:
    def __init__(
        self,
        urdf_path: str,
        data_dir: str,
        train_trajectory_key: str,
        val_trajectory_key: str,
        num_robot_points: int,
        num_obstacle_points: int,
        num_target_points: int,
        random_scale: float,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        include_reward: Optional[bool] = None,
        goal_reward: Optional[float] = None,
        collision_reward: Optional[float] = None,
        step_reward: Optional[float] = None,
        reward_scale: Optional[float] = None,
        shuffle: Optional[bool] = None,
    ):
        super().__init__()
        self.robot = Robot(urdf_path)
        self.data_dir = Path(data_dir)
        self.train_trajectory_key = train_trajectory_key
        self.val_trajectory_key = val_trajectory_key
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_robot_points = num_robot_points
        self.num_obstacle_points = num_obstacle_points
        self.num_target_points = num_target_points
        self.num_workers = num_workers
        self.random_scale = random_scale
        self.include_reward = include_reward if include_reward is not None else False
        self.reward_scale = reward_scale if reward_scale is not None else 1.0
        self.goal_reward = goal_reward * self.reward_scale if goal_reward is not None else 0.0
        self.collision_reward = collision_reward * self.reward_scale if collision_reward is not None else 0.0
        self.step_reward = step_reward * self.reward_scale if step_reward is not None else 0.0
        self.shuffle = shuffle if shuffle is not None else True

    def setup(self, stage: Optional[str] = None):
        """
        A Pytorch Lightning method that is called per-device in when doing
        distributed training.

        :param stage Optional[str]: Indicates whether we are in the training
                                    procedure or if we are doing ad-hoc testing
        """
        if stage == "fit" or stage is None:
            if self.include_reward:
                self.data_train = StateRewardDataset.load_from_directory(
                    self.robot,
                    self.data_dir,
                    dataset_type=DatasetType.COL,
                    trajectory_key=self.train_trajectory_key,
                    num_robot_points=self.num_robot_points,
                    num_obstacle_points=self.num_obstacle_points,
                    num_target_points=self.num_target_points,
                    random_scale=self.random_scale,
                    goal_reward=self.goal_reward,
                    collision_reward=self.collision_reward,
                    step_reward=self.step_reward,
                )
                cprint("Loaded StateRewardDataset for training", "green")
            else:
                self.data_train = StateDataset.load_from_directory(
                    self.robot,
                    self.data_dir,
                    dataset_type=DatasetType.TRAIN,
                    trajectory_key=self.train_trajectory_key,
                    num_robot_points=self.num_robot_points,
                    num_obstacle_points=self.num_obstacle_points,
                    num_target_points=self.num_target_points,
                    random_scale=self.random_scale,
                )
                cprint("Loaded StateDataset for training", "green")
            if self.include_reward:
                self.data_val_state = StateRewardDataset.load_from_directory(
                    self.robot,
                    self.data_dir,
                    dataset_type=DatasetType.VAL_STATE,
                    trajectory_key=self.val_trajectory_key,
                    num_robot_points=self.num_robot_points,
                    num_obstacle_points=self.num_obstacle_points,
                    num_target_points=self.num_target_points,
                    random_scale=0.0,
                    goal_reward=self.goal_reward,
                    collision_reward=self.collision_reward,
                    step_reward=self.step_reward,
                )
                cprint("Loaded StateRewardDataset for validation", "green")
            else:
                self.data_val_state = StateDataset.load_from_directory(
                    self.robot,
                    self.data_dir,
                    dataset_type=DatasetType.VAL_STATE,
                    trajectory_key=self.val_trajectory_key,
                    num_robot_points=self.num_robot_points,
                    num_obstacle_points=self.num_obstacle_points,
                    num_target_points=self.num_target_points,
                    random_scale=0.0,
                )
                cprint("Loaded StateDataset for validation", "green")
            self.data_val = TrajectoryDataset.load_from_directory(
                self.robot,
                self.data_dir,
                dataset_type=DatasetType.VAL,
                trajectory_key=self.val_trajectory_key,
                num_robot_points=self.num_robot_points,
                num_obstacle_points=self.num_obstacle_points,
                num_target_points=self.num_target_points,
                random_scale=0.0,
            )
            cprint("Loaded TrajectoryDataset for validation", "green")
            # Handle missing optional validation files gracefully
            mini_train_path = Path(self.data_dir) / "val" / "mini_train.hdf5"
            if mini_train_path.exists():
                try:
                    self.data_mini_train = TrajectoryDataset.load_from_directory(
                        self.robot,
                        self.data_dir,
                        dataset_type=DatasetType.MINI_TRAIN,
                        trajectory_key=self.val_trajectory_key,
                        num_robot_points=self.num_robot_points,
                        num_obstacle_points=self.num_obstacle_points,
                        num_target_points=self.num_target_points,
                        random_scale=0.0,
                    )
                    cprint("Loaded TrajectoryDataset for mini_train", "green")
                except Exception:
                    self.data_mini_train = self.data_val
            else:
                # Use validation data as fallback for mini_train if file doesn't exist
                self.data_mini_train = self.data_val
                
            val_pretrain_path = Path(self.data_dir) / "val" / "val_pretrain.hdf5"
            if val_pretrain_path.exists():
                try:
                    self.data_val_pretrain = TrajectoryDataset.load_from_directory(
                        self.robot,
                        self.data_dir,
                        dataset_type=DatasetType.VAL_PRETRAIN,
                        trajectory_key=self.val_trajectory_key,
                        num_robot_points=self.num_robot_points,
                        num_obstacle_points=self.num_obstacle_points,
                        num_target_points=self.num_target_points,
                        random_scale=0.0,
                    )
                    cprint("Loaded TrajectoryDataset for val_pretrain", "green")
                except:
                    self.data_val_pretrain = self.data_val
            else:
                # Use validation data as fallback for val_pretrain if file doesn't exist
                self.data_val_pretrain = self.data_val
        if stage == "test" or stage is None:
            self.data_test = StateDataset.load_from_directory(
                self.robot,
                self.data_dir,
                self.train_trajectory_key,  # TODO change this
                self.num_robot_points,
                self.num_obstacle_points,
                self.num_target_points,
                dataset_type=DatasetType.TEST,
                random_scale=self.random_scale,
            )
            cprint("Loaded StateDataset for testing", "green")
        if stage == "dagger":
            self.data_dagger = TrajectoryDataset.load_from_directory(
                self.robot,
                self.data_dir,
                dataset_type=DatasetType.TRAIN,
                trajectory_key=self.val_trajectory_key,
                num_robot_points=self.num_robot_points,
                num_obstacle_points=self.num_obstacle_points,
                num_target_points=self.num_target_points,
                random_scale=0.0,
            )
            cprint("Loaded TrajectoryDataset for dagger", "green")

    def train_dataloader(self) -> DataLoader:
        """
        A Pytorch lightning method to get the dataloader for training

        :rtype DataLoader: The training dataloader
        """
        return DataLoader(
            self.data_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=self.shuffle,
            persistent_workers=False if self.num_workers == 0 else True,
            drop_last=True,
        )
    
    def train_trajectory_dataloader(self, batch_size=None) -> DataLoader:
        """
        Method to get the dataloader for training trajectories

        :rtype DataLoader: The training trajectory dataloader
        """
        if not hasattr(self, "data_train_trajectory"):
            self.data_train_trajectory = TrajectoryDataset.load_from_directory(
                self.robot,
                self.data_dir,
                dataset_type=DatasetType.TRAIN,
                trajectory_key=self.train_trajectory_key, # TODO: what? here?
                num_robot_points=self.num_robot_points,
                num_obstacle_points=self.num_obstacle_points,
                num_target_points=self.num_target_points,
                random_scale=0.0,
            )
        return DataLoader(
            self.data_train_trajectory,
            batch_size if batch_size is not None else self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=self.shuffle,
            persistent_workers=False if self.num_workers == 0 else True,
            drop_last=True,
        )

    def dagger_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_dagger,
            self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

    def val_state_dataloader(self) -> DataLoader:
        """
        Method to get the dataloader for validation states

        :rtype DataLoader: The validation state dataloader
        """
        return DataLoader(
            self.data_val_state,
            self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=self.shuffle,
            persistent_workers=False if self.num_workers == 0 else True,
            drop_last=True,
        )

    def val_trajectory_dataloader(self) -> DataLoader:
        """
        Method to get the dataloader for validation trajectories

        :rtype DataLoader: The validation trajectory dataloader
        """
        return DataLoader(
            self.data_val,
            self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=self.shuffle,
            persistent_workers=False if self.num_workers == 0 else True,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        A Pytorch lightning method to get the dataloader for testing

        :rtype DataLoader: The dataloader for testing
        """
        assert NotImplementedError("Not implemented")

    def md5_checksums(self):
        """
        Currently too lazy to figure out how to fit this into Lightning with the whole
        setup() thing and the data being initialized in that call and when to get
        hyperparameters etc etc, so just hardcoding the paths right now
        """
        paths = [
            ("train", self.data_dir / "train" / "train.hdf5"),
            ("val", self.data_dir / "val" / "val.hdf5"),
            ("mini_train", self.data_dir / "val" / "mini_train.hdf5"),
        ]
        checksums = {}
        for key, path in paths:
            if path.exists():
                with MPNDataset(self.robot, path) as f:
                    checksums[key] = f.md5_checksum
        return checksums
