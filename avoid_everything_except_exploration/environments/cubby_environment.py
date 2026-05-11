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

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from geometrout.primitive import Cuboid, CuboidArray, Cylinder, CylinderArray
from geometrout.transform import SE3, SO3
from robofin.collision import FrankaCollisionSpheres
from robofin.old.kinematics.numba import franka_arm_link_fk
from old.robot_constants import RealFrankaConstants
from robofin.robots import FrankaRealRobot

from avoid_everything.environments.base import (
    Environment,
    min_franka_arm_distance,
    min_franka_eef_distance,
    radius_sample,
)
from avoid_everything.type_defs import NeutralCandidate, TaskOrientedCandidate


@dataclass
class CubbyCandidate(TaskOrientedCandidate):
    """
    Represents a configuration, its end-effector pose (in right_gripper frame), and
    some metadata about the cubby (i.e. which cubby pocket it belongs to and the free space
    inside that cubby)
    """

    pocket_idx: int
    support_volume: Cuboid


class Cubby:
    """
    The actual cubby construction itself, without any robot info.
    """

    def __init__(self):
        self.cubby_left = radius_sample(0.7, 0.1)
        self.cubby_right = radius_sample(-0.7, 0.1)
        self.cubby_bottom = radius_sample(0.2, 0.1)
        self.cubby_front = radius_sample(0.55, 0.1)
        self.cubby_back = self.cubby_front + radius_sample(0.35, 0.2)
        self.cubby_top = radius_sample(0.7, 0.1)
        self.cubby_mid_h_z = radius_sample(0.45, 0.1)
        self.cubby_mid_v_y = radius_sample(0.0, 0.1)
        self.thickness = radius_sample(0.02, 0.01)
        self.middle_shelf_thickness = self.thickness
        self.center_wall_thickness = self.thickness
        self.in_cabinet_rotation = radius_sample(0, np.pi / 18)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Cubbies are essentially represented as unrotated boxes that are then rotated around
        their central yaw axis by `self.in_cabinet_rotation`. This function produces the
        rotation matrix corresponding to that value and axis.

        :rtype np.ndarray: The rotation matrix
        """
        cabinet_T_world = np.array(
            [
                [1, 0, 0, -(self.cubby_front + self.cubby_back) / 2],
                [0, 1, 0, -(self.cubby_left + self.cubby_right) / 2],
                [0, 0, 1, -(self.cubby_top + self.cubby_bottom) / 2],
                [0, 0, 0, 1],
            ]
        )
        in_cabinet_rotation = np.array(
            [
                [
                    np.cos(self.in_cabinet_rotation),
                    -np.sin(self.in_cabinet_rotation),
                    0,
                    0,
                ],
                [
                    np.sin(self.in_cabinet_rotation),
                    np.cos(self.in_cabinet_rotation),
                    0,
                    0,
                ],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        world_T_cabinet = np.array(
            [
                [1, 0, 0, (self.cubby_front + self.cubby_back) / 2],
                [0, 1, 0, (self.cubby_left + self.cubby_right) / 2],
                [0, 0, 1, (self.cubby_top + self.cubby_bottom) / 2],
                [0, 0, 0, 1],
            ]
        )
        pivot = np.matmul(
            world_T_cabinet, np.matmul(in_cabinet_rotation, cabinet_T_world)
        )
        return pivot

    def _unrotated_cuboids(self) -> List[Cuboid]:
        """
        Returns the unrotated cuboids that must then be rotated to produce the final cubby.

        :rtype List[Cuboid]: All the cuboids in the cubby
        """
        cuboids = [
            # Floor
            Cuboid(
                center=np.array([0.0, 0.0, -0.01]),
                dims=np.array([2.0, 2.0, 0.0045]),
                quaternion=SO3.unit().q,
            ),
            # Back Wall
            Cuboid(
                center=np.array(
                    [
                        self.cubby_back,
                        (self.cubby_left + self.cubby_right) / 2,
                        self.cubby_top / 2,
                    ]
                ),
                dims=np.array(
                    [
                        self.thickness,
                        (self.cubby_left - self.cubby_right),
                        self.cubby_top,
                    ]
                ),
                quaternion=SO3.unit().q,
            ),
            # Bottom Shelf
            Cuboid(
                center=np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_right) / 2,
                        self.cubby_bottom,
                    ]
                ),
                dims=np.array(
                    [
                        self.cubby_back - self.cubby_front,
                        self.cubby_left - self.cubby_right,
                        self.thickness,
                    ]
                ),
                quaternion=SO3.unit().q,
            ),
            # Top Shelf
            Cuboid(
                center=np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_right) / 2,
                        self.cubby_top,
                    ]
                ),
                dims=np.array(
                    [
                        self.cubby_back - self.cubby_front,
                        self.cubby_left - self.cubby_right,
                        self.thickness,
                    ]
                ),
                quaternion=SO3.unit().q,
            ),
            # Right Wall
            Cuboid(
                center=np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        self.cubby_right,
                        (self.cubby_top + self.cubby_bottom) / 2,
                    ]
                ),
                dims=np.array(
                    [
                        self.cubby_back - self.cubby_front,
                        self.thickness,
                        (self.cubby_top - self.cubby_bottom) + self.thickness,
                    ]
                ),
                quaternion=SO3.unit().q,
            ),
            # Left Wall
            Cuboid(
                center=np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        self.cubby_left,
                        (self.cubby_top + self.cubby_bottom) / 2,
                    ]
                ),
                dims=np.array(
                    [
                        self.cubby_back - self.cubby_front,
                        self.thickness,
                        (self.cubby_top - self.cubby_bottom) + self.thickness,
                    ]
                ),
                quaternion=SO3.unit().q,
            ),
        ]
        if not np.isclose(self.cubby_mid_v_y, 0.0):
            # Center Wall (vertical)
            cuboids.append(
                Cuboid(
                    center=np.array(
                        [
                            (self.cubby_front + self.cubby_back) / 2,
                            self.cubby_mid_v_y,
                            (self.cubby_top + self.cubby_bottom) / 2,
                        ]
                    ),
                    dims=np.array(
                        [
                            self.cubby_back - self.cubby_front,
                            self.center_wall_thickness,
                            self.cubby_top - self.cubby_bottom + self.thickness,
                        ]
                    ),
                    quaternion=SO3.unit().q,
                )
            )
        if not np.isclose(self.cubby_mid_h_z, 0.0):
            # Middle Shelf
            cuboids.append(
                Cuboid(
                    center=np.array(
                        [
                            (self.cubby_front + self.cubby_back) / 2,
                            (self.cubby_left + self.cubby_right) / 2,
                            self.cubby_mid_h_z,
                        ]
                    ),
                    dims=np.array(
                        [
                            self.cubby_back - self.cubby_front,
                            self.cubby_left - self.cubby_right,
                            self.middle_shelf_thickness,
                        ]
                    ),
                    quaternion=SO3.unit().q,
                )
            )
        return cuboids

    @property
    def cuboids(self) -> List[Cuboid]:
        """
        Returns the cuboids that make up the cubby

        :rtype List[Cuboid]: The cuboids that make up each section of the cubby
        """
        cuboids: List[Cuboid] = []
        for cuboid in self._unrotated_cuboids():
            center = cuboid.center
            new_matrix = np.matmul(
                self.rotation_matrix,
                np.array(
                    [
                        [1, 0, 0, center[0]],
                        [0, 1, 0, center[1]],
                        [0, 0, 1, center[2]],
                        [0, 0, 0, 1],
                    ]
                ),
            )
            center = new_matrix[:3, 3]
            cuboids.append(
                Cuboid(
                    center,
                    cuboid.dims,
                    quaternion=SO3.from_matrix(new_matrix[:3, :3]).q,
                )
            )
        return cuboids

    @property
    def support_volumes(self) -> List[Cuboid]:
        """
        Returns the support volumes inside each of the cubby pockets.
        These support volumes could be tighter to make for more efficient environment
        queries, but right now they include half of the surrounding shelves.

        :rtype List[Cuboid]: The list of support volumes
        """
        if np.isclose(self.center_wall_thickness, 0) and np.isclose(
            self.middle_shelf_thickness, 0
        ):
            centers = [
                np.array(
                    [
                        self.cubby_front + self.cubby_back,
                        self.cubby_left + self.cubby_right,
                        self.cubby_top + self.cubby_bottom,
                    ]
                )
                / 2
            ]
            dims = [
                np.array(
                    [
                        self.cubby_back - self.cubby_front,
                        self.cubby_left - self.cubby_right,
                        self.cubby_top - self.cubby_bottom,
                    ]
                )
            ]
        elif np.isclose(self.center_wall_thickness, 0):
            centers = [
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_right) / 2,
                        (self.cubby_top + self.cubby_mid_h_z) / 2,
                    ]
                ),
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_right) / 2,
                        (self.cubby_bottom + self.cubby_mid_h_z) / 2,
                    ]
                ),
            ]
            dims = [
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_left - self.cubby_right),
                        (self.cubby_top - self.cubby_mid_h_z),
                    ]
                ),
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_left - self.cubby_right),
                        (self.cubby_mid_h_z - self.cubby_bottom),
                    ]
                ),
            ]
        elif np.isclose(self.middle_shelf_thickness, 0):
            centers = [
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_mid_v_y) / 2,
                        (self.cubby_top + self.cubby_bottom) / 2,
                    ]
                ),
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_mid_v_y + self.cubby_right) / 2,
                        (self.cubby_top + self.cubby_bottom) / 2,
                    ]
                ),
            ]
            dims = [
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_left - self.cubby_mid_v_y),
                        (self.cubby_top - self.cubby_bottom),
                    ]
                ),
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_mid_v_y - self.cubby_right),
                        (self.cubby_top - self.cubby_bottom),
                    ]
                ),
            ]
        else:
            centers = [
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_mid_v_y) / 2,
                        (self.cubby_top + self.cubby_mid_h_z) / 2,
                    ]
                ),
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_mid_v_y + self.cubby_right) / 2,
                        (self.cubby_top + self.cubby_mid_h_z) / 2,
                    ]
                ),
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_left + self.cubby_mid_v_y) / 2,
                        (self.cubby_bottom + self.cubby_mid_h_z) / 2,
                    ]
                ),
                np.array(
                    [
                        (self.cubby_front + self.cubby_back) / 2,
                        (self.cubby_mid_v_y + self.cubby_right) / 2,
                        (self.cubby_bottom + self.cubby_mid_h_z) / 2,
                    ]
                ),
            ]
            dims = [
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_left - self.cubby_mid_v_y),
                        (self.cubby_top - self.cubby_mid_h_z),
                    ]
                ),
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_mid_v_y - self.cubby_right),
                        (self.cubby_top - self.cubby_mid_h_z),
                    ]
                ),
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_left - self.cubby_mid_v_y),
                        (self.cubby_mid_h_z - self.cubby_bottom),
                    ]
                ),
                np.array(
                    [
                        (self.cubby_back - self.cubby_front),
                        (self.cubby_mid_v_y - self.cubby_right),
                        (self.cubby_mid_h_z - self.cubby_bottom),
                    ]
                ),
            ]

        volumes = []
        for c, d in zip(centers, dims):
            unrotated_pose = np.eye(4)
            unrotated_pose[:3, 3] = c
            pose = SE3.from_matrix(np.matmul(self.rotation_matrix, unrotated_pose))
            volumes.append(Cuboid(center=pose.pos, dims=d, quaternion=pose.so3.q))
        return volumes


class CubbyEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.demo_candidates = []
        pass

    def _gen(
        self,
        cooo: FrankaCollisionSpheres,
        prismatic_joint: float,
        scene_buffer: float,
        self_collision_buffer: float,
        joint_range_scalar: float,
    ) -> bool:
        """
        Generates an environment and a pair of start/end candidates
        """
        self.cubby = Cubby()
        self.prismatic_joint = prismatic_joint
        self.scene_buffer = scene_buffer
        self.self_collision_buffer = self_collision_buffer
        self.joint_range_scalar = joint_range_scalar
        support_idxs = np.arange(len(self.cubby.support_volumes))
        random.shuffle(support_idxs)
        supports = self.cubby.support_volumes

        (
            start_pose,
            start_q,
            start_support_volume,
            start_buffer,
            target_pose,
            target_q,
            target_support_volume,
            target_buffer,
        ) = (None, None, None, None, None, None, None, None)

        for ii, idx in enumerate(support_idxs):
            start_support_volume = supports[idx]
            start_pose, start_q, start_buffer = self.random_pose_and_config(
                cooo, start_support_volume
            )
            if start_pose is None or start_q is None:
                continue
            for jdx in support_idxs[ii + 1 :]:
                target_support_volume = supports[jdx]
                target_pose, target_q, target_buffer = self.random_pose_and_config(
                    cooo,
                    target_support_volume,
                )
                if target_pose is not None and target_q is not None:
                    break
            if target_pose is not None and target_q is not None:
                break

        if start_q is None or target_q is None:
            self.demo_candidates = None
            return False
        assert start_buffer is not None and target_buffer is not None
        self.demo_candidates = (
            CubbyCandidate(
                pose=start_pose,
                config=start_q,
                pocket_idx=idx,
                support_volume=start_support_volume,
                scene_buffer=start_buffer,
                self_collision_buffer=self.self_collision_buffer,
                joint_range_scalar=self.joint_range_scalar,
            ),
            CubbyCandidate(
                pose=target_pose,
                config=target_q,
                pocket_idx=jdx,
                support_volume=target_support_volume,
                scene_buffer=target_buffer,
                self_collision_buffer=self.self_collision_buffer,
                joint_range_scalar=self.joint_range_scalar,
            ),
        )
        return True

    def random_pose_and_config(
        self, cooo, support_volume: Cuboid
    ) -> Tuple[Optional[SE3], Optional[np.ndarray], Optional[float]]:
        """
        Creates a random end effector pose in the desired support volume and solves for
        collision free IK

        """
        samples = support_volume.sample_volume(100)
        pose, q, scene_buffer = None, None, self.scene_buffer
        for sample in samples:
            theta = radius_sample(0, np.pi / 4)
            z = np.array([np.cos(theta), np.sin(theta), 0])
            x = np.array([0.0, 0, -1])
            y = np.cross(z, x)
            pose = SE3.from_unit_axes(
                sample,
                x,
                y,
                z,
            )
            if cooo.franka_eef_collides_fast(
                pose,
                self.prismatic_joint,
                self.obstacle_arrays,
                "right_gripper",
                scene_buffer=self.scene_buffer,
            ):
                pose = None
                continue
            q = FrankaRealRobot.collision_free_ik(
                pose,
                self.prismatic_joint,
                cooo,
                self.obstacle_arrays,
                scene_buffer=self.scene_buffer,
                self_collision_buffer=self.self_collision_buffer,
                joint_range_scalar=self.joint_range_scalar,
                retries=1000,
            )
            if q is not None:
                break
        if pose is not None and q is not None:
            scene_buffer = np.max(
                [
                    self.scene_buffer,
                    min_franka_eef_distance(
                        pose,
                        self.prismatic_joint,
                        cooo,
                        self.obstacle_arrays,
                        "right_gripper",
                    ),
                    min_franka_arm_distance(
                        q,
                        self.prismatic_joint,
                        cooo,
                        self.obstacle_arrays,
                    ),
                ]
            )

        return (pose, q, scene_buffer)

    def _gen_neutral_candidates(
        self,
        how_many: int,
        cooo: FrankaCollisionSpheres,
    ) -> List[NeutralCandidate]:
        """
        Generates a set of neutral candidates (all collision free)

        :param how_many int: How many to generate ideally--can be less if there are a lot of failures
        :param cooo FrankaCollisionSpheres: TODO
        :rtype List[NeutralCandidate]: A list of neutral candidates
        """
        candidates: List[NeutralCandidate] = []
        for _ in range(how_many * 50):
            if len(candidates) >= how_many:
                break
            sample = FrankaRealRobot.random_neutral(method="uniform")
            if not cooo.franka_arm_collides_fast(
                sample,
                self.prismatic_joint,
                self.obstacle_arrays,
                scene_buffer=self.scene_buffer,
                self_collision_buffer=self.self_collision_buffer,
                check_self=True,
            ):
                pose = franka_arm_link_fk(sample, self.prismatic_joint, np.eye(4))[
                    RealFrankaConstants.ARM_LINKS.right_gripper
                ]
                candidates.append(
                    NeutralCandidate(
                        config=sample,
                        pose=SE3.from_matrix(pose),
                        scene_buffer=np.max(
                            [
                                self.scene_buffer,
                                min_franka_arm_distance(
                                    sample,
                                    self.prismatic_joint,
                                    cooo,
                                    self.obstacle_arrays,
                                ),
                            ]
                        ),
                        self_collision_buffer=self.self_collision_buffer,
                        joint_range_scalar=self.joint_range_scalar,
                    )
                )
        return candidates

    def _gen_additional_candidate_sets(
        self,
        how_many: int,
        cooo: FrankaCollisionSpheres,
    ) -> List[List[TaskOrientedCandidate]]:
        """
        Creates additional candidates, where the candidates correspond to the support volumes
        of the environment's generated candidates (created by the `gen` function)

        :param how_many int: How many candidates to generate in each support volume (the result is guaranteed
                             to match this number or the function will run forever)
        :param cooo TODO: TODO
        :rtype List[List[TaskOrientedCandidate]]: A pair of candidate sets from random pockets
        """
        supports = self.cubby.support_volumes
        candidates: List[TaskOrientedCandidate] = []
        while len(candidates) < 2 * how_many:
            idx = random.choice(range(len(supports)))
            pose, q, buffer = self.random_pose_and_config(cooo, supports[idx])
            if pose is not None and q is not None and buffer is not None:
                candidates.append(
                    CubbyCandidate(
                        pose=pose,
                        config=q,
                        pocket_idx=idx,
                        support_volume=supports[idx],
                        scene_buffer=self.scene_buffer,
                        self_collision_buffer=self.self_collision_buffer,
                        joint_range_scalar=self.joint_range_scalar,
                    )
                )
        return [candidates[:how_many], candidates[how_many:]]

    @property
    def obstacles(self) -> List[Cuboid]:
        """
        Returns all cuboids in the scene (and the scene is entirely made of cuboids)

        :rtype List[Cuboid]: The list of cuboids in this scene
        """
        return self.cubby.cuboids

    @property
    def obstacle_arrays(self) -> List[Union[CuboidArray, CylinderArray]]:
        """
        Returns all cuboids in the scene (and the scene is entirely made of cuboids)

        :rtype List[Cuboid]: The list of cuboids in this scene
        """
        return [CuboidArray(self.cubby.cuboids)]

    @property
    def cuboids(self) -> List[Cuboid]:
        """
        Returns all cuboids in the scene (and the scene is entirely made of cuboids)

        :rtype List[Cuboid]: The list of cuboids in this scene
        """
        return self.cubby.cuboids

    @property
    def cylinders(self) -> List[Cylinder]:
        """
        Returns an empty list because there are no cylinders in this scene, but left in
        to conform to the standard

        :rtype List[Cuboid]: The list of cuboids in this scene
        """
        return []
