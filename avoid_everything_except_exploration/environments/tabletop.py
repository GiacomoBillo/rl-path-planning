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

from typing import List, Optional, Union

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
)
from avoid_everything.type_defs import NeutralCandidate, TaskOrientedCandidate


def random_linear_decrease():
    """
    Generates a random number according a distribution between 0 and 1 where the PDF looks
    like a linear line with slope -1. Useful for generating numbers within a range where
    the lower numbers are more preferred than the larger ones.
    """
    return 1 - np.sqrt(np.random.uniform())


class TabletopEnvironment(Environment):
    """
    A randomly constructed tabletop environment with random objects placed on it. The tabletop
    Table setup can be L-shaped or l-shaped, but will always include an obstacle free base
    table under the robot and some obstacle free space on the tables.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        """
        Resets the state of the environment
        """
        self.objects = []
        self.tables = []  # The tables with objects on it
        self.clear_tables = []  # The tables without objects

    def _gen(
        self,
        cooo: FrankaCollisionSpheres,
        prismatic_joint: float,
        scene_buffer: float,
        self_collision_buffer: float,
        joint_range_scalar: float,
        how_many: int,
    ) -> bool:
        """
        Generates the environment and a pair of valid candidates. The environment has a
        object-free table under the robot's base, as well as possibly some object-free
        sections on the table that are outside of the workspace.

        :param cooo FrankaCollisionSpheres: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :param how_many int: How many objects to put on the tables
        :rtype bool: Whether the environment was successfully generated
        """
        self.reset()
        self.setup_tables()
        self.place_objects(how_many)
        self.prismatic_joint = prismatic_joint
        self.scene_buffer = scene_buffer
        self.self_collision_buffer = self_collision_buffer
        self.joint_range_scalar = joint_range_scalar
        cand1 = self.gen_candidate(cooo)
        if cand1 is None:
            self.objects = []
            return False
        cand2 = self.gen_candidate(cooo)
        if cand2 is None:
            self.objects = []
            return False
        self.demo_candidates = [cand1, cand2]
        for c in self.cuboids:
            if np.any(c.dims < 0):
                return False
        for c in self.cylinders:
            if c.radius < 0 or c.height < 0:
                return False
        return True

    def _gen_neutral_candidates(
        self, how_many: int, cooo: FrankaCollisionSpheres
    ) -> List[NeutralCandidate]:
        """
        Generate a set of collision free neutral poses (represented as NeutralCandidate object)

        :param how_many int: How many neutral poses to generate
        :param cooo FrankaCollisionSpheres: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[NeutralCandidate]: A list of neutral poses
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

    def place_objects(self, how_many: int):
        """
        Places random objects on the table's surface, which _should_ not be overlapping.
        Overlap is calculated with a heuristic, so it's possible for them to overlap sometimes.

        :param how_many int: How many objects
        """
        center_candidates = self.random_points_on_table(10 * how_many)
        objects: List[Union[Cuboid, Cylinder]] = []
        for candidate in center_candidates:
            if len(objects) >= how_many:
                break
            candidate_is_good = True
            min_sdf = 1000
            for o in objects:
                sdf_value = o.sdf(candidate)
                if min_sdf is None or sdf_value < min_sdf:
                    min_sdf = sdf_value
                if o.sdf(candidate) <= 0.05:
                    candidate_is_good = False
            if candidate_is_good:
                x, y, z = candidate
                objects.append(self.random_object(x, y, z, 0.05, min_sdf))
        self.objects.extend(objects)

    @property
    def obstacles(self) -> List[Union[Cuboid, Cylinder]]:
        """
        Returns all obstacles in the scene.
        :rtype List[Union[Cuboid, Cylinder]]: The list of obstacles in the scene
        """
        return self.tables + self.clear_tables + self.objects

    @property
    def cuboids(self) -> List[Cuboid]:
        """
        Returns just the cuboids in the scene
        :rtype List[Cuboid]: The list of obstacles in the scene
        """
        return [o for o in self.obstacles if isinstance(o, Cuboid)]

    @property
    def cylinders(self) -> List[Cylinder]:
        """
        Returns just the cylinders in the scene
        :rtype List[Cylinder]: The list of obstacles in the scene
        """
        return [o for o in self.obstacles if isinstance(o, Cylinder)]

    @property
    def obstacle_arrays(self) -> List[Union[CuboidArray, CylinderArray]]:
        """
        Returns all cuboids in the scene (and the scene is entirely made of cuboids)

        :rtype List[Cuboid]: The list of cuboids in this scene
        """
        arrays = []
        cuboids = self.cuboids
        cylinders = self.cylinders
        if cuboids:
            arrays.append(CuboidArray(cuboids))
        if cylinders:
            arrays.append(CylinderArray(cylinders))
        return arrays

    def random_points_on_table(self, how_many: int) -> np.ndarray:
        """
        Generated random points on the table surface (to be used for obstacle placement).
        Points should be distributed _roughly_ evenly across all tables.

        :param how_many int: How many points to generate
        :rtype np.ndarray: A set of points, has dim [how_many, 3]
        """
        areas = []
        for t in self.tables:
            x, y, _ = t.dims
            areas.append(x * y)
        # We want to evenly sample points from each of the tables
        # First define which point will correspond to which tabletop
        table_choices = np.random.choice(
            np.arange(len(self.tables)),
            size=how_many,
            p=np.array(areas) / np.sum(areas),
        )
        # Sample points from the surface and then
        # only take those on the top
        # This relies on the fact that the tabletop is horizontal
        pointsets = []
        for t in self.tables:
            x_min, y_min, _ = np.min(t.corners, axis=0)
            x_max, y_max, z_max = np.max(t.corners, axis=0)
            # Always sample with points on the top surface
            pointsets.append(
                np.random.uniform(
                    [x_min, y_min, z_max],
                    [x_max, y_max, z_max],
                    size=(how_many, 3),
                )
            )
        return np.stack(pointsets)[table_choices, np.arange(how_many), :]

    def setup_tables(self):
        """
        Generate the random tables. Table setup can be L-shaped or l-shaped, but will always include
        an obstacle free base table under the robot. Additionally, objects are only placed
        within a randomly generated workspace within the table. The `self.tables` object
        has the tables that can have stuff on them. The `self.clear_tables` will have no objects
        placed on them.
        """
        table_height = np.random.choice(
            (np.random.uniform(0, 0.4), 0.0), p=[0.65, 0.35]
        )
        z = (table_height + -0.02) / 2
        dim_z = table_height + 0.02
        # Setup front table
        front_x_min = self.rand(0.3, 0.15)
        front_x_max = self.rand(1.05, 0.15)
        front_y_max = self.rand(0.72, 0.22)
        front_y_min = self.rand(-0.77, 0.18)

        whole_front_table = Cuboid(
            center=np.array(
                [
                    (front_x_min + front_x_max) / 2,
                    (front_y_min + front_y_max) / 2,
                    z,
                ]
            ),
            dims=np.array(
                [
                    (front_x_max - front_x_min),
                    (front_y_max - front_y_min),
                    dim_z,
                ]
            ),
            quaternioa=SO3.unit().q,
        )

        self.tables = [whole_front_table]
        self.clear_tables = []

        # Setup table that robot is mounted to
        # TODO (fishy) Finish this after lunch
        mount_table = Cuboid.random(
            center_range=np.array([0.0, 0.0, -0.01])
            + np.array([[-0.02, -0.02, 0.0], [0.02, 0.02, 0.0]]),
            dimension_range=np.array([1, 0.6, 0.02])
            + np.array(
                [
                    [0, -0.1, 0],
                    [0, 0.1, 0],
                ]
            ),
            random_orientation=False,
        )
        xdim_mean = 2 * (
            np.min(whole_front_table.corners[:, 0]) - mount_table.center[0]
        )
        xdim = min(
            xdim_mean + np.random.uniform(-0.03, 0.03),
            xdim_mean,
        )
        mount_table.dims[0] = xdim
        self.clear_tables.append(mount_table)

    def _gen_additional_candidate_sets(
        self, how_many: int, cooo: FrankaCollisionSpheres
    ) -> List[List[TaskOrientedCandidate]]:
        """
        Problems in the tabletop environment are symmetric, meaning that all candidates
        generated on the surface of the table are valid start/end poses. However, not all
        environments are symmetric, so this function is implemented here to match the
        general environment interface. This creates two sets of `how_many` candidates that
        are intended to be used as start/end respectively. Take the cartesian product of these
        two sets and you'll have a bunch of valid problems.

        :param how_many int: How many candidates to generate in each candidate set (the result
                             is guaranteed to match this number or the function will run forever)
        :param cooo FrankaCollisionSpheres: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[List[TaskOrientedCandidate]]: A list of candidate sets, where each has `how_many`
                                      candidates on the table.
        """
        candidate_sets = []
        for _ in range(2):
            candidate_set = []
            while len(candidate_set) <= how_many:
                candidate = self.gen_candidate(cooo)
                if candidate is not None:
                    candidate_set.append(candidate)
            candidate_sets.append(candidate_set)
        return candidate_sets

    def gen_candidate(
        self, cooo: FrankaCollisionSpheres
    ) -> Optional[TaskOrientedCandidate]:
        """
        Generates a valid, collision-free end effector pose (according to this
        environment's distribution) and a corresponding collision-free inverse kinematic
        solution. The poses will always be along the tabletop or on top of objects.
        They are densely distributed close to the surface and less frequently further
        above the table.

        :param cooo FrankaCollisionSpheres: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype TaskOrientedCandidate: A valid candidate
        :raises Exception: Raises an exception if there are unsupported objects on the table
        """
        points = self.random_points_on_table(100)
        q = None
        pose = None
        for p in points:
            for o in self.objects:
                if o.sdf(p) <= 0.01:
                    if isinstance(o, Cuboid):
                        p[2] = o.center[2] + o.half_extents[2]
                    elif isinstance(o, Cylinder):
                        p[2] = o.center[2] + o.height / 2
                    else:
                        raise Exception("Object can only be cuboid or cylinder")
            p[2] = p[2] + random_linear_decrease() * (0.12 - 0.01) / (1 - 0) + 0.01
            pose = SE3(p, self.random_orientation())
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
                choose_close_to=RealFrankaConstants.NEUTRAL,
            )
            if q is not None:
                break
        if pose is None or q is None:
            return None
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

        return TaskOrientedCandidate(
            pose=pose,
            config=q,
            scene_buffer=scene_buffer,
            self_collision_buffer=self.self_collision_buffer,
            joint_range_scalar=self.joint_range_scalar,
        )

    def random_object(
        self,
        x: float,
        y: float,
        table_top: float,
        xy_dim_min: float,
        xy_dim_max: float,
        z_dim_min: float = 0.05,
        z_dim_max: float = 0.35,
        cylinder_probability: float = 0.3,
    ) -> Union[Cuboid, Cylinder]:
        """
        Generate a random object on the table top. If a cylinder, will always be oriented
        so that the round face is parallel to the tabletop.

        :param x float: The x position of the object in the world frame
        :param y float: The y position of the object in the world frame
        :param table_top float: The height of the tabletop
        :param dim_min float: The minimum value to use for either the radius or the x or y dimension
        :param dim_max float: The maximum value to use for either the radius or the x or y dimension.
                              The value is clamped at 0.15.
        :rtype Union[Cuboid, Cylinder]: The primitive
        """
        xy_dim_max = min(xy_dim_max, 0.15)
        if np.random.rand() < cylinder_probability:
            c = Cylinder.random(
                radius_range=np.array([xy_dim_min, xy_dim_max]),
                height_range=np.array([0.05, 0.35]),
                random_orientation=False,
            )
            c.pose = SE3(np.array([x, y, c.height / 2 + table_top]), SO3.unit().q)
        else:
            c = Cuboid.random(
                dimension_range=np.array(
                    [
                        [xy_dim_min, xy_dim_min, z_dim_min],
                        [xy_dim_max, xy_dim_max, z_dim_max],
                    ]
                ),
                random_orientation=False,
            )
            c.pose = SE3(
                np.array([x, y, c.half_extents[2] + table_top]),
                SO3.from_rpy(0, 0, np.random.uniform(0, np.pi / 2)).q,
            )
        return c

    def random_orientation(self):
        roll = np.pi + np.random.uniform(-np.pi / 2, np.pi / 2)
        pitch = np.random.uniform(-np.pi / 2, np.pi / 2)
        yaw = np.random.uniform(-np.pi / 2, np.pi / 2)
        return SO3.from_rpy(roll, pitch, yaw).q

    @classmethod
    def rand(cls, center, dist):
        """
        Returns a normal distribution where the bottom and top are clipped to create
        something that looks mostly flat with a bump in the middle

        """
        min = center - dist
        sigma = 0.55 * dist
        return np.clip(np.random.normal(center, sigma), min, center + dist)
