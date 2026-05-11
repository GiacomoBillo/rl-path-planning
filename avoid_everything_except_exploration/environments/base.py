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

from abc import ABC, abstractmethod
from typing import Any, List, Union

import numpy as np
from geometrout.primitive import Cuboid, Cylinder
from robofin.collision import FrankaCollisionSpheres

from avoid_everything.type_defs import NeutralCandidate, TaskOrientedCandidate


def radius_sample(center: float, radius: float):
    """
    Helper function to draw a uniform sample with a fixed radius around a center

    :param center float: The center of the distribution
    :param radius float: The radius of the distribution
    """
    return np.random.uniform(center - radius, center + radius)


class Environment(ABC):
    def __init__(self):
        self.generated = False
        self.demo_candidates = []
        pass

    @property
    @abstractmethod
    def obstacles(self) -> List[Union[Cuboid, Cylinder]]:
        """
        Returns all obstacles in the scene.
        :rtype List[Union[Cuboid, Cylinder]]: The list of obstacles in the scene
        """
        pass

    @property
    @abstractmethod
    def cuboids(self) -> List[Cuboid]:
        """
        Returns just the cuboids in the scene
        :rtype List[Cuboid]: The list of cuboids in the scene
        """
        pass

    @property
    @abstractmethod
    def cylinders(self) -> List[Cylinder]:
        """
        Returns just the cylinders in the scene
        :rtype List[Cylinder]: The list of cylinders in the scene
        """
        pass

    def gen(
        self,
        cooo: FrankaCollisionSpheres,
        prismatic_joint: float,
        scene_buffer: float,
        self_collision_buffer: float,
        joint_range_scalar: float,
        **kwargs: Any,
    ) -> bool:
        """
        Generates an environment and a pair of start/end candidates

        :param cooo FrankaCollisionSpheres: TODO
        :prismatic_joint float: The value for the Franka's prismatic joint
        :buffer float: The collision buffer for distance to obstacles

        :rtype bool: Whether the environment was successfully generated
        """
        self.generated = self._gen(
            cooo,
            prismatic_joint,
            scene_buffer,
            self_collision_buffer,
            joint_range_scalar,
            **kwargs,
        )
        if self.generated:
            assert len(self.demo_candidates) == 2
            cand1, cand2 = self.demo_candidates
            assert cand1 is not None and cand2 is not None
        return self.generated

    def gen_additional_candidate_sets(
        self, how_many: int, cooo: FrankaCollisionSpheres
    ) -> List[List[TaskOrientedCandidate]]:
        """
        This creates two sets of `how_many` candidates that
        are intended to be used as start/end respectively. Take the cartesian product of these
        two sets and you'll have a bunch of valid problems.

        :param how_many int: How many candidates to generate in each candidate set (the result
                             is guaranteed to match this number or the function will run forever)
        :param cooo FrankaCollisionSpheres: TODO
        :rtype List[List[TaskOrientedCandidate]]: A list of candidate sets, where each has `how_many`
                                      candidates on the table.
        """
        assert (
            self.generated
        ), "Must run generate the environment before requesting additional candidates"
        return self._gen_additional_candidate_sets(how_many, cooo)

    def gen_neutral_candidates(
        self, how_many: int, cooo: FrankaCollisionSpheres
    ) -> List[NeutralCandidate]:
        """
        Generate a set of collision free neutral poses and corresponding configurations
        (represented as Candidate object)

        :param how_many int: How many neutral poses to generate
        :param cooo FrankaCollisionSpheres: TODO
        :rtype List[NeutralCandidate]: A list of neutral poses
        """
        assert (
            self.generated
        ), "Must run generate the environment before requesting additional candidates"
        return self._gen_neutral_candidates(how_many, cooo)

    @abstractmethod
    def _gen(
        self,
        cooo: FrankaCollisionSpheres,
        prismatic_joint,
        scene_buffer,
        self_collision_buffer,
        joint_range_scalar,
        **kwargs,
    ) -> bool:
        """
        The internal implementation of the gen function.

        :param cooo FranakCollisionSpheres: TODO
        :rtype bool: Whether the environment was successfully generated
        """
        pass

    @abstractmethod
    def _gen_additional_candidate_sets(
        self, how_many: int, cooo: FrankaCollisionSpheres
    ) -> List[List[TaskOrientedCandidate]]:
        """
        This creates two sets of `how_many` candidates that
        are intended to be used as start/end respectively. Take the cartesian product of these
        two sets and you'll have a bunch of valid problems.

        :param how_many int: How many candidates to generate in each candidate set (the result
                             is guaranteed to match this number or the function will run forever)
        :param cooo FrankaCollisionSpheres: TODO: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[List[TaskOrientedCandidate]]: A list of candidate sets, where each has `how_many`
                                      candidates on the table.
        """
        pass

    @abstractmethod
    def _gen_neutral_candidates(
        self, how_many: int, cooo: FrankaCollisionSpheres
    ) -> List[NeutralCandidate]:
        """
        Generate a set of collision free neutral poses and corresponding configurations
        (represented as NeutralCandidate object)

        :param how_many int: How many neutral poses to generate
        :param cooo FrankaCollisionSpheres: TODO: Checks for self collisions using spheres that
                                                  mimic the internal Franka collision checker.
        :rtype List[NeutralCandidate]: A list of neutral poses
        """
        pass


def min_franka_eef_distance(
    pose: np.ndarray,
    prismatic_joint: float,
    cooo: FrankaCollisionSpheres,
    primitive_arrays: List[Union[Cuboid, Cylinder]],
    frame: str,
) -> float:
    """
    Computes the minimum distance between the Franka's end effector and the primitives in the scene.
    """
    cspheres = cooo.eef_csphere_info(pose, prismatic_joint, frame)
    distances = [
        np.min(arr.scene_sdf(cspheres.centers) - cspheres.radii)
        for arr in primitive_arrays
    ]
    return np.min(distances) + 1e-6


def min_franka_arm_distance(
    q: np.ndarray,
    prismatic_joint: float,
    cooo: FrankaCollisionSpheres,
    primitive_arrays: List[Union[Cuboid, Cylinder]],
) -> float:
    """
    Computes the minimum distance between the Franka's arm and the primitives in the scene.
    """
    cspheres = cooo.csphere_info(q, prismatic_joint)
    distances = [
        np.min(arr.scene_sdf(cspheres.centers) - cspheres.radii)
        for arr in primitive_arrays
    ]
    return np.min(distances) + 1e-6
