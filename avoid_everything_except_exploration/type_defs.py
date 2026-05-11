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

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from geometrout.primitive import Cuboid, Cylinder, Sphere
from geometrout.transform import SE3

Obstacles = List[Union[Cuboid, Cylinder, Sphere]]
Trajectory = Sequence[Union[Sequence, np.ndarray]]


class DatasetType(IntEnum):
    """
    A simple enum class to indicate whether a dataloader is for training, validating, or testing
    """

    VAL_STATE = 0
    VAL = 1
    MINI_TRAIN = 2
    VAL_PRETRAIN = 3
    TEST = 4
    TRAIN = 5
    COL = 6 # Cycle of Learning (RL transitions)


@dataclass(kw_only=True)
class PlanningProblem:
    """
    Defines a common interface to describe planning problems
    """

    target: Union[SE3, np.ndarray, Tuple[SE3, np.ndarray]]
    q0: np.ndarray  # The starting configuration
    target_volume: Optional[Union[Cuboid, Cylinder]] = None
    obstacles: Optional[Obstacles] = None  # The obstacles in the scene
    obstacle_point_cloud: Optional[np.ndarray] = None
    target_negative_volumes: Obstacles = field(default_factory=lambda: [])

    @property
    def cuboids(self):
        if self.obstacles is None:
            return []
        return [o for o in self.obstacles if isinstance(o, Cuboid)]

    @property
    def cylinders(self):
        if self.obstacles is None:
            return []
        return [o for o in self.obstacles if isinstance(o, Cylinder)]

    @property
    def spheres(self):
        if self.obstacles is None:
            return []
        return [o for o in self.obstacles if isinstance(o, Sphere)]


@dataclass(kw_only=True)
class SolvedPlanningProblem(PlanningProblem):
    global_solution: Optional[np.ndarray] = field(default_factory=lambda: np.array([]))


ProblemSet = Dict[str, Dict[str, List[PlanningProblem]]]


class EnvironmentType(IntEnum):
    tabletop = auto()
    cubby = auto()


@dataclass
class Candidate:
    """
    A candidate problem to be fed to the expert data generation pipeline.

    Represents a configuration and the corresponding end-effector pose
    (in the right_gripper frame by default).
    """

    pose: SE3
    config: np.ndarray
    scene_buffer: float
    self_collision_buffer: float
    joint_range_scalar: float
    eff_frame: str = "right_gripper"


@dataclass
class TaskOrientedCandidate(Candidate):
    """
    Represents a configuration and the corresponding end-effector pose
    (in the right_gripper frame) for a task oriented pose.
    """

    pass


@dataclass
class NeutralCandidate(Candidate):
    """
    Represents a configuration and the corresponding end-effector pose
    (in the right_gripper frame) for a neutral pose.
    """

    pass
