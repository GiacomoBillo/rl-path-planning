"""
This file contains the CoLLossFn class, which is responsible for calculating
the loss functions for the CoL motion policy.
"""

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

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from robofin.robots import Robot
from robofin.samplers import TorchRobotSampler

from avoid_everything_except_exploration.geometry import TorchCuboids, TorchCylinders


def point_match_loss(
    input_pc: torch.Tensor, target_pc: torch.Tensor, reduction="mean"
) -> torch.Tensor:
    """
    A combination L1 and L2 loss to penalize large and small deviations between
    two point clouds

    :param input_pc torch.Tensor: Point cloud sampled from the network's output.
                                  Has dim [B, N, 3]
    :param target_pc torch.Tensor: Point cloud sampled from the supervision
                                   Has dim [B, N, 3]
    :rtype torch.Tensor: The loss
    """
    return F.mse_loss(input_pc, target_pc, reduction=reduction) + F.l1_loss(
        input_pc, target_pc, reduction=reduction
    )

def collision_loss(
    input_pc: torch.Tensor,
    cuboid_centers: torch.Tensor,
    cuboid_dims: torch.Tensor,
    cuboid_quaternions: torch.Tensor,
    cylinder_centers: torch.Tensor,
    cylinder_radii: torch.Tensor,
    cylinder_heights: torch.Tensor,
    cylinder_quaternions: torch.Tensor,
    margin,
    reduction="mean",
) -> torch.Tensor:
    """
    Calculates the hinge loss, calculating whether the robot (represented as a
    point cloud) is in collision with any obstacles in the scene. Collision
    here actually means within 3cm of the obstacle--this is to provide stronger
    gradient signal to encourage the robot to move out of the way. Also, some 
    of the primitives can have zero volume (i.e. a dim is zero for cuboids or 
    radius or height is zero for cylinders). If these are zero volume, they 
    will have infinite sdf values (and therefore be ignored by the loss).

    :param input_pc torch.Tensor: Points sampled from the robot's surface after it
                                  is placed at the network's output prediction. Has dim [B, N, 3]
    :param cuboid_centers torch.Tensor: Has dim [B, M1, 3]
    :param cuboid_dims torch.Tensor: Has dim [B, M1, 3]
    :param cuboid_quaternions torch.Tensor: Has dim [B, M1, 4]. Quaternion is formatted as w, x, y, z.
    :param cylinder_centers torch.Tensor: Has dim [B, M2, 3]
    :param cylinder_radii torch.Tensor: Has dim [B, M2, 1]
    :param cylinder_heights torch.Tensor: Has dim [B, M2, 1]
    :param cylinder_quaternions torch.Tensor: Has dim [B, M2, 4]. Quaternion is formatted as w, x, y, z.
    :rtype torch.Tensor: Returns the loss value aggregated over the batch
    """

    cuboids = TorchCuboids(
        cuboid_centers,
        cuboid_dims,
        cuboid_quaternions,
    )
    cylinders = TorchCylinders(
        cylinder_centers,
        cylinder_radii,
        cylinder_heights,
        cylinder_quaternions,
    )
    sdf_values = torch.minimum(cuboids.sdf(input_pc), cylinders.sdf(input_pc))
    return F.hinge_embedding_loss(
        sdf_values,
        -torch.ones_like(sdf_values),
        margin=margin,
        reduction=reduction,
    )

class CoLLossFn:
    """
    A container class to hold the various losses. This is structured as a
    container because that allows it to cache the robot pointcloud sampler
    object. By caching this, we reduce parsing time when processing the URDF
    and allow for a consistent random pointcloud (consistent per-GPU, that is)
    """

    def __init__(
        self,
        urdf_path: str,
        collision_loss_margin: float,
    ):
        self.urdf_path = urdf_path
        self.robot = None
        self.fk_sampler = None
        self.num_points = 1024
        self.collision_loss_margin = collision_loss_margin

    def _ensure_robot_sampler(self, device: torch.device):
        if self.fk_sampler is None:
            self.robot = Robot(self.urdf_path, device=device)
            self.fk_sampler = TorchRobotSampler(
                self.robot,
                num_robot_points=self.num_points,
                num_eef_points=128,
                with_base_link=False,
                use_cache=True,
                device=device,
            )
        assert self.fk_sampler.device == device, "Sampler device mismatch"

    def sample(self, q: torch.Tensor) -> torch.Tensor:
        assert self.fk_sampler is not None
        return self.fk_sampler.sample(q)

    def bc_pointcloud_loss(
        self,
        *,
        pred_q_unnorm: torch.Tensor,
        expert_q_unnorm: torch.Tensor,
        is_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Behavioral cloning loss in point-cloud space, masked to expert rows.

        Returns a scalar: mean over expert samples of
        MSE(pred_pc, expert_pc) + L1(pred_pc, expert_pc)
        where both MSE and L1 are computed elementwise (reduction='none'),
        then averaged over points & coords per sample.
        """
        self._ensure_robot_sampler(pred_q_unnorm.device)
        pred_pc   = self.sample(pred_q_unnorm)[..., :3]   # [B, N, 3]
        expert_pc = self.sample(expert_q_unnorm)[..., :3] # [B, N, 3]

        # elementwise loss: [B, N, 3]
        per_elem = point_match_loss(pred_pc, expert_pc, reduction="none")
        # per-sample scalar: [B]
        per_sample = per_elem.mean(dim=(1, 2))

        if is_expert is None:
            return per_sample.mean()

        # mask to expert rows only
        mask = is_expert
        if mask.dim() > 1:
            mask = mask.squeeze(-1)
        mask = mask.to(dtype=per_sample.dtype)

        denom = mask.sum().clamp_min(1.0)
        return (per_sample * mask).sum() / denom

    def collision_loss(
        self,
        unnormalized_q: torch.Tensor,
        cuboid_centers: torch.Tensor,
        cuboid_dims: torch.Tensor,
        cuboid_quaternions: torch.Tensor,
        cylinder_centers: torch.Tensor,
        cylinder_radii: torch.Tensor,
        cylinder_heights: torch.Tensor,
        cylinder_quaternions: torch.Tensor,
        reduction="mean",
    ) -> torch.Tensor:
        """
        This method calculates the collision loss.

        :param unnormalized_q torch.Tensor: Unnormalized configuration, has dim [B, DOF]
        :param cuboid_centers torch.Tensor: Has dim [B, M1, 3]
        :param cuboid_dims torch.Tensor: Has dim [B, M1, 3]
        :param cuboid_quaternions torch.Tensor: Has dim [B, M1, 4]. Quaternion is formatted as w, x, y, z.
        :param cylinder_centers torch.Tensor: Has dim [B, M2, 3]
        :param cylinder_radii torch.Tensor: Has dim [B, M2, 1]
        :param cylinder_heights torch.Tensor: Has dim [B, M2, 1]
        :param cylinder_quaternions torch.Tensor: Has dim [B, M2, 4]. Quaternion is formatted as w, x, y, z.
        :rtype torch.Tensor: The collision loss value aggregated over the batch
        """
        self._ensure_robot_sampler(unnormalized_q.device)
        input_pc = self.sample(unnormalized_q)[..., :3]
        return collision_loss(
            input_pc,
            cuboid_centers,
            cuboid_dims,
            cuboid_quaternions,
            cylinder_centers,
            cylinder_radii,
            cylinder_heights,
            cylinder_quaternions,
            self.collision_loss_margin,
            reduction=reduction,
        )