"""
SB3 feature extractor that wraps the MPiFormer BC encoder.
Used for transfer learning of the pretrained BC model into the RL agent

The BC encoder (point_cloud_embedder + transformer) maps raw observations to a
512-dim feature vector. This extractor plugs it into SB3's actor/critic as a
shared representation, enabling warm-start of the SAC policy from BC weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

if TYPE_CHECKING:
    from avoid_everything.mpiformer import MotionPolicyTransformer


class MPiFormerExtractor(BaseFeaturesExtractor):
    """Feature extractor backed by the MPiFormer BC encoder.

    Wraps ``bc_model.encode()`` so SB3's actor and critic can share the pre-trained representation.  
    The encoder can be frozen during early RL training and later unfrozen for end-2-end fine-tuning

    Args:
        observation_space: Dict observation space from AvoidEverythingEnv.
        bc_model: Loaded MotionPolicyTransformer (or subclass) instance.
        pc_bounds: Point-cloud normalisation bounds, shape (2, 3) or equivalent
            nested list ``[min_xyz, max_xyz]``.  Registered as a buffer so it
            moves to the correct device automatically.
        freeze: If True, freeze all BC encoder parameters on construction.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        bc_model: MotionPolicyTransformer,
        pc_bounds: list,
        freeze: bool = True,
    ):
        super().__init__(observation_space, features_dim=bc_model.d_model)
        self.bc_model = bc_model
        self.register_buffer(
            "pc_bounds", torch.tensor(pc_bounds, dtype=torch.float32)
        )
        if freeze:
            self.freeze()

    def freeze(self) -> None:
        """Freeze all BC encoder parameters (no gradients)."""
        for p in self.bc_model.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze BC encoder parameters for end-to-end fine-tuning."""
        for p in self.bc_model.parameters():
            p.requires_grad = True

    def forward(self, observations: dict) -> torch.Tensor:
        pc = observations["point_cloud"]  # (B, N, 3) float32
        pc_labels = observations["point_cloud_labels"]  # (B, N, 1) int32; .long() applied internally
        q = observations["configuration"]  # (B, dof) float32
        return self.bc_model.encode(pc_labels, pc, q, self.pc_bounds)  # (B, d_model)
