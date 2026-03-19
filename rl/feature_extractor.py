"""
SB3 feature extractors for MPiFormer RL training.

Splits the BC model into:
- MPiFormerBackbone: frozen PointNet++ + joint encoder (shared across actor/critic)
- MPiFormerTransformerExtractor: per-instance transformer + action token decoder (not shared)

This enables selective freezing and per-actor/critic training of the transformer.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Literal

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch_geometric.utils import to_dense_batch

if TYPE_CHECKING:
    from avoid_everything.mpiformer import MotionPolicyTransformer


class MPiFormerPerceptionEncoder(nn.Module):
    """ Perception encoder (first part of the encoder) of MPiFormer.

    contains:
    - PointNet++ point cloud embedder
    - robot joint state encoder
    - action token
    - positional encoding layer (not learnable)
    - type embeddings

    Args:
        bc_model: MotionPolicyTransformer to extract perception encoder components from.
        deep_copy: If True, deep copy the components. If False, share the components.
    """

    def __init__(self, bc_model: MotionPolicyTransformer, deep_copy: bool = False):
        super().__init__()
        self.d_model = bc_model.d_model
        if deep_copy:
            self.point_cloud_embedder = copy.deepcopy(bc_model.point_cloud_embedder)
            self.feature_embedder = copy.deepcopy(bc_model.feature_embedder)
            self.action_tokens = nn.Parameter(bc_model.action_tokens.clone())
            self.pe_layer = copy.deepcopy(bc_model.pe_layer)
            self.token_type_embedding = copy.deepcopy(bc_model.token_type_embedding)
        else:
            self.point_cloud_embedder = bc_model.point_cloud_embedder
            self.feature_embedder = bc_model.feature_embedder
            self.action_tokens = bc_model.action_tokens
            self.pe_layer = bc_model.pe_layer
            self.token_type_embedding = bc_model.token_type_embedding

    def forward(
        self,
        point_cloud_labels: torch.Tensor,
        point_cloud: torch.Tensor,
        q: torch.Tensor,
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            embedded_sequence: (SeqLen, B, D)
        """
        assert point_cloud_labels.shape[:2] == point_cloud.shape[:2]
        pc_embedding, pos = self.point_cloud_embedder(point_cloud_labels, point_cloud)
        feature_embedding = self.feature_embedder(q).unsqueeze(1)
        B = point_cloud.size(0)
        sequence = torch.cat(
            (
                pc_embedding,
                feature_embedding,
                self.action_tokens.expand((B, -1, -1)),
            ),
            dim=1,
        ).transpose(0, 1)

        pc_type_emb = self.token_type_embedding(
            torch.tensor(0, dtype=torch.long, device=point_cloud.device)
        )
        joint_state_type_emb = self.token_type_embedding(
            torch.tensor(1, dtype=torch.long, device=point_cloud.device)
        )[None, None, :]
        action_type_emb = self.token_type_embedding(
            torch.tensor(2, dtype=torch.long, device=point_cloud.device)
        )[None, None, :]

        pos_emb = torch.cat(
            (
                self.pe_layer(pos, bounds) + pc_type_emb,
                joint_state_type_emb.expand((B, -1, -1)),
                action_type_emb.expand((B, 1, -1)),
            ),
            dim=1,
        ).transpose(0, 1)
        embedded_sequence = sequence + pos_emb
        return embedded_sequence

    def freeze(self, components: list[Literal["pointnet", "joint_encoder", "action_token", "type_embeddings"]] | None = None) -> None:
        """Freeze components.
        
        Args:
            components: List of component names to freeze. None means all.
        """
        if components is None:
            components = ["pointnet", "joint_encoder", "action_token", "type_embeddings"]
        
        for comp in components:
            if comp == "pointnet":
                for p in self.point_cloud_embedder.parameters():
                    p.requires_grad = False
            elif comp == "joint_encoder":
                for p in self.feature_embedder.parameters():
                    p.requires_grad = False
            elif comp == "action_token":
                self.action_tokens.requires_grad = False
            elif comp == "type_embeddings":
                for p in self.token_type_embedding.parameters():
                    p.requires_grad = False
            else:
                raise ValueError(f"Unknown component name: {comp}")

    def unfreeze(self, components: list[Literal["pointnet", "joint_encoder", "action_token", "type_embeddings"]] | None = None) -> None:
        """Unfreeze components.
        
        Args:
            components: List of component names to unfreeze. None means all.
        """
        if components is None:
            components = ["pointnet", "joint_encoder", "action_token", "type_embeddings"]
        
        for comp in components:
            if comp == "pointnet":
                for p in self.point_cloud_embedder.parameters():
                    p.requires_grad = True
            elif comp == "joint_encoder":
                for p in self.feature_embedder.parameters():
                    p.requires_grad = True
            elif comp == "action_token":
                self.action_tokens.requires_grad = True
            elif comp == "type_embeddings":
                for p in self.token_type_embedding.parameters():
                    p.requires_grad = True
            else:
                raise ValueError(f"Unknown component name: {comp}")


class MPiFormerTransformerEncoder(nn.Module):
    """Transformer encoder (second part of the encoder) of MPiFormer.

    contains:
    - transformer layers

    Args:
        bc_model: MotionPolicyTransformer to extract transformer encoder components from.
        deep_copy: If True, deep copy the components. If False, share the components.
    """

    def __init__(self, bc_model: MotionPolicyTransformer, deep_copy: bool = False):
        super().__init__()
        self.d_model = bc_model.d_model
        if deep_copy:
            self.transformer = copy.deepcopy(bc_model.encoder)
        else:
            self.transformer = bc_model.encoder

    def forward(self, embedded_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedded_sequence: (SeqLen, B, D)
            
        Returns:
            action_embedding: (B, D)
        """
        action_embedding = self.transformer(embedded_sequence, mask=None)[-1:]  # (1, B, D)
        return action_embedding.squeeze(0)  # (B, D)

    def freeze(self) -> None:
        """Freeze transformer encoder parameters."""
        for p in self.transformer.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze transformer encoder parameters."""
        for p in self.transformer.parameters():
            p.requires_grad = True
   

class MPiFormerExtractor(BaseFeaturesExtractor):
    """Feature extractor that wraps/manages the MPiFormer BC encoder 
    for loading a pre-trained model into the SB3 RL policy.
    
    Wraps ``bc_model.encode()`` via split encoders so SB3's actor and critic 
    can share or copy the pre-trained representation.
    
    The encoders can be frozen/unfrozen independently.

    Args:
        observation_space: Dict observation space from AvoidEverythingEnv.
        bc_model: Loaded MotionPolicyTransformer (or subclass) instance.
        pc_bounds: Point-cloud normalisation bounds, shape (2, 3) or equivalent
            nested list ``[min_xyz, max_xyz]``.  Registered as a buffer so it
            moves to the correct device automatically.
        freeze_perception: If True, freeze perception encoder parameters on construction. Default: True.
        freeze_transformer: If True, freeze transformer encoder parameters on construction. Default: False.
        deep_copy_perception: If True, deep copy the perception components from bc_model. Default: True.
        deep_copy_transformer: If True, deep copy the transformer components from bc_model. Default: True.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        bc_model: MotionPolicyTransformer,
        pc_bounds: list,
        freeze_perception: bool = True,
        freeze_transformer: bool = False,
        deep_copy_perception: bool = True,
        deep_copy_transformer: bool = True,
    ):
        super().__init__(observation_space, features_dim=bc_model.d_model)
        
        self.perception_encoder = MPiFormerPerceptionEncoder(bc_model, deep_copy=deep_copy_perception)
        self.transformer_encoder = MPiFormerTransformerEncoder(bc_model, deep_copy=deep_copy_transformer)
        
        self.register_buffer(
            "pc_bounds", torch.tensor(pc_bounds, dtype=torch.float32)
        )
        
        if freeze_perception:
            self.freeze_perception()
        else:
            self.unfreeze_perception()
            
        if freeze_transformer:
            self.freeze_transformer()
        else:
            self.unfreeze_transformer()

    def freeze_perception(self) -> None:
        """Freeze perception encoder parameters."""
        self.perception_encoder.freeze()

    def unfreeze_perception(self) -> None:
        """Unfreeze perception encoder parameters."""
        self.perception_encoder.unfreeze()

    def freeze_transformer(self) -> None:
        """Freeze transformer encoder parameters."""
        self.transformer_encoder.freeze()

    def unfreeze_transformer(self) -> None:
        """Unfreeze transformer encoder parameters."""
        self.transformer_encoder.unfreeze()

    def forward(self, observations: dict) -> torch.Tensor:
        """
        Extract features from observations by running through the perception encoder
        and then use the transformer encoder to get the final action embedding.
        """
        pc = observations["point_cloud"]  # (B, N, 3)
        pc_labels = observations["point_cloud_labels"]  # (B, N, 1)
        q = observations["configuration"]  # (B, dof)
        
        embedded_sequence = self.perception_encoder(pc_labels, pc, q, self.pc_bounds)
        action_embedding = self.transformer_encoder(embedded_sequence)
        return action_embedding