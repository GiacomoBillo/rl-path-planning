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


class MPiFormerBackbone(nn.Module):
    """Frozen shared perception backbone: PointNet++ + joint encoder.
    
    Extracts point cloud and joint features from observations.
    Intended to be frozen and shared across actor and critic.
    
    Args:
        bc_model: MotionPolicyTransformer instance with point_cloud_embedder, feature_embedder, etc.
    """

    def __init__(self, bc_model: MotionPolicyTransformer):
        super().__init__()
        self.point_cloud_embedder = bc_model.point_cloud_embedder
        self.feature_embedder = bc_model.feature_embedder
        self.pe_layer = bc_model.pe_layer
        self.token_type_embedding = bc_model.token_type_embedding
        self.d_model = bc_model.d_model

    def forward(
        self,
        pc_labels: torch.Tensor,
        pc: torch.Tensor,
        q: torch.Tensor,
        bounds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract embeddings and positional encoding.
        
        Returns:
            pc_embedding: (B, N', D) point cloud embeddings after PointNet++
            pos: (B, N', 3) point cloud positions after PointNet++
            joint_embedding: (B, 1, D) joint state embedding
            pos_emb: (B, N'+2, D) positional encodings for transformer sequence
        """
        # Point cloud through PointNet++
        pc_embedding, pos = self.point_cloud_embedder(pc_labels, pc)
        
        # Joint state through encoder
        B = pc.size(0)
        joint_embedding = self.feature_embedder(q).unsqueeze(1)  # (B, 1, D)
        
        # Build positional encodings for transformer
        pc_type_emb = self.token_type_embedding(
            torch.tensor(0, dtype=torch.long, device=pc.device)
        )
        joint_state_type_emb = self.token_type_embedding(
            torch.tensor(1, dtype=torch.long, device=pc.device)
        )[None, None, :]
        
        pos_emb = torch.cat(
            (
                self.pe_layer(pos, bounds) + pc_type_emb,
                joint_state_type_emb.expand((B, -1, -1)),
            ),
            dim=1,
        )
        
        return pc_embedding, joint_embedding, pos_emb

    def freeze(self, components: list[Literal["pointnet", "joint_encoder"]] | None = None) -> None:
        """Freeze backbone components.
        
        Args:
            components: List of component names to freeze. None means all.
        """
        if components is None:
            components = ["pointnet", "joint_encoder"]
        
        for comp in components:
            if comp == "pointnet":
                for p in self.point_cloud_embedder.parameters():
                    p.requires_grad = False
            elif comp == "joint_encoder":
                for p in self.feature_embedder.parameters():
                    p.requires_grad = False

    def unfreeze(self, components: list[Literal["pointnet", "joint_encoder"]] | None = None) -> None:
        """Unfreeze backbone components.
        
        Args:
            components: List of component names to unfreeze. None means all.
        """
        if components is None:
            components = ["pointnet", "joint_encoder"]
        
        for comp in components:
            if comp == "pointnet":
                for p in self.point_cloud_embedder.parameters():
                    p.requires_grad = True
            elif comp == "joint_encoder":
                for p in self.feature_embedder.parameters():
                    p.requires_grad = True


class MPiFormerTransformerExtractor(BaseFeaturesExtractor):
    """Per-instance transformer feature extractor for SB3 actor/critic.
    
    Takes frozen backbone embeddings, applies transformer, extracts action token.
    Each actor/critic can have its own transformer instance (separate by default).
    
    Args:
        observation_space: Dict observation space from AvoidEverythingEnv.
        shared_backbone: MPiFormerBackbone instance (shared reference, frozen).
        bc_model: MotionPolicyTransformer for copying transformer weights.
        pc_bounds: Point-cloud normalisation bounds, shape (2, 3).
        freeze_backbone: If True, freeze backbone parameters on construction.
        shared_transformer: If True, share transformer across actor/critic (same object).
            If False (default), each gets its own deepcopy of the transformer.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        shared_backbone: MPiFormerBackbone,
        bc_model: MotionPolicyTransformer,
        pc_bounds: list,
        freeze_backbone: bool = True,
        shared_transformer: bool = False,
    ):
        super().__init__(observation_space, features_dim=bc_model.d_model)
        self.shared_backbone = shared_backbone
        self.register_buffer(
            "pc_bounds", torch.tensor(pc_bounds, dtype=torch.float32)
        )
        
        # Transformer encoder and action tokens (shared only if specified)
        if shared_transformer:
            self.encoder = bc_model.encoder
            self.action_tokens = bc_model.action_tokens  # Same Parameter object
        else:
            self.encoder = copy.deepcopy(bc_model.encoder)  # New independent module
            self.action_tokens = nn.Parameter(bc_model.action_tokens.clone())  # New Parameter with cloned data
        self.d_model = bc_model.d_model
        
        if freeze_backbone:
            self.freeze(components=["pointnet", "joint_encoder"])

    def forward(self, observations: dict) -> torch.Tensor:
        """Extract features via backbone + transformer.
        
        Args:
            observations: Dict with "point_cloud", "point_cloud_labels", "configuration"
        
        Returns:
            (B, d_model) action token embeddings from transformer
        """
        pc = observations["point_cloud"]  # (B, N, 3)
        pc_labels = observations["point_cloud_labels"]  # (B, N, 1)
        q = observations["configuration"]  # (B, dof)
        B = pc.size(0)
        
        # Get embeddings from backbone
        pc_embedding, joint_embedding, pos_emb = self.shared_backbone(
            pc_labels, pc, q, self.pc_bounds
        )
        
        # Build full sequence for transformer
        action_token_type_emb = self.shared_backbone.token_type_embedding(
            torch.tensor(2, dtype=torch.long, device=pc.device)
        )[None, None, :]
        
        sequence = torch.cat(
            (
                pc_embedding,
                joint_embedding,
                self.action_tokens.expand((B, -1, -1)),
            ),
            dim=1,
        ).transpose(0, 1)  # (N'+2, B, D)
        
        action_type_emb = action_token_type_emb.expand((B, 1, -1))
        pos_emb_full = torch.cat((pos_emb, action_type_emb), dim=1).transpose(0, 1)  # (N'+3, B, D)
        
        embedded_sequence = sequence + pos_emb_full
        action_token = self.encoder(embedded_sequence, mask=None)[-1:]  # (1, B, D)
        
        return action_token.squeeze(0)  # (B, D)

    def freeze(self, components: list[Literal["pointnet", "joint_encoder", "transformer"]] | None = None) -> None:
        """Freeze components.
        
        Args:
            components: List of component names to freeze. None means all.
        """
        if components is None:
            components = ["pointnet", "joint_encoder", "transformer"]
        
        for comp in components:
            if comp in ["pointnet", "joint_encoder"]:
                self.shared_backbone.freeze(components=[comp])
            elif comp == "transformer":
                for p in self.encoder.parameters():
                    p.requires_grad = False
                self.action_tokens.requires_grad = False

    def unfreeze(self, components: list[Literal["pointnet", "joint_encoder", "transformer"]] | None = None) -> None:
        """Unfreeze components.
        
        Args:
            components: List of component names to unfreeze. None means all.
        """
        if components is None:
            components = ["pointnet", "joint_encoder", "transformer"]
        
        for comp in components:
            if comp in ["pointnet", "joint_encoder"]:
                self.shared_backbone.unfreeze(components=[comp])
            elif comp == "transformer":
                for p in self.encoder.parameters():
                    p.requires_grad = True
                self.action_tokens.requires_grad = True


class MPiFormerExtractor(BaseFeaturesExtractor):
    """Legacy feature extractor that wraps the MPiFormer BC encoder.
    
    (DEPRECATED: Use MPiFormerTransformerExtractor + MPiFormerBackbone instead)

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
