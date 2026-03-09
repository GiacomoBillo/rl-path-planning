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

from typing import Tuple

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import MLP, PointNetConv, fps, radius
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.utils import to_dense_batch

from avoid_everything.transformer import (
    Encoder,
    FeedForward,
    MultiHeadAttention,
    TransformerLayer,
)

if not WITH_TORCH_CLUSTER:
    quit("This code requires 'torch-cluster'")


class PositionEncoding3D(nn.Module):
    """
    Generate sinusoidal positional encoding.

    f(p) = (sin(2^0 pi p), cos(2^0 pi p), ..., sin(2^L pi pi), cos(2^L pi p))
    From M2t2:
    https://github.com/NVlabs/M2T2/blob/734a5251e7ca36405c2b7056407db90db6c8e695/m2t2/contact_decoder.py#L51
    The primary difference from the source is that there was
    a bug in the positional encoding, which is fixed here.
    """

    def __init__(self, enc_dim, scale=np.pi, temperature=10000):
        super(PositionEncoding3D, self).__init__()
        self.enc_dim = enc_dim
        self.freq = np.ceil(enc_dim / 6)
        self.scale = scale
        self.temperature = temperature

    def forward(self, pos, bounds):
        pos_min = bounds[0]
        pos_max = bounds[1]
        
        # Avoid division by zero in position encoding
        pos_range = pos_max - pos_min
        if torch.any(pos_range == 0):
            pos_range = torch.clamp(pos_range, min=1e-8)
            
        pos = ((pos - pos_min) / pos_range - 0.5) * 2 * np.pi
        dim_t = torch.arange(self.freq, dtype=torch.float32, device=pos.device)
        dim_t = self.temperature ** (dim_t / self.freq)
        pos = pos[..., None] * self.scale / dim_t  # (B, N, 3, F)
        pos = torch.stack([pos.sin(), pos.cos()], dim=-1).flatten(start_dim=2)
        pos = pos[..., : self.enc_dim]
        return pos.detach()


class SAModule(nn.Module):
    """
    Set aggregation module from PointNet++ (based on implementation in pytorch geometric).
    """

    def __init__(self, ratio: float, r: float, net: nn.Module):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(net, add_self_loops=False)

    def forward(
        self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class MPiFormerPointNet(nn.Module):
    def __init__(self, num_robot_points: int, input_feature_dim: int, d_model: int):
        super().__init__()
        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(
            ratio=0.25, r=0.05, net=MLP([3 + input_feature_dim, 64, 64, 64])
        )
        self.sa2_module = SAModule(ratio=0.25, r=0.3, net=MLP([64 + 3, 128, 128, 256]))
        self.sa3_module = SAModule(
            ratio=0.25, r=0.5, net=MLP([256 + 3, 256, 512, d_model])
        )
        self.point_id_embedding = nn.Parameter(
            torch.randn((1, num_robot_points, input_feature_dim))
        )
        self.feature_encoder = nn.Embedding(3, input_feature_dim)
        self.num_robot_points = num_robot_points

    def forward(
        self,
        point_cloud_features: torch.Tensor,
        point_cloud: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass of the network

        :param point_cloud torch.Tensor: Has dimensions (B, N, 4)
                                              B is the batch size
                                              N is the number of points
                                              4 is x, y, z, segmentation_mask
                                              This tensor must be on the GPU (CPU tensors not supported)
        :rtype torch.Tensor: The output from the network
        """
        B, N, _ = point_cloud.shape
        pos = point_cloud.reshape(B * N, 3)  # Hard coded to fail if dimensions change
        x = self.feature_encoder(point_cloud_features.squeeze(-1).long())
        robot_features = x[:, : self.num_robot_points, :]
        other_features = x[:, self.num_robot_points :, :]
        robot_features = robot_features + self.point_id_embedding
        x = torch.cat((robot_features, other_features), dim=1)
        x = x.reshape(B * N, -1)

        batch_indices = torch.arange(B, device=point_cloud.device).unsqueeze(1)
        batch_indices = batch_indices.repeat(1, N)
        batch_indices = batch_indices.view(-1)

        x, pos, batch_indices = self.sa1_module(x, pos, batch_indices)
        x, pos, batch_indices = self.sa2_module(x, pos, batch_indices)
        x, pos, batch_indices = self.sa3_module(x, pos, batch_indices)
        x, x_mask = to_dense_batch(x, batch_indices)
        assert torch.all(x_mask), "Should be true because this PC has consistent size"
        pos, pos_mask = to_dense_batch(pos, batch_indices)
        assert torch.all(pos_mask), "Should be true because this PC has consistent size"
        return x, pos


class MotionPolicyTransformer(pl.LightningModule):
    """
    The MPiFormer architecture described by Fishman, et al.

    Uses a PointNet++ and an encoder-only transformer (with GELU activations)
    """

    def __init__(
        self,
        num_robot_points: int,
        robot_dof: int,
        *,
        feature_dim: int = 4,
        n_heads: int = 8,
        d_model: int = 512,
        n_layers: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.point_cloud_embedder = MPiFormerPointNet(
            num_robot_points, feature_dim, d_model
        )
        self.feature_embedder = nn.Linear(robot_dof, d_model)
        self.action_decoder = nn.Linear(d_model, robot_dof)
        encoder_layer = TransformerLayer(
            d_model=d_model,
            self_attn=MultiHeadAttention(
                heads=n_heads,
                d_model=d_model,
                dropout_prob=dropout,
            ),
            src_attn=None,  # No cross attention
            feed_forward=FeedForward(
                d_model=d_model,
                d_ff=4 * d_model,
                dropout=dropout,
                activation=nn.GELU,  # GELU gating
                is_gated=False,  # GELU gating
                bias1=True,  # GELU gating
                bias2=True,  # GELU gating
                bias_gate=True,  # GELU gating
            ),
            dropout_prob=dropout,
        )
        self.encoder = Encoder(encoder_layer, n_layers=n_layers)
        self.action_tokens = nn.Parameter(torch.randn((1, 1, d_model)))
        # Embedding instead of nn.parameter because it does gaussian initialization
        self.token_type_embedding = nn.Embedding(3, d_model)
        self.pe_layer = PositionEncoding3D(d_model)

    def encode(
        self,
        point_cloud_labels: torch.Tensor,
        point_cloud: torch.Tensor,
        q: torch.Tensor,
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        """Run the full BC encoder and return action token features.

        Returns:
            Tensor of shape (B, d_model) — the action token after the transformer,
            before the action decoder.
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
            torch.tensor(0, dtype=torch.long, device=self.device)
        )
        joint_state_type_emb = self.token_type_embedding(
            torch.tensor(1, dtype=torch.long, device=self.device)
        )[None, None, :]
        action_type_emb = self.token_type_embedding(
            torch.tensor(2, dtype=torch.long, device=self.device)
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
        action_token = self.encoder(embedded_sequence, mask=None)[-1:]  # (1, B, d_model)
        return action_token.squeeze(0)  # (B, d_model)

    def forward(
        self,
        point_cloud_labels: torch.Tensor,
        point_cloud: torch.Tensor,
        q: torch.Tensor,
        bounds: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore[override]
        features = self.encode(point_cloud_labels, point_cloud, q, bounds)  # (B, d_model)
        return self.action_decoder(features.unsqueeze(0)).transpose(0, 1)  # (B, 1, robot_dof)
