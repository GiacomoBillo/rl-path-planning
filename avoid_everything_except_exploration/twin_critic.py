import torch
from torch import nn

from avoid_everything_except_exploration.mpiformer import (
    MPiFormerPointNet,
    Encoder, TransformerLayer, MultiHeadAttention, FeedForward,
    PositionEncoding3D,
)

class _CriticTrunk(nn.Module):
    """Takes precomputed (pc_emb, pos), plus (q, a), and outputs Q."""
    def __init__(self, robot_dof: int, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.state_embedder  = nn.Linear(robot_dof, d_model)
        self.action_embedder = nn.Linear(robot_dof, d_model)
        self.token_type_embedding = nn.Embedding(3, d_model)  # 0=pc,1=state,2=action
        self.pe_layer = PositionEncoding3D(d_model)

        enc_layer = TransformerLayer(
            d_model=d_model,
            self_attn=MultiHeadAttention(heads=n_heads, d_model=d_model, dropout_prob=dropout),
            src_attn=None,
            feed_forward=FeedForward(
                d_model=d_model, d_ff=4*d_model, dropout=dropout,
                activation=nn.GELU, is_gated=False, bias1=True, bias2=True, bias_gate=True
            ),
            dropout_prob=dropout,
        )
        self.encoder = Encoder(enc_layer, n_layers=n_layers)
        self.q_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1),
        )

    def forward(self, pc_emb: torch.Tensor, pos: torch.Tensor,
                q: torch.Tensor, a: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        B = q.size(0)
        s_tok = self.state_embedder(q).unsqueeze(1)
        a_tok = self.action_embedder(a).unsqueeze(1)

        seq = torch.cat((pc_emb, s_tok, a_tok), dim=1).transpose(0, 1)  # [S,B,D]

        dev = pc_emb.device
        pc_type = self.token_type_embedding(torch.tensor(0, device=dev))
        s_type  = self.token_type_embedding(torch.tensor(1, device=dev))[None, None, :]
        a_type  = self.token_type_embedding(torch.tensor(2, device=dev))[None, None, :]

        pos_emb = torch.cat(
            (self.pe_layer(pos, bounds) + pc_type, s_type.expand(B, -1, -1), a_type.expand(B, 1, -1)),
            dim=1
        ).transpose(0, 1)

        h = self.encoder(seq + pos_emb, mask=None)   # [S,B,D]
        h_a = h[-1]                                   # action token
        return self.q_head(h_a)                       # [B,1]


class TwinCritic(nn.Module):
    """
    Shared point-cloud encoder -> two independent Q heads (TD3-style).
    Saves ~1x PointNet++ forward per step.
    """
    def __init__(self, 
                 num_robot_points: int,
                 robot_dof: int,
                 *,
                 feature_dim: int = 4, # match actor feature_dim
                 n_heads: int = 4,
                 d_model: int = 512,   # match actor d_model
                 n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.pc_encoder = MPiFormerPointNet(num_robot_points, feature_dim, d_model)
        self.q1 = _CriticTrunk(robot_dof, d_model, n_heads, n_layers, dropout)
        self.q2 = _CriticTrunk(robot_dof, d_model, n_heads, n_layers, dropout)

    def forward(self, point_cloud_labels: torch.Tensor, point_cloud: torch.Tensor,
                q: torch.Tensor, a: torch.Tensor, bounds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pc_emb, pos = self.pc_encoder(point_cloud_labels, point_cloud)
        return (
            self.q1(pc_emb, pos, q, a, bounds),
            self.q2(pc_emb, pos, q, a, bounds),
        )
