from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Sequence

import torch
from torch import nn

try:
    from tensordict.nn import TensorDictModule, TensorDictSequential
    from torchrl.modules import ProbabilisticActor, TanhNormal
except ImportError:  # TorchRL is optional until integrated in the runtime path.
    TensorDictModule = None
    TensorDictSequential = None
    ProbabilisticActor = None
    TanhNormal = None

if TYPE_CHECKING:
    from avoid_everything.mpiformer import MotionPolicyTransformer


def _require_torchrl() -> None:
    if TensorDictModule is None or ProbabilisticActor is None or TanhNormal is None:
        raise ImportError(
            "TorchRL components are required. Install torchrl and tensordict first."
        )


def _validate_split_layer(split_layer: int, total_layers: int) -> int:
    if split_layer < 0 or split_layer > total_layers:
        raise ValueError(
            f"split_layer must be in [0, {total_layers}], got {split_layer}."
        )
    return split_layer


class MPiFormerBackboneNetwork(nn.Module):
    """Frozen backbone network: perception + first split_layer transformer layers."""

    def __init__(
        self,
        bc_model: MotionPolicyTransformer,
        *,
        pc_bounds: Sequence[Sequence[float]] | torch.Tensor,
        split_layer: int = 7,
        deep_copy: bool = True,
    ):
        super().__init__()
        total_layers = len(bc_model.encoder.layers)
        split_layer = _validate_split_layer(split_layer, total_layers)

        self.split_layer = split_layer
        self.total_layers = total_layers

        if deep_copy:
            self.point_cloud_embedder = copy.deepcopy(bc_model.point_cloud_embedder)
            self.feature_embedder = copy.deepcopy(bc_model.feature_embedder)
            self.action_tokens = nn.Parameter(bc_model.action_tokens.detach().clone())
            self.pe_layer = copy.deepcopy(bc_model.pe_layer)
            self.token_type_embedding = copy.deepcopy(bc_model.token_type_embedding)
            self.transformer_layers = nn.ModuleList(
                copy.deepcopy(list(bc_model.encoder.layers[:split_layer]))
            )
        else:
            self.point_cloud_embedder = bc_model.point_cloud_embedder
            self.feature_embedder = bc_model.feature_embedder
            self.action_tokens = bc_model.action_tokens
            self.pe_layer = bc_model.pe_layer
            self.token_type_embedding = bc_model.token_type_embedding
            self.transformer_layers = nn.ModuleList(list(bc_model.encoder.layers[:split_layer]))

        self.register_buffer("pc_bounds", torch.as_tensor(pc_bounds, dtype=torch.float32))

        # Backbone is always frozen.
        for param in self.parameters():
            param.requires_grad = False

    def _embed(
        self,
        point_cloud_labels: torch.Tensor,
        point_cloud: torch.Tensor,
        configuration: torch.Tensor,
    ) -> torch.Tensor:
        pc_embedding, pos = self.point_cloud_embedder(point_cloud_labels, point_cloud)
        q_embedding = self.feature_embedder(configuration).unsqueeze(1)
        batch_size = point_cloud.size(0)

        sequence = torch.cat(
            (pc_embedding, q_embedding, self.action_tokens.expand((batch_size, -1, -1))),
            dim=1,
        ).transpose(0, 1)

        pc_type_emb = self.token_type_embedding(
            torch.tensor(0, dtype=torch.long, device=point_cloud.device)
        )
        q_type_emb = self.token_type_embedding(
            torch.tensor(1, dtype=torch.long, device=point_cloud.device)
        )[None, None, :]
        action_type_emb = self.token_type_embedding(
            torch.tensor(2, dtype=torch.long, device=point_cloud.device)
        )[None, None, :]

        pos_emb = torch.cat(
            (
                self.pe_layer(pos, self.pc_bounds) + pc_type_emb,
                q_type_emb.expand((batch_size, -1, -1)),
                action_type_emb.expand((batch_size, 1, -1)),
            ),
            dim=1,
        ).transpose(0, 1)

        return sequence + pos_emb

    def forward(
        self,
        point_cloud_labels: torch.Tensor,
        point_cloud: torch.Tensor,
        configuration: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._embed(point_cloud_labels, point_cloud, configuration)
        for layer in self.transformer_layers:
            hidden = layer(x=hidden, mask=None)
        return hidden


class _TailNetwork(nn.Module):
    def __init__(
        self,
        bc_model: MotionPolicyTransformer,
        *,
        split_layer: int = 7,
        deep_copy: bool = True,
    ):
        super().__init__()
        total_layers = len(bc_model.encoder.layers)
        split_layer = _validate_split_layer(split_layer, total_layers)
        self.d_model = bc_model.d_model

        if deep_copy:
            self.transformer_layers = nn.ModuleList(
                copy.deepcopy(list(bc_model.encoder.layers[split_layer:]))
            )
            self.final_norm = copy.deepcopy(bc_model.encoder.norm)
        else:
            self.transformer_layers = nn.ModuleList(list(bc_model.encoder.layers[split_layer:]))
            self.final_norm = bc_model.encoder.norm

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = features
        for layer in self.transformer_layers:
            hidden = layer(x=hidden, mask=None)
        hidden = self.final_norm(hidden)
        return hidden[-1]


class MPiFormerActorHeadNetwork(nn.Module):
    """Actor head: remaining transformer layers + output layers for policy distribution."""

    def __init__(
        self,
        bc_model: MotionPolicyTransformer,
        *,
        split_layer: int = 7,
        deep_copy: bool = True,
        log_std_init: float = -10.0,
    ):
        super().__init__()
        self.tail = _TailNetwork(bc_model, split_layer=split_layer, deep_copy=deep_copy)
        action_dim = bc_model.action_decoder.out_features

        self.mu = nn.Linear(self.tail.d_model, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

        # Same actor-head initialization currently done in rl/common.py.
        with torch.no_grad():
            self.mu.weight.copy_(bc_model.action_decoder.weight)
            self.mu.bias.copy_(bc_model.action_decoder.bias)

    def forward(
        self,
        features: torch.Tensor,
        *,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.tail(features)
        loc = self.mu(latent)
        log_std = self.log_std.clamp(log_std_min, log_std_max).expand_as(loc)
        scale = log_std.exp()
        return loc, scale


class MPiFormerCriticHeadNetwork(nn.Module):
    """Critic head: remaining transformer layers + twin Q outputs."""

    def __init__(
        self,
        bc_model: MotionPolicyTransformer,
        *,
        split_layer: int = 7,
        deep_copy: bool = True,
    ):
        super().__init__()
        self.tail = _TailNetwork(bc_model, split_layer=split_layer, deep_copy=deep_copy)
        action_dim = bc_model.action_decoder.out_features
        critic_in_dim = self.tail.d_model + action_dim
        self.q1 = nn.Linear(critic_in_dim, 1)
        self.q2 = nn.Linear(critic_in_dim, 1)

    def forward(self, features: torch.Tensor, rl_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.tail(features)
        x = torch.cat([latent, rl_action], dim=-1)
        return self.q1(x), self.q2(x)


class MPiFormerTorchRLArchitecture:
    """
    Encapsulated TorchRL wiring:
      1) backbone_module: observation -> features
      2) head_module: features -> loc, scale
      3) rl_actor: ProbabilisticActor(head_module)
    """

    def __init__(
        self,
        bc_model: MotionPolicyTransformer,
        *,
        pc_bounds: Sequence[Sequence[float]] | torch.Tensor,
        split_layer: int = 7,
        deep_copy: bool = True,
        log_std_init: float = -10.0,
    ):
        self.backbone = MPiFormerBackboneNetwork(
            bc_model,
            pc_bounds=pc_bounds,
            split_layer=split_layer,
            deep_copy=deep_copy,
        )
        self.actor_head = MPiFormerActorHeadNetwork(
            bc_model,
            split_layer=split_layer,
            deep_copy=deep_copy,
            log_std_init=log_std_init,
        )
        self.critic_head = MPiFormerCriticHeadNetwork(
            bc_model,
            split_layer=split_layer,
            deep_copy=deep_copy,
        )

    def build_backbone_module(
        self,
        *,
        in_keys: Sequence[str] = ("point_cloud_labels", "point_cloud", "configuration"),
        out_key: str = "features",
    ) -> Any:
        _require_torchrl()
        return TensorDictModule(  # type: ignore[misc]
            self.backbone,
            in_keys=list(in_keys),
            out_keys=[out_key],
        )

    def build_head_module(
        self,
        *,
        in_key: str = "features",
        out_keys: Sequence[str] = ("loc", "scale"),
    ) -> Any:
        _require_torchrl()
        return TensorDictModule(  # type: ignore[misc]
            self.actor_head,
            in_keys=[in_key],
            out_keys=list(out_keys),
        )

    def build_probabilistic_actor(
        self,
        *,
        action_spec: Any,
        in_keys: Sequence[str] = ("loc", "scale"),
        out_key: str = "rl_action",
        return_log_prob: bool = True,
    ) -> Any:
        _require_torchrl()
        head_module = self.build_head_module(in_key="features", out_keys=in_keys)
        return ProbabilisticActor(  # type: ignore[misc]
            module=head_module,
            in_keys=list(in_keys),
            out_keys=[out_key],
            distribution_class=TanhNormal,
            return_log_prob=return_log_prob,
            spec=action_spec,
        )

    def build_actor_pipeline(
        self,
        *,
        action_spec: Any,
        observation_in_keys: Sequence[str] = ("point_cloud_labels", "point_cloud", "configuration"),
        features_key: str = "features",
        out_key: str = "rl_action",
        return_log_prob: bool = True,
    ) -> Any:
        _require_torchrl()
        backbone_module = self.build_backbone_module(
            in_keys=observation_in_keys,
            out_key=features_key,
        )
        actor = self.build_probabilistic_actor(
            action_spec=action_spec,
            in_keys=("loc", "scale"),
            out_key=out_key,
            return_log_prob=return_log_prob,
        )
        return TensorDictSequential(backbone_module, actor)  # type: ignore[misc]

    def build_critic_module(
        self,
        *,
        features_key: str = "features",
        action_key: str = "rl_action",
        out_keys: Sequence[str] = ("q1", "q2"),
    ) -> Any:
        _require_torchrl()
        return TensorDictModule(  # type: ignore[misc]
            self.critic_head,
            in_keys=[features_key, action_key],
            out_keys=list(out_keys),
        )


__all__ = [
    "MPiFormerBackboneNetwork",
    "MPiFormerActorHeadNetwork",
    "MPiFormerCriticHeadNetwork",
    "MPiFormerTorchRLArchitecture",
]

