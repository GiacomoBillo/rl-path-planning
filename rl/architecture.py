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
    """
    Critic head: remaining transformer layers + single Q output.
    -> Instantiate two separate heads for double critic
    """

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
        self.q = nn.Linear(critic_in_dim, 1)

    def forward(self, features: torch.Tensor, rl_action: torch.Tensor) -> torch.Tensor:
        latent = self.tail(features)
        x = torch.cat([latent, rl_action], dim=-1)
        return self.q(x)


class MPiFormerTorchRLArchitecture:
    """
    Wrap MPiFormer components into modular TorchRL components to use TensorDictModules as inputs and outputs
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
        # split bc model at split_layer into frozen backbone and actor/critic heads that can be fine-tuned
        # torch modules
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
        self.critic_head_q1 = MPiFormerCriticHeadNetwork(
            bc_model,
            split_layer=split_layer,
            deep_copy=deep_copy,
        )
        self.critic_head_q2 = MPiFormerCriticHeadNetwork(
            bc_model,
            split_layer=split_layer,
            deep_copy=deep_copy,
        )

    class _BatchFirst(nn.Module):
        """Wrapper to let modules that expect sequence-first tensors ([S,B,...])
        accept batch-first inputs ([B,S,...]) and optionally transpose outputs.

        - S: transformer sequence length
        - B: batch size
        - D: feature dimension

        Why is it needed?
            Replay stores features as [B,S,D] (batch_first) to be memory-friendly. 
            Heads/tails still expect sequence-first [S,B,D], 
            so _BatchFirst centralizes the transpose logic and avoids duplicating small adapters.

        Usage:
          - For actor/critic heads: transpose_in=True, transpose_out=False
          - For backbone: transpose_in=False, transpose_out=True
        """

        def __init__(self, module: nn.Module, *, transpose_in: bool = False, transpose_out: bool = False):
            super().__init__()
            self.module = module
            self.transpose_in = transpose_in
            self.transpose_out = transpose_out

        def forward(self, *args, **kwargs):
            # transpose incoming 'features' if present in kwargs
            if self.transpose_in:
                if "features" in kwargs:
                    f = kwargs["features"]
                    if isinstance(f, torch.Tensor) and f.ndim == 3:
                        kwargs["features"] = f.transpose(0, 1).contiguous()
                elif len(args) >= 1 and isinstance(args[0], torch.Tensor) and args[0].ndim == 3:
                    f = args[0]
                    args = (f.transpose(0, 1).contiguous(),) + args[1:]

            out = self.module(*args, **kwargs)

            if self.transpose_out:
                # transpose sequence-first outputs like [S,B,D] -> [B,S,D]
                if isinstance(out, torch.Tensor) and out.ndim == 3:
                    return out.transpose(0, 1).contiguous()
                if isinstance(out, (tuple, list)):
                    out_list = []
                    for v in out:
                        if isinstance(v, torch.Tensor) and v.ndim == 3:
                            out_list.append(v.transpose(0, 1).contiguous())
                        else:
                            out_list.append(v)
                    return tuple(out_list) if isinstance(out, tuple) else out_list

            return out

    def build_backbone_module(
        self,
        *,
        in_keys: Sequence[str] = ("point_cloud_labels", "point_cloud", "configuration"),
        out_key: str = "features",
        batch_first: bool = False,
    ) -> Any:
        _require_torchrl()
        module = self.backbone if not batch_first else self._BatchFirst(self.backbone, transpose_in=False, transpose_out=True)
        return TensorDictModule(  # type: ignore[misc]
            module,
            in_keys=list(in_keys),
            out_keys=[out_key],
        )

    def build_head_module(
        self,
        *,
        in_key: str = "features",
        out_keys: Sequence[str] = ("loc", "scale"),
        batch_first: bool = False,
    ) -> Any:
        _require_torchrl()
        module = self.actor_head if not batch_first else self._BatchFirst(self.actor_head, transpose_in=True, transpose_out=False)
        return TensorDictModule(  # type: ignore[misc]
            module,
            in_keys=[in_key],
            out_keys=list(out_keys),
        )

    def build_probabilistic_actor(
        self,
        *,
        action_spec: Any | None = None,
        in_keys: Sequence[str] = ("loc", "scale"),
        out_key: str = "action",
        distribution_kwargs: dict | None = None,
        return_log_prob: bool = True,
        batch_first: bool = False,
    ) -> Any:
        _require_torchrl()
        head_module = self.build_head_module(in_key="features", out_keys=in_keys, batch_first=batch_first)
        return ProbabilisticActor(  # type: ignore[misc]
            module=head_module,
            in_keys=list(in_keys),
            out_keys=[out_key],
            distribution_class=TanhNormal,
            distribution_kwargs=distribution_kwargs,
            return_log_prob=return_log_prob,
            spec=action_spec,
        )

    def build_actor_pipeline(
        self,
        *,
        action_spec: Any,
        observation_in_keys: Sequence[str] = ("point_cloud_labels", "point_cloud", "configuration"),
        features_key: str = "features",
        out_key: str = "action",
        return_log_prob: bool = True,
        batch_first: bool = False,
    ) -> Any:
        _require_torchrl()
        backbone_module = self.build_backbone_module(
            in_keys=observation_in_keys,
            out_key=features_key,
            batch_first=batch_first,
        )
        actor = self.build_probabilistic_actor(
            action_spec=action_spec,
            in_keys=("loc", "scale"),
            out_key=out_key,
            return_log_prob=return_log_prob,
            batch_first=batch_first,
        )
        return TensorDictSequential(backbone_module, actor)  # type: ignore[misc]

    def build_critic_modules(
        self,
        *,
        features_key: str = "features",
        action_key: str = "action",
        batch_first: bool = False,
    ) -> list[Any]:
        """Build list with 2 separate critic tensordict modules for double critic."""
        _require_torchrl()
        q1_mod = self.critic_head_q1 if not batch_first else self._BatchFirst(self.critic_head_q1, transpose_in=True, transpose_out=False)
        q2_mod = self.critic_head_q2 if not batch_first else self._BatchFirst(self.critic_head_q2, transpose_in=True, transpose_out=False)
        q1_td = TensorDictModule(q1_mod, in_keys=[features_key, action_key], out_keys=["state_action_value"])  # type: ignore[misc]
        q2_td = TensorDictModule(q2_mod, in_keys=[features_key, action_key], out_keys=["state_action_value"])  # type: ignore[misc]
        return [q1_td, q2_td]


__all__ = [
    "MPiFormerBackboneNetwork",
    "MPiFormerActorHeadNetwork",
    "MPiFormerCriticHeadNetwork",
    "MPiFormerTorchRLArchitecture",
]

