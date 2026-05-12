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
    from avoid_everything.pretraining import PretrainingMotionPolicyTransformer


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


class MPiFormerBackbone(nn.Module):
    """Frozen backbone network: perception + first split_layer transformer layers."""

    def __init__(
        self,
        bc_model: PretrainingMotionPolicyTransformer,
        *,
        split_layer: int = 7,
        deep_copy: bool = True,
    ):
        super().__init__()
        total_layers = len(bc_model.encoder.layers)
        self.split_layer = _validate_split_layer(split_layer, total_layers)
        self.pc_bounds = bc_model.pc_bounds.to(bc_model.device)
        print(f"PC Bounds: {self.pc_bounds.device}")
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

        # Backbone is always frozen.
        for param in self.parameters():
            param.requires_grad = False

        # output dim (tokens, features)


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
        # point_cloud_labels: torch.Tensor,
        # point_cloud: torch.Tensor,
        # configuration: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        hidden = self._embed(
            point_cloud_labels=state["point_cloud_labels"],
            point_cloud=state["point_cloud"],
            configuration=state["configuration"]
        )
        for layer in self.transformer_layers:
            hidden = layer(x=hidden, mask=None)
        return hidden


class _TailNetwork(nn.Module):
    def __init__(
        self,
        bc_model: PretrainingMotionPolicyTransformer,
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


# class MPiFormerActorHead(nn.Module):
#     """Actor head: remaining transformer layers + output layers for policy distribution."""

#     LOG_STD_MIN = -20.0
#     LOG_STD_MAX = 2.0

#     def __init__(
#         self,
#         bc_model: MotionPolicyTransformer,
#         *,
#         split_layer: int = 7,
#         deep_copy: bool = True,
#         log_std_init: float = -10.0,
#         action_scale: float = 1.0,
#         action_bias: float = 0.0,
#     ):
#         super().__init__()
#         self.tail = _TailNetwork(bc_model, split_layer=split_layer, deep_copy=deep_copy)
#         self.action_dim = bc_model.action_decoder.out_features

#         # scale and bias for action output, applied after tanh squashing in SAC implementation (by default it leaves the action unchanged)
#         self.action_scale = action_scale
#         self.action_bias = action_bias

#         self.mu = nn.Linear(self.tail.d_model, self.action_dim)
#         # self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init))) # State-independent log std
#         self.log_std = nn.Linear(self.tail.d_model, self.action_dim) # State-dependent log std, initialized to log_std_init

#         # Same actor-head initialization currently done in rl/common.py.
#         with torch.no_grad():
#             self.mu.weight.copy_(bc_model.action_decoder.weight)
#             self.mu.bias.copy_(bc_model.action_decoder.bias)

#             # initialize log_std
#             self.log_std.weight.zero_() 
#             self.log_std.bias.fill_(log_std_init)

#     def forward(
#         self,
#         features: torch.Tensor,
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         latent = self.tail(features)
#         mean = self.mu(latent)
#         log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).expand_as(mean)
#         return mean, log_std


class MPiFormerSACActorHead(nn.Module):
    """
    SAC actor head
    - actor head: remaining transformer layers + output layers for policy distribution.
    - based on CleanRL SAC implementation 
      https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
    """
    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        bc_model: PretrainingMotionPolicyTransformer,
        *,
        split_layer: int = 7,
        deep_copy: bool = True,
        log_std_init: float = -10.0,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
    ):
        super().__init__()
        self.tail = _TailNetwork(bc_model, split_layer=split_layer, deep_copy=deep_copy)
        self.action_dim = bc_model.action_decoder.out_features

        # scale and bias for action output, applied after tanh squashing in SAC implementation (by default it leaves the action unchanged)
        self.action_scale = action_scale
        self.action_bias = action_bias

        self.mu = nn.Linear(self.tail.d_model, self.action_dim)
        # self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init))) # State-independent log std
        self.log_std = nn.Linear(self.tail.d_model, self.action_dim) # State-dependent log std, initialized to log_std_init

        # Same actor-head initialization currently done in rl/common.py.
        with torch.no_grad():
            self.mu.weight.copy_(bc_model.action_decoder.weight)
            self.mu.bias.copy_(bc_model.action_decoder.bias)

            # initialize log_std
            self.log_std.weight.zero_() 
            self.log_std.bias.fill_(log_std_init)


    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.tail(features)
        mean = self.mu(hidden)
        log_std = self.log_std(hidden)
        log_std = torch.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std
    
    def get_action(self, x: torch.Tensor):
        """
        Returns: action, log_prob (i.e. entropy), mean, std
        """
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, log_std



class MPiFormerCriticHead(nn.Module):
    """
    Critic head: remaining transformer layers + single Q output.
    -> Instantiate two separate heads for double critic
    """

    def __init__(
        self,
        bc_model: PretrainingMotionPolicyTransformer,
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


if __name__ == "__main__":
    from rl.common import load_bc_checkpoint
    from robofin.robots import Robot
    from avoid_everything.data_loader import TrajectoryDataset, DatasetType

    # cfg from file model_configs/rl_sac_cubbies.yaml
    import yaml
    # config_path = "/workspace/model_configs/evaluation.yaml"
    # with open(config_path, 'r') as f:
    #     cfg = yaml.safe_load(f)
    cfg = yaml.safe_load(open("model_configs/rl_sac_cubbies.yaml"))

    # import dataset
    # Load validation dataset
    # urdf_path = cfg['shared_parameters']['urdf_path']
    # robot = Robot(urdf_path, device='cpu')
    # dataset = TrajectoryDataset.load_from_directory(
    #     robot=robot,
    #     directory=cfg['data_module_parameters']['data_dir'],
    #     dataset_type=DatasetType.VAL,
    #     trajectory_key=cfg['data_module_parameters']['val_trajectory_key'],
    #     num_robot_points=cfg['shared_parameters']['num_robot_points'],
    #     num_obstacle_points=cfg['data_module_parameters']['num_obstacle_points'],
    #     num_target_points=cfg['data_module_parameters']['num_target_points'],
    #     random_scale=0.0  # No noise for validation
    # )

    bc_model = load_bc_checkpoint(cfg)
    backbone = MPiFormerBackbone(bc_model, pc_bounds=bc_model.pc_bounds, split_layer=7)
    # print(backbone)

    # x = backbone(dataset[0]["point_cloud_labels"].unsqueeze(0), dataset[0]["point_cloud"].unsqueeze(0), dataset[0]["configuration"].unsqueeze(0))
    # print(x.shape)

    from torchinfo import summary
    num_points = cfg["num_obstacle_points"] + cfg["num_robot_points"] + cfg["num_target_points"]
    print(f"Input shapes: [B,{num_points},3] point_cloud, [B,{num_points}] labels, [B,7] configuration values")
    summary(backbone, input_size=[
        (1, num_points,),  # point_cloud_labels
        (1, num_points, 3),  # point_cloud
        (1, 7,)  # configuration
    ])

    from rl.feature_extractor import MPiFormerPerceptionEncoder
    feature_extractor = MPiFormerPerceptionEncoder(bc_model)
    summary(feature_extractor, input_size=[
        (1, num_points,),  # point_cloud_labels
        (1, num_points, 3),  # point_cloud
        (1, 7,),  # configuration
    ])