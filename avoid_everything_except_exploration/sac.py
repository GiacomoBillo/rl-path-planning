"""
SAC (Soft Actor-Critic) Trainer for motion planning.

Pure SAC algorithm: entropy-regularized actor-critic with dual Q-networks and soft target updates.
- Actor: outputs mean and log_std for stochastic policy
- Critic: dual Q-networks (Q1, Q2) with shared perception encoder
- Training: Q-loss + actor loss + entropy coefficient update
- Rollouts: collect RL transitions in replay buffer with reward signals from environment

Does NOT use:
- Behavior cloning loss
- Collision avoidance loss (handled by reward function)
- CoL-specific algorithm details
"""

from typing import Dict, Tuple, Callable
# from lightning import Fabric

import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics

from robofin.old.collision import TorchFrankaCollisionSpheres
from robofin.robots import Robot
from robofin.samplers import TorchRobotSampler

from avoid_everything_except_exploration.geometry import TorchCuboids, TorchCylinders
from avoid_everything.pretraining import PretrainingMotionPolicyTransformer
from avoid_everything_except_exploration.replay import ReplayBuffer
from avoid_everything_except_exploration.architecture import MPiFormerCriticHead, MPiFormerSACActorHead, MPiFormerBackbone


class SACMotionPolicyTrainer():
    """
    Holds the models and optimizers for the CoL algorithm. Provides methods for
    training, validation, and inference.
    Hold a reference to the replay buffer and directly inserts samples into it.
    """

    def __init__(
        self,
        urdf_path: str,
        num_robot_points: int,
        collision_reward: float,
        goal_reward: float,
        step_reward: float,
        reward_scale: float,
        robot_dof: int,
        point_match_loss_weight: float,
        collision_loss_weight: float,
        actor_loss_weight: float,
        collision_loss_margin: float,
        gamma: float,
        # exploration_noise: float,
        # target_actor_noise: float,
        # target_actor_noise_clip: float,
        # use_huber_loss: bool,
        tau: float,
        # grad_clip_norm: float,
        pc_bounds: list[list[float]],
        rollout_length: int,
        entropy_coef_init: float | str,
        target_entropy: float,

        # finetuning args
        bc_model: PretrainingMotionPolicyTransformer,
        split_layer: int,
        log_std_init: int,
        target_network_frequency: int = 1,
        policy_frequency: int = 1,
        action_clip: float | None = None,
        # Separate LR configs for actor and critic (with schedule support)
        actor_lr: dict | None = None,
        critic_lr: dict | None = None,
        alpha_lr: float | None = None,

        device=None,
        *args, **kwargs
    ):
        self.urdf_path = urdf_path
        self.robot = None
        self.fk_sampler = None
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_robot_points = num_robot_points
        self.point_match_loss_weight = point_match_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.actor_loss_weight = actor_loss_weight
        # self.loss_fun = CoLLossFn(self.urdf_path, collision_loss_margin)
        
        # Store learning rate configs
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        
        self.pc_bounds = torch.as_tensor(pc_bounds)
        self.rollout_length = rollout_length
        self.robot_dof = robot_dof
        self.reward_scale = reward_scale
        self.collision_reward = collision_reward * reward_scale
        self.goal_reward = goal_reward * reward_scale
        self.step_reward = step_reward * reward_scale
        self.gamma = gamma
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency

        self.action_clip = action_clip
        # self.use_huber_loss = use_huber_loss
        self.tau = tau
        # self.grad_clip_norm = grad_clip_norm
        # Cached per-joint half-range tensor, initialized once setup() builds the robot.
        self._joint_range_half: torch.Tensor | None = None
        
        # Simulation state (for simulation_step)
        self.first_simulation_step = True
        self.simulation_batch: dict | None = None
        self.next_simulation_batch: dict | None = None
        self.simulation_q: torch.Tensor | None = None
        self.simulation_pc: torch.Tensor | None = None
        # Cached obstacle objects (rebuilt only when episodes reset)
        self.simulation_cuboids = None
        self.simulation_cylinders = None
        self.simulation_active = None
        # Per-episode tracking for info dict
        self.simulation_step_counts: torch.Tensor | None = None  # steps per active episode
        self.simulation_cumulative_rewards: torch.Tensor | None = None  # cumulative reward per active episode

        self.val_position_error = torchmetrics.MeanMetric()
        self.val_orientation_error = torchmetrics.MeanMetric()
        self.val_collision_rate = torchmetrics.MeanMetric()
        self.val_funnel_collision_rate = torchmetrics.MeanMetric() # collision rate of successful reaches
        self.val_reaching_success_rate = torchmetrics.MeanMetric()
        self.val_success_rate = torchmetrics.MeanMetric() # reaching success without collision
        self.val_waypoint_count = torchmetrics.MeanMetric() # number of waypoints in successful reaches
        self.val_step_size_unnorm = torchmetrics.MeanMetric()
        self.val_step_size_norm = torchmetrics.MeanMetric()
        self.val_point_match_loss = torchmetrics.MeanMetric()
        self.val_collision_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

        self.bc_model = bc_model # keep original model
        # TODO: original model head
        # split original model in backbone and actor and critic heads
        self.backbone = MPiFormerBackbone(bc_model=bc_model, split_layer=split_layer, deep_copy=False).to(self.device)
        self.actor = MPiFormerSACActorHead(bc_model=bc_model, split_layer=split_layer, log_std_init=log_std_init, deep_copy=True).to(self.device)
        self.qf1 = MPiFormerCriticHead(bc_model=bc_model, split_layer=split_layer, deep_copy=True).to(self.device)
        self.qf2 = MPiFormerCriticHead(bc_model=bc_model, split_layer=split_layer, deep_copy=True).to(self.device)
        # Hard-deep-copy critics weights into targets; then call polyak_update() each step.
        self.qf1_target = copy.deepcopy(self.qf1).to(self.device)
        self.qf2_target = copy.deepcopy(self.qf2).to(self.device)

        self.actor_optimizer: torch.optim.Optimizer
        self.critic_optimizer: torch.optim.Optimizer
        self.actor_scheduler: torch.optim.lr_scheduler.LambdaLR
        self.critic_scheduler: torch.optim.lr_scheduler.LambdaLR
        # learnable entropy
        if isinstance(entropy_coef_init, str) and entropy_coef_init.startswith("auto_"): # e.g. "auto_0.1" to initialize target entropy to 0.1
            self.entropy_autotune = True
            self.target_entropy = target_entropy
            self.log_alpha = torch.tensor(np.log(float(entropy_coef_init[len("auto_"):])), requires_grad=True)
        else:
            self.entropy_autotune = False
            self.log_alpha = torch.tensor(np.log(float(entropy_coef_init)), requires_grad=False)

        if self.entropy_autotune:
            self.alpha_optimizer: torch.optim.Optimizer
            self.alpha_scheduler: torch.optim.lr_scheduler.LambdaLR

        def return_bounds(gamma, step_reward, goal_reward, collision_reward, horizon):
            t = np.arange(horizon)
            geom = (1.0 - gamma**t) / (1.0 - gamma)  # sum_{i=0}^{t-1} gamma^i
            step_sums = step_reward * geom
            goal_returns = step_sums + (gamma**t) * goal_reward      # terminate with goal at t
            coll_returns = step_sums + (gamma**t) * collision_reward # terminate with collision at t
            no_term = step_reward * ((1.0 - gamma**horizon) / (1.0 - gamma)) # no termination

            max_ret = max(goal_returns.max(), no_term)
            min_ret = min(coll_returns.min(), no_term)
            return float(min_ret), float(max_ret)

        self.min_cumulative_reward, self.max_cumulative_reward = return_bounds(
            self.gamma,
            self.step_reward,
            self.goal_reward,
            self.collision_reward,
            self.rollout_length,
        )

    def configure_optimizers(self):
        """
        Build separate optimizers/schedulers for actor and critic with independent learning rates.
        Alpha optimizer (for entropy) has constant learning rate (no schedule).
        Actor and critic can have either constant or linear decay schedules.
        """        
        # Helper to get LR from config (actor_lr and critic_lr are dicts with type/start_lr/end_lr)
        def get_optimizer_lr(lr_config):
            if isinstance(lr_config, dict):
                lr = lr_config["start_lr"]
            else:
                lr = lr_config
            if lr is None:
                raise ValueError("Learning rate config cannot be None")
            return lr
        
        # Actor optimizer with separate LR
        actor_base_lr = get_optimizer_lr(self.actor_lr)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=actor_base_lr,
        )
        
        # Critic optimizer with separate LR
        critic_base_lr = get_optimizer_lr(self.critic_lr)
        critic_param_groups = [
            {"params": self.qf1.parameters(), "lr": critic_base_lr, "name": "critic_q1"},
            {"params": self.qf2.parameters(), "lr": critic_base_lr, "name": "critic_q2"},
        ]
        self.critic_optimizer = torch.optim.AdamW(
            critic_param_groups,
            lr=critic_base_lr,
        )
        
        # Create LR schedulers based on schedule type (constant or linear_warmup)
        def make_lr_schedule(lr_config):
            """
            Create a LambdaLR schedule based on config type.
            
            Types:
            - "constant": Fixed learning rate (start_lr)
            - "linear_warmup": Ramp from start_lr to end_lr over warmup_steps, then constant at end_lr
            """
            if lr_config is None:
                return lambda step: 1.0  # Identity schedule
            
            if isinstance(lr_config, dict):
                schedule_type = lr_config.get("type", "constant")
                start_lr = lr_config["start_lr"]
                
                if schedule_type == "constant":
                    return lambda step: 1.0  # Constant learning rate
                
                elif schedule_type == "linear_warmup":
                    # Ramp from start_lr to end_lr over warmup_steps, then stay at end_lr
                    end_lr = lr_config.get("end_lr", start_lr)
                    warmup_steps = lr_config.get("warmup_steps", 1000)
                    
                    def warmup_schedule(step):
                        lr = start_lr + (end_lr - start_lr) * min(1.0, step / max(1, warmup_steps))
                        return lr / start_lr if start_lr > 0 else 1.0
                    return warmup_schedule
            
            return lambda step: 1.0  # Default: constant
        
        self.actor_scheduler = LambdaLR(
            self.actor_optimizer,
            make_lr_schedule(self.actor_lr)
        )
        self.critic_scheduler = LambdaLR(
            self.critic_optimizer,
            make_lr_schedule(self.critic_lr)
        )
        
        # Alpha optimizer: constant learning rate only (no schedule)
        if self.entropy_autotune:
            alpha_lr = self.alpha_lr if isinstance(self.alpha_lr, (int, float)) else 1.0e-5
            self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=alpha_lr)
            self.alpha_scheduler = LambdaLR(self.alpha_optimizer, lambda step: 1.0)  # Constant schedule
        
        return {
            "actor_optimizer": self.actor_optimizer,
            "critic_optimizer": self.critic_optimizer,
            "actor_scheduler": self.actor_scheduler,
            "critic_scheduler": self.critic_scheduler,
            "entropy_coef_optimizer": self.alpha_optimizer if self.entropy_autotune else None,
            "entropy_coef_scheduler": self.alpha_scheduler if self.entropy_autotune else None,
        }

    # @torch.no_grad()
    # def _polyak_update(self, tau: float = 0.005):
    #     """Soft update of target networks: θ' ← τ θ + (1-τ) θ'"""
    #     for tgt, src in zip(self.target_actor.parameters(), self.actor.parameters()):
    #         tgt.data.lerp_(src.data, tau)
    #     for tgt, src in zip(self.target_critic.parameters(), self.critic.parameters()):
    #         tgt.data.lerp_(src.data, tau)

    def _verify_device(self) -> torch.device:
        """
        Resolve current device from model parameters.
        """
        assert next(self.actor.parameters()).device == next(self.qf1.parameters()).device
        assert next(self.actor.parameters()).device == next(self.qf2.parameters()).device
        assert next(self.actor.parameters()).device == next(self.qf1_target.parameters()).device
        assert next(self.actor.parameters()).device == next(self.qf2_target.parameters()).device
        return next(self.actor.parameters()).device

    # def _apply_noise_in_joint_space(
    #     self,
    #     normalized_action: torch.Tensor,
    #     noise_std: float,
    #     noise_clip: float | None = None,
    #     clamp_unnormalized: float | None = None,
    # ) -> torch.Tensor:
    #     """
    #     Convert normalized deltas to real joint space, add/clamp noise there, 
    #     return normalized result.

    #     :param normalized_action: The normalized action to apply noise to.
    #     :param noise_std: The standard deviation of the noise in joint space.
    #     :param noise_clip: The clip value for the noise in joint space.
    #     :param clamp_unnormalized: The clamp value for the unnormalized action in joint space.
    #     :return: The normalized action with noise applied (normalize joint space).
    #     """
    #     if noise_std <= 0 and clamp_unnormalized is None:
    #         return normalized_action
    #     assert self.robot is not None
    #     assert self._joint_range_half is not None
    #     joint_range_half = self._joint_range_half.to(dtype=normalized_action.dtype, device=normalized_action.device)
    #     while joint_range_half.dim() < normalized_action.dim():
    #         joint_range_half = joint_range_half.unsqueeze(0)

    #     action_unn = normalized_action * joint_range_half
    #     if noise_std > 0:
    #         noise = torch.randn_like(action_unn) * noise_std
    #         if noise_clip is not None:
    #             noise = torch.clamp(noise, -noise_clip, noise_clip)
    #         action_unn = action_unn + noise

    #     if clamp_unnormalized is not None:
    #         action_unn = torch.clamp(action_unn, -clamp_unnormalized, clamp_unnormalized)

    #     return action_unn / joint_range_half

    # def setup(self, fabric: Fabric):
    def setup(self):
        """
        Device-critical initialization. Call after moving model to the desired 
        device. Initializes robot and point cloud sampler on current device.
        """

        # fabric setup: wrap trainable modules w/ their optimizers
        # self.actor,  self.actor_optimizer  = fabric.setup(self.actor,  self.actor_optimizer)
        # self.qf1, self.critic_optimizer = fabric.setup(self.qf1, self.critic_optimizer)
        # self.qf2 = fabric.setup(self.qf2) # shared optimizer for both critics
        # # target networks have no optimizers
        # self.qf1_target  = fabric.setup(self.qf1_target)
        # self.qf2_target = fabric.setup(self.qf2_target)

        self.device = self._verify_device()
        # assert str(self.device) != "cpu", "You do not want to train on CPU"
        print(f"Using device: {self.device}", "green")
        self.robot = Robot(self.urdf_path, device=self.device)
        self.fk_sampler = TorchRobotSampler(
            self.robot,
            num_robot_points=self.num_robot_points,
            num_eef_points=128,
            use_cache=True,
            with_base_link=True,
            device=self.device,
        )
        self._prismatic_joint = float(
            self.robot.auxiliary_joint_defaults.get("panda_finger_joint1", 0.04)
        )
        self._collision_checker = TorchFrankaCollisionSpheres(device=str(self.device))
        assert self.robot.MAIN_DOF == self.robot_dof
        self.pc_bounds = self.pc_bounds.to(self.device)
        self.actor.train()
        self.qf1.train()
        self.qf2.train()

        # don't need gradients for target networks and frozen backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.qf1_target.parameters():
            p.requires_grad_(False)
        for p in self.qf2_target.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self.qf1_target.eval()
        self.qf2_target.eval()

        # pre-compute joint range half for efficient action noise application
        joint_limits = torch.as_tensor(self.robot.main_joint_limits, dtype=torch.float32, device=self.device)
        self._joint_range_half = 0.5 * (joint_limits[:, 1] - joint_limits[:, 0])

        # move all metrics to the same device
        for m in [self.val_position_error, self.val_orientation_error,
                  self.val_collision_rate, self.val_funnel_collision_rate,
                  self.val_reaching_success_rate, self.val_success_rate,
                  self.val_waypoint_count, self.val_step_size_unnorm,
                  self.val_step_size_norm,
                  self.val_point_match_loss, self.val_collision_loss,
                  self.val_loss]:
            m.to(self.device)

    def get_device(self) -> torch.device:
        """
        Get the device of the trainer.
        """
        assert self.device is not None, "You must call setup() before getting the device"
        return self.device

    def move_batch_to_device(
        self, batch: Dict[str, torch.Tensor], device: torch.device, non_blocking: bool=False
    ) -> Dict[str, torch.Tensor]:
        """
        Move a batch of data to the desired device.
        """
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(device, non_blocking=non_blocking)
            else:
                out[k] = v
        return out

    def _refresh_simulation_obstacles(
        self,
        reset_mask: torch.Tensor,
        reset_batch: Dict[str, torch.Tensor],
    ) -> None:
        """
        Refresh cached obstacle tensors only for rows that were reset.
        reset_batch contains ONLY the samples that reset (already filtered by num_resets).
        """
        assert self.simulation_cuboids is not None
        assert self.simulation_cylinders is not None

        new_cuboids = TorchCuboids(
            reset_batch["cuboid_centers"],
            reset_batch["cuboid_dims"],
            reset_batch["cuboid_quats"],
        )
        new_cylinders = TorchCylinders(
            reset_batch["cylinder_centers"],
            reset_batch["cylinder_radii"],
            reset_batch["cylinder_heights"],
            reset_batch["cylinder_quats"],
        )

        self.simulation_cuboids.centers[reset_mask] = new_cuboids.centers
        self.simulation_cuboids.dims[reset_mask] = new_cuboids.dims
        self.simulation_cuboids.quats[reset_mask] = new_cuboids.quats
        self.simulation_cuboids.inv_frames[reset_mask] = new_cuboids.inv_frames
        self.simulation_cuboids.mask[reset_mask] = new_cuboids.mask

        self.simulation_cylinders.centers[reset_mask] = new_cylinders.centers
        self.simulation_cylinders.radii[reset_mask] = new_cylinders.radii
        self.simulation_cylinders.heights[reset_mask] = new_cylinders.heights
        self.simulation_cylinders.quats[reset_mask] = new_cylinders.quats
        self.simulation_cylinders.inv_frames[reset_mask] = new_cylinders.inv_frames
        self.simulation_cylinders.mask[reset_mask] = new_cylinders.mask


    def _sample(self, q: torch.Tensor) -> torch.Tensor:
        """
        Samples a point cloud from the surface of all the robot's links

        :param q torch.Tensor: Batched configuration in joint space
        :rtype torch.Tensor: Batched point cloud of size [B, self.num_robot_points, 3]
        """
        assert self.fk_sampler is not None
        points = self.fk_sampler.sample(q)
        return points

    # def _bc_loss(
    #     self,
    #     q_pred: torch.Tensor,
    #     q_next: torch.Tensor,
    #     is_expert: torch.Tensor | None = None,
    # ) -> torch.Tensor:
    #     """
    #     BC loss in point-cloud space
    #     """
    #     assert self.robot is not None
    #     q_next_unn = self.robot.unnormalize_joints(q_next)
    #     return self.loss_fun.bc_pointcloud_loss(
    #         pred_q_unnorm=self.robot.unnormalize_joints(q_pred),
    #         expert_q_unnorm=q_next_unn,
    #         is_expert=is_expert,
    #     )

    # def _collision_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    #     """
    #     Collision loss
    #     """
    #     q_next = batch["next_configuration"]
    #     assert self.robot is not None
    #     return self.loss_fun.collision_loss(
    #         unnormalized_q=self.robot.unnormalize_joints(q_next),
    #         cuboid_centers=batch["cuboid_centers"],
    #         cuboid_dims=batch["cuboid_dims"],
    #         cuboid_quaternions=batch["cuboid_quats"],
    #         cylinder_centers=batch["cylinder_centers"],
    #         cylinder_radii=batch["cylinder_radii"],
    #         cylinder_heights=batch["cylinder_heights"],
    #         cylinder_quaternions=batch["cylinder_quats"],
    #     )

    # def _actor_loss(
    #     self,
    #     pc_labels: torch.Tensor,
    #     pc: torch.Tensor,
    #     q: torch.Tensor,
    #     a_pred: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     RL Actor loss
    #     """
    #     # don't backprop into critic
    #     for p in self.critic.parameters():
    #         p.requires_grad_(False)
    #     q1, _ = self.critic(pc_labels, pc, q, a_pred, self.pc_bounds)
    #     loss_actor = -q1.mean()
    #     for p in self.critic.parameters():
    #         p.requires_grad_(True)
    #     return loss_actor

    # def _critic_loss(
    #     self,
    #     batch: dict[str, torch.Tensor],
    #     metrics: dict[str, float],
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     One-step TD loss for critic using target networks.
    #     """
    #     q      = batch["configuration"]
    #     a      = batch["action"]
    #     q_next = batch["next_configuration"]
    #     r      = batch["reward"]
    #     done   = batch.get("done", torch.zeros_like(r))

    #     # build next state's (s') point cloud by sampling robot at q_next
    #     assert self.robot is not None
    #     q_next_unn = self.robot.unnormalize_joints(q_next)
    #     robot_pc = self._sample(q_next_unn)[..., :3]
    #     next_pc = torch.cat([robot_pc, batch["point_cloud"][:, robot_pc.size(1):]], dim=1)

    #     # target action a' = π'(s')
    #     with torch.no_grad():
    #         a_next = self.target_actor(batch["point_cloud_labels"], next_pc, q_next, self.pc_bounds)
    #         if a_next.dim() == 3:
    #             a_next = a_next[:, -1, :]
    #         a_next = self._apply_noise_in_joint_space(
    #             a_next,
    #             noise_std=self.target_actor_noise,
    #             noise_clip=self.target_actor_noise_clip,
    #             clamp_unnormalized=self.action_clip,
    #         )

    #     with torch.no_grad():
    #         q_next_target, q_next_target2 = self.target_critic(
    #             batch["point_cloud_labels"], next_pc, q_next, a_next, self.pc_bounds
    #         )
    #         y = r + self.gamma * (1.0 - done) * torch.min(q_next_target, q_next_target2)

    #     q_sa, q_sa2 = self.critic(
    #         batch["point_cloud_labels"], batch["point_cloud"], q, a, self.pc_bounds)

    #     with torch.no_grad():
    #         metrics["q_target_mean_(y)"] = float(y.mean().item())
    #         metrics["q_sa_mean"]     = float(q_sa.mean().item())
    #     if self.use_huber_loss:
    #         return torch.nn.functional.huber_loss(q_sa, y, delta=1.0), \
    #                torch.nn.functional.huber_loss(q_sa2, y, delta=1.0)
    #     return torch.nn.functional.mse_loss(q_sa, y), \
    #            torch.nn.functional.mse_loss(q_sa2, y)

    # def train_step(
    #     self,
    #     batch: dict[str, torch.Tensor],
    #     fabric: Fabric,
    #     update_targets: bool,
    #     use_actor_loss: bool,
    # ) -> Dict[str, float]:
    #     """
    #     One training iteration on a mixed (expert + actor) batch.
    #     Calculates losses and performs optimization. Critic and BC losses are
    #     always computed and optimized on. Target networks are updated only if
    #     update_targets is True. The RL actor loss is only computed and optimized 
    #     on if use_actor_loss is True.

    #     Computes L_BC, L_Q1, L_A via self._state_based_step(), then:
    #     1) critic step on L_Q1
    #     2) actor step on (λ_A * L_A + λ_BC * L_BC)
    #     3) Polyak soft-update of targets

    #     :param update_targets: Whether to update the target networks
    #     :param use_actor_loss: Whether to use the actor loss (else just BC and critic updates)
    #     :return: A flat dict of scalars for logging
    #     """
    #     # critic update on one-step TD loss first (keeps actor graph clean)
    #     metrics = {}
    #     loss_q1, loss_q2 = self._critic_loss(batch, metrics)
    #     metrics.update({
    #         "critic_loss_1": float(loss_q1.detach().item()),
    #         "critic_loss_2": float(loss_q2.detach().item()),
    #     })
    #     self.critic_optim.zero_grad(set_to_none=True)
    #     fabric.backward(loss_q1 + loss_q2)
    #     clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip_norm)
    #     self.critic_optim.step()
    #     self.critic_scheduler.step()

    #     # actor predict next q via Δq
    #     q = batch["configuration"]
    #     pc_labels = batch["point_cloud_labels"]
    #     pc = batch["point_cloud"]
    #     qdeltas = self.actor(pc_labels, pc, q, self.pc_bounds)
    #     a_pred  = qdeltas[:, -1, :] if qdeltas.dim() == 3 else qdeltas
    #     if self.action_clip is not None:
    #         a_pred = torch.clamp(a_pred, -self.action_clip, self.action_clip)
    #     q_pred  = torch.clamp(q + a_pred, -1, 1)

    #     loss_bc = self._bc_loss(q_pred, batch["next_configuration"], batch["is_expert"])
    #     metrics["point_match_loss"] = float(loss_bc.detach().item())

    #     collision_loss = self._collision_loss(batch)
    #     metrics["collision_loss"] = float(collision_loss.detach().item())

    #     # actor update on critic-guided actor loss (+ optional BC)
    #     if use_actor_loss:
    #         loss_actor = self._actor_loss(pc_labels, pc, q, a_pred)
    #         metrics["actor_loss"] = float(loss_actor.detach().item())
    #         actor_total = (self.point_match_loss_weight * loss_bc +
    #                         self.collision_loss_weight * collision_loss +
    #                         self.actor_loss_weight * loss_actor)
    #     else:
    #         actor_total = (self.point_match_loss_weight * loss_bc +
    #                        self.collision_loss_weight * collision_loss)

    #     self.actor_optim.zero_grad(set_to_none=True)
    #     fabric.backward(actor_total)
    #     clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip_norm)
    #     self.actor_optim.step()
    #     self.actor_scheduler.step()

    #     if update_targets:
    #         self._polyak_update(tau=self.tau) # soft target update

    #     metrics["lr"]  = float(self.actor_optim.param_groups[0]["lr"])

    #     return metrics

    def torch_compile(self, mode: str = "default"):
        # Standard compilation
        self.backbone = torch.compile(self.backbone, mode=mode)
        self.actor = torch.compile(self.actor, mode=mode)
        self.qf1 = torch.compile(self.qf1, mode=mode)
        self.qf2 = torch.compile(self.qf2, mode=mode)
        self.qf1_target = torch.compile(self.qf1_target, mode=mode)
        self.qf2_target = torch.compile(self.qf2_target, mode=mode)

    def train_step(self,
                    batch: dict[str, torch.Tensor],
                    # fabric: Fabric,
                    global_step: int,
                    critic_warmup: bool = False,
                    ):
        """
        Compute one training step and return metrics as GPU tensors (detached).
        
        Do NOT call .item() here—defer to caller at log time to avoid GPU sync stalls.
        Returning detached tensors allows caller to accumulate them and sync once per
        log_every_n_steps instead of every step.
        
        Returns:
            dict: Metrics with GPU tensors
                - Always: qf_loss, qf1_loss, qf2_loss, current_q_min, current_q1, current_q2,
                          target_q_min, target_q1, target_q2
                - On policy_frequency steps: actor_loss, actor_q_min, entropy, action_mean, 
                                           log_std_mean, alpha_loss (if autotune), alpha (if autotune)
        """
        metrics = {
            "actor_lr": float(self.actor_optimizer.param_groups[0]["lr"]),
            "critic_lr": float(self.critic_optimizer.param_groups[0]["lr"]),
        }
        if self.entropy_autotune:
            metrics["alpha_lr"] = float(self.alpha_optimizer.param_groups[0]["lr"])

        states = batch["state"]
        next_states = batch["next_state"]
        actions = batch["action"]
        rewards = batch["reward"]
        dones = batch["done"]

        alpha = self.log_alpha.exp()
        with torch.no_grad():
            next_state_features = self.backbone(next_states)
            next_state_actions, next_state_log_pi, _, _ = self.actor.get_action(next_state_features)
            qf1_next_target = self.qf1_target(next_state_features, next_state_actions)
            qf2_next_target = self.qf2_target(next_state_features, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
            next_q_value = rewards.flatten() + (1 - dones.flatten()) * self.gamma * (min_qf_next_target).view(-1)

        state_features = self.backbone(states)
        qf1_a_values = self.qf1(state_features, actions).view(-1)
        qf2_a_values = self.qf2(state_features, actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        self.critic_optimizer.zero_grad()
        # fabric.backward(qf_loss) 
        qf_loss.backward()
        self.critic_optimizer.step()
        # TODO: clip gradients?

        # Critic metrics (always computed) - return as detached tensors (no .item() calls!)
        metrics["qf_loss"] = qf_loss.detach()
        metrics["qf1_loss"] = qf1_loss.detach()
        metrics["qf2_loss"] = qf2_loss.detach()
        metrics["current_q_min"] = torch.min(qf1_a_values, qf2_a_values).mean().detach()
        metrics["current_q1"] = qf1_a_values.mean().detach()
        metrics["current_q2"] = qf2_a_values.mean().detach()
        metrics["target_q_min"] = min_qf_next_target.mean().detach()
        metrics["target_q1"] = qf1_next_target.mean().detach()
        metrics["target_q2"] = qf2_next_target.mean().detach()
        metrics["critic_warmup"] = critic_warmup

        # Delayed policy update (TD3-style)
        if not critic_warmup and global_step % self.policy_frequency == 0:
            for _ in range(self.policy_frequency):
                pi, log_pi, mu, log_std = self.actor.get_action(state_features)
                qf1_pi = self.qf1(state_features, pi)
                qf2_pi = self.qf2(state_features, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Actor metrics (only on policy_frequency steps)
                metrics["actor_loss"] = actor_loss.detach()
                metrics["actor_q_min"] = min_qf_pi.mean().detach()
                metrics["entropy"] = (-log_pi.mean()).detach()
                metrics["action_mean"] = mu.abs().mean().detach()
                metrics["log_std_mean"] = log_std.mean().detach()
                # TODO: per-joint action/log_std metrics?

                if self.entropy_autotune:
                    alpha_loss = (-self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)).mean()
                    metrics["alpha_loss"] = alpha_loss.detach()

                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp().detach()
                    metrics["alpha"] = self.alpha

        # Update target networks
        if global_step % self.target_network_frequency == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return metrics

    @torch.no_grad()
    def simulation_step(self, dataloader_iterator: iter):
        """
        Run one step of the environment simulation with the current policy.
        Used for data collection. Maintains state across calls.
        Uses a tensor-based batch queue for reset episodes to avoid sample waste.
        """
        # Initialize batch on first call
        if self.first_simulation_step:
            self.first_simulation_step = False
            self.simulation_batch = self.move_batch_to_device(next(dataloader_iterator), self.device, non_blocking=True)
            self.simulation_cuboids = TorchCuboids(self.simulation_batch["cuboid_centers"], self.simulation_batch["cuboid_dims"], self.simulation_batch["cuboid_quats"])
            self.simulation_cylinders = TorchCylinders(self.simulation_batch["cylinder_centers"], self.simulation_batch["cylinder_radii"],
                                    self.simulation_batch["cylinder_heights"], self.simulation_batch["cylinder_quats"])
            # Initialize per-episode tracking
            B = self.simulation_batch["configuration"].size(0)
            self.simulation_step_counts = torch.zeros(B, dtype=torch.long, device=self.device)
            self.simulation_cumulative_rewards = torch.zeros(B, dtype=torch.float32, device=self.device)
            # Initialize reset batch queue with pre-loaded batch (async)
            self.reset_queue = self.move_batch_to_device(next(dataloader_iterator), self.device, non_blocking=True)
            self.reset_queue_idx = 0
            # PREFETCH a second batch to sit idle on the GPU    
            self.prefetched_batch = self.move_batch_to_device(next(dataloader_iterator), self.device, non_blocking=True)
        
        # Extract current state
        batch: Dict = self.simulation_batch
        q = batch["configuration"].clone() # clone for safety since we'll modify q_next and pc_next in-place later
        pc = batch["point_cloud"].clone()
        idx = batch["pidx"]
        
        # Actor step
        # action_bc = self.bc_model(self.simulation_batch["point_cloud_labels"], pc, q, self.pc_bounds) # original pretrained BC action
        q_features = self.backbone(self.simulation_batch)
        action, log_prob, mean, log_std= self.actor.get_action(q_features)
        # action=mean
        # action = torch.zeros_like(q)
        if self.action_clip is not None:
            action = torch.clamp(action, -self.action_clip, self.action_clip)
        
        q_next = (q + action).clamp(-1, 1)
        q_next_unn = self.robot.unnormalize_joints(q_next)

        cuboids = self.simulation_cuboids
        cylinders = self.simulation_cylinders
        
        # Compute termination and reward
        reached, pos_err, orient_err = self._compute_target_reached(q_next_unn, batch["target_position"], batch["target_orientation"])
        collided = self._check_for_collisions(q_next_unn, cuboids, cylinders)
        # Determine timeout: reached step limit (num_steps == rollout_length)
        timeout = self.simulation_step_counts >= self.rollout_length
        done = reached | collided | timeout
        r_t = torch.where(collided, self.collision_reward,
                          torch.where(reached, self.goal_reward, self.step_reward)).unsqueeze(1)
        
        # Update point cloud for next step
        samples = self._sample(q_next_unn)[..., :3]
        batch["point_cloud"][:, :samples.size(1)] = samples
        batch["configuration"] = q_next
        
        # Update tracking for all episodes
        self.simulation_step_counts.add_(1)
        self.simulation_cumulative_rewards.add_(r_t.squeeze(1))
        
        # Reset done episodes using tensor-based batch queue
        reset_mask = done.squeeze(-1) if done.dim() > 1 else done
        num_resets = reset_mask.sum().item()
        if num_resets > 0:    
            available = self.reset_queue["configuration"].size(0) - self.reset_queue_idx
            
            if num_resets <= available:
                # Happy path: enough in queue, just slice and increment pointer (O(1))
                reset_batch = {
                    k: v[self.reset_queue_idx:self.reset_queue_idx + num_resets]
                    for k, v in self.reset_queue.items()
                }
                self.reset_queue_idx += num_resets
            else:
                # Need to refill: split resets between remaining queue and prefetch
                num_from_queue = available
                num_from_prefetch = num_resets - num_from_queue
                
                reset_batch = {
                    k: torch.cat([self.reset_queue[k][self.reset_queue_idx:],
                                  self.prefetched_batch[k][:num_from_prefetch]], dim=0)
                    for k in self.reset_queue.keys()
                }
                
                # Promote prefetched batch to queue and prefetch next batch
                self.reset_queue = self.prefetched_batch
                self.reset_queue_idx = num_from_prefetch
                self.prefetched_batch = self.move_batch_to_device(next(dataloader_iterator), self.device, non_blocking=True)
            
            # Apply resets to done episodes
            batch["configuration"][reset_mask] = reset_batch["configuration"]
            batch["point_cloud"][reset_mask] = reset_batch["point_cloud"]
            # Update ALL batch fields
            batch["cuboid_centers"][reset_mask] = reset_batch["cuboid_centers"]
            batch["cuboid_dims"][reset_mask] = reset_batch["cuboid_dims"]
            batch["cuboid_quats"][reset_mask] = reset_batch["cuboid_quats"]
            batch["cylinder_centers"][reset_mask] = reset_batch["cylinder_centers"]
            batch["cylinder_radii"][reset_mask] = reset_batch["cylinder_radii"]
            batch["cylinder_heights"][reset_mask] = reset_batch["cylinder_heights"]
            batch["cylinder_quats"][reset_mask] = reset_batch["cylinder_quats"]
            batch["target_position"][reset_mask] = reset_batch["target_position"]
            batch["target_orientation"][reset_mask] = reset_batch["target_orientation"]
            # Reset tracking for episodes that ended
            self.simulation_step_counts[reset_mask].zero_()
            self.simulation_cumulative_rewards[reset_mask].zero_()
            # Update cached obstacle objects for reset indices
            self._refresh_simulation_obstacles(reset_mask, reset_batch)
        
        # Create info dict for each episode
        info_dict = {
            "collision": collided,
            "target_reached": reached,
            "position_error": pos_err,
            "orientation_error": orient_err,
            "num_steps": self.simulation_step_counts.float(),
            "timeout": timeout,
            "cumulative_reward": self.simulation_cumulative_rewards,
        }
        
        # Save state for next call
        self.simulation_batch = batch
        # print(f"\tIdx 1: {idx[0].item()}, \n\tReached: {reached[0].item()}, Collision: {collided[0].item()}\n\tReward 1: {r_t[0].item()}, \n\tDone 1: {done[0].item()}, \n\tQ : {q[0].cpu().numpy()}, \n\taction 1: {action[0].cpu().numpy()}, \n\tbc action : {pretrained_a[0].cpu().numpy()}\n\tQ next 1: {q_next[0].cpu().numpy()}")
        # Return all episode data with info dict
        return idx, q, action, q_next, r_t, done, info_dict 

    @torch.no_grad()
    def simulation_step_v2(self, dataloader_iterator: iter):
        """
        Optimized simulation step with active episode masking.
        Only computes on active (non-finished) episodes, avoiding wasted computation.
        Resets entire batch when all episodes are done.
        
        Pattern similar to actor_rollout(): track active mask, only forward/compute on subset.
        """
        # Initialize or reset batch when ALL episodes are done
        if self.simulation_batch is None or not self.simulation_active.any():
            self.simulation_batch = self.move_batch_to_device(next(dataloader_iterator), self.device, non_blocking=True)
            self.simulation_q = self.simulation_batch["configuration"].clone()
            self.simulation_pc = self.simulation_batch["point_cloud"].clone()
            B = self.simulation_q.size(0)
            self.simulation_active = torch.ones(B, dtype=torch.bool, device=self.simulation_q.device)
            self.simulation_cuboids = TorchCuboids(self.simulation_batch["cuboid_centers"], 
                                                    self.simulation_batch["cuboid_dims"], 
                                                    self.simulation_batch["cuboid_quats"])
            self.simulation_cylinders = TorchCylinders(self.simulation_batch["cylinder_centers"], 
                                                       self.simulation_batch["cylinder_radii"],
                                                       self.simulation_batch["cylinder_heights"], 
                                                       self.simulation_batch["cylinder_quats"])
            # Initialize per-episode tracking
            self.simulation_step_counts = torch.zeros(B, dtype=torch.long, device=self.simulation_q.device)
            self.simulation_cumulative_rewards = torch.zeros(B, dtype=torch.float32, device=self.simulation_q.device)
        
        # Extract current state
        q = self.simulation_q
        pc = self.simulation_pc
        batch: Dict = self.simulation_batch
        active = self.simulation_active
        idx = batch["pidx"]
        
        # Extract active subset for computation
        q_act = q[active]
        pc_act = pc[active]
        batch_act = {key: val[active] if isinstance(val, torch.Tensor) else val 
                     for key, val in batch.items()}
        
        # Actor step on active subset
        # Create minimal batch dict for backbone
        batch_for_backbone = {
            "point_cloud": batch_act["point_cloud"],
            "point_cloud_labels": batch_act["point_cloud_labels"],
            "configuration": batch_act["configuration"],
        }
        q_features = self.backbone(batch_for_backbone)
        action, log_prob, mean, log_std = self.actor.get_action(q_features)
        if self.action_clip is not None:
            action = torch.clamp(action, -self.action_clip, self.action_clip)
        q_next_act = (q_act + action).clamp(-1, 1)
        q_next_unn_act = self.robot.unnormalize_joints(q_next_act)
        
        # Get cuboids/cylinders for active subset only
        cuboids_act = self.simulation_cuboids[active]
        cylinders_act = self.simulation_cylinders[active]
        
        # Compute termination and reward on active subset
        reached_act, pos_err_act, orient_err_act = self._compute_target_reached(q_next_unn_act, 
                                                    batch_act["target_position"], 
                                                    batch_act["target_orientation"])
        collided_act = self._check_for_collisions(q_next_unn_act, cuboids_act, cylinders_act)
        done_act = reached_act | collided_act
        r_t_act = torch.where(collided_act, self.collision_reward,
                              torch.where(reached_act, self.goal_reward, self.step_reward)).unsqueeze(1)
        
        # Update point cloud for active subset
        samples_act = self._sample(q_next_unn_act)[..., :3]
        pc_act[: samples_act.size(0), :samples_act.size(1)] = samples_act
        
        # Mark newly done episodes as inactive
        still_active_act = ~done_act
        
        # BEFORE updating active mask, extract data to return
        # (idx_act is from current active set before it's updated)
        idx_act = idx[active]
        q_act_for_return = q_act  # This is the starting state for active episodes
        
        # Update active mask for next step
        tmp = active.nonzero(as_tuple=False).squeeze(1)
        active[tmp] = still_active_act
        
        # Write back active subset into full tensors (for still-active episodes only)
        q[active] = q_next_act[still_active_act]
        pc[active] = pc_act[still_active_act]
        
        # Update step counts for active episodes
        self.simulation_step_counts[tmp] += 1
        # Update cumulative rewards for all active episodes
        self.simulation_cumulative_rewards[tmp] += r_t_act.squeeze(1)
        
        # Determine timeout: still active but reached step limit (num_steps == rollout_length)
        timeout_act = still_active_act & (self.simulation_step_counts[tmp] >= self.rollout_length)
        
        # Create info dict for each active episode
        info_dict = {
            "collision": collided_act,
            "target_reached": reached_act,
            "position_error": pos_err_act,
            "orientation_error": orient_err_act,
            "num_steps": self.simulation_step_counts[tmp].float(),
            "timeout": timeout_act,
            "reward": r_t_act.squeeze(1),
            "cumulative_reward": self.simulation_cumulative_rewards[tmp],
            "action_magnitude": action.abs(),
        }
        
        # Save state for next call
        self.simulation_q = q
        self.simulation_pc = pc
        self.simulation_active = active
        
        # Return active episode data
        # All tensors are in "active" space: idx_act, q_act_for_return, a_act, q_next_act, r_t_act, done_act, info_dict
        return idx_act, q_act_for_return, action, q_next_act, r_t_act, done_act, info_dict


    @torch.no_grad()
    def actor_rollout(
        self,
        batch: dict[str, torch.Tensor],
        replay_buffer: ReplayBuffer,
    ) -> Dict[str, float]:
        """
        Run a batch of actor rollouts to collect transitions for a replay 
        buffer. 
        Performs a batched rollout with self.rollout_length steps. Reaching the 
        goal (within 1cm and 15 degrees) or colliding with obstacles marks the 
        end of the episode for each rollout in the batch.

        :param batch: a batch of starting states
        :param replay_buffer: a replay buffer to push the collected transitions to
        :return: a metrics dict including average episode reward across the batch.
        """
        # use idx from dataset batch so the replay can fetch scenes later
        idx = batch["idx"]                     # [B]
        q = batch["configuration"].clone()     # [B, DOF], normalized
        B = q.size(0)

        # track cumulative rewards per episode (per batch element)
        cumulative_rewards = torch.zeros(B, device=q.device)
        transitions_collected: int = 0

        cuboids = TorchCuboids(batch["cuboid_centers"], batch["cuboid_dims"], batch["cuboid_quats"])
        cylinders = TorchCylinders(batch["cylinder_centers"], batch["cylinder_radii"],
                                batch["cylinder_heights"], batch["cylinder_quats"])

        # point cloud for the actor input; will be updated only for active rows
        pc = batch["point_cloud"].clone()
        labels = batch["point_cloud_labels"]

        active = torch.ones(B, dtype=torch.bool, device=q.device)
        assert self.robot is not None
        for _ in range(self.rollout_length):
            if not active.any():
                break

            # actor action for active rows
            q_act = q[active]
            pc_act = pc[active]
            lbl_act = labels[active]
            qdeltas = self.actor(lbl_act, pc_act, q_act, self.pc_bounds)  # [B, 1, DOF] or [B, DOF]
            a = (qdeltas[:, -1, :] if qdeltas.dim()==3 else qdeltas)
            a = self._apply_noise_in_joint_space(
                a,
                noise_std=self.exploration_noise,
            )

            if self.action_clip is not None:
                a = torch.clamp(a, -self.action_clip, self.action_clip)
            q_next = (q_act + a).clamp(-1, 1)
            q_next_unn = self.robot.unnormalize_joints(q_next)

            # termination & reward at q_next
            reached = self._check_reaching_success(
                q_next_unn,
                batch["target_position"][active],
                batch["target_orientation"][active])
            collided = self._check_for_collisions(
                q_next_unn, cuboids[active], cylinders[active])
            done = reached | collided
            r_t = torch.where(collided, self.collision_reward,
                torch.where(reached, self.goal_reward, self.step_reward)).float().unsqueeze(1)

            replay_buffer.push(
                idx=idx[active],
                q=q_act,
                a=a,
                q_next=q_next,
                r=r_t,
                done=done.unsqueeze(1),
            )

            # accumulate rewards for active episodes
            cumulative_rewards[active] += r_t.squeeze(1)
            transitions_collected += int(active.sum().item())

            # update only those still active
            still = ~done
            if still.any():
                samples = self._sample(q_next_unn[still])[..., :3] # [B_still, num_robot_points, 3]
                pc_act[still, :samples.size(1)] = samples
                q_act[still] = q_next[still]

            # write back into full tensors
            q[active] = q_act
            pc[active] = pc_act

            # mark finished rows inactive
            tmp = active.nonzero(as_tuple=False).squeeze(1)
            active[tmp] = still

        # average reward per episode across batch
        avg_episode_reward = float(cumulative_rewards.mean().item())
        return {
            "avg_episode_reward": avg_episode_reward, 
            "transitions_collected": transitions_collected,
        }


    def _compute_target_reached(
        self,
        q_unnorm: torch.Tensor,
        target_position: torch.Tensor,
        target_orientation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Efficiently compute target reached status and errors with single FK call.
        Returns: (target_reached, pos_errors, orient_errors)
        """
        assert self.fk_sampler is not None
        eff_pose = self.fk_sampler.end_effector_pose(q_unnorm)
        pos_errors = torch.linalg.vector_norm(
            eff_pose[:, :3, -1] - target_position, dim=-1
        )
        R = torch.matmul(
            eff_pose[:, :3, :3],
            target_orientation.transpose(-1, -2),
        )
        trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        cos_value = torch.clamp((trace - 1) / 2, -1, 1)
        orient_errors = torch.abs(torch.rad2deg(torch.acos(cos_value)))
        target_reached = torch.logical_and(orient_errors < 15, pos_errors < 0.01)
        return target_reached, pos_errors, orient_errors


    def _check_for_collisions(
        self, q_unnorm: torch.Tensor, cuboids: TorchCuboids, cylinders: TorchCylinders
    ) -> torch.Tensor:
        """
        Checks if a batch of joint configurations has collided with their environments.
        Check both self-collision and environment collision. Returns a boolean tensor of shape [B] indicating collision status for each configuration.
        """
        # assert self.robot is not None, "Robot not initialized"
        # collision_spheres = self.robot.compute_spheres(q_unnorm)
        # has_collision = torch.zeros(
        #     (q_unnorm.shape[0],), dtype=torch.bool, device=q_unnorm.device
        # )
        # for radii, spheres in collision_spheres:
        #     num_spheres = spheres.shape[-2]
        #     sphere_sequence = spheres.reshape((q_unnorm.shape[0], -1, num_spheres, 3))
        #     sdf_values = torch.minimum(
        #         cuboids.sdf_sequence(sphere_sequence),
        #         cylinders.sdf_sequence(sphere_sequence),
        #     )
        #     assert (
        #         sdf_values.size(0) == q_unnorm.shape[0]
        #         and sdf_values.size(2) == num_spheres
        #     )
        #     radius_collisions = torch.any(
        #         sdf_values.reshape((sdf_values.size(0), -1)) <= radii, dim=-1
        #     )
        #     has_collision = torch.logical_or(radius_collisions, has_collision)
        # return has_collision
        # cuboids   = self._scene_primitives["cuboids"]
        # cylinders = self._scene_primitives["cylinders"]


        collides = self._collision_checker.franka_arm_collides(
            q=q_unnorm,
            prismatic_joint=self._prismatic_joint,
            primitives=[cuboids, cylinders],
            scene_buffer=0,
            check_self=True,
        )
        return collides

    # ---------- Validation ----------

    def reset_state_val_metrics(self):
        self.val_point_match_loss.reset()
        self.val_collision_loss.reset()
        self.val_loss.reset()

    def compute_state_val_metrics(self) -> Dict[str, float]:
        return {
            "val_point_match_loss": float(self.val_point_match_loss.compute().item()),
            "val_collision_loss":   float(self.val_collision_loss.compute().item()),
            "val_loss":             float(self.val_loss.compute().item()),
        }

    def reset_rollout_val_metrics(self):
        self.val_position_error.reset()
        self.val_orientation_error.reset()
        self.val_collision_rate.reset()
        self.val_funnel_collision_rate.reset()
        self.val_reaching_success_rate.reset()
        self.val_success_rate.reset()
        self.val_waypoint_count.reset()
        self.val_step_size_unnorm.reset()
        self.val_step_size_norm.reset()

    def compute_rollout_val_metrics(self) -> Dict[str, float]:
        return {
            "val_position_error":        float(self.val_position_error.compute().item()),
            "val_orientation_error":     float(self.val_orientation_error.compute().item()),
            "val_collision_rate":        float(self.val_collision_rate.compute().item()),
            "val_funnel_collision_rate": float(self.val_funnel_collision_rate.compute().item()),
            "val_reaching_success_rate": float(self.val_reaching_success_rate.compute().item()),
            "val_success_rate":          float(self.val_success_rate.compute().item()),
            "val_waypoint_count":        float(self.val_waypoint_count.compute().item()),
            "val_step_size_unnorm":  float(self.val_step_size_unnorm.compute().item()),
            "val_step_size_norm":    float(self.val_step_size_norm.compute().item()),
        }

    def rollout(
        self,
        batch: dict[str, torch.Tensor],
        rollout_length: int,
        sampler: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Rolls out the policy an arbitrary length by calling it iteratively

        :param batch Dict[str, torch.Tensor]: A data batch coming from the
                                              data loader--should already be
                                              on the correct device
        :param rollout_length int: The number of steps to roll out (not including the start)
        :param sampler Callable[[torch.Tensor], torch.Tensor]: A function that takes a batch of robot
                                                               configurations [B x self.robot.MAIN_DOF] and returns a batch of
                                                               point clouds samples on the surface of that robot
        
        :rtype torch.Tensor: The entire trajectory batch, i.e. a tensor of
                             unnormalized configuration batches including the 
                             starting configurations where each element in the 
                             tensor corresponds to a timestep. For example, the
                             first element of each batch in the tensor would
                             be a single trajectory.
        """
        point_cloud_labels, point_cloud, q = (
            batch["point_cloud_labels"],
            batch["point_cloud"],
            batch["configuration"],
        )

        B = q.size(0)
        actual_rollout_length = rollout_length + 1 # include starting configuration
        assert self.robot is not None
        device = q.device
        trajectory = torch.zeros((B, actual_rollout_length, self.robot.MAIN_DOF), device=device)
        q_unnorm = self.robot.unnormalize_joints(q)
        assert isinstance(q_unnorm, torch.Tensor)
        trajectory[:, 0, :] = q_unnorm

        for i in range(1, actual_rollout_length):
            # Have to copy the scene and target pc's because they are being re-used and
            # sparse tensors are stateful
            qdeltas = self.actor(point_cloud_labels, point_cloud, q, self.pc_bounds)
            # Support models returning either [B, DOF] or [B, 1, DOF]
            qdelta = (qdeltas[:, -1, :] if qdeltas.dim()==3 else qdeltas)
            y_hat = torch.clamp(q + qdelta, min=-1, max=1)
            q_unnorm = self.robot.unnormalize_joints(y_hat)
            trajectory[:, i, :] = q_unnorm
            samples = sampler(q_unnorm)[..., :3]
            point_cloud[:, : samples.size(1)] = samples
            q = y_hat

        return trajectory

    def _target_error(
        self, batch: dict[str, torch.Tensor], rollouts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the position and orientation errors between the rollouts and the target.

        :param batch: The batch of data that was used to generate the rollouts.
        :param rollouts: The rollouts to calculate the errors for.
        :return: A tuple containing the position error and the orientation error.
        """
        assert self.fk_sampler is not None
        eff = self.fk_sampler.end_effector_pose(rollouts[:, -1])
        position_error = torch.norm(
            eff[:, :3, -1] - batch["target_position"], dim=1
        )

        R = torch.matmul(eff[:, :3, :3], batch["target_orientation"].transpose(1, 2))
        trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        cos_value = torch.clamp((trace - 1) / 2, -1, 1)
        orientation_error = torch.abs(torch.rad2deg(torch.acos(cos_value)))
        return position_error, orientation_error

    def _collision_error(self, batch, rollouts):
        """
        Calculates the collision rate of a batch of rollouts, using the robot's 
        collision sphere representation and the SDF's of the obstacle 
        primitives.
        """
        cuboids = TorchCuboids(
            batch["cuboid_centers"],
            batch["cuboid_dims"],
            batch["cuboid_quats"],
        )
        cylinders = TorchCylinders(
            batch["cylinder_centers"],
            batch["cylinder_radii"],
            batch["cylinder_heights"],
            batch["cylinder_quats"],
        )

        B = batch["cuboid_centers"].size(0)

        # Here is some Pytorch broadcasting voodoo to calculate whether each
        # rollout has a collision or not (looking to calculate the collision rate)
        assert self.robot is not None
        assert rollouts.size(0) == B
        assert rollouts.size(2) == self.robot.MAIN_DOF

        rollout_steps = rollouts.reshape(-1, self.robot.MAIN_DOF)
        has_collision = torch.zeros(B, dtype=torch.bool, device=rollouts.device)
        
        collision_spheres = self.robot.compute_spheres(rollout_steps)
        for radii, spheres in collision_spheres: # spheres: torch.Tensor [B, num_spheres, 3], 3-dim is x,y,z
            num_spheres = spheres.shape[-2]
            sphere_sequence = spheres.reshape((B, -1, num_spheres, 3))
            sdf_values = torch.minimum(
                cuboids.sdf_sequence(sphere_sequence),
                cylinders.sdf_sequence(sphere_sequence),
            )
            assert sdf_values.size(0) == B and sdf_values.size(2) == num_spheres
            radius_collisions = torch.any(
                sdf_values.reshape((sdf_values.size(0), -1)) <= radii, dim=-1
            )
            has_collision = torch.logical_or(radius_collisions, has_collision)
        return has_collision

    def state_validation_step(self, batch: dict[str, torch.Tensor]):
        """
        Validation (state-based): compute single-step BC loss only (fast & stable).
        """
        pc_labels = batch["point_cloud_labels"]
        pc        = batch["point_cloud"]
        q         = batch["configuration"]
        q_next     = batch["next_configuration"]
        is_expert  = batch["is_expert"]

        with torch.no_grad():
            qdeltas = self.actor(pc_labels, pc, q, self.pc_bounds)
            a_pred  = qdeltas[:, -1, :] if qdeltas.dim() == 3 else qdeltas
            q_pred  = torch.clamp(q + a_pred, -1, 1)
            assert self.robot is not None
            q_pred_unn = self.robot.unnormalize_joints(q_pred)
            q_next_unn = self.robot.unnormalize_joints(q_next)
            loss_bc    = self.loss_fun.bc_pointcloud_loss(
                pred_q_unnorm=q_pred_unn, expert_q_unnorm=q_next_unn, is_expert=is_expert)
            l_collision = self.loss_fun.collision_loss(
                unnormalized_q=q_next_unn,
                cuboid_centers=batch["cuboid_centers"],
                cuboid_dims=batch["cuboid_dims"],
                cuboid_quaternions=batch["cuboid_quats"],
                cylinder_centers=batch["cylinder_centers"],
                cylinder_radii=batch["cylinder_radii"],
                cylinder_heights=batch["cylinder_heights"],
                cylinder_quaternions=batch["cylinder_quats"],
            )

        self.val_point_match_loss.update(loss_bc)
        self.val_collision_loss.update(l_collision)
        val_loss = self.point_match_loss_weight * loss_bc + self.collision_loss_weight * l_collision
        self.val_loss.update(val_loss)

    def _end_rollouts_at_target(
        self, batch: dict[str, torch.Tensor], rollouts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Ends the rollouts at the target position and orientation, padding the
        rest of each successful trajectory with the last successful
        configuration. Also returns the length up until success and a mask
        indicating which rollouts have successful configurations.
        """
        B = rollouts.size(0)
        assert self.fk_sampler is not None
        assert self.robot is not None
        eff_poses = self.fk_sampler.end_effector_pose(
            rollouts.reshape(-1, self.robot.MAIN_DOF)
        ).reshape(B, -1, 4, 4)
        pos_errors = torch.norm(
            eff_poses[:, :, :3, -1] - batch["target_position"][:, None, :], dim=2
        )
        R = torch.matmul(
            eff_poses[:, :, :3, :3],
            batch["target_orientation"][:, None, :, :].transpose(-1, -2),
        )
        trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        cos_value = torch.clamp((trace - 1) / 2, -1, 1)
        orien_errors = torch.abs(torch.rad2deg(torch.acos(cos_value)))

        # Use the whole trajectory if there are no successes
        whole_trajectory_mask = torch.zeros_like(pos_errors, dtype=torch.bool)
        whole_trajectory_mask[:, -1] = True

        # Mask elements that meet criterion for success
        reaching_success = torch.logical_and(orien_errors < 15, pos_errors < 0.01)

        # If a trajectory has any "successful" configurations, keep marking those as true
        # If a trajectory does not have successful configurations, mark the
        # configuration with minimal position error as true
        has_reaching_success = reaching_success.any(dim=1)
        best_solution_mask = torch.where(
            has_reaching_success.unsqueeze(1), reaching_success, whole_trajectory_mask
        )

        # Find the first indices where success is true (i.e. trajectory lengths)
        lengths = torch.argmax(
            best_solution_mask
            * torch.arange(
                best_solution_mask.shape[1], 0, -1, device=best_solution_mask.device
            ),
            1,
            keepdim=True,
        ).squeeze(1)
        expanded_lengths = lengths[:, None, None].expand_as(rollouts)
        selection_mask = (
            torch.arange(rollouts.size(1), device=lengths.device)[None, :, None]
            > expanded_lengths
        )
        final_values = rollouts[torch.arange(rollouts.size(0)), lengths.squeeze()]
        final_values = final_values.unsqueeze(1).expand_as(rollouts)
        rollouts[selection_mask] = final_values[selection_mask]
        return rollouts, lengths, has_reaching_success

    def trajectory_validation_step(
        self, batch: dict[str, torch.Tensor]
    ) -> None:
        """
        Performs a validation step by calculating metrics on rollouts.
        """
        rollouts = self.rollout(batch, self.rollout_length, self._sample)
        rollouts, lengths, has_reaching_success = self._end_rollouts_at_target(batch, rollouts)
        position_error, orientation_error = self._target_error(batch, rollouts)
        has_collision = self._collision_error(batch, rollouts)

        self.val_position_error.update(position_error)
        self.val_orientation_error.update(orientation_error)
        self.val_collision_rate.update(has_collision.float().detach())
        self.val_funnel_collision_rate.update(
            has_collision[has_reaching_success].float().detach()
        )
        self.val_reaching_success_rate.update(has_reaching_success.float().detach())
        self.val_success_rate.update(
            torch.logical_and(~has_collision, has_reaching_success).float().detach()
        )
        self.val_waypoint_count.update((lengths[has_reaching_success]).float())

        # Average step size between consecutive configurations (masked to valid steps)
        T = rollouts.size(1)
        if T > 1:
            # Build mask over steps (t -> t+1) that occur before "lengths" per rollout
            step_idx = torch.arange(T - 1, device=rollouts.device).unsqueeze(0)  # [1, T-1]
            step_mask = step_idx < lengths.unsqueeze(1)  # [B, T-1]

            # Unnormalized step sizes
            deltas_unn = rollouts[:, 1:, :] - rollouts[:, :-1, :]  # [B, T-1, D]
            step_norms_unn = torch.norm(deltas_unn, dim=-1)  # [B, T-1]
            valid_unn = step_norms_unn[step_mask]
            if valid_unn.numel() > 0:
                self.val_step_size_unnorm.update(valid_unn)

            # Normalized step sizes (normalize each configuration, then difference)
            assert self.robot is not None
            rollouts_norm = self.robot.normalize_joints(rollouts)
            deltas_norm = rollouts_norm[:, 1:, :] - rollouts_norm[:, :-1, :]  # [B, T-1, D]
            step_norms_norm = torch.norm(deltas_norm, dim=-1)  # [B, T-1]
            valid_norm = step_norms_norm[step_mask]
            if valid_norm.numel() > 0:
                self.val_step_size_norm.update(valid_norm)

    @torch.no_grad()
    def rollout_value_visualization(
        self,
        batch: dict[str, torch.Tensor],
        index: int | None = None,
    ) -> None:
        """
        Rollout the actor policy and visualize the state transition values 
        estimated by the critic.
        """
        q = batch["configuration"].clone()     # [B, DOF], normalized
        B = q.size(0)
        cuboids = TorchCuboids(batch["cuboid_centers"], batch["cuboid_dims"], batch["cuboid_quats"])
        cylinders = TorchCylinders(batch["cylinder_centers"], batch["cylinder_radii"],
                                batch["cylinder_heights"], batch["cylinder_quats"])

        # point cloud for the actor input; will be updated only for active rows
        pc = batch["point_cloud"].clone()
        labels = batch["point_cloud_labels"]

        active = torch.ones(B, dtype=torch.bool, device=q.device)
        assert self.robot is not None
        
        if index is not None:
            viz_idx = index
        else:
            viz_idx = int(torch.randint(0, B, ()).item())
        
        viz_sample = {k: v[viz_idx] for k, v in batch.items()}
        viz_rollout_q_nexts = torch.zeros(
            (self.rollout_length, self.robot.MAIN_DOF), device=q.device, requires_grad=False)
        viz_rollout_values = torch.zeros(
            (self.rollout_length,), device=q.device, requires_grad=False)
        viz_rollout_length = 0

        for i in range(self.rollout_length):
            if not active.any():
                break

            # actor action for active rows
            q_act = q[active]
            pc_act = pc[active]
            lbl_act = labels[active]
            qdeltas = self.actor(lbl_act, pc_act, q_act, self.pc_bounds)  # [B, 1, DOF] or [B, DOF]
            a = (qdeltas[:, -1, :] if qdeltas.dim()==3 else qdeltas)
            a = a + torch.randn_like(a) * self.exploration_noise
            if self.action_clip is not None:
                a = torch.clamp(a, -self.action_clip, self.action_clip)
            q_next = (q_act + a).clamp(-1, 1)
            q_next_unn = self.robot.unnormalize_joints(q_next)

            # termination & reward at q_next
            reached = self._check_reaching_success(
                q_next_unn,
                batch["target_position"][active],
                batch["target_orientation"][active])
            collided = self._check_for_collisions(
                q_next_unn, cuboids[active], cylinders[active])
            done = reached | collided

            q1, _ = self.critic(lbl_act, pc_act, q_act, a, self.pc_bounds)

            # map original batch index to compressed active indices
            if bool(active[viz_idx].item()):
                vi = int(active[:viz_idx].sum().item())
                viz_rollout_q_nexts[i] = q_next[vi]
                viz_rollout_values[i] = q1[vi, 0]
                viz_rollout_length += 1
            else:
                break

            # update only those still active
            still = ~done
            if still.any():
                samples = self._sample(q_next_unn[still])[..., :3] # [B_still, num_robot_points, 3]
                pc_act[still, :samples.size(1)] = samples
                q_act[still] = q_next[still]

            # write back into full tensors
            q[active] = q_act
            pc[active] = pc_act

            # mark finished rows inactive
            tmp = active.nonzero(as_tuple=False).squeeze(1)
            active[tmp] = still

        # visualize randomly selected rollout from the batch, with Q-values
        # visualize_problem(self.robot, viz_sample)
        # visualize_rollout_values(
        #     self.robot,
        #     viz_rollout_q_nexts[:viz_rollout_length],
        #     viz_rollout_values[:viz_rollout_length],
        #     self.max_cumulative_reward,
        #     self.min_cumulative_reward
        # )

    @torch.no_grad()
    def rollout_rewards_visualization(
        self,
        batch: dict[str, torch.Tensor],
        index: int | None = None,
    ) -> None:
        """
        Rollout the actor policy and visualize the state transition values 
        estimated by the critic.
        """
        q = batch["configuration"].clone()     # [B, DOF], normalized
        B = q.size(0)
        cuboids = TorchCuboids(batch["cuboid_centers"], batch["cuboid_dims"], batch["cuboid_quats"])
        cylinders = TorchCylinders(batch["cylinder_centers"], batch["cylinder_radii"],
                                batch["cylinder_heights"], batch["cylinder_quats"])

        # point cloud for the actor input; will be updated only for active rows
        pc = batch["point_cloud"].clone()
        labels = batch["point_cloud_labels"]

        active = torch.ones(B, dtype=torch.bool, device=q.device)
        assert self.robot is not None
        
        if index is not None:
            viz_idx = index
        else:
            viz_idx = int(torch.randint(0, B, ()).item())
        
        viz_rollout_q_nexts = torch.zeros(
            (self.rollout_length, self.robot.MAIN_DOF), device=q.device, requires_grad=False)
        viz_rollout_rewards = torch.zeros(
            (self.rollout_length,), device=q.device, requires_grad=False)
        viz_rollout_length = 0

        for i in range(self.rollout_length):
            if not active.any():
                break

            # actor action for active rows
            q_act = q[active]
            pc_act = pc[active]
            lbl_act = labels[active]
            qdeltas = self.actor(lbl_act, pc_act, q_act, self.pc_bounds)  # [B, 1, DOF] or [B, DOF]
            a = (qdeltas[:, -1, :] if qdeltas.dim()==3 else qdeltas)
            a = a + torch.randn_like(a) * self.exploration_noise
            if self.action_clip is not None:
                a = torch.clamp(a, -self.action_clip, self.action_clip)
            q_next = (q_act + a).clamp(-1, 1)
            q_next_unn = self.robot.unnormalize_joints(q_next)

            # termination & reward at q_next
            reached = self._check_reaching_success(
                q_next_unn,
                batch["target_position"][active],
                batch["target_orientation"][active])
            collided = self._check_for_collisions(
                q_next_unn, cuboids[active], cylinders[active])
            done = reached | collided
            r_t = torch.where(collided, self.collision_reward,
                torch.where(reached, self.goal_reward, self.step_reward)).float().unsqueeze(1)

            # map original batch index to compressed active indices
            if bool(active[viz_idx].item()):
                vi = int(active[:viz_idx].sum().item())
                viz_rollout_q_nexts[i] = q_next[vi]
                viz_rollout_rewards[i] = r_t[vi, 0]
                viz_rollout_length += 1
            else:
                break

            # update only those still active
            still = ~done
            if still.any():
                samples = self._sample(q_next_unn[still])[..., :3] # [B_still, num_robot_points, 3]
                pc_act[still, :samples.size(1)] = samples
                q_act[still] = q_next[still]

            # write back into full tensors
            q[active] = q_act
            pc[active] = pc_act

            # mark finished rows inactive
            tmp = active.nonzero(as_tuple=False).squeeze(1)
            active[tmp] = still

        # visualize randomly selected rollout from the batch, with Q-values
        # visualize_rollout_rewards(
        #     self.robot,
        #     viz_rollout_q_nexts[:viz_rollout_length],
        #     viz_rollout_rewards[:viz_rollout_length],
        #     self.goal_reward,
        #     self.collision_reward
        # )


    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        actor_only: bool = False,
        map_location: str | torch.device | None = None,
        **init_kwargs,
    ) -> "CoLMotionPolicyTrainer":
        """
        Load a trainer instance from a checkpoint.

        - If actor_only=True, only initializes and loads the actor weights
          (compatible with legacy single-module checkpoints). Target actor
          weights are mirrored from the actor. Critics remain randomly
          initialized.
        - Otherwise, attempts to restore all modules (actor, critics, targets)
          from a Fabric-style checkpoint saved in run_training.py. Optimizers
          and schedulers are configured and restored when available.

        Any required __init__ kwargs (e.g., shared/training parameters) must be
        provided via init_kwargs.
        """

        ckpt = torch.load(checkpoint_path, map_location=map_location or "cpu")

        trainer = cls(**init_kwargs)
        trainer.configure_optimizers()

        def _load_actor_from_state_dict(state_dict: dict):
            trainer.actor.load_state_dict(state_dict, strict=False)
            trainer.target_actor.load_state_dict(trainer.actor.state_dict())

        if actor_only:
            if "actor" in ckpt and isinstance(ckpt["actor"], dict):
                _load_actor_from_state_dict(ckpt["actor"])
            elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                _load_actor_from_state_dict(ckpt["state_dict"])
            else:
                sd = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
                if sd:
                    _load_actor_from_state_dict(sd)

            # Also mirror the actor's point-cloud embedder weights into the critic's (and target critic's)
            # pc_encoder. Rely on strict state_dict load to validate dimension compatibility.
            try:
                actor_pc_encoder = trainer.actor.point_cloud_embedder
                critic_pc_encoder = trainer.critic.pc_encoder
                target_critic_pc_encoder = trainer.target_critic.pc_encoder
            except AttributeError as e:
                raise AttributeError(
                    "Expected actor to have 'point_cloud_embedder' and critic to have 'pc_encoder'"
                ) from e

            try:
                critic_pc_encoder.load_state_dict(
                    actor_pc_encoder.state_dict(), strict=True
                )
                target_critic_pc_encoder.load_state_dict(
                    actor_pc_encoder.state_dict(), strict=True
                )
            except RuntimeError as e:
                raise RuntimeError(
                    "Failed to load actor point-cloud embedder weights into critic pc_encoder. "
                    "Ensure encoder hyperparameters match between actor and critic (e.g., "
                    "num_robot_points, feature_dim, d_model). Original error: " + str(e)
                ) from e
            return trainer

        loaded_any = False
        if isinstance(ckpt, dict):
            if "actor" in ckpt:
                trainer.actor.load_state_dict(ckpt["actor"], strict=False)
                loaded_any = True
            if "critic" in ckpt:
                trainer.critic.load_state_dict(ckpt["critic"], strict=False)
                loaded_any = True
            if "target_actor" in ckpt:
                trainer.target_actor.load_state_dict(ckpt["target_actor"], strict=False)
            else:
                trainer.target_actor.load_state_dict(trainer.actor.state_dict())
            if "target_critic" in ckpt:
                trainer.target_critic.load_state_dict(ckpt["target_critic"], strict=False)

            if "actor_optim" in ckpt:
                try:
                    trainer.actor_optim.load_state_dict(ckpt["actor_optim"])  # type: ignore[arg-type]
                except Exception:
                    pass
            if "critic_optim" in ckpt:
                try:
                    trainer.critic_optim.load_state_dict(ckpt["critic_optim"])  # type: ignore[arg-type]
                except Exception:
                    pass
            if "actor_sch" in ckpt:
                try:
                    trainer.actor_scheduler.load_state_dict(ckpt["actor_sch"])  # type: ignore[arg-type]
                except Exception:
                    pass
            if "critic_sch" in ckpt:
                try:
                    trainer.critic_scheduler.load_state_dict(ckpt["critic_sch"])  # type: ignore[arg-type]
                except Exception:
                    pass

        if not loaded_any and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            trainer.actor.load_state_dict(ckpt["state_dict"], strict=False)
            trainer.target_actor.load_state_dict(trainer.actor.state_dict())

        return trainer
