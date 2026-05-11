"""
This file contains the CoLMotionPolicyTrainer class, which is responsible for
training the CoL motion policy.
"""

from typing import Tuple, Callable, Dict
from lightning import Fabric

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
import torchmetrics

from robofin.robots import Robot
from robofin.samplers import TorchRobotSampler

from avoid_everything_except_exploration.geometry import TorchCuboids, TorchCylinders
from avoid_everything_except_exploration.mpiformer import MotionPolicyTransformer
from avoid_everything_except_exploration.twin_critic import TwinCritic
from avoid_everything_except_exploration.loss import CoLLossFn
from avoid_everything_except_exploration.replay import ReplayBuffer
from avoid_everything_except_exploration.utils.visualization import visualize_problem, visualize_rollout_rewards, visualize_rollout_values


class CoLMotionPolicyTrainer():
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
        min_lr: float,
        max_lr: float,
        warmup_steps: int,
        weight_decay: float,
        gamma: float,
        exploration_noise: float,
        target_actor_noise: float,
        target_actor_noise_clip: float,
        action_clip: float | str,
        use_huber_loss: bool,
        tau: float,
        grad_clip_norm: float,
        pc_bounds: list[list[float]],
        rollout_length: int,
    ):
        self.urdf_path = urdf_path
        self.robot = None
        self.fk_sampler = None
        self.device = None
        self.num_robot_points = num_robot_points
        self.point_match_loss_weight = point_match_loss_weight
        self.collision_loss_weight = collision_loss_weight
        self.actor_loss_weight = actor_loss_weight
        self.loss_fun = CoLLossFn(self.urdf_path, collision_loss_margin)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.pc_bounds = torch.as_tensor(pc_bounds)
        self.rollout_length = rollout_length
        self.robot_dof = robot_dof
        self.reward_scale = reward_scale
        self.collision_reward = collision_reward * reward_scale
        self.goal_reward = goal_reward * reward_scale
        self.step_reward = step_reward * reward_scale
        self.gamma = gamma
        self.exploration_noise = exploration_noise
        self.target_actor_noise = target_actor_noise
        self.target_actor_noise_clip = target_actor_noise_clip
        if isinstance(action_clip, str):
            self.action_clip = None
        else:
            self.action_clip = action_clip
        self.use_huber_loss = use_huber_loss
        self.tau = tau
        self.grad_clip_norm = grad_clip_norm
        # Cached per-joint half-range tensor, initialized once setup() builds the robot.
        self._joint_range_half: torch.Tensor | None = None

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

        self.actor = MotionPolicyTransformer(
            num_robot_points=num_robot_points,
            robot_dof=robot_dof,
        )
        self.target_actor = MotionPolicyTransformer(
            num_robot_points=num_robot_points,
            robot_dof=robot_dof,
        )
        self.critic = TwinCritic(
            num_robot_points=num_robot_points,
            robot_dof=robot_dof,
        )
        self.target_critic = TwinCritic(
            num_robot_points=num_robot_points,
            robot_dof=robot_dof,
        )

        self.actor_optim: torch.optim.Optimizer
        self.critic_optim: torch.optim.Optimizer
        self.actor_scheduler: torch.optim.lr_scheduler.LambdaLR
        self.critic_scheduler: torch.optim.lr_scheduler.LambdaLR

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
        Build separate optimizers/schedulers for actor and critic.
        Target networks are updated via Polyak soft updates (no optimizer).
        """
        betas = (0.9, 0.95)
        self.actor_optim  = torch.optim.AdamW(
            self.actor.parameters(),  lr=self.min_lr,  weight_decay=self.weight_decay, betas=betas)
        critic_param_groups = [
            {"params": self.critic.pc_encoder.parameters(), "lr": self.min_lr, "name": "critic_shared"},
            {"params": self.critic.q1.parameters(),         "lr": self.min_lr, "name": "critic_q1"},
            {"params": self.critic.q2.parameters(),         "lr": self.min_lr, "name": "critic_q2"},
        ]
        self.critic_optim = torch.optim.AdamW(
            critic_param_groups, lr=self.min_lr, weight_decay=self.weight_decay, betas=betas)

        def lr_lambda(step):
            lr = self.min_lr + (self.max_lr - self.min_lr) * min(1.0, step / self.warmup_steps)
            return lr / self.min_lr

        self.actor_scheduler  = LambdaLR(self.actor_optim,  lr_lambda)
        self.critic_scheduler = LambdaLR(self.critic_optim, lr_lambda)

        # Hard-copy weights into targets once here; call polyak_update() each step.
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())


        return {
            "actor_optim": self.actor_optim,
            "critic_optim": self.critic_optim,
            "actor_scheduler": self.actor_scheduler,
            "critic_scheduler": self.critic_scheduler,
        }

    @torch.no_grad()
    def _polyak_update(self, tau: float = 0.005):
        """Soft update of target networks: θ' ← τ θ + (1-τ) θ'"""
        for tgt, src in zip(self.target_actor.parameters(), self.actor.parameters()):
            tgt.data.lerp_(src.data, tau)
        for tgt, src in zip(self.target_critic.parameters(), self.critic.parameters()):
            tgt.data.lerp_(src.data, tau)

    def _verify_device(self) -> torch.device:
        """
        Resolve current device from model parameters.
        """
        assert next(self.actor.parameters()).device == next(self.critic.parameters()).device
        assert next(self.actor.parameters()).device == next(self.target_actor.parameters()).device
        assert next(self.actor.parameters()).device == next(self.target_critic.parameters()).device
        return next(self.actor.parameters()).device

    def _apply_noise_in_joint_space(
        self,
        normalized_action: torch.Tensor,
        noise_std: float,
        noise_clip: float | None = None,
        clamp_unnormalized: float | None = None,
    ) -> torch.Tensor:
        """
        Convert normalized deltas to real joint space, add/clamp noise there, 
        return normalized result.

        :param normalized_action: The normalized action to apply noise to.
        :param noise_std: The standard deviation of the noise in joint space.
        :param noise_clip: The clip value for the noise in joint space.
        :param clamp_unnormalized: The clamp value for the unnormalized action in joint space.
        :return: The normalized action with noise applied (normalize joint space).
        """
        if noise_std <= 0 and clamp_unnormalized is None:
            return normalized_action
        assert self.robot is not None
        assert self._joint_range_half is not None
        joint_range_half = self._joint_range_half.to(dtype=normalized_action.dtype, device=normalized_action.device)
        while joint_range_half.dim() < normalized_action.dim():
            joint_range_half = joint_range_half.unsqueeze(0)

        action_unn = normalized_action * joint_range_half
        if noise_std > 0:
            noise = torch.randn_like(action_unn) * noise_std
            if noise_clip is not None:
                noise = torch.clamp(noise, -noise_clip, noise_clip)
            action_unn = action_unn + noise

        if clamp_unnormalized is not None:
            action_unn = torch.clamp(action_unn, -clamp_unnormalized, clamp_unnormalized)

        return action_unn / joint_range_half

    def setup(self, fabric: Fabric):
        """
        Device-critical initialization. Call after moving model to the desired 
        device. Initializes robot and point cloud sampler on current device.
        """

        # fabric setup: wrap trainable modules w/ their optimizers
        self.actor,  self.actor_optim  = fabric.setup(self.actor,  self.actor_optim)
        self.critic, self.critic_optim = fabric.setup(self.critic, self.critic_optim)
        # target networks have no optimizers
        self.target_actor  = fabric.setup(self.target_actor)
        self.target_critic = fabric.setup(self.target_critic)

        self.device = self._verify_device()
        assert str(self.device) != "cpu", "You do not want to train on CPU"
        self.robot = Robot(self.urdf_path, device=self.device)
        self.fk_sampler = TorchRobotSampler(
            self.robot,
            num_robot_points=self.num_robot_points,
            num_eef_points=128,
            use_cache=True,
            with_base_link=True,
            device=self.device,
        )
        assert self.robot.MAIN_DOF == self.robot_dof
        self.pc_bounds = self.pc_bounds.to(self.device)
        self.actor.train()
        self.critic.train()

        # never need gradients for target networks
        self.target_actor.eval()
        self.target_critic.eval()

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

    def _sample(self, q: torch.Tensor) -> torch.Tensor:
        """
        Samples a point cloud from the surface of all the robot's links

        :param q torch.Tensor: Batched configuration in joint space
        :rtype torch.Tensor: Batched point cloud of size [B, self.num_robot_points, 3]
        """
        assert self.fk_sampler is not None
        points = self.fk_sampler.sample(q)
        return points

    def _bc_loss(
        self,
        q_pred: torch.Tensor,
        q_next: torch.Tensor,
        is_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        BC loss in point-cloud space
        """
        assert self.robot is not None
        q_next_unn = self.robot.unnormalize_joints(q_next)
        return self.loss_fun.bc_pointcloud_loss(
            pred_q_unnorm=self.robot.unnormalize_joints(q_pred),
            expert_q_unnorm=q_next_unn,
            is_expert=is_expert,
        )

    def _collision_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Collision loss
        """
        q_next = batch["next_configuration"]
        assert self.robot is not None
        return self.loss_fun.collision_loss(
            unnormalized_q=self.robot.unnormalize_joints(q_next),
            cuboid_centers=batch["cuboid_centers"],
            cuboid_dims=batch["cuboid_dims"],
            cuboid_quaternions=batch["cuboid_quats"],
            cylinder_centers=batch["cylinder_centers"],
            cylinder_radii=batch["cylinder_radii"],
            cylinder_heights=batch["cylinder_heights"],
            cylinder_quaternions=batch["cylinder_quats"],
        )

    def _actor_loss(
        self,
        pc_labels: torch.Tensor,
        pc: torch.Tensor,
        q: torch.Tensor,
        a_pred: torch.Tensor,
    ) -> torch.Tensor:
        """
        RL Actor loss
        """
        # don't backprop into critic
        for p in self.critic.parameters():
            p.requires_grad_(False)
        q1, _ = self.critic(pc_labels, pc, q, a_pred, self.pc_bounds)
        loss_actor = -q1.mean()
        for p in self.critic.parameters():
            p.requires_grad_(True)
        return loss_actor

    def _critic_loss(
        self,
        batch: dict[str, torch.Tensor],
        metrics: dict[str, float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One-step TD loss for critic using target networks.
        """
        q      = batch["configuration"]
        a      = batch["action"]
        q_next = batch["next_configuration"]
        r      = batch["reward"]
        done   = batch.get("done", torch.zeros_like(r))

        # build next state's (s') point cloud by sampling robot at q_next
        assert self.robot is not None
        q_next_unn = self.robot.unnormalize_joints(q_next)
        robot_pc = self._sample(q_next_unn)[..., :3]
        next_pc = torch.cat([robot_pc, batch["point_cloud"][:, robot_pc.size(1):]], dim=1)

        # target action a' = π'(s')
        with torch.no_grad():
            a_next = self.target_actor(batch["point_cloud_labels"], next_pc, q_next, self.pc_bounds)
            if a_next.dim() == 3:
                a_next = a_next[:, -1, :]
            a_next = self._apply_noise_in_joint_space(
                a_next,
                noise_std=self.target_actor_noise,
                noise_clip=self.target_actor_noise_clip,
                clamp_unnormalized=self.action_clip,
            )

        with torch.no_grad():
            q_next_target, q_next_target2 = self.target_critic(
                batch["point_cloud_labels"], next_pc, q_next, a_next, self.pc_bounds
            )
            y = r + self.gamma * (1.0 - done) * torch.min(q_next_target, q_next_target2)

        q_sa, q_sa2 = self.critic(
            batch["point_cloud_labels"], batch["point_cloud"], q, a, self.pc_bounds)

        with torch.no_grad():
            metrics["q_target_mean_(y)"] = float(y.mean().item())
            metrics["q_sa_mean"]     = float(q_sa.mean().item())
        if self.use_huber_loss:
            return torch.nn.functional.huber_loss(q_sa, y, delta=1.0), \
                   torch.nn.functional.huber_loss(q_sa2, y, delta=1.0)
        return torch.nn.functional.mse_loss(q_sa, y), \
               torch.nn.functional.mse_loss(q_sa2, y)

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        fabric: Fabric,
        update_targets: bool,
        use_actor_loss: bool,
    ) -> Dict[str, float]:
        """
        One training iteration on a mixed (expert + actor) batch.
        Calculates losses and performs optimization. Critic and BC losses are
        always computed and optimized on. Target networks are updated only if
        update_targets is True. The RL actor loss is only computed and optimized 
        on if use_actor_loss is True.

        Computes L_BC, L_Q1, L_A via self._state_based_step(), then:
        1) critic step on L_Q1
        2) actor step on (λ_A * L_A + λ_BC * L_BC)
        3) Polyak soft-update of targets

        :param update_targets: Whether to update the target networks
        :param use_actor_loss: Whether to use the actor loss (else just BC and critic updates)
        :return: A flat dict of scalars for logging
        """
        # critic update on one-step TD loss first (keeps actor graph clean)
        metrics = {}
        loss_q1, loss_q2 = self._critic_loss(batch, metrics)
        metrics.update({
            "critic_loss_1": float(loss_q1.detach().item()),
            "critic_loss_2": float(loss_q2.detach().item()),
        })
        self.critic_optim.zero_grad(set_to_none=True)
        fabric.backward(loss_q1 + loss_q2)
        clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_clip_norm)
        self.critic_optim.step()
        self.critic_scheduler.step()

        # actor predict next q via Δq
        q = batch["configuration"]
        pc_labels = batch["point_cloud_labels"]
        pc = batch["point_cloud"]
        qdeltas = self.actor(pc_labels, pc, q, self.pc_bounds)
        a_pred  = qdeltas[:, -1, :] if qdeltas.dim() == 3 else qdeltas
        if self.action_clip is not None:
            a_pred = torch.clamp(a_pred, -self.action_clip, self.action_clip)
        q_pred  = torch.clamp(q + a_pred, -1, 1)

        loss_bc = self._bc_loss(q_pred, batch["next_configuration"], batch["is_expert"])
        metrics["point_match_loss"] = float(loss_bc.detach().item())

        collision_loss = self._collision_loss(batch)
        metrics["collision_loss"] = float(collision_loss.detach().item())

        # actor update on critic-guided actor loss (+ optional BC)
        if use_actor_loss:
            loss_actor = self._actor_loss(pc_labels, pc, q, a_pred)
            metrics["actor_loss"] = float(loss_actor.detach().item())
            actor_total = (self.point_match_loss_weight * loss_bc +
                            self.collision_loss_weight * collision_loss +
                            self.actor_loss_weight * loss_actor)
        else:
            actor_total = (self.point_match_loss_weight * loss_bc +
                           self.collision_loss_weight * collision_loss)

        self.actor_optim.zero_grad(set_to_none=True)
        fabric.backward(actor_total)
        clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_clip_norm)
        self.actor_optim.step()
        self.actor_scheduler.step()

        if update_targets:
            self._polyak_update(tau=self.tau) # soft target update

        metrics["lr"]  = float(self.actor_optim.param_groups[0]["lr"])

        return metrics

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

    def _check_reaching_success(
        self,
        q_unnorm: torch.Tensor,
        target_position: torch.Tensor,
        target_orientation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Checks if a batch of trajectories has reached the target position and 
        orientation within a tolerance.
        """
        assert self.fk_sampler is not None
        eff_poses = self.fk_sampler.end_effector_pose(q_unnorm)
        pos_errors = torch.linalg.vector_norm(
            eff_poses[:, :3, -1] - target_position, dim=-1
        )
        R = torch.matmul(
            eff_poses[:, :3, :3],
            target_orientation.transpose(-1, -2),
        )
        trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        cos_value = torch.clamp((trace - 1) / 2, -1, 1)
        orient_errors = torch.abs(torch.rad2deg(torch.acos(cos_value)))
        return torch.logical_and(orient_errors < 15, pos_errors < 0.01) # less than 15 degrees and 1cm

    def _check_for_collisions(
        self, q_unnorm: torch.Tensor, cuboids: TorchCuboids, cylinders: TorchCylinders
    ) -> torch.Tensor:
        """
        Checks if a batch of joint configurations has collided with their environments.
        """
        assert self.robot is not None, "Robot not initialized"
        collision_spheres = self.robot.compute_spheres(q_unnorm)
        has_collision = torch.zeros(
            (q_unnorm.shape[0],), dtype=torch.bool, device=q_unnorm.device
        )
        for radii, spheres in collision_spheres:
            num_spheres = spheres.shape[-2]
            sphere_sequence = spheres.reshape((q_unnorm.shape[0], -1, num_spheres, 3))
            sdf_values = torch.minimum(
                cuboids.sdf_sequence(sphere_sequence),
                cylinders.sdf_sequence(sphere_sequence),
            )
            assert (
                sdf_values.size(0) == q_unnorm.shape[0]
                and sdf_values.size(2) == num_spheres
            )
            radius_collisions = torch.any(
                sdf_values.reshape((sdf_values.size(0), -1)) <= radii, dim=-1
            )
            has_collision = torch.logical_or(radius_collisions, has_collision)
        return has_collision

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
        visualize_rollout_values(
            self.robot,
            viz_rollout_q_nexts[:viz_rollout_length],
            viz_rollout_values[:viz_rollout_length],
            self.max_cumulative_reward,
            self.min_cumulative_reward
        )

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
        visualize_rollout_rewards(
            self.robot,
            viz_rollout_q_nexts[:viz_rollout_length],
            viz_rollout_rewards[:viz_rollout_length],
            self.goal_reward,
            self.collision_reward
        )

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
