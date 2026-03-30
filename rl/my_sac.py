import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import SAC
from typing import Any, Callable, Optional, Tuple, Dict
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from tqdm.auto import tqdm
from datetime import datetime
import os
import wandb
from wandb.integration.sb3 import WandbCallback
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

np.set_printoptions(precision=3, suppress=False)


class SACDebug(SAC):
    """
    Custom SAC class for debugging, with options for deterministic rollout and verbose logging
    """
    def __init__(self, *args, force_deterministic=True, debug_verbose=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_deterministic = force_deterministic
        self.debug_verbose_level = debug_verbose
        self.log_indent = ""
        if self.debug_verbose_level > 0:
            print(f"Using SACDebug, force_deterministic={self.force_deterministic}, debug_level={self.debug_verbose_level}")
        
        self.previous_weights = {}
        
        # Components to monitor for Level 2 summary
        # Format: (Display Name, Path, Indent Level)
        self.COMPONENTS_TO_MONITOR = [
            ("Perception", "features_extractor.perception_encoder", 0),
            ("PointNet", "features_extractor.perception_encoder.point_cloud_embedder", 1),
            ("Features", "features_extractor.perception_encoder.feature_embedder", 1), 
            ("Action Token", "features_extractor.perception_encoder.action_tokens", 1),
            ("Type Emb", "features_extractor.perception_encoder.token_type_embedding", 1),
            ("Transformer", "features_extractor.transformer_encoder", 0),
            ("Mu", "mu", 0),
            ("LogStd", "log_std", 0),
            ("Q-Nets", ["qf0", "qf1"], 0),
        ]

    def log(self, msg: str, level: int = 0) -> None:
        """
        Log message if debug level is sufficient.
        Level 0: Minimal, always print (only major events)
        Level 1: Summary (high-level status)
        Level 2: Component Summary (status of key components)
        Level 3: Detailed (per-parameter details)

        Default level is 0, meaning the message will always be printed. 
        """
        if self.debug_verbose_level >= level:
            print(self.log_indent + msg)
    
    def indent(self):
        self.log_indent += "  "
        return self
    
    def unindent(self):
        self.log_indent = self.log_indent[:-2]
        return self


    def _get_submodule(self, model: th.nn.Module, path: str) -> Optional[th.nn.Module]:
        """Safely retrieve a submodule by dot-separated path."""
        try:
            curr = model
            for part in path.split("."):
                curr = getattr(curr, part)
            return curr
        except AttributeError:
            return None


    def monitor_weights_sum(self, model: th.nn.Module, name: str, context="", verbose_level=1) -> None:
        """
        Compute and log the sum of all weights in the given model.
        
        Args:
            model: The module to check (e.g. self.policy.actor)
            name: Human-readable name (e.g. "Actor")
            context: Additional context to include in the log
            verbose_level: Minimum debug level to print this message
        """
        s = 0
        with th.no_grad():
            for p in model.parameters():
                s += p.data.sum().item()
        self.log(f"[{name} weights sum] {context}: {s:.4f}", level=verbose_level)


    def monitor_weight_changes(self, model: th.nn.Module, name: str, threshold: float = 1e-6) -> None:
        """
        Check which parameters have changed since the last check.
        
        Args:
            model: The module to check (e.g. self.policy.actor)
            name: Human-readable name (e.g. "Actor")
            threshold: Minimum difference to consider a parameter changed
        """
        # Move to CPU to save GPU memory
        current_params = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        total_scalars = sum(p.numel() for p in current_params.values())
        
        if name in self.previous_weights:
            prev_params = self.previous_weights[name]
            changed_scalar_count = 0
            
            # Level 2: Component Summary
            if self.debug_verbose_level == 2:
                self.log(f"[{name} Weights Changes Summary]", level=2)
                for comp_name, comp_path, indent in self.COMPONENTS_TO_MONITOR:
                    # Check if component exists in this model (Actor or Critic)
                    # We can check if any parameter starts with comp_path
                    comp_diff = 0.0
                    comp_found = False
                    
                    # Iterate all params to find matches (inefficient but safe)
                    # Convert list to tuple for startswith
                    paths_to_check = tuple(comp_path) if isinstance(comp_path, list) else comp_path

                    comp_scalar_count = 0
                    for n, p in current_params.items():
                        if n.startswith(paths_to_check):
                            comp_found = True
                            comp_scalar_count += p.numel()
                            if n in prev_params:
                                comp_diff += (p - prev_params[n]).abs().sum().item()
                    
                    if comp_found:
                        status = "CHANGED" if comp_diff > threshold else "Unchanged"
                        prefix = "  " * (indent + 1)
                        
                        # Normalize diff by scalar count
                        norm_diff = comp_diff / comp_scalar_count if comp_scalar_count > 0 else 0.0
                        
                        diff_str = f"(Avg Diff: {norm_diff*100:.3f}%)" if status == "CHANGED" else ""
                        self.log(f"{prefix}> {name} {comp_name}: {status} {diff_str}", level=2)

            # Level 3: Detailed per-parameter check
            if self.debug_verbose_level >= 3:
                for n, p in current_params.items():
                    num_scalars = p.numel()
                    if n in prev_params:
                        diff = (p - prev_params[n]).abs().sum().item()
                        if diff > threshold:
                            changed_scalar_count += num_scalars
                            self.log(f"  !!! CHANGED: {n} ({num_scalars} scalars) | Diff: {diff:.6f}", level=3)
                        else:
                            self.log(f"  ✓ Unchanged: {n} ({num_scalars} scalars) | Diff: {diff:.6f}", level=3)
            
            # Calculate total changed count for summary (always needed)
            if self.debug_verbose_level < 3: # Avoid double counting if loop didn't run
                 for n, p in current_params.items():
                    if n in prev_params:
                        diff = (p - prev_params[n]).abs().sum().item()
                        if diff > threshold:
                            changed_scalar_count += p.numel()

            self.log(f"[Count {name} Weights Changed] Changed scalars: {changed_scalar_count/total_scalars*100:.2f}% out of {total_scalars} params", level=1)
        
        # update previous weights
        self.previous_weights[name] = current_params


    def monitor_freeze_status(self, model: th.nn.Module, name: str) -> None:
        """
        Check which parameters are currently frozen.
        
        Args:
            model: The module to check
            name: Human-readable name
        """
        total_params = 0
        unfrozen_count = 0

        # Level 2: Component Summary
        if self.debug_verbose_level == 2:
            self.log(f"[{name} Freeze Status Summary]", level=2)
            for comp_name, comp_path, indent in self.COMPONENTS_TO_MONITOR:
                # Handle list of paths
                if isinstance(comp_path, list):
                    paths = comp_path
                else:
                    paths = [comp_path]

                c_total = 0
                c_frozen = 0
                module_found = False

                for path in paths:
                    module = self._get_submodule(model, path)
                    if module is not None:
                        module_found = True
                        
                        if isinstance(module, th.nn.Module):
                            params_to_check = module.parameters()
                        elif isinstance(module, th.nn.Parameter):
                            params_to_check = [module]
                        else:
                            continue

                        for p in params_to_check:
                            num_scalars = p.numel()
                            c_total += num_scalars
                            if not p.requires_grad:
                                c_frozen += num_scalars
                
                if module_found and c_total > 0:
                    if c_frozen == c_total:
                        status = "all Frozen"
                    elif c_frozen == 0:
                        status = "all Trainable"
                    else:
                        status = f"Mixed ({c_frozen}/{c_total} frozen scalars)"
                    
                    prefix = "  " * (indent + 1)
                    self.log(f"{prefix}> {name} {comp_name}: {status}", level=2)

        # Level 3: Detailed per-parameter check
        if self.debug_verbose_level >= 3:
            for n, p in model.named_parameters():
                num_scalars = p.numel()
                total_params += num_scalars
                if p.requires_grad:
                    self.log(f"  !!! UNFROZEN: {n} ({num_scalars} scalars)", level=3)
                    unfrozen_count += num_scalars
                else:
                    self.log(f"  ✓ Frozen: {n} ({num_scalars} scalars)", level=3)
        else:
             # Just count for summary
             for n, p in model.named_parameters():
                num_scalars = p.numel()
                total_params += num_scalars
                if p.requires_grad:
                    unfrozen_count += num_scalars

        self.log(f"[{name} Count Frozen Weights] Frozen scalars: {(total_params - unfrozen_count)/total_params*100:.2f}%  | Trainable scalars: {unfrozen_count/total_params*100:.2f}% | out of {total_params} params", level=1)

    def monitor_agent(self, 
                      description: str = "",
                      monitor_actor: bool = True,
                      monitor_critic: bool = True
        ) -> None:
        """
        High-level function to monitor and debug both Actor and Critic status.
        """
        if self.debug_verbose_level < 1:
            return
        
        if description != "":
            description = f": {description} "
        self.log("\n"+"-"*50, level=1)
        self.log(f"--- Start Monitoring Policy Status {description}---", level=1)

        if monitor_actor:
            self.log(f"[DEBUG ACTOR{description}]", level=1)
            self.indent()
            self.monitor_weights_sum(self.policy.actor, "Actor")
            self.monitor_freeze_status(self.policy.actor, "Actor")
            self.monitor_weight_changes(self.policy.actor, "Actor")
            self.unindent()
        if monitor_critic:
            self.log(f"\n[DEBUG CRITIC{description}]", level=1)
            self.indent()
            self.monitor_weights_sum(self.policy.critic, "Critic")
            self.monitor_freeze_status(self.policy.critic, "Critic")
            self.monitor_weight_changes(self.policy.critic, "Critic")
            self.unindent()

        # TODO: potentially monitor also target critic
        self.log(f"--- End Monitoring Policy Status {description}---", level=1)
        self.log("-"*50+"\n", level=1)


    # to overwrite off_policy_algorithm.py:OffPolicyAlgorithm._sample_action
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            
            if self.debug_verbose_level >= 3:
                # DEBUG: Check weights sum
                self.monitor_weights_sum(self.policy.actor, "Actor", verbose_level=3)

                # DEBUG: Check action distribution parameters
                obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
                with th.no_grad():
                    mean_actions, log_std, kwargs = self.actor.get_action_dist_params(obs_tensor)
                    # Note: the action is squashed
                    sampled_action = self.actor.action_dist.actions_from_params(mean_actions, log_std, deterministic=self.force_deterministic, **kwargs)
                    
                    self.log("[debug]\n"
                          f"mean_actions=\t{mean_actions.detach().cpu().numpy().squeeze()},\n"
                          f"log_std=\t{log_std.detach().cpu().numpy().squeeze()},\n"
                          f"squashed_sampled_action=\t{sampled_action.detach().cpu().numpy().squeeze()}\n"
                          f"difference sample-mean=\t{th.abs(sampled_action - mean_actions).detach().cpu().numpy().squeeze()}", level=3)

            unscaled_action, _ = self.predict(self._last_obs, deterministic=self.force_deterministic)
            
        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action

        return action, buffer_action
    

    def log_std_status(self, context: str = "Prefill Start") -> None:
        """
        Log the current log_std values and corresponding standard deviations.
        
        Args:
            context: Description of when this logging occurs (e.g., "Prefill Start", "Step 5")
        """
        if self.debug_verbose_level < 1:
            return
        
        with th.no_grad():
            log_std_values = self.policy.actor.log_std.bias
            std_values = th.exp(log_std_values)
            self.log(f"[{context}] log_std.bias={log_std_values.cpu().numpy()}", level=1)
            self.log(f"[{context}] std (exp(log_std))={std_values.cpu().numpy()}", level=1)
    
    def log_action_sampling(self, obs: dict, actions: np.ndarray, step: int, deterministic: bool) -> None:
        """
        Log detailed action sampling information for debugging stochastic vs deterministic sampling.
        Shows log_std, mean actions (before/after tanh), sampled actions, and differences.
        
        Args:
            obs: Observation dict to get action distribution for
            actions: Already sampled actions from predict()
            step: Current step number (for logging)
            deterministic: Whether deterministic sampling was used
        """
        if self.debug_verbose_level < 2:
            return
        
        with th.no_grad():
            obs_tensor, _ = self.policy.obs_to_tensor(obs)
            mean_actions, log_std, kwargs = self.policy.actor.get_action_dist_params(obs_tensor)
            std = th.exp(log_std)
            mean_actions_squashed = th.tanh(mean_actions)
            
            self.log(f"\n[Prefill Step {step}] deterministic={deterministic}", level=2)
            self.log(f"  log_std={log_std.cpu().numpy().squeeze()}", level=2)
            self.log(f"  std (exp(log_std))={std.cpu().numpy().squeeze()}", level=2)
            self.log(f"  mean_actions (before tanh)={mean_actions.cpu().numpy().squeeze()}", level=2)
            self.log(f"  mean_actions (after tanh)={mean_actions_squashed.cpu().numpy().squeeze()}", level=2)
            self.log(f"  sampled actions={actions.squeeze()}", level=2)
            self.log(f"  |sampled - mean|={np.abs(actions.squeeze() - mean_actions_squashed.cpu().numpy().squeeze())}", level=2)




class MySAC(SACDebug):
    """
    Expantion of SB3 SAC class

    - Added functions to freeze/unfreeze learnable parts of the actor and critic
    - Added function to warm up the critic by training with frozen actor
    - Added explicit log_std initialization (SB3 bug workaround)
    """

    def initialize_log_std(self, log_std_value: float = -20.0, state_independent_start: bool = True) -> None:
        """
        Explicitly initialize actor's log_std parameter.
        
        This is a workaround for SB3 SAC bug where log_std_init in policy_kwargs is ignored
        for standard SAC (only works with use_sde=True).
        
        Args:
            log_std_value: Value to initialize log_std to (default: -20.0 for near-deterministic)
            state_independent_start: If True, set weight=0 for constant log_std across all states.
                             If False, only set bias (keeps state-dependent behavior).
        """
        with th.no_grad():
            # set constant initial log_std.bias
            self.policy.actor.log_std.bias.fill_(log_std_value)
            if state_independent_start:
                # State-independent log_std (constant across all states) at start
                self.policy.actor.log_std.weight.fill_(0.0)  # No state dependence
                self.log(f"✓ Initialized actor log_std (log_std.weight=0, state-independent start)", level=0)
            else:
                # State-dependent log_std (can vary with input features) at start
                self.log(f"✓ Initialized actor log_std (rand init log_std.weight, state-dependent start)", level=0)
            
            # Verify and report
            log_std_values = self.policy.actor.log_std.bias
            std_values = th.exp(log_std_values)
            self.log(f"  log_std.bias = {log_std_values.cpu().numpy()}", level=1)
            self.log(f"  std (exp(log_std)) = {std_values.cpu().numpy()}", level=1)

    def setup_logger(
        self,
        logger_config: Dict,
        hyperparameters: Optional[Dict] = None,
        run_name: Optional[str] = None,
    ) -> Tuple[str, Optional[BaseCallback]]:
        """
        Configure multi-format logger for training metrics.
        
        Sets up Stable Baselines3's logger to write to multiple outputs:
        - stdout: Console output for real-time monitoring
        - tensorboard: TensorBoard logs for visualization
        - wandb: Weights & Biases cloud logging
        ...
        
        Args:
            logger_config: Dictionary with logging configuration:
                - log_dir: Base directory for logs
                - run_name: Name for this run (timestamp will be appended if run_name param not provided)
                - formats: List like ["stdout", "tensorboard", "wandb"]
                - wandb_project: WandB project name (required if "wandb" in formats)
                - wandb_entity: WandB username/team (optional)
            hyperparameters: Dictionary of hyperparameters to log to WandB (optional)
            run_name: Pre-generated run name with timestamp (optional). If provided, uses this 
                instead of generating a new timestamp. Useful when the log directory was already 
                created earlier (e.g., in get_args_and_cfg()).
        
        Returns:
            Tuple of (run_name, wandb_callback):
            - run_name: name + timestamp
            - wandb_callback: WandbCallback if "wandb" in formats, None otherwise
            Add the callback to model.learn(callback=...).
        """
        log_dir = logger_config.get("log_dir", "./logs")
        formats = logger_config.get("formats", ["stdout", "log", "tensorboard", "wandb"])
        
        # Use provided run_name or generate new one with timestamp
        if run_name is None:
            base_run_name = logger_config.get("run_name")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if base_run_name:
                run_name = f"{base_run_name}-{timestamp}"
            else:
                run_name = timestamp
        
        full_log_dir = os.path.join(log_dir, run_name)
        os.makedirs(full_log_dir, exist_ok=True)
        
        # Filter out wandb from SB3 formats (we handle it separately)
        sb3_formats = [f for f in formats if f != "wandb"]
        
        # Configure SB3 logger with standard formats
        if sb3_formats:
            new_logger = configure(full_log_dir, sb3_formats)
            self.set_logger(new_logger)
            self.log(f"  Logging to: {', '.join(sb3_formats)}", level=1)
        
        # Setup WandB if in formats list
        wandb_callback = None
        if "wandb" in formats:
            wandb_project = logger_config.get("wandb_project")
            if not wandb_project:
                raise ValueError("wandb_project is required when 'wandb' in formats")
            
            # Login to WandB using API key from .env if available
            wandb_api_key = os.getenv("WANDB_API_KEY")
            if wandb_api_key:
                wandb.login(key=wandb_api_key)
            
            # Initialize WandB run (following official API pattern)
            wandb.init(
                project=wandb_project,
                name=run_name,
                dir=full_log_dir,  # Tell WandB where logs are
                config=hyperparameters or {},
                sync_tensorboard=True,  # Auto-upload SB3's tensorboard metrics from this dir
                reinit="finish_previous",  # Allow multiple runs in same process
            )
            
            # Create WandbCallback (only takes model_save_path and verbose)
            wandb_callback = WandbCallback(
                model_save_path=None,  # Don't auto-save to wandb
                verbose=self.verbose,
            )
            self.log(f"✓ WandB enabled: project={wandb_project}, run={run_name}", level=0)
        
        return run_name, wandb_callback

    def freeze_learnable_actor(self) -> None:
        """
        Freeze learnable part of the actor
        """
        if "perception" not in self.policy.actor.features_extractor.permanently_frozen_components:
            self.log("Freezing actor perception module...", level=1)
            self.policy.actor.features_extractor.freeze_perception()
        if "transformer" not in self.policy.actor.features_extractor.permanently_frozen_components:
            self.log("Freezing actor transformer module...", level=1)
            self.policy.actor.features_extractor.freeze_transformer()

        # always freeze policy heads (mu and log_std)
        self.log("Freezing actor policy heads (mu and log_std)...", level=1)
        for param in self.policy.actor.mu.parameters():
            param.requires_grad = False
        for param in self.policy.actor.log_std.parameters():
            param.requires_grad = False

    def unfreeze_learnable_actor(self) -> None:
        """
        Unfreeze learnable part of the actor
        """
        if "perception" not in self.policy.actor.features_extractor.permanently_frozen_components:
            self.log("Unfreezing actor perception module...", level=1)
            self.policy.actor.features_extractor.unfreeze_perception()
        if "transformer" not in self.policy.actor.features_extractor.permanently_frozen_components:
            self.log("Unfreezing actor transformer module...", level=1)
            self.policy.actor.features_extractor.unfreeze_transformer()
        
        self.log("Unfreezing actor policy heads (mu and log_std)...", level=1)
        for param in self.policy.actor.mu.parameters():
            param.requires_grad = True
        for param in self.policy.actor.log_std.parameters():
            param.requires_grad = True

    def freeze_learnable_critic(self) -> None:
        """
        Freeze learnable part of the critic
        """
        if "perception" not in self.policy.critic.features_extractor.permanently_frozen_components:
            self.log("Freezing critic perception module...", level=1)
            self.policy.critic.features_extractor.freeze_perception()
        if "transformer" not in self.policy.critic.features_extractor.permanently_frozen_components:
            self.log("Freezing critic transformer module...", level=1)
            self.policy.critic.features_extractor.freeze_transformer()
        # always freeze Q-value head
        self.log("Freezing critic Q-value head...", level=1)
        for param in self.policy.critic.q_networks.parameters():
            param.requires_grad = False

    def unfreeze_learnable_critic(self) -> None:
        """
        Unfreeze learnable part of the critic
        """
        if "perception" not in self.policy.critic.features_extractor.permanently_frozen_components:
            self.log("Unfreezing critic perception module...", level=1)
            self.policy.critic.features_extractor.unfreeze_perception()
        if "transformer" not in self.policy.critic.features_extractor.permanently_frozen_components:
            self.log("Unfreezing critic transformer module...", level=1)
            self.policy.critic.features_extractor.unfreeze_transformer()
        self.log("Unfreezing critic Q-value head...", level=1)
        for param in self.policy.critic.q_networks.parameters():
            param.requires_grad = True


    def warmup_critic(
        self,
        critic_warmup_steps: int,
        callback: BaseCallback
    ) -> None:
        """
        Warm up the critic by training for a specified number of steps with the actor frozen.
        """
        if critic_warmup_steps <= 0:
            self.log("✓ No critic warmup steps specified, skipping warmup.")
            return
        self.log(f"\nWarming up critic for {critic_warmup_steps} steps with frozen actor")

        # Freeze actor, set eval mode, and prevent SB3 from switching back to train mode during learn()
        self.freeze_learnable_actor()
        self.policy.actor.set_training_mode(False) # disable Dropout/BatchNorm
        original_actor_set_mode = self.policy.actor.set_training_mode
        self.policy.actor.set_training_mode = lambda mode: None
        
        # Monitor and debug before warmup
        self.monitor_agent("BEFORE WARMUP")

        # --- Critic warmup = learn with frozen actor ---
        self.learn(
            total_timesteps=critic_warmup_steps,
            progress_bar=True,
            callback=callback,
            log_interval=1,
            tb_log_name="critic_warmup"  # separate TB logs for warmup phase
            )

        # Restore actor set_training_mode and Unfreeze actor components
        self.unfreeze_learnable_actor()
        self.policy.actor.set_training_mode = original_actor_set_mode
        self.policy.actor.set_training_mode(True) # enable dropout/batchnorm
        
        self.log("✓ Critic warmup complete. Actor unfrozen")
        # Monitor and debug after warmup
        self.monitor_agent("AFTER WARMUP")


    def prefill_replay_buffer(
        self,
        cfg: dict,
        callback: Optional[Callable[[dict[str, Any], dict[str, Any]], Any]] = None,
    ) -> None:
        """Pre-fill replay buffer with policy rollouts before learn().
        
        Reads configuration parameters and calls internal _prefill_replay_buffer.
        Initializes and closes callback progress bar if present.
        
        Args:
            cfg: Configuration dictionary containing:
                - transitions_before_learn: number of transitions to prefill the buffer with (default: buffer_size)
                - prefill_deterministic: whether to use deterministic policy/actor (default: True)
                - prefill_render: whether callback rendering during prefill is enabled (default: False)
            callback: Optional callback invoked on each prefill env step.
                Receives locals()/globals() like SB3 callbacks.
        """
        prefill_transitions = int(
            cfg.get("transitions_before_learn", self.buffer_size)
        )
        prefill_deterministic = bool(cfg.get("prefill_deterministic", True))
        prefill_render = bool(cfg.get("prefill_render", False))
        
        if prefill_transitions <= 0:
            self.log("✓ No replay buffer prefill requested, skipping.")
            return
        if prefill_render and callback is None:
            raise ValueError("cfg['prefill_render']=True requires a prefill callback.")
            
        self.log(
            f"\nPrefilling replay buffer with {prefill_transitions} transitions "
            f"(using deterministic actor = {prefill_deterministic})..."
        )
        
        transitions_added, episodes_added, mean_r, mean_len, num_collisions = self._prefill_replay_buffer(
            target_transitions=prefill_transitions,
            force_deterministic=prefill_deterministic,
            callback=callback,
        )
        
        # Close callback progress bar (releases resources: file handles, threads)
        if callback is not None and hasattr(callback, 'close_progress_bar'):
            callback.close_progress_bar()
        
        self.log(
            "✓ Replay prefill complete: "
            f"\n\t{transitions_added} transitions added, "
            f"\n\t{episodes_added} episodes,"
            f"\n\tmean_ep_reward={mean_r:.2f},\n\tmean_ep_len={mean_len:.2f},\n\tcollisions={num_collisions}"
        )
        self.log(f"Current replay buffer status: {self.replay_buffer.size()}/{self.replay_buffer.buffer_size} transitions stored.", level=1)


    def _prefill_replay_buffer(
        self,
        target_transitions: int,
        force_deterministic: bool = True,
        callback: Optional[Callable[[dict[str, Any], dict[str, Any]], Any]] = None,
    ) -> Tuple[int, int, float, float, int]:
        """Internal method to pre-fill replay buffer with policy rollouts.

        Uses an SB3 evaluate_policy-style loop over env and stores each
        transition with replay_buffer.add(...).
        
        Args:
            target_transitions: Number of transitions to collect
            force_deterministic: Whether to use deterministic policy for collection
            callback: Optional callback invoked once per env step.
                Receives locals()/globals() like SB3 callbacks (e.g., on_step).
            
        Returns:
            Tuple of (transitions_added, episodes_added, mean_reward, mean_length, num_collisions)
        """
        if target_transitions <= 0:
            return 0, 0, 0.0, 0.0

        env = self.get_env()
        if env is None:
            raise ValueError("Model has no environment attached for replay prefill.")

        n_envs = env.num_envs
        current_rewards = np.zeros(n_envs, dtype=np.float32)
        current_lengths = np.zeros(n_envs, dtype=np.int32)

        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        num_collisions = 0
        transitions_added = 0
        progress_delta = 0
        pbar_total = target_transitions
        pbar_unit = "tr"

        obs = env.reset()
        
        # Log initial log_std status
        self.log_std_status("Prefill Start")
        
        while transitions_added < target_transitions:
            # Predict action with torch.no_grad() for efficiency        
            with th.no_grad():
                actions, _ = self.predict(obs, deterministic=force_deterministic)
            
            # Log detailed action sampling debug info for first few steps
            if transitions_added < 3:
                self.log_action_sampling(obs, actions, step=transitions_added, deterministic=force_deterministic)
            
            next_obs, rewards, dones, infos = env.step(actions)
            
            # Store one transition per env until the transition target is reached.
            progress_delta = 0
            for i in range(n_envs):
                if transitions_added >= target_transitions:
                    break

                obs_i = {k: np.array(v[i : i + 1]) for k, v in obs.items()}
                next_obs_i = {k: np.array(v[i : i + 1]) for k, v in next_obs.items()}
                action_i = np.array(actions[i : i + 1], dtype=np.float32)
                reward_i = np.array([rewards[i]], dtype=np.float32)
                done_i = np.array([dones[i]], dtype=np.float32)

                self.replay_buffer.add(obs_i, next_obs_i, action_i, reward_i, done_i, [infos[i]])
                transitions_added += 1
                progress_delta += 1
                num_collisions += int(infos[i].get("collision", False))
                current_rewards[i] += rewards[i]
                current_lengths[i] += 1

                if dones[i]:
                    episode_rewards.append(float(current_rewards[i]))
                    episode_lengths.append(int(current_lengths[i]))
                    current_rewards[i] = 0.0
                    current_lengths[i] = 0

            if callback is not None:
                callback(locals(), {})

            obs = next_obs
            
        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        mean_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
        return transitions_added, len(episode_lengths), mean_reward, mean_length, num_collisions
