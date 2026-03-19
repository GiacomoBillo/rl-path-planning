import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import SAC
from typing import Optional, Tuple
from stable_baselines3.common.noise import ActionNoise


np.set_printoptions(precision=3, suppress=False)


class SACDebug(SAC):
    """
    Custom SAC class for debugging, with options for deterministic rollout and verbose logging
    """
    def __init__(self, *args, deterministic_rollout=True, verbose=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.deterministic_rollout = deterministic_rollout
        self.debug_verbose = verbose
        print(f"Using SACDebug, deterministic_rollout={self.deterministic_rollout}")

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
            
            if self.debug_verbose:
                # DEBUG: Check weights
                s = 0
                with th.no_grad():
                    for p in self.policy.actor.parameters():
                         s += p.data.sum().item()
                print(f"[SACDebug._sample_action] Actor weight sum: {s:.4f}")

                # DEBUG: Check action distribution parameters
                obs_tensor, _ = self.policy.obs_to_tensor(self._last_obs)
                with th.no_grad():
                    mean_actions, log_std, kwargs = self.actor.get_action_dist_params(obs_tensor)
                    # Note: the action is squashed
                    sampled_action = self.actor.action_dist.actions_from_params(mean_actions, log_std, deterministic=self.deterministic_rollout, **kwargs)
                    
                    print("[debug]\n"
                          f"mean_actions=\t{mean_actions.detach().cpu().numpy().squeeze()},\n"
                          f"log_std=\t{log_std.detach().cpu().numpy().squeeze()},\n"
                          f"squashed_sampled_action=\t{sampled_action.detach().cpu().numpy().squeeze()}\n"
                          f"difference sample-mean=\t{th.abs(sampled_action - mean_actions).detach().cpu().numpy().squeeze()}")

            unscaled_action, _ = self.predict(self._last_obs, deterministic=self.deterministic_rollout)
            
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
