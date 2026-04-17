from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import tqdm
import numpy as np
from typing import Any, Optional



class DebugCallback(BaseCallback):
    """Unified callback for training, evaluation, and prefill.

    Handles:
    - Optional progress bar (created internally if enabled)
    - Step logging (reward, done, info) with formatting
    - Rendering (safely restricted to single-env to avoid clutter)

    'locals' refers to the local variables of the scope where the callback is called.

    About 'infos':
    - In Stable Baselines3, environments are typically wrapped in a VecEnv.
    - 'infos' is a list of dictionaries, one per environment.
    - We log/render only the first environment (index 0) to avoid clutter.
    """

    def __init__(
        self,
        description: str = "train",
        log_steps: bool = False,
        render: bool = False,
        progress_bar: bool = False,
        total: int = None, # total number of steps or episodes for progress bar (if None, will try to infer from locals)
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.description = description
        self.log_steps = log_steps
        self.render_enabled = render
        self.progress_bar_enabled = progress_bar
        self.pbar = None
        self.last_done_count = 0
        self.total = total

    def _ensure_progress_bar(self, locals_: dict) -> None:
        """Lazily initialize progress bar on first call if enabled and metadata available."""
        if not self.progress_bar_enabled or self.pbar is not None:
            return
        
        # Extract total and unit from locals
        pbar_total = self.total or locals_.get("pbar_total", None)
        pbar_unit = locals_.get("pbar_unit", "step")
        
        self.pbar = tqdm(total=pbar_total, desc=self.description.capitalize(), unit=pbar_unit)
        self.last_done_count = 0

    def close_progress_bar(self) -> None:
        """Close and cleanup the progress bar."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
            self.last_done_count = 0

    def _on_rollout_start(self):
        """Called by model.learn() at the beginning of every rollout.
        TODO: missing equivalent for evaluate_policy()
        """
        if self.log_steps:
            print(f"[{self.description}-rollout-start] ")

        env = self.training_env
        if self.render_enabled and env is not None:
            # Check if it's a Stable Baselines3 VecEnv
            if hasattr(env, "env_method"):
                # Safely call render ONLY on the first environment (index 0)
                env.env_method("render", indices=0)
            else:
                # Fallback for standard, non-vectorized Gymnasium environments
                env.render()

    def _on_step(self) -> bool:
        """Called by model.learn() at every step."""
        return self._process_step(self.locals, self.training_env)

    def __call__(self, locals_: dict, globals_: dict) -> None:
        """Called by evaluate_policy() and prefill loop at every step."""
        # evaluate_policy and prefill pass 'env' in locals_
        env = locals_.get("env")
        self._process_step(locals_, env)

    def _process_step(self, locals_: dict, env: Any) -> bool:
        # Lazy-init progress bar if enabled
        self._ensure_progress_bar(locals_)
        
        # 1. Normalize variable names
        # Check keys explicitly to avoid False-y issues with boolean/numpy arrays
        infos = locals_.get("infos", locals_.get("info", []))
        dones = locals_.get("dones", locals_.get("done", []))
        rewards = locals_.get("rewards", locals_.get("reward", []))

        # 2. Select first env (index 0) in case of parallel envs.
        # Supports list/tuple containers because SubprocVecEnv may return tuples.
        info = infos[0] if (isinstance(infos, (list, tuple)) and len(infos) > 0) else infos
        done_val = dones[0] if (isinstance(dones, (list, tuple, np.ndarray)) and len(dones) > 0) else dones
        reward_val = rewards[0] if (isinstance(rewards, (list, tuple, np.ndarray)) and len(rewards) > 0) else rewards

        # 3. Log steps (only first env to avoid clutter)
        if self.log_steps:
            reward_str = f"{float(reward_val):.3f}"
            info_str = self._format_info(info)
            collision = info.get("collision", None) if isinstance(info, dict) else None
            print(f"[{self.description}-step] reward={reward_str} terminated={done_val} collision={collision} info={info_str}")

        # 4. Log episode summary when any episode ends.
        # Parallel VecEnv branch.
        if isinstance(infos, (list, tuple)) and isinstance(dones, (list, tuple, np.ndarray)):
            for i, (done_val, info) in enumerate(zip(dones, infos)):
                if done_val and isinstance(info, dict):
                    self._log_episode_summary(info, i)
        else:
            # Single-env branch.
            if done_val and isinstance(info, dict):
                self._log_episode_summary(info)

        # 5. Render
        if self.render_enabled and env is not None:
            if hasattr(env, "env_method"):
                env.env_method("render", indices=0)
            else:
                env.render()

        # 5. Optional progress bar support
        if self.pbar:
            progress_delta = int(locals_.get("progress_delta", 0))
            if progress_delta > 0:
                self.pbar.update(progress_delta)

            episode_rewards = locals_.get("episode_rewards", [])
            n_done = len(episode_rewards)
            if n_done > self.last_done_count:
                self.pbar.update(n_done - self.last_done_count)
                self.last_done_count = n_done

            if episode_rewards:
                self.pbar.set_postfix({"mean_r": f"{np.mean(episode_rewards):.2f}"})

        return True

    def _format_info(self, info: Any) -> Any:
        """Recursively round floats in the info dict."""
        if isinstance(info, float):
            return round(info, 3)
        elif isinstance(info, dict):
            return {k: self._format_info(v) for k, v in info.items()}
        elif isinstance(info, list):
            return [self._format_info(x) for x in info]
        elif isinstance(info, tuple):
            return tuple(self._format_info(x) for x in info)
        return info

    def _log_episode_summary(self, info: dict, env_idx: int = None) -> None:
        """Log episode summary when an episode completes.
        
        Args:
            info: Info dict from environment containing episode statistics
            env_idx: Optional environment index for multi-env setups
        """
        # Extract episode statistics from info dict
        episode_return = info.get("episode_return", 0.0)
        episode_steps = info.get("episode_num_steps", 0)
        episode_collisions = info.get("episode_num_collisions", 0)
        position_error = info.get("position_error", 0.0)
        orientation_error = info.get("orientation_error", 0.0)
        target_reached = info.get("target_reached", False)
        truncated = info.get("TimeLimit.truncated", False)
        
        # Determine termination reason
        if target_reached:
            termination_msg = "Target reached!"
        elif truncated:
            termination_msg = "Episode truncated (max steps)"
        else:
            termination_msg = "Episode ended"
        
        # Format prefix with optional environment index
        prefix = f"[{self.description}-episode"
        if env_idx is not None:
            prefix += f"-env{env_idx}"
        prefix += "]"
        
        # Print episode summary
        print(f"{prefix} {termination_msg} "
              f"return={episode_return:.2f} steps={episode_steps} "
              f"collisions={episode_collisions} "
              f"pos_err={position_error:.4f}m orient_err={orientation_error:.2f}deg")


class EvalMetricsCallback:
    """Callback for evaluate_policy that aggregates task metrics at episode end."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.episodes = 0
        self.target_reached = []
        self.num_collisions = []
        self.position_errors = []
        self.orientation_errors = []
        self.action_abs_means = []

    def __call__(self, locals_: dict, globals_: dict) -> None:
        # Single-env callback branch used by SB3 evaluate_policy internals:
        # it passes scalar `info` and `done` in locals() for each env index.
        info_single = locals_.get("info")
        done_single = locals_.get("done")
        if info_single is not None and done_single is not None:
            if bool(done_single) and isinstance(info_single, dict):
                self._consume_info(info_single)
            return

        # Parallel VecEnv callback branch (e.g. custom evaluation loops).
        infos = locals_.get("infos", [])
        dones = locals_.get("dones", [])
        if isinstance(infos, (list, tuple)) and isinstance(dones, (list, tuple, np.ndarray)):
            for done, info in zip(dones, infos):
                if done and isinstance(info, dict):
                    self._consume_info(info)
            return

        # Fallback single-env branch for code that provides `infos` as dict.
        if isinstance(infos, dict):
            done_flag = bool(dones) if not isinstance(dones, np.ndarray) else (bool(dones.item()) if dones.shape == () else False)
            if done_flag:
                self._consume_info(infos)

    def _consume_info(self, info: dict) -> None:
        self.episodes += 1
        self.target_reached.append(float(info.get("target_reached", False)))
        self.num_collisions.append(float(info.get("episode_num_collisions", 0)))
        self.position_errors.append(float(info.get("position_error", 0.0)))
        self.orientation_errors.append(float(info.get("orientation_error", 0.0)))
        action_abs_mean = np.asarray(info.get("episode_action_abs_mean", []), dtype=np.float32)
        if action_abs_mean.size > 0:
            self.action_abs_means.append(action_abs_mean)

    def summary(self) -> dict:
        if self.episodes == 0:
            return {
                "episodes": 0,
                "target_reached_rate": 0.0,
                "mean_num_collisions": 0.0,
                "mean_position_error": 0.0,
                "mean_orientation_error": 0.0,
                "ep_action_abs_mean": 0.0,
            }

        summary = {
            "episodes": self.episodes,
            "target_reached_rate": float(np.mean(self.target_reached)),
            "mean_num_collisions": float(np.mean(self.num_collisions)),
            "mean_position_error": float(np.mean(self.position_errors)),
            "mean_orientation_error": float(np.mean(self.orientation_errors)),
        }
        if self.action_abs_means:
            mean_per_joint = np.stack(self.action_abs_means).mean(axis=0)
            summary["ep_action_abs_mean"] = float(np.mean(mean_per_joint))
            for joint_idx, joint_value in enumerate(mean_per_joint):
                summary[f"ep_action_abs_j{joint_idx}"] = float(joint_value)

        return summary


def evaluate_policy_with_metrics(
    model: Any,
    eval_env: Any,
    n_eval_episodes: int,
    deterministic: bool = True,
    debug_callback: Optional[Any] = None,
) -> dict:
    """Run evaluate_policy once and return reward/length plus aggregated task metrics."""
    metrics_callback = EvalMetricsCallback()
    metrics_callback.reset()

    if debug_callback is None:
        callback = metrics_callback
    else:
        def callback(locals_: dict, globals_: dict) -> None:
            debug_callback(locals_, globals_)
            metrics_callback(locals_, globals_)

    episode_rewards, episode_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        callback=callback,
        return_episode_rewards=True,
    )

    summary = metrics_callback.summary()
    summary.update({
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "mean_ep_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "std_ep_length": float(np.std(episode_lengths)) if episode_lengths else 0.0,
    })
    return summary


class PeriodicEvalMetricsCallback(BaseCallback):
    """Run one evaluation pass and log aggregated task metrics."""

    def __init__(
        self,
        eval_env: Any,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic: bool = True,
        verbose: int = 0,
        logger_prefix: str = "eval",
        print_prefix: str = "Periodic evaluation",
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.logger_prefix = logger_prefix
        self.print_prefix = print_prefix

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        metrics = evaluate_policy_with_metrics(
            model=self.model,
            eval_env=self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
        )

        print(f"\n=== {self.print_prefix} @ step {self.num_timesteps} ===")
        print(
            f"mean_reward={metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}, "
            f"mean_len={metrics['mean_ep_length']:.2f} +/- {metrics['std_ep_length']:.2f}"
        )
        print(
            f"target_reached_rate={metrics['target_reached_rate']:.3f}, "
            f"mean_collisions={metrics['mean_num_collisions']:.3f}, "
            f"mean_pos_err={metrics['mean_position_error']:.4f}m, "
            f"mean_orient_err={metrics['mean_orientation_error']:.2f}deg, "
            f"mean_action_abs={metrics['ep_action_abs_mean']:.4f}"
        )

        for key, value in metrics.items():
            self.logger.record(f"{self.logger_prefix}/{key}", value)
        self.logger.dump(step=self.num_timesteps)
        return True
