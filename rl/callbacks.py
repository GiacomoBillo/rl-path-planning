from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import numpy as np
from typing import Any



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

        # 2. Select first env (index 0) in case of parallel envs
        info = infos[0] if (isinstance(infos, list) and len(infos) > 0) else infos
        done_val = dones[0] if (isinstance(dones, (list, np.ndarray)) and len(dones) > 0) else dones
        reward_val = rewards[0] if (isinstance(rewards, (list, np.ndarray)) and len(rewards) > 0) else rewards

        # 3. Log steps
        if self.log_steps:
            reward_str = f"{float(reward_val):.3f}"
            info_str = self._format_info(info)
            collision = info.get("collision", None) if isinstance(info, dict) else None
            print(f"[{self.description}-step] reward={reward_str} terminated={done_val} collision={collision} info={info_str}")

        # 4. Render
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
        return info