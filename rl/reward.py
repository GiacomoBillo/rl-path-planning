from __future__ import annotations

from typing import Any

import numpy as np


class RewardCalculator:
    DEFAULT_REWARD_CONFIG: dict[str, float] = {
        "total_magnitude": 1.0,
        "goal_weight": 1.0,
        "collision_weight": -1.0,
        "goal_distance_weight": 0.0,
        "goal_distance_progress_weight": 0.0,
        "orientation_position_weight": 0.0, # weight for orientation error wrt position error
        "step_penalty_weight": 0.0,
        "action_l2_weight": 0.0,
        "sdf_obstacle_weight": 0.0,
        "sdf_obstacle_threshold": 0.0,
        "sdf_obstacle_max_penalty": 0.0,
        "sdf_self_collision_weight": 0.0,
        "sdf_self_collision_threshold": 0.0,
        "sdf_self_collision_max_penalty": 0.0,
        "joint_limit_weight": 0.0,
        "delta_joint_limit_weight": 0.0,
    }

    def __init__(
        self,
        reward_config: dict[str, Any] | None,
        position_threshold: float,
        orientation_threshold: float,
    ) -> None:
        if position_threshold <= 0:
            raise ValueError("position_threshold must be > 0.")
        if orientation_threshold <= 0:
            raise ValueError("orientation_threshold must be > 0.")

        self.position_threshold = position_threshold
        self.orientation_threshold = orientation_threshold
        self.reward_config = self.build_reward_config(reward_config)
        self.reset()

    @classmethod
    def build_reward_config(cls, reward_config: dict[str, Any] | None) -> dict[str, float]:
        reward_config = reward_config or {}
        if not isinstance(reward_config, dict):
            raise ValueError("reward_config must be a dict when provided.")

        config = {
            key: float(reward_config.get(key, default_value))
            for key, default_value in cls.DEFAULT_REWARD_CONFIG.items()
        }

        if config["total_magnitude"] <= 0:
            raise ValueError("reward.total_magnitude must be > 0.")
        if config["goal_weight"] < 0:
            raise ValueError("reward.goal_weight must be >= 0.")
        if config["collision_weight"] > 0:
            raise ValueError("reward.collision_weight must be <= 0.")
        if config["goal_distance_weight"] < 0:
            raise ValueError("reward.goal_distance_weight must be >= 0.")
        if config["goal_distance_progress_weight"] < 0:
            raise ValueError("reward.goal_distance_progress_weight must be >= 0.")
        if config["step_penalty_weight"] > 0:
            raise ValueError("reward.step_penalty_weight must be <= 0.")
        if config["action_l2_weight"] > 0:
            raise ValueError("reward.action_l2_weight must be <= 0.")
        if config["sdf_obstacle_weight"] > 0:
            raise ValueError("reward.sdf_obstacle_weight must be <= 0.")
        if config["sdf_self_collision_weight"] > 0:
            raise ValueError("reward.sdf_self_collision_weight must be <= 0.")
        if config["joint_limit_weight"] > 0:
            raise ValueError("reward.joint_limit_weight must be <= 0.")
        if config["sdf_obstacle_threshold"] < 0:
            raise ValueError("reward.sdf_obstacle_threshold must be >= 0.")
        if config["sdf_self_collision_threshold"] < 0:
            raise ValueError("reward.sdf_self_collision_threshold must be >= 0.")
        if config["sdf_obstacle_max_penalty"] < 0:
            raise ValueError("reward.sdf_obstacle_max_penalty must be >= 0.")
        if config["sdf_self_collision_max_penalty"] < 0:
            raise ValueError("reward.sdf_self_collision_max_penalty must be >= 0.")
        return config

    def reset(self) -> None:
        self.prev_position_error: float | None = None
        self.prev_orientation_error: float | None = None

    def compute_reward(
        self,
        collision: bool,
        target_reached: bool,
        position_error: float,
        orientation_error: float,
        sdf_obstacles: float,
        sdf_self_collision: float,
        joint_limit_violation: float,
        action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        # normalize errors wrt thresolds
        position_error = position_error / self.position_threshold
        orientation_error = orientation_error / self.orientation_threshold

        reward_terms = {
            "goal": self.rwd_goal(collision, target_reached),
            "collision": self.rwd_collision(collision),
            "goal_distance": self.rwd_goal_distance(position_error, orientation_error),
            "goal_distance_progress": self.rwd_goal_distance_progress(position_error, orientation_error),
            "step_penalty": self.rwd_step_penalty(),
            "action_l2": self.rwd_action_l2(action),
            "sdf_obstacles": self.rwd_sdf_obstacles(sdf_obstacles),
            "sdf_self_collision": self.rwd_sdf_self_collision(sdf_self_collision),
            "joint_limit": self.rwd_joint_limit(joint_limit_violation),
        }

        self.prev_position_error = position_error
        self.prev_orientation_error = orientation_error
        reward = self.reward_config["total_magnitude"] * sum(reward_terms.values())
        return reward, reward_terms


    def _margin_deficit(self, sdf_margin: float, threshold: float, max_penalty: float) -> float:
        deficit = max(0.0, threshold - sdf_margin)
        return min(deficit, max_penalty)

    def rwd_goal(self, collision: bool, target_reached: bool) -> float:
        if not collision and target_reached:
            return self.reward_config["goal_weight"]
        return 0.0

    def rwd_collision(self, collision: bool) -> float:
        if collision:
            return self.reward_config["collision_weight"]
        return 0.0

    def rwd_goal_distance(self, position_error: float, orientation_error: float) -> float:
        reward = self.reward_config["goal_distance_weight"] * (
                    np.exp(-(position_error ** 2))
                    + self.reward_config["orientation_position_weight"] * np.exp(-(orientation_error ** 2))
                ) 
        return reward

    def rwd_goal_distance_progress(self, position_error: float, orientation_error: float) -> float:
        if self.prev_position_error is None or self.prev_orientation_error is None:
            return 0.0
        delta_position = self.prev_position_error - position_error
        delta_orientation = self.prev_orientation_error - orientation_error
        return self.reward_config["goal_distance_progress_weight"] * (delta_position + self.reward_config["orientation_position_weight"] * delta_orientation)

    def rwd_step_penalty(self) -> float:
        return self.reward_config["step_penalty_weight"]

    def rwd_action_l2(self, action: np.ndarray) -> float:
        return self.reward_config["action_l2_weight"] * np.mean(np.square(action))

    def rwd_sdf_obstacles(self, sdf_margin: float) -> float:
        deficit = self._margin_deficit(
            sdf_margin=sdf_margin,
            threshold=self.reward_config["sdf_obstacle_threshold"],
            max_penalty=self.reward_config["sdf_obstacle_max_penalty"],
        )
        return self.reward_config["sdf_obstacle_weight"] * deficit

    def rwd_sdf_self_collision(self, sdf_margin: float) -> float:
        deficit = self._margin_deficit(
            sdf_margin=sdf_margin,
            threshold=self.reward_config["sdf_self_collision_threshold"],
            max_penalty=self.reward_config["sdf_self_collision_max_penalty"],
        )
        return self.reward_config["sdf_self_collision_weight"] * deficit

    def rwd_joint_limit(self, joint_limit_violation: float) -> float:
        return self.reward_config["joint_limit_weight"] * np.mean(np.square(joint_limit_violation))


