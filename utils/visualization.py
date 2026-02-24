import viz_client
import torch
import numpy as np
from robofin.robots import Robot
import time

def convert_to_numpy_f32(arr: np.ndarray | torch.Tensor) -> np.ndarray:
    """
    Convert a NumPy array or Torch tensor to a NumPy float32 array.
    
    Parameters
    ----------
    arr : np.ndarray or torch.Tensor
        Input array to convert.
    
    Returns
    -------
    np.ndarray
        Converted array with dtype float32.
    """
    if isinstance(arr, torch.Tensor):
        np_arr: np.ndarray = arr.cpu().numpy()
    elif isinstance(arr, np.ndarray):
        np_arr = arr
    else:
        raise TypeError("convert_to_numpy_f32: Input must be a NumPy array or Torch tensor")
    return np_arr.astype(np.float32)

def visualize_problem(
    robot: Robot,
    sample: dict[str, torch.Tensor],
    target_alpha: float = 0.7,
    target_color: list[float] = [0.8, 0.2, 0.8],
    obstacle_color: list[float] = [0.8, 0.5, 0.6],
):
    """
    Visualize a problem setup - target, obstacles and start configuration
    """
    assert viz_client is not None, "viz_client is not imported"
    if not viz_client.is_connected():
        viz_client.connect(str(robot.urdf_path))

    target_pose = np.eye(4)
    target_pose[:3, 3] = convert_to_numpy_f32(sample["target_position"])
    target_pose[:3, :3] = convert_to_numpy_f32(sample["target_orientation"])

    viz_client.publish_ghost_end_effector(
        pose=target_pose,
        frame=robot.tcp_link_name,
        color=target_color,
        alpha=target_alpha
    )

    cuboid_dims = convert_to_numpy_f32(sample["cuboid_dims"])
    cuboid_centers = convert_to_numpy_f32(sample["cuboid_centers"])
    cuboid_quaternions = convert_to_numpy_f32(sample["cuboid_quats"])
    cylinder_radii = convert_to_numpy_f32(sample["cylinder_radii"])
    cylinder_heights = convert_to_numpy_f32(sample["cylinder_heights"])
    cylinder_centers = convert_to_numpy_f32(sample["cylinder_centers"])
    cylinder_quaternions = convert_to_numpy_f32(sample["cylinder_quats"])

    viz_client.publish_config(robot.unnormalize_joints(sample["configuration"]))
    viz_client.publish_obstacles(cuboid_centers=cuboid_centers,
                                cuboid_dims=cuboid_dims,
                                cuboid_quaternions=cuboid_quaternions,
                                cylinder_centers=cylinder_centers,
                                cylinder_radii=cylinder_radii,
                                cylinder_heights=cylinder_heights,
                                cylinder_quaternions=cylinder_quaternions,
                                color=obstacle_color)


# visualize trajectory
def visualize_trajectory(
    robot: Robot,
    sample: dict[str, torch.Tensor],
    trajectory: torch.Tensor,
    target_alpha: float = 0.7,
    target_color: list[float] = [0.8, 0.2, 0.8],
    obstacle_color: list[float] = [0.8, 0.5, 0.6],
    normalized: bool = True,
    mode: str = "interpolation",  # "interpolation" or "discrete"
    segment_duration: float = 0.01,
    rate_hz: float = 30.0,
):
    """
    Visualize a trajectory
    
    This function sets up the scene (target, obstacles) and then animates the robot
    
    Parameters
    ----------
    robot : Robot
        Robot instance
    sample : dict[str, torch.Tensor]
        Sample containing scene information (target, obstacles)
    trajectory : torch.Tensor
        Normalized joint configurations [T, MAIN_DOF] where T is trajectory length.
        This can be the expert trajectory from sample["expert"] or a custom trajectory.
    target_alpha : float
        Transparency of target end-effector
    target_color : list[float]
        RGB color for target end-effector
    obstacle_color : list[float]
        RGB color for obstacles
    normalized : bool
        Whether the input trajectory is normalized. If True, it will be unnormalized before visualization.
    mode : str
        Visualization mode: "interpolation" for smooth linear interpolation of the waypoints, "discrete" for discrete steps
    segment_duration : float
        Seconds spent interpolating between each pair of waypoints, only used in "interpolation" mode
    rate_hz : float
        Animation framerate in Hz (higher = smoother for "interpolation", faster playback for "discrete")
    """
    assert viz_client is not None, "viz_client is not imported"
    if not viz_client.is_connected():
        viz_client.connect(str(robot.urdf_path))
    
    # Set up the scene (target and obstacles)
    target_pose = np.eye(4)
    target_pose[:3, 3] = convert_to_numpy_f32(sample["target_position"])
    target_pose[:3, :3] = convert_to_numpy_f32(sample["target_orientation"])
    
    viz_client.publish_ghost_end_effector(
        pose=target_pose,
        frame=robot.tcp_link_name,
        color=target_color,
        alpha=target_alpha
    )
    
    cuboid_dims = convert_to_numpy_f32(sample["cuboid_dims"])
    cuboid_centers = convert_to_numpy_f32(sample["cuboid_centers"])
    cuboid_quaternions = convert_to_numpy_f32(sample["cuboid_quats"])
    cylinder_radii = convert_to_numpy_f32(sample["cylinder_radii"])
    cylinder_heights = convert_to_numpy_f32(sample["cylinder_heights"])
    cylinder_centers = convert_to_numpy_f32(sample["cylinder_centers"])
    cylinder_quaternions = convert_to_numpy_f32(sample["cylinder_quats"])
    
    viz_client.publish_obstacles(
        cuboid_centers=cuboid_centers,
        cuboid_dims=cuboid_dims,
        cuboid_quaternions=cuboid_quaternions,
        cylinder_centers=cylinder_centers,
        cylinder_radii=cylinder_radii,
        cylinder_heights=cylinder_heights,
        cylinder_quaternions=cylinder_quaternions,
        color=obstacle_color
    )

    if mode == "interpolation":
        waypoints = []
        for i in range(trajectory.shape[0]):
            if normalized:
                config = robot.unnormalize_joints(trajectory[i])
            else:
                config = trajectory[i]
            waypoint = {name: float(val) for name, val in zip(robot.main_joint_names, config)}
            waypoint.update(robot.auxiliary_joint_defaults)
            # print(f"Waypoint {i}: {waypoint}")
            waypoints.append(waypoint)
        
        # Animate the trajectory
        viz_client.publish_trajectory(
            waypoints,
            segment_duration=segment_duration,
            rate_hz=rate_hz
        )

    elif mode == "discrete":
        for i in range(trajectory.shape[0]):
            if normalized:
                config = robot.unnormalize_joints(trajectory[i])
            else:
                config = trajectory[i]
            viz_client.publish_config(config)
            time.sleep(1/rate_hz)

    
def remove_trajectory_padding(trajectory: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
    """
    Remove padding from trajectory (repeated final configurations).
    
    Parameters
    ----------
    trajectory : torch.Tensor
        Trajectory tensor [T, DOF]
    atol : float
        Absolute tolerance for detecting identical configurations
        
    Returns
    -------
    torch.Tensor
        Trajectory with padding removed
    """
    if trajectory.shape[0] <= 1:
        return trajectory
    
    # Find first index where config matches the previous one
    for i in range(1, trajectory.shape[0]):
        if torch.allclose(trajectory[i], trajectory[i-1], atol=atol):
            # Check if rest of trajectory is also the same (padding)
            if all(torch.allclose(trajectory[i], trajectory[j], atol=atol) 
                   for j in range(i, trajectory.shape[0])):
                # Found padding, return only up to i
                return trajectory[:i]
    
    # No padding found
    return trajectory

def visualize_expert_trajectory(robot: Robot, sample: dict[str, torch.Tensor], verbose=True, **kwargs):
    """
    Visualize expert trajectory from a sample
    remove padding from trajectory (repeated final configurations) before visualizing
    """
    trajectory = sample["expert"]  # [T, MAIN_DOF]
    trajectory = remove_trajectory_padding(trajectory)
    if verbose:
        print(f"Original trajectory length: {sample['expert'].shape[0]} states")
        print(f"Trimmed trajectory length: {trajectory.shape[0]} states")
    visualize_trajectory(robot=robot, sample=sample, trajectory=trajectory, normalized=False, **kwargs)


# visualization for RL
def reward_color(
    reward: torch.Tensor | float,
    goal_reward: float,
    collision_reward: float,
):
    """
    Color-code a reward
    """
    if abs(reward - goal_reward) < 1e-6:
        return [0.1, 0.9, 0.1] # green
    elif abs(reward - collision_reward) < 1e-6:
        return [0.9, 0.1, 0.1] # red
    else:
        return [0.9, 0.9, 0.1] # yellow

def value_color(
    value: torch.Tensor | float, 
    min_cumulative_reward: float, 
    max_cumulative_reward: float) -> list[float]:
    """
    Color-code a Q-value or cumulative reward
    """
    if isinstance(value, torch.Tensor):
        value = value.item()
    denom = max_cumulative_reward - min_cumulative_reward
    assert denom != 0, "visualize_sample_with_value(): Lerp denominator is 0"
    lerp = (value - min_cumulative_reward) / denom

    if lerp < 0:
        x = 1 - np.exp(lerp)        # saturates toward 1 below bound
        return [1., 0., x]          # more magenta as lerp decreases below bounds
    elif lerp > 1:
        x = 1 - np.exp(-(lerp - 1)) # saturates toward 1 above bound
        return [0., 1., x]          # more cyan as lerp increases over bounds
    else:
        return [1 - lerp, lerp, 0.] # red→yellow→green when value is in range

def visualize_sample(
    robot: Robot,
    sample: dict[str, torch.Tensor],
    goal_reward: float, 
    collision_reward: float,
    robot_alpha: float = 0.5,
):
    """
    Visualize a sample transition - next state color depends on the reward
    """
    visualize_problem(robot, sample)

    color = reward_color(
        sample["reward"], goal_reward, collision_reward)

    viz_client.publish_ghost_robot(
        robot.unnormalize_joints(sample["next_configuration"]),
        color=color,
        alpha=robot_alpha)
    print(sample["reward"].item())

def visualize_sample_with_value(
    robot: Robot,
    sample: dict[str, torch.Tensor],
    value: torch.Tensor | float,
    min_cumulative_reward: float,
    max_cumulative_reward: float,
    robot_alpha: float = 0.5,
):
    """
    Visualize a sample transition - next state color depends on the value 
    (estimated Q-value or calculated cumulative reward)
    """
    visualize_problem(robot,sample)

    color = value_color(
        value, min_cumulative_reward, max_cumulative_reward)

    viz_client.publish_ghost_robot(
        robot.unnormalize_joints(sample["next_configuration"]),
        color=color,
        alpha=robot_alpha)

def visualize_rollout_rewards(
    robot: Robot,
    sample: dict[str, torch.Tensor],
    rollout: torch.Tensor,
    rewards: torch.Tensor,
    goal_reward: float,
    collision_reward: float,
    robot_alpha: float = 0.5
):
    """
    Parameters
    ----------
    sample : dict[str, torch.Tensor]
        Sample from a batch - containing start config and scene information
    rollout : torch.Tensor
        Normalized configurations of main joints [N, MAIN_DOF] (values from ["next_configuration"])
    rewards : torch.Tensor
        Rewards for each step in the rollout [N] (values from ["reward"])
    """

    visualize_problem(robot, sample)
    for i in range(rollout.shape[0]):
        color = reward_color(
            rewards[i], goal_reward, collision_reward)
        viz_client.publish_ghost_robot(
            robot.unnormalize_joints(rollout[i]),
            color=color,
            alpha=robot_alpha,
            index=i
        )

def visualize_rollout_values(
    robot: Robot,
    rollout: torch.Tensor,
    values: torch.Tensor,
    max_cumulative_reward: float,
    min_cumulative_reward: float,
    robot_alpha: float = 0.5
):
    """
    Parameters
    ----------
    sample : dict[str, torch.Tensor]
        Sample from a batch - containing start config and scene information
    rollout : torch.Tensor
        Normalized configurations of main joints [N, MAIN_DOF] (values from ["next_configuration"])
    rewards : torch.Tensor
        Rewards for each step in the rollout [N] (values from ["reward"])
    """

    for i in range(rollout.shape[0]):
        color = value_color(
            values[i], min_cumulative_reward, max_cumulative_reward)

        viz_client.publish_ghost_robot(
            robot.unnormalize_joints(rollout[i]),
            color=color,
            alpha=robot_alpha,
            index=i
        )