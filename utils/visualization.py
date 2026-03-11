import viz_client
import torch
import numpy as np
from robofin.robots import Robot
from robofin.collision import CollisionSpheres
import time

import ipywidgets as widgets
from IPython.display import display, clear_output
from avoid_everything.data_loader import TrajectoryDataset
from geometrout.primitive import Cuboid, Cylinder


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

def _obstacle_primitives_from_sample(sample: dict[str, torch.Tensor]) -> list:
    """Build a list of geometrout Cuboid/Cylinder obstacles from a sample dict."""
    primitives = []

    cuboid_centers = convert_to_numpy_f32(sample["cuboid_centers"])
    cuboid_dims    = convert_to_numpy_f32(sample["cuboid_dims"])
    cuboid_quats   = convert_to_numpy_f32(sample["cuboid_quats"])
    for c, d, q in zip(cuboid_centers, cuboid_dims, cuboid_quats):
        cub = Cuboid(c, d, q)
        if not cub.is_zero_volume():
            primitives.append(cub)

    cylinder_centers = convert_to_numpy_f32(sample["cylinder_centers"])
    cylinder_radii   = convert_to_numpy_f32(sample["cylinder_radii"])
    cylinder_heights = convert_to_numpy_f32(sample["cylinder_heights"])
    cylinder_quats   = convert_to_numpy_f32(sample["cylinder_quats"])
    for c, r, h, q in zip(cylinder_centers, cylinder_radii, cylinder_heights, cylinder_quats):
        cyl = Cylinder(c, float(r), float(h), q)
        if not cyl.is_zero_volume():
            primitives.append(cyl)

    return primitives


def check_trajectory_collisions(
    robot: Robot,
    sample: dict[str, torch.Tensor],
    trajectory: torch.Tensor,
    normalized: bool = False,
) -> list[bool]:
    """
    Check each configuration in a trajectory for collision using robofin's CollisionSpheres.

    Parameters
    ----------
    robot : Robot
        Robot instance.
    sample : dict[str, torch.Tensor]
        Sample containing obstacle geometry (cuboid_*/cylinder_* keys).
    trajectory : torch.Tensor
        Joint configurations [T, MAIN_DOF].
    normalized : bool
        Whether the trajectory is normalized; if True, unnormalize before checking.

    Returns
    -------
    list[bool]
        Boolean list of length T; True if the configuration at that timestep is in collision.
    """
    checker = CollisionSpheres(robot)
    primitives = _obstacle_primitives_from_sample(sample)

    collisions = []
    for i in range(trajectory.shape[0]):
        config = trajectory[i]
        if normalized:
            config = robot.unnormalize_joints(config)
        config_np = convert_to_numpy_f32(config)
        collides = checker.robot_collides(
            config_np,
            auxiliary_joint_values=None,
            primitives=primitives,
        )
        collisions.append(bool(collides))
    return collisions


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
    normalized: bool = False,
    mode: str = "interpolation",  # "interpolation" or "discrete"
    segment_duration: float = 0.01,
    rate_hz: float = 30.0,
    highlight_collisions: bool = True,
    collision_color: list[float] = [0.9, 0.1, 0.1],
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
    highlight_collisions : bool
        If True, overlay a ghost robot in collision_color at timesteps where robofin detects a collision.
    collision_color : list[float]
        RGB color used for the collision-highlight ghost robot.
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

    if not isinstance(trajectory, torch.Tensor):
        trajectory = torch.tensor(trajectory, dtype=torch.float32)

    # remove padding from trajectory (repeated final configurations)
    trajectory = remove_trajectory_padding(trajectory)

    # Unnormalize the full trajectory once before playback.
    if normalized:
        configs = [robot.unnormalize_joints(trajectory[i]) for i in range(trajectory.shape[0])]
    else:
        configs = [trajectory[i] for i in range(trajectory.shape[0])]

    # Run collision check once and overlay highlights before playback.
    if highlight_collisions:
        _highlight_collisions(robot, sample, configs, collision_color)

    if mode == "interpolation":
        waypoints = []
        for config in configs:
            waypoint = {name: float(val) for name, val in zip(robot.main_joint_names, config)}
            waypoint.update(robot.auxiliary_joint_defaults)
            waypoints.append(waypoint)
        
        # Animate the trajectory
        viz_client.publish_trajectory(
            waypoints,
            segment_duration=segment_duration,
            rate_hz=rate_hz
        )

    elif mode == "discrete":
        for config in configs:
            viz_client.publish_config(config)
            time.sleep(1/rate_hz)

    
def _highlight_collisions(
    robot: Robot,
    sample: dict[str, torch.Tensor],
    configs: list,
    collision_color: list[float],
    verbose: bool = True,
) -> list[int]:
    """
    Check for collisions across a list of (unnormalized) configs and overlay
    a ghost robot in collision_color at the first colliding configuration.

    Collision checking
    - done using the robofin library generalized for custom robots
    - check self-collisions and collisions with obstacles in the environment

    Returns the list of colliding timestep indices.
    """
    viz_client.clear_ghost_robot()  # Clear previous collision highlights
    checker = CollisionSpheres(robot)
    primitives = _obstacle_primitives_from_sample(sample)
    collision_indices = [
        i for i, cfg in enumerate(configs)
        if checker.robot_collides(convert_to_numpy_f32(cfg), auxiliary_joint_values=None, primitives=primitives)
    ]
    if verbose and len(configs)>1:
        print(f"Trajectory length: {len(configs)}")
        if collision_indices:
            print(f"⚠ Collision detected at timestep(s): {collision_indices}")
        else:
            print("The trajectory is collision-free")
    if collision_indices:           
        viz_client.publish_ghost_robot(configs[collision_indices[0]], color=collision_color, alpha=0.6)
    return collision_indices


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

    def _allclose(a, b):
        if isinstance(a, np.ndarray):
            return np.allclose(a, b, atol=atol)
        return torch.allclose(a, b, atol=atol)

    # Find first index where config matches the previous one
    for i in range(1, trajectory.shape[0]):
        if _allclose(trajectory[i], trajectory[i-1]):
            # Check if rest of trajectory is also the same (padding)
            if all(_allclose(trajectory[i], trajectory[j])
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



# Sliders for visualization of samples and trajectories
def create_slider(
    dataset,
    callback,
    description='Sample Index:',
    **callback_args,
):
    """
    Create a reusable interactive slider for visualizing dataset samples.
    
    Parameters:
    -----------
    dataset : Dataset
        The dataset to iterate through
    callback : callable
        Function to call on slider change. Signature: callback(index, output, dataset=..., robot=..., animation_mode=...)
        The callback should handle visualization logic within the `with output:` context that's already set up.
    robot : Robot, optional
        Robot object to pass to callback
    animation_mode : str, optional
        Animation mode (e.g., "discrete", "interpolation") to pass to callback
    description : str, default='Sample Index:'
        Slider label text
    
    Returns:
    --------
    tuple : (slider, output)
        The slider and output widgets. Call display(slider, output) if auto_display=False.
    """
    # Create the widget objects
    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(dataset) - 1, # Dynamically set max based on your dataset size
        step=1,
        description=description,
        continuous_update=False, # Highly recommended for ROS/Robot vis
        style={'description_width': 'initial'}, # to adapt description width
    )

    output = widgets.Output()

    # Callback
    def on_value_change(change):
        index = change['new']
        sample_i = dataset[index]
        
        with output:
            clear_output(wait=True)
            # callback
            callback(
                index=index,
                sample=sample_i,
                **callback_args
            )

    # Link the slider to the function
    slider.observe(on_value_change, names='value')
    # Display
    display(slider, output)
    # Trigger the first sample immediately so the screen isn't blank
    on_value_change({'new': slider.value})
    return slider


def create_nested_slider(
    dataset,
    get_trajectory_fn,
    inner_callback,
    outer_callback=None,
    outer_description='Trajectory Index:',
    inner_description='Timestep Index:',
    outer_callback_args: dict = {},
    inner_callback_args: dict = {},
):
    """
    Create a two-level interactive slider:
    - Outer slider: selects an entry from a dataset
    - Inner slider: scrubs timesteps within the selected trajectory

    Parameters
    ----------
    dataset : Dataset
        The dataset to iterate through.
    get_trajectory_fn : callable(sample) -> torch.Tensor
        Given a dataset sample, returns the (trimmed, unnormalized) trajectory
        tensor [T, MAIN_DOF] used to set the inner slider's max.
    inner_callback : callable
        Called on every inner slider change.
        Signature: inner_callback(index, sample, trajectory,
                                  collision_indices, **inner_callback_args)
        where index is the current timestep and collision_indices is the list
        of colliding timesteps computed by outer_callback (or None).
    outer_callback : callable, optional
        Called when the outer slider changes (i.e. a new trajectory is loaded).
        Signature: outer_callback(sample, trajectory, **outer_callback_args)
        If it returns a value it is stored in state["collision_indices"] and
        forwarded to inner_callback on every subsequent call.
    outer_description : str
        Label for the outer (trajectory) slider.
    inner_description : str
        Label for the inner (timestep) slider.
    outer_callback_args : dict
        Extra keyword arguments forwarded to outer_callback on every call.
    inner_callback_args : dict
        Extra keyword arguments forwarded to inner_callback on every call.
    """
    # Mutable shared state between the two closures
    state = {"sample": None, "trajectory": None, "collision_indices": None}

    def _load(outer_index):
        state["sample"] = dataset[outer_index]
        state["trajectory"] = get_trajectory_fn(state["sample"])
        state["collision_indices"] = None  # reset until outer_callback recomputes

    _load(0)
    initial_traj_len = len(state["trajectory"])

    outer_slider = widgets.IntSlider(
        value=0, min=0, max=len(dataset) - 1, step=1,
        description=outer_description, continuous_update=False,
        style={'description_width': 'initial'}, # to adapt description width
    )
    inner_slider = widgets.IntSlider(
        value=0, min=0, max=max(initial_traj_len - 1, 0), step=1,
        description=inner_description, continuous_update=False,
        style={'description_width': 'initial'}, # to adapt description width
    )
    outer_output = widgets.Output()
    inner_output = widgets.Output()

    def _run_outer_callback():
        if outer_callback is None:
            return
        
        with outer_output:
            clear_output(wait=True)
            result = outer_callback(
                sample=state["sample"],
                trajectory=state["trajectory"],
                **outer_callback_args,
            )
            if result is not None:
                state["collision_indices"] = result

    def _run_inner_callback():
        with inner_output:
            clear_output(wait=True)
            inner_callback(
                index=inner_slider.value,
                sample=state["sample"],
                trajectory=state["trajectory"],
                collision_indices=state["collision_indices"],
                **inner_callback_args,
            )

    def on_outer_change(change):
        _load(change['new'])
        new_max = max(len(state["trajectory"]) - 1, 0)
        inner_slider.max = new_max
        _run_outer_callback()

        # reset inner slider index
        inner_slider.value = 0  
        # else reset to first collision
        # inner_slider.value = state["collision_indices"][0] if state["collision_indices"] else 0  # triggers on_inner_change

    def on_inner_change(change):
        _run_inner_callback()

    outer_slider.observe(on_outer_change, names='value')
    inner_slider.observe(on_inner_change, names='value')

    display(widgets.VBox([outer_slider, outer_output, inner_slider, inner_output]))
    _run_outer_callback()  # populate scene on first render
    # _run_inner_callback()  # populate robot state on first render


def visualize_expert_trajectory_slider(
        dataset: TrajectoryDataset,
        robot: Robot,
        animation_mode="discrete",
    ):
    """
    Create an interactive slider to visualize expert trajectories

    """
    def callback(index, sample, **kwargs): 
        robot = kwargs.get("robot")
        animation_mode = kwargs.get("animation_mode")
        # Trigger ROS/Robot visualization
        visualize_expert_trajectory(robot=robot, sample=sample, mode=animation_mode)

        config_str = ", ".join([f"{angle:.3f}" for angle in sample['configuration'].numpy()])
        print(f"Robot starting configuration: [{config_str}]")
        print(f"✓ Visualization updated!")

    slider = create_slider(
        dataset=dataset,
        callback=callback,
        description='Trajectory Index:',
        robot=robot,
        animation_mode=animation_mode
    )


def visualize_saved_trajectory_slider(
        dataset: TrajectoryDataset,
        robot: Robot,
        animation_mode="discrete",
        segment_duration=0.01,
        rate_hz=30.0,
    ):
    """
    Create an interactive slider to visualize saved trajectories
    """
    def callback(index, sample, **kwargs): 
        robot = kwargs.get("robot")
        animation_mode = kwargs.get("animation_mode")
        segment_duration = kwargs.get("segment_duration")
        rate_hz = kwargs.get("rate_hz")
        # Trigger ROS/Robot visualization
        visualize_trajectory(robot=robot, 
                             sample=sample, 
                             trajectory=sample["trajectory"], 
                             mode=animation_mode,
                             segment_duration=segment_duration,
                             rate_hz=rate_hz)
        
        config_str = ", ".join([f"{angle:.3f}" for angle in sample['configuration'].numpy()])
        print(f"Robot starting configuration: [{config_str}]")
        print(f"✓ Visualization updated!")

    slider = create_slider(
        dataset=dataset,
        callback=callback,
        description='Trajectory Index:',
        robot=robot,
        animation_mode=animation_mode,
        segment_duration=segment_duration,
        rate_hz=rate_hz
    )



# visualize
def visualize_state_from_trajectory(
    robot: Robot,
    sample: dict[str, torch.Tensor],
    index: int,
    target_alpha: float = 0.7,
    target_color: list[float] = [0.8, 0.2, 0.8],
    obstacle_color: list[float] = [0.8, 0.5, 0.6],
    expert=False,
    normalized=False,
    highlight_collisions: bool = True,
    collision_color: list[float] = [0.9, 0.1, 0.1],
    ):
    """
    Visualize step at timestep index in the trajectory - target, obstacles and configuration(index).
    If highlight_collisions is True, a ghost robot in collision_color is overlaid when
    robofin's CollisionSpheres detects that the configuration is in collision.
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

    # select config at index in expert trajectory
    if expert:
        config = sample["expert"][index]
    # select config at index in saved trajectory
    else:
        config = sample["trajectory"][index]  

    if normalized:
        config = robot.unnormalize_joints(config)
    viz_client.publish_config(config)
    viz_client.publish_obstacles(cuboid_centers=cuboid_centers,
                                cuboid_dims=cuboid_dims,
                                cuboid_quaternions=cuboid_quaternions,
                                cylinder_centers=cylinder_centers,
                                cylinder_radii=cylinder_radii,
                                cylinder_heights=cylinder_heights,
                                cylinder_quaternions=cylinder_quaternions,
                                color=obstacle_color)

    if highlight_collisions:
        _highlight_collisions(robot, sample, [config], collision_color)
    else:
        viz_client.clear_ghost_robot()


def visualize_states_in_trajectories_slider(
        sample: dict[str, torch.Tensor], # sample from TrajectoryDataset
        robot: Robot,
        expert=False,
        description='Timestep Index:',
        initial_index=0,
        highlight_collisions: bool = True,
        collision_color: list[float] = [0.9, 0.1, 0.1],
    ):
    """
    Create an interactive slider to visualize the states in a given trajectory  
    """
    def callback(index, trajectory, sample, **kwargs): 
        robot = kwargs.get("robot")
        # Trigger ROS/Robot visualization
        visualize_state_from_trajectory(
            robot=robot, sample=sample, index=index, expert=expert,
            highlight_collisions=highlight_collisions, collision_color=collision_color,
        )
        
        config_str = ", ".join([f"{angle:.3f}" for angle in trajectory[index]])
        print(f"Robot in configuration: [{config_str}]")
        print(f"✓ Visualization updated!")

    if expert:
        trajectory = sample["expert"]
    else:
        trajectory = sample["trajectory"]
    trajectory = remove_trajectory_padding(trajectory)

    # Create the widget objects
    slider = widgets.IntSlider(
        value=initial_index,
        min=0,
        max=len(trajectory) - 1, # Dynamically set max based on your trajectory length
        step=1,
        description=description,
        continuous_update=False # Highly recommended for ROS/Robot vis
    )

    output = widgets.Output()

    # Callback
    def on_value_change(change):
        index = change['new']
        
        with output:
            clear_output(wait=True)
            # callback
            callback(
                index=index,
                sample=sample,
                trajectory=trajectory,
                robot=robot,
            )

    # Link the slider to the function
    slider.observe(on_value_change, names='value')
    # Display
    display(slider, output)
    # Trigger the first sample immediately so the screen isn't blank
    on_value_change({'new': slider.value})


def visualize_states_nested_slider(
    dataset,
    robot: Robot,
    expert: bool = False,
    animate_traj: bool = False,
    highlight_collisions: bool = True,
    collision_color: list[float] = [0.9, 0.1, 0.1],
    mode: str = "discrete",
    rate_hz: float = 30.0,
    segment_duration: float = 0.01,
    outer_description: str = 'Trajectory Index:',
    inner_description: str = 'Timestep Index:',
):
    """
    Two-level interactive slider: outer slider selects a trajectory from the
    dataset, inner slider scrubs individual timesteps within that trajectory.

    The outer slider computes collision indices once and publishes the first
    ghost robot collision (if any). Optionally it animates the full trajectory.
    The inner slider reuses the cached collision indices and only republishes
    the robot config and ghost robot, avoiding redundant collision computation.

    Parameters
    ----------
    dataset : Dataset
        Dataset whose entries contain either sample["expert"] or
        sample["trajectory"] depending on `expert`.
    robot : Robot
        Robot instance used for visualization.
    expert : bool
        If True, visualize sample["expert"]; otherwise sample["trajectory"].
    animate_traj : bool
        If True, the outer slider animates the full trajectory before handing
        control to the inner slider.
    highlight_collisions : bool
        If True, overlay a ghost robot at colliding timesteps.
    collision_color : list[float]
        RGB color used for the collision-highlight ghost robot.
    mode : str
        Animation mode forwarded to visualize_trajectory: "discrete" or
        "interpolation". Only used when animate_traj=True.
    rate_hz : float
        Animation frame rate in Hz. Only used when animate_traj=True.
    segment_duration : float
        Seconds per waypoint segment for "interpolation" mode.
        Only used when animate_traj=True.
    outer_description : str
        Label for the trajectory-selection slider.
    inner_description : str
        Label for the timestep-selection slider.
    """
    traj_key = "expert" if expert else "trajectory"

    def get_trajectory_fn(sample):
        return remove_trajectory_padding(sample[traj_key])

    def outer_callback(sample, trajectory, **kwargs):
        """
        Called when the outer slider selects a new trajectory.
        Sets up the scene, computes collision indices, publishes the first
        collision ghost, and optionally animates the trajectory.
        Returns collision_indices so create_nested_slider can cache them.
        """
        configs = [trajectory[i] for i in range(len(trajectory))]

        if animate_traj:
            # _highlight_collisions sets the ghost robot before animation starts
            # so it remains visible during playback.
            collision_indices = _highlight_collisions(
                robot, sample, configs, collision_color
            )
            # visualize_trajectory handles scene setup (obstacles + target);
            # highlight_collisions=False avoids redundant collision computation.
            visualize_trajectory(
                robot=robot,
                sample=sample,
                trajectory=trajectory,
                normalized=False,
                mode=mode,
                rate_hz=rate_hz,
                segment_duration=segment_duration,
                highlight_collisions=False,
            )
        else:
            # Set up scene (target + obstacles + start config placeholder)
            visualize_problem(robot, sample)
            if highlight_collisions:
                collision_indices = _highlight_collisions(
                    robot, sample, configs, collision_color
                )
            else:
                viz_client.clear_ghost_robot()
                collision_indices = []

        return collision_indices

    def inner_callback(index, sample, trajectory, collision_indices=None, **kwargs):
        """
        Called on every inner slider change.
        Republishes only the robot config and ghost robot using the collision
        indices already computed by outer_callback.
        """
        config = trajectory[index]
        viz_client.publish_config(config)

        if highlight_collisions and collision_indices is not None:
            if index in collision_indices:
                viz_client.publish_ghost_robot(
                    config, color=collision_color, alpha=0.6
                )

        config_str = ", ".join([f"{v:.3f}" for v in config.tolist()])
        print(f"Trajectory timestep {index}: [{config_str}]")
        if collision_indices is not None and index in collision_indices:
            print(f"⚠ Collision at this timestep!")
        print("✓ Visualization updated!")

    create_nested_slider(
        dataset=dataset,
        get_trajectory_fn=get_trajectory_fn,
        inner_callback=inner_callback,
        outer_callback=outer_callback,
        outer_description=outer_description,
        inner_description=inner_description,
    )

