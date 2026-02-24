"""
Thin helper that hides the ZeroMQ details; no ROS imports needed.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import yaml
import zmq
from termcolor import cprint


PORT        = 5556
SERVER_CMD  = "source /opt/ros/humble/setup.bash && python3 -c \"import sys; sys.path.insert(0, '/workspace/viz_server/src'); from viz_server.server import main; main()\""

_ctx        = zmq.Context.instance()
_sock: zmq.Socket[Any] | None = None
_connected  = False
_base_link_name: str = "base_link"  # Default, will be loaded from config


# ====================================================================== #
# Server bootstrap helpers
# ====================================================================== #
def _server_alive() -> bool:
    """Return True iff a viz_server REP socket is already up."""
    try:
        s = _ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        s.connect(f"tcp://127.0.0.1:{PORT}")
        s.send_json({"cmd": "ping"})
        s.recv_json(flags=0)
        s.close(0)
        return True
    except Exception:   # noqa: BLE001
        return False


def is_connected() -> bool:
    """Return True if viz_client has an active connection to viz_server."""
    return _connected


def connect(urdf: str, *, port: int = 5556) -> None:
    """
    Ensure viz_server is running and obtain a REQ socket.

    Parameters
    ----------
    urdf : str
        Path to the robot URDF (only needed on first call).
    port : int, default 5556
        ZeroMQ port to connect to.
    """
    global _sock, _connected, PORT, _base_link_name
    PORT = port

    # Check if robot_config.yaml exists before starting server
    urdf_dir = Path(urdf).parent
    config_path = urdf_dir / "robot_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"robot_config.yaml not found at {config_path}. This file is required for viz_server to function.")

    if not _server_alive():
        cprint("viz_server not running — starting…", "yellow")
        if not Path(urdf).is_file():
            raise FileNotFoundError(urdf)
        
        cmd = f"{SERVER_CMD} --urdf {urdf}"
        
        # Use pipes to capture output for debugging
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=True,
            executable="/bin/bash"
        )
        
        # Wait up to 10 seconds for server to start
        for i in range(100):  # 10 seconds with 0.1s sleep
            if _server_alive():
                break
            time.sleep(0.1)
            
            # Check if process has failed
            if process.poll() is not None:
                # Process has exited, get the output
                stdout, stderr = process.communicate()
                error_msg = f"viz_server failed to start. Exit code: {process.returncode}"
                if stderr:
                    error_msg += f"\nStderr: {stderr.decode()}"
                if stdout:
                    error_msg += f"\nStdout: {stdout.decode()}"
                raise RuntimeError(error_msg)
        
        if not _server_alive():
            # Process is still running but server didn't respond
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
            raise RuntimeError("Server failed to start within 10 seconds")

    # Load base link name from config
    _base_link_name = _load_base_link_name(urdf)
    
    _sock = _ctx.socket(zmq.REQ)
    assert _sock is not None
    _sock.connect(f"tcp://127.0.0.1:{PORT}")
    _connected = True; cprint("Connected to viz_server", "green")


# ====================================================================== #
# Low-level send helper
# ====================================================================== #
def _send(hdr: dict, payload: bytes | None = None) -> None:
    if not _connected or _sock is None:
        raise RuntimeError("Call viz_client.connect() first")
    if payload is None:
        _sock.send_json(hdr)
    else:
        _sock.send_json(hdr, zmq.SNDMORE)
        _sock.send(payload, copy=False)
    resp = _sock.recv_json()
    if not isinstance(resp, dict) or resp.get("status") != "ok":
        if isinstance(resp, dict):
            msg = resp.get("msg", "unknown error") 
            cprint(f"Server error: {msg}", "red")
        else:
            msg = str(resp)
            cprint(f"Server unknown non-ok response: {msg}", "red")
        


# ====================================================================== #
# Helper functions
# ====================================================================== #
def _load_base_link_name(urdf_path: str) -> str:
    """Load base link name from robot_config.yaml in the same directory as URDF."""
    try:
        urdf_dir = Path(urdf_path).parent
        config_path = urdf_dir / "robot_config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"robot_config.yaml not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        base_link = config.get("robot_config", {}).get("base_link_name")
        if not base_link:
            raise ValueError(f"base_link_name not found in robot_config.yaml at {config_path}")
        
        return base_link
        
    except Exception as e:
        raise RuntimeError(f"Error loading robot_config.yaml: {e}")


def _convert_to_numpy_f32(arr: np.ndarray | torch.Tensor) -> np.ndarray:
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
        np_arr: np.ndarray = arr
    else:
        raise TypeError("_convert_to_numpy_f32: Input must be a NumPy array or Torch tensor")
    return np_arr.astype(np.float32)

# ====================================================================== #
# Public API
# ====================================================================== #
def publish_robot_pointcloud(
    points: np.ndarray | torch.Tensor,
    *,
    frame: str | None = None,
    name: str = "robot_cloud"
) -> None:
    """
    Publish robot point cloud.

    Parameters
    ----------
    points : (N, 3) NumPy array or Torch tensor
    frame  : TF frame name (default: base link from config)
    name   : logical name of the cloud
    """
    if frame is None:
        frame = _base_link_name
    
    arr = _convert_to_numpy_f32(points)
    hdr = {
        "cmd":"pointcloud", "frame":frame, "pc_type":"robot_points",
        "dtype":str(arr.dtype), "shape":arr.shape,
        "name":name
    }
    _send(hdr, arr.tobytes())


def publish_target_pointcloud(
    points: np.ndarray | torch.Tensor,
    *,
    frame: str | None = None,
    name: str = "target_cloud"
) -> None:
    """
    Publish target point cloud.

    Parameters
    ----------
    points : (N, 3) NumPy array or Torch tensor
    frame  : TF frame name (default: base link from config)  
    name   : logical name of the cloud
    """
    if frame is None:
        frame = _base_link_name
    
    arr = _convert_to_numpy_f32(points)
    hdr = {
        "cmd":"pointcloud", "frame":frame, "pc_type":"target_points",
        "dtype":str(arr.dtype), "shape":arr.shape,
        "name":name
    }
    _send(hdr, arr.tobytes())


def publish_obstacle_pointcloud(
    points: np.ndarray | torch.Tensor,
    *,
    frame: str | None = None,
    name: str = "obstacle_cloud"
) -> None:
    """
    Publish obstacle point cloud.

    Parameters
    ----------
    points : (N, 3) NumPy array or Torch tensor
    frame  : TF frame name (default: base link from config)
    name   : logical name of the cloud
    """
    if frame is None:
        frame = _base_link_name
    
    arr = _convert_to_numpy_f32(points)
    hdr = {
        "cmd":"pointcloud", "frame":frame, "pc_type":"obstacle_points",
        "dtype":str(arr.dtype), "shape":arr.shape,
        "name":name
    }
    _send(hdr, arr.tobytes())


def clear_robot_pointcloud(*, frame: str | None = None, name: str = "robot_cloud") -> None:
    """Clear robot point cloud by publishing empty cloud."""
    if frame is None:
        frame = _base_link_name
    empty_points = np.array([], dtype=np.float32).reshape(0, 3)
    hdr = {
        "cmd":"pointcloud", "frame":frame, "pc_type":"robot_points",
        "dtype":str(empty_points.dtype), "shape":empty_points.shape,
        "name":name
    }
    _send(hdr, empty_points.tobytes())


def clear_target_pointcloud(*, frame: str | None = None, name: str = "target_cloud") -> None:
    """Clear target point cloud by publishing empty cloud."""
    if frame is None:
        frame = _base_link_name
    empty_points = np.array([], dtype=np.float32).reshape(0, 3)
    hdr = {
        "cmd":"pointcloud", "frame":frame, "pc_type":"target_points",
        "dtype":str(empty_points.dtype), "shape":empty_points.shape,
        "name":name
    }
    _send(hdr, empty_points.tobytes())


def clear_obstacle_pointcloud(*, frame: str | None = None, name: str = "obstacle_cloud") -> None:
    """Clear obstacle point cloud by publishing empty cloud."""
    if frame is None:
        frame = _base_link_name
    empty_points = np.array([], dtype=np.float32).reshape(0, 3)
    hdr = {
        "cmd":"pointcloud", "frame":frame, "pc_type":"obstacle_points",
        "dtype":str(empty_points.dtype), "shape":empty_points.shape,
        "name":name
    }
    _send(hdr, empty_points.tobytes())


# Legacy functions for backward compatibility
def publish_pointcloud(
    points: np.ndarray | torch.Tensor,
    *,
    frame: str | None = None,
    pc_type: str = "robot_points",
    name: str = "cloud"
) -> None:
    """
    Stream an XYZ point cloud to the server.
    
    DEPRECATED: Use publish_robot_pointcloud(), publish_target_pointcloud(), 
    or publish_obstacle_pointcloud() instead.

    Parameters
    ----------
    points : (N, 3) NumPy array or Torch tensor
    frame  : TF frame name (default: base link from config)
    pc_type : point cloud type - 'robot_points', 'target_points', or 'obstacle_points'
    name   : logical name of the cloud (colors set in RViz config)
    """
    if frame is None:
        frame = _base_link_name
    
    valid_types = ["robot_points", "target_points", "obstacle_points"]
    if pc_type not in valid_types:
        raise ValueError(f"pc_type must be one of {valid_types}, got '{pc_type}'")
    
    arr = _convert_to_numpy_f32(points)
    hdr = {
        "cmd":"pointcloud", "frame":frame, "pc_type":pc_type,
        "dtype":str(arr.dtype), "shape":arr.shape,
        "name":name
    }
    _send(hdr, arr.tobytes())

def publish_target_points(
    points: np.ndarray | torch.Tensor,
    *,
    frame: str | None = None,
    name: str = "target_cloud"
) -> None:
    """DEPRECATED: Use publish_target_pointcloud() instead."""
    publish_target_pointcloud(points, frame=frame, name=name)


def publish_obstacle_points(
    points: np.ndarray | torch.Tensor,
    *,
    frame: str | None = None,
    name: str = "obstacle_cloud"
) -> None:
    """DEPRECATED: Use publish_obstacle_pointcloud() instead."""
    publish_obstacle_pointcloud(points, frame=frame, name=name)

def publish_config(config: np.ndarray | torch.Tensor | list[float]) -> None:
    """
    Show the robot in a single configuration. Auxiliary joints take their 
    default values.

    Parameters
    ----------
    config (np.ndarray): Main joint angles in radians (excluding auxiliary joints)
    """
    if isinstance(config, np.ndarray) or isinstance(config, torch.Tensor):
        if config.ndim == 2:
            assert config.shape[0] == 1
            config = config.squeeze()
        config = config.tolist()
    
    assert isinstance(config, list)
    _send({"cmd":"config", "config": config})

def publish_joints(joints: Dict[str, float]) -> None:
    """
    Show the robot in a single configuration.

    Parameters
    ----------
    joints : Dict[str, float]
        Joint angles in radians. Must include all movable joints OR parent joints
        of mimic relationships. Mimic joints are automatically computed from 
        their parent joints.
        
    Examples
    --------
    # Traditional: set all joints including mimics
    publish_joints({"joint1": 0.0, "mimic_joint1": 0.02, "mimic_joint2": 0.02})
    
    # Alternative: only set parent joints, mimics computed automatically  
    publish_joints({"joint1": 0.0, "mimic_joint1": 0.02}) # mimic_joint2 
    resolved in server.
    """
    _send({"cmd":"joints", "joints":joints})


def publish_trajectory(
    waypoints: List[Dict[str, float]],
    *,
    segment_duration: float = 1.0,
    rate_hz: float = 30.0
) -> None:
    """
    Animate a list of joint dictionaries.

    Parameters
    ----------
    waypoints : ordered list of full joint maps
    segment_duration : seconds spent interpolating between each pair
    rate_hz : animation framerate in Hz (higher = smoother)
    """
    _send({"cmd":"trajectory",
           "waypoints":waypoints,
           "segment_duration":segment_duration,
           "rate_hz":rate_hz})


def publish_ghost_end_effector(
    pose: List[float] | np.ndarray,
    frame: Optional[str]=None,
    auxiliary_joint_values: Optional[Dict[str, float]]=None,
    *,
    color: List[float] | None = None,
    alpha: float = 0.7
) -> None:
    """
    Display a translucent mesh of the entire end effector at an arbitrary pose.
    
    This shows the end effector base link plus all visual links (e.g., fingers)
    with proper forward kinematics applied based on current joint states.
    
    Parameters
    ----------
    pose : List[float] | np.ndarray
        Either a 4x4 homogeneous transformation matrix or a list of 7 elements
        [x, y, z, qx, qy, qz, qw] position and orientation for end effector base
    color : List[float] | None
        [r, g, b] color values in 0-1 range (default green)
    alpha : float
        Alpha/transparency value in 0-1 range (0=transparent, 1=opaque)
    """
    if isinstance(pose, np.ndarray):
        if pose.ndim == 3:
            assert pose.shape[0] <= 1, "Batch dim cannot be greater than 1"
            pose = pose.squeeze()
        if pose.shape == (4, 4):
            # convert 4x4 transformation matrix to [x, y, z, qx, qy, qz, qw]
            qx,qy,qz,qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
            pose = np.concatenate((pose[:3, 3], [qx, qy, qz, qw])).tolist()
        elif pose.shape == (7,):
            pose = pose.tolist()
        else:
            raise ValueError("pose must be a 4x4 matrix or a list of 7 elements")


    if color is None:
        color = [0, 1, 0]  # default green

    hdr = {"cmd":"ghost_end_effector", "pose":pose, "color":color, "alpha":alpha}
    if frame is not None:
        hdr["frame"] = frame
    if auxiliary_joint_values is not None:
        for joint_name, joint_value in auxiliary_joint_values.items():
            hdr[joint_name] = joint_value

    _send(hdr)


def publish_ghost_robot(
    config: np.ndarray | torch.Tensor | List[float],
    auxiliary_joint_values: Optional[Dict[str, float]]=None,
    *,
    color: List[float] | None = None,
    alpha: float = 0.5
) -> None:
    """
    Display a translucent mesh of the entire robot at an arbitrary configuration.
    
    This shows all robot links with visual geometry using forward kinematics
    to compute the pose of each link based on the given joint configuration.
    
    Parameters
    ----------
    config (List[float] | np.ndarray): Configuration of main joints
    auxiliary_joint_values (Dict[str, float]): Auxiliary joint values (optional)
    color : List[float] | None
        [r, g, b] color values in 0-1 range (default green)
    alpha : float
        Alpha/transparency value in 0-1 range (0=transparent, 1=opaque)

    """
    if color is None:
        color = [0, 1, 0]  # default green

    if isinstance(config, np.ndarray) or isinstance(config, torch.Tensor):
        if config.ndim == 2:
            assert config.shape[0] == 1
            config = config.squeeze()
        config = config.tolist()

    hdr = {"cmd":"ghost_robot", 
           "config":config, 
           "color":color, 
           "alpha":alpha}
    if auxiliary_joint_values is not None:
        for joint_name, joint_value in auxiliary_joint_values.items():
            hdr[joint_name] = joint_value
    _send(hdr)


def clear_ghost_end_effector() -> None:
    """
    Clear all ghost end effector markers from RViz.
    
    This removes all translucent end effector meshes that were previously
    published with publish_ghost_end_effector().
    """
    _send({"cmd":"clear_ghost_end_effector"})


def clear_ghost_robot() -> None:
    """
    Clear all ghost robot markers from RViz.
    
    This removes all translucent robot meshes that were previously
    published with publish_ghost_robot().
    """
    _send({"cmd":"clear_ghost_robot"})


def publish_obstacles(cuboid_dims,
                      cuboid_centers,
                      cuboid_quaternions,
                      cylinder_radii,
                      cylinder_heights,
                      cylinder_centers,
                      cylinder_quaternions,
                      color: Optional[List[float]]=None) -> None:
    """
    Publish cylinder and cuboid obstacles to the viz_server.
    """

    cub_dims: np.ndarray = _convert_to_numpy_f32(cuboid_dims)
    cub_centers: np.ndarray = _convert_to_numpy_f32(cuboid_centers)
    cub_quaternions: np.ndarray = _convert_to_numpy_f32(cuboid_quaternions)
    cyl_radii: np.ndarray = _convert_to_numpy_f32(cylinder_radii)
    cyl_heights: np.ndarray = _convert_to_numpy_f32(cylinder_heights)
    cyl_centers: np.ndarray = _convert_to_numpy_f32(cylinder_centers)
    cyl_quaternions: np.ndarray = _convert_to_numpy_f32(cylinder_quaternions)

    if color is None:
        color = [0.8, 0.5, 0.6]  # light-red default

    hdr = {
        "cmd": "obstacles",
        "color": color,
        "dtype":str(cub_dims.dtype), # assumed all arrays are the same dtype
        "cuboid_dims_shape": cub_dims.shape,
        "cuboid_dims_bytesize": cub_dims.size * cub_dims.itemsize,
        "cuboid_centers_shape": cub_centers.shape,
        "cuboid_centers_bytesize": cub_centers.size * cub_centers.itemsize,
        "cuboid_quaternions_shape": cub_quaternions.shape,
        "cuboid_quaternions_bytesize": cub_quaternions.size * cub_quaternions.itemsize,
        "cylinder_radii_shape": cyl_radii.shape,
        "cylinder_radii_bytesize": cyl_radii.size * cyl_radii.itemsize,
        "cylinder_heights_shape": cyl_heights.shape,
        "cylinder_heights_bytesize": cyl_heights.size * cyl_heights.itemsize,
        "cylinder_centers_shape": cyl_centers.shape,
        "cylinder_centers_bytesize": cyl_centers.size * cyl_centers.itemsize,
        "cylinder_quaternions_shape": cyl_quaternions.shape,
    }

    payload = (
        cub_dims.tobytes() +
        cub_centers.tobytes() +
        cub_quaternions.tobytes() +
        cyl_radii.tobytes() +
        cyl_heights.tobytes() +
        cyl_centers.tobytes() +
        cyl_quaternions.tobytes()
    )

    _send(hdr, payload)

def publish_obstacles_from_flobs(flobs) -> None:
    """
    Publish cylinder and cuboid obstacles to the viz_server.

    Parameters
    ----------
    flobs : object containing obstacle data as floating point torch tensors or 
        numpy arrays.
    """

    publish_obstacles(
        cuboid_dims=flobs.cuboid_dims,
        cuboid_centers=flobs.cuboid_centers,
        cuboid_quaternions=flobs.cuboid_quaternions,
        cylinder_radii=flobs.cylinder_radii,
        cylinder_heights=flobs.cylinder_heights,
        cylinder_centers=flobs.cylinder_centers,
        cylinder_quaternions=flobs.cylinder_quaternions
    )


def clear_obstacles() -> None:
    """
    Clear all obstacle markers from RViz.
    
    This removes all obstacle markers that were previously
    published with publish_obstacles().
    """
    _send({"cmd":"clear_obstacles"})


def shutdown() -> None:
    """
    Shutdown the viz_server and its robot_state_publisher.
    
    This will cleanly terminate the viz_server process and its
    robot_state_publisher subprocess.
    """
    global _connected
    
    if not _connected or _sock is None:
        cprint("Not connected to viz_server", "yellow")
        return
    
    try:
        _send({"cmd":"shutdown"})
        cprint("Sent shutdown command to viz_server", "green")
    except Exception as e:
        cprint(f"Error sending shutdown command: {e}", "red")
    
    # Reset connection state
    _connected = False
