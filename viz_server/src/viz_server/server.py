"""
ROS 2 lifecycle node that listens on a ZeroMQ REP socket and republishes
incoming data as standard ROS topics so Foxglove / RViz can visualise them.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Union

import fasteners
import numpy as np
import rclpy
import zmq
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from termcolor import cprint
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
from urdf_parser_py.urdf import URDF  # should maybe use urchin instead
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from robofin.robots import Robot


LOCK_FILE = "/tmp/viz_server.lock"
ZMQ_PORT  = 5556

MAX_GHOST_ROBOT_MARKERS = 69
MAX_OBSTACLES = 40

def convert_mesh_paths_to_absolute(urdf_path, output_path=None):
    """
    Convert relative mesh paths in a URDF to absolute file:// paths.
    Useful for making URDFs compatible with RViz and Foxglove.
    
    Args:
        urdf_path: Path to input URDF file
        output_path: Path for output URDF (optional, defaults to input path with _absolute suffix)
    """
    urdf_path = Path(urdf_path)
    
    if output_path is None:
        output_path = urdf_path.parent / f"{urdf_path.stem}_abs.urdf"
    
    print(f"Converting mesh paths to absolute: {urdf_path} -> {output_path}")
    
    # Parse URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = urdf_path.parent
    
    converted_count = 0
    
    # Convert mesh paths
    for mesh in root.findall('.//mesh'):
        filename = mesh.get('filename')
        if filename and not filename.startswith('file://'):
            if filename.startswith('package://'):
                filename = filename[len('package://'):]
            
            abs_path = urdf_dir / filename
            if abs_path.exists():
                mesh.set('filename', f'file://{abs_path.absolute()}')
                converted_count += 1
            else:
                print(f"WARNING: Mesh file not found: {abs_path}")
    
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    print(f"SUCCESS: Converted {converted_count} mesh paths to absolute")
    print(f"  Output: {output_path}")
    
    return output_path

class VizServer(Node):
    """
    A single-process visualisation bridge.

    Parameters
    ----------
    urdf_path : str
        Path to the robot URDF to parse (needed for joint names & ghost meshes).
    hz : float
        Internal update rate used for trajectory interpolation.
    segment_dur : float
        Default seconds spent between two successive trajectory waypoints.
    """

    def __init__(self, urdf_path: str, hz: float = 30.0, segment_dur: float = 1.0):
        super().__init__("viz_server")

        # ------------------------------------------------------------------ #
        # singleton lock so only one server owns the port
        # ------------------------------------------------------------------ #
        self.lock = fasteners.InterProcessLock(LOCK_FILE)
        if not self.lock.acquire(blocking=False):
            cprint("Another viz_server is already running – aborting", "red")
            sys.exit(1)

        # ------------------------------------------------------------------ #
        # URDF parsing -> movable joint list
        # ------------------------------------------------------------------ #
        self.urdf_path: str = urdf_path
        self.robot = Robot(self.urdf_path)
        self.abs_urdf_path = "/tmp/abs_urdf.urdf"
        convert_mesh_paths_to_absolute(self.urdf_path, self.abs_urdf_path)
        self.urdf_robot: URDF = URDF.from_xml_file(self.abs_urdf_path)
        self.joint_names: list = [j.name for j in self.urdf_robot.joints if j.type != "fixed"]
        self.num_dof: int = len(self.joint_names)
        cprint(f"Robot DoF ({self.num_dof}): {self.joint_names}", "cyan")

        # Parse mimic joint relationships
        self.mimic_joints = self._parse_mimic_joints()
        if self.mimic_joints:
            cprint(f"Mimic joints found: {self.mimic_joints}", "cyan")

        # ------------------------------------------------------------------ #
        # ROS publishers
        # ------------------------------------------------------------------ #
        # Use more robust QoS to avoid drops under bursty publishing
        js_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # latch last marker state
        )
        pc_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.js_pub     = self.create_publisher(JointState,   "/joint_states",  js_qos)
        self.marker_pub = self.create_publisher(MarkerArray,  "/viz/markers",   marker_qos)

        # Multiple point cloud publishers for different types
        self.pc_robot_pub    = self.create_publisher(PointCloud2, "/viz/robot_points",    pc_qos)
        self.pc_target_pub   = self.create_publisher(PointCloud2, "/viz/target_points",   pc_qos)
        self.pc_obstacle_pub = self.create_publisher(PointCloud2, "/viz/obstacle_points", pc_qos)

        # Static transform broadcaster for world frame
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # tunables
        self.rate_hz          = hz
        self.segment_duration = segment_dur
        self.rsp_proc         = None         # robot_state_publisher subprocess

        # ------------------------------------------------------------------ #
        # Start robot_state_publisher immediately
        # ------------------------------------------------------------------ #
        
        cprint("Starting robot_state_publisher...", "green")
        self._start_robot_state_publisher()
        # Keep robot_state_publisher alive in the background
        threading.Thread(target=self._rsp_watchdog, daemon=True).start()
        
        # Publish static transform from world to base link
        self._publish_world_to_base_transform()

        # ------------------------------------------------------------------ #
        # ZeroMQ REP socket
        # ------------------------------------------------------------------ #
        self.zmq_ctx = zmq.Context.instance()
        self.sock    = self.zmq_ctx.socket(zmq.REP)
        self.sock.bind(f"tcp://127.0.0.1:{ZMQ_PORT}")
        threading.Thread(target=self._zmq_loop, daemon=True).start()

    def __del__(self):
        """Clean up when the node is destroyed."""
        self._cleanup()

    # ====================================================================== #
    # ZeroMQ – request loop
    # ====================================================================== #
    def _zmq_loop(self) -> None:
        """Blocking REP loop; one JSON + optional binary frame per request."""
        while rclpy.ok():
            try:
                hdr     = self.sock.recv_json()
                payload = self.sock.recv() if self.sock.getsockopt(zmq.RCVMORE) else None
                match hdr.get("cmd"):
                    case "ping":       self.sock.send_json({"status": "ok"})
                    case "joints":     self._handle_joints(hdr)
                    case "config":     self._handle_config(hdr)
                    case "trajectory": self._handle_trajectory(hdr)
                    case "pointcloud": self._handle_pointcloud(hdr, payload)
                    case "ghost_end_effector": self._handle_ghost_end_effector(hdr)
                    case "ghost_robot": self._handle_ghost_robot(hdr)
                    case "clear_ghost_end_effector": self._handle_clear_ghost_end_effector()
                    case "clear_ghost_robots": self._handle_clear_ghost_robots()
                    case "obstacles":  self._handle_obstacles(hdr, payload)
                    case "clear_obstacles": self._handle_clear_obstacles()
                    case "shutdown":   self._handle_shutdown()
                    case _:            self.sock.send_json({"status": "error", "msg": "unknown cmd"})
            except Exception as exc:     # noqa: BLE001
                self.get_logger().error(str(exc))
                self.sock.send_json({"status": "error", "msg": f"_zmq_loop Exception: {str(exc)}"})

    # ====================================================================== #
    # Command handlers
    # ====================================================================== #
    def _handle_joints(self, hdr: Dict) -> None:
        """Publish one JointState message."""
        try:
            self._publish_joints(hdr["joints"])
            self.sock.send_json({"status": "ok"})
        except Exception as e:
            self.sock.send_json({"status": "error", "msg": f"joints: {e}"})

    def _handle_config(self, hdr: Dict) -> None:
        """
        Publish one JointState message, based on main joint config vector. 
        Auxiliary joints take default values.
        """
        try:
            config = hdr["config"]
            joints = dict(zip(self.robot.main_joint_names, config))
            for joint_name, joint_value in self.robot.auxiliary_joint_defaults.items():
                joints[joint_name] = joint_value

            self._publish_joints(joints)
            self.sock.send_json({"status": "ok"})
        except Exception as e:
            self.sock.send_json({"status": "error", "msg": f"config: {e}"})

    def _publish_joints(self, joints: Dict[str, float]) -> None:
        """Publish joint state without ZMQ response (for internal use)."""
        # Ensure robot_state_publisher is alive; restart if needed
        self._ensure_robot_state_publisher()
        resolved_joints = self._resolve_mimic_joints(joints)
        
        js               = JointState()
        js.header.stamp  = self.get_clock().now().to_msg()
        js.name          = self.joint_names
        js.position      = [resolved_joints[n] for n in self.joint_names]
        self.js_pub.publish(js)

    # ------------------------------------------------------------------ #
    # robot_state_publisher management
    # ------------------------------------------------------------------ #
    def _is_robot_state_publisher_alive(self) -> bool:
        try:
            return hasattr(self, 'rsp_proc') and self.rsp_proc is not None and self.rsp_proc.poll() is None
        except Exception:
            return False

    def _start_robot_state_publisher(self) -> None:
        """Start robot_state_publisher if it isn't running."""
        if self._is_robot_state_publisher_alive():
            return

        try:
            xml = Path(self.abs_urdf_path).read_text()
        except Exception as e:
            cprint(f"Failed reading URDF XML for robot_state_publisher: {e}", "red")
            return

        # Write params file to avoid huge/quoted CLI args
        try:
            params = {
                "robot_state_publisher": {
                    "ros__parameters": {
                        "robot_description": xml
                    }
                }
            }
            params_path = Path("/tmp/rsp_params.yaml")
            with open(params_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(params, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            cprint(f"Failed writing params file for robot_state_publisher: {e}", "red")
            return

        try:
            # Source ROS before launching to ensure 'ros2' is on PATH
            cmd = (
                "source /opt/ros/humble/setup.bash && "
                f"ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file {params_path}"
            )
            log_path = Path("/tmp/robot_state_publisher.log")
            log_file = open(log_path, "ab", buffering=0)
            self.rsp_proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                shell=True,
                executable="/bin/bash"
            )

            # Give it a moment and check if it died immediately
            time.sleep(0.5)
            if self.rsp_proc.poll() is not None:
                cprint("robot_state_publisher failed to start", "red")
                try:
                    tail = Path("/tmp/robot_state_publisher.log").read_bytes()[-4000:]
                    cprint(tail.decode(errors="ignore"), "yellow")
                except Exception:
                    pass
                self.rsp_proc = None
                return

            # Publish a neutral state shortly after startup so RSP has data
            threading.Thread(target=self._publish_neutral_state, daemon=True).start()
        except Exception as e:
            cprint(f"Failed to start robot_state_publisher: {e}", "red")

    def _ensure_robot_state_publisher(self) -> None:
        """Ensure robot_state_publisher is running; restart if it died."""
        if not self._is_robot_state_publisher_alive():
            cprint("robot_state_publisher not running — restarting…", "yellow")
            self._start_robot_state_publisher()

    def _rsp_watchdog(self) -> None:
        """Background watchdog that restarts robot_state_publisher if it dies."""
        while rclpy.ok():
            if not self._is_robot_state_publisher_alive():
                cprint("[watchdog] robot_state_publisher down — attempting restart", "yellow")
                self._start_robot_state_publisher()
            time.sleep(1.0)

    def _handle_trajectory(self, hdr: Dict) -> None:
        """Launch background interpolation thread."""
        waypoints   = hdr["waypoints"]
        segment_dur = hdr.get("segment_duration", self.segment_duration)
        rate_hz     = hdr.get("rate_hz", self.rate_hz)
        threading.Thread(target=self._run_traj,
                         args=(waypoints, segment_dur, rate_hz), daemon=True).start()
        self.sock.send_json({"status": "ok"})

    def _run_traj(self, wps: List[Dict[str, float]], seg_dur: float, rate_hz: float | None = None) -> None:
        """Simple linear interpolation at specified rate."""
        if rate_hz is None:
            rate_hz = self.rate_hz
        rate = self.create_rate(rate_hz)
        for i in range(len(wps) - 1):
            a, b  = wps[i], wps[i+1]
            steps = max(1, int(rate_hz * seg_dur))
            for s in range(steps + 1):
                alpha  = s / steps
                interp = {jn: (1-alpha)*a[jn] + alpha*b[jn] for jn in self.joint_names}
                self._publish_joints(interp)
                rate.sleep()
        time.sleep(1.0)

    def _handle_pointcloud(self, hdr: Dict, payload: bytes | None) -> None:
        """Convert raw XYZ buffer => PointCloud2."""
        if payload is None:
            self.sock.send_json({"status": "error", "msg": "missing payload"}); return
        
        pts     = np.frombuffer(payload, hdr["dtype"]).reshape(hdr["shape"])   # type: ignore[arg-type]
        frame_id   = hdr.get("frame_id", self.robot.base_link_name)  # Default to base link
        pc_type = hdr.get("pc_type", "robot_points")     # Default type
        
        # Create simple XYZ point cloud
        header = Header(frame_id=frame_id, stamp=self.get_clock().now().to_msg())
        cloud  = pc2.create_cloud_xyz32(header, pts)
        
        # Publish to the appropriate topic based on type
        if pc_type == "robot_points":
            self.pc_robot_pub.publish(cloud)
        elif pc_type == "target_points":
            self.pc_target_pub.publish(cloud)
        elif pc_type == "obstacle_points":
            self.pc_obstacle_pub.publish(cloud)
        else:
            self.sock.send_json({"status": "error", "msg": f"unknown pc_type: {pc_type}"}); return
            
        self.sock.send_json({"status": "ok"})

    def _handle_ghost_end_effector(self, hdr: Dict) -> None:
        """Spawn a translucent mesh for the visual links of the end-effector."""
        [x, y, z, qx, qy, qz, qw] = hdr["pose"]
        frame: Union[str, None] = hdr.get("frame", None) # end-effector link name that has pose
        color: list[float] = hdr.get("color", [0.0, 1.0, 0.0])
        alpha: float = hdr.get("alpha", 0.5)

        auxiliary_joint_values = {}
        for joint_name in self.robot.auxiliary_joint_names:
            auxiliary_joint_values[joint_name] = \
                hdr.get(joint_name, self.robot.auxiliary_joint_defaults[joint_name])

        if frame is None:
            frame = self.robot.tcp_link_name
        assert isinstance(frame, str)
        if frame not in self.robot.fixed_eef_link_transforms:
            raise ValueError(f"Frame {frame} is not a valid end effector frame (must be a link fixed to tcp)")

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        pose[:3, 3] = [x,y,z]

        visual_fk = self.robot.eef_visual_fk(pose, frame, auxiliary_joint_values)

        markers = []
        for link_name, link_pose in visual_fk.items():

            link = self.urdf_robot.link_map.get(link_name)
            if not link:
                self.sock.send_json({"status": "error", "msg": f"link {link_name} not found"})
                return
            
            if not (link.visual and hasattr(link.visual, 'geometry')
                    and link.visual.geometry
                    and hasattr(link.visual.geometry, 'filename')):
                continue
            link_mesh_uri = link.visual.geometry.filename

            qx,qy,qz,qw = Rotation.from_matrix(link_pose[:3, :3]).as_quat()
            x,y,z = link_pose[:3, 3]
            
            m = Marker()
            m.header.frame_id = self.robot.base_link_name
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns   = f"ghost_ee_{link_name}"
            m.type = Marker.MESH_RESOURCE
            m.mesh_resource = link_mesh_uri
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x,y,z
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = qx,qy,qz,qw
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color.r = float(color[0])
            m.color.g = float(color[1])
            m.color.b = float(color[2])
            m.color.a = float(alpha)
            markers.append(m)

        self.marker_pub.publish(MarkerArray(markers=markers))
        self.sock.send_json({"status": "ok"})

    def _handle_ghost_robot(self, hdr: Dict) -> None:
        """Spawn a translucent mesh for the visual links of the robot."""
        config: list[float] = hdr["config"]
        color: list[float] = hdr.get("color", [0.0, 1.0, 0.0])
        alpha: float = hdr.get("alpha", 0.5)
        index: int = hdr.get("index", 0)

        if index >= MAX_GHOST_ROBOT_MARKERS:
            self.sock.send_json({"status": "error", "msg": f"index {index} is too large"})
            return

        auxiliary_joint_values = {}
        for joint_name in self.robot.auxiliary_joint_names:
            auxiliary_joint_values[joint_name] = \
                hdr.get(joint_name, self.robot.auxiliary_joint_defaults[joint_name])

        visual_fk = self.robot.visual_fk(np.array(config), auxiliary_joint_values)
        assert isinstance(visual_fk, dict)

        markers = []
        for link_name, link_pose in visual_fk.items():


            link = self.urdf_robot.link_map.get(link_name)
            if not link:
                self.sock.send_json({"status": "error", "msg": f"link {link_name} not found"})
                return

            if not (link.visual and hasattr(link.visual, 'geometry')
                    and link.visual.geometry
                    and hasattr(link.visual.geometry, 'filename')):
                continue
            link_mesh_uri = link.visual.geometry.filename

            squeezed_pose = link_pose.squeeze()
            qx,qy,qz,qw = Rotation.from_matrix(squeezed_pose[:3, :3]).as_quat()
            x,y,z = squeezed_pose[:3, 3]

            m = Marker()
            m.id = index
            m.header.frame_id = self.robot.base_link_name
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns   = f"ghost_robot_{link_name}"
            m.type = Marker.MESH_RESOURCE
            m.mesh_resource = link_mesh_uri
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x,y,z
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = qx,qy,qz,qw
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color.r = float(color[0])
            m.color.g = float(color[1])
            m.color.b = float(color[2])
            m.color.a = float(alpha)
            markers.append(m)

        self.marker_pub.publish(MarkerArray(markers=markers))
        self.sock.send_json({"status": "ok"})

    def _handle_clear_ghost_end_effector(self) -> None:
        """Clear all ghost end effector markers."""
        markers = []
        timestamp = self.get_clock().now().to_msg()
        
        for i in range(MAX_GHOST_ROBOT_MARKERS):
            for link_name in self.robot.eef_visual_link_names:
                m = Marker()
                m.id = i
                m.header.frame_id = self.robot.base_link_name
                m.header.stamp = timestamp
                m.ns = f"ghost_ee_{link_name}"
                m.action = Marker.DELETEALL
                markers.append(m)

        self.marker_pub.publish(MarkerArray(markers=markers))
        self.sock.send_json({"status": "ok"})

    def _handle_clear_ghost_robots(self) -> None:
        """Clear all ghost robot markers."""
        markers = []
        timestamp = self.get_clock().now().to_msg()
        
        # Clear all robot links
        for link_name in self.robot.arm_visual_link_names:
            m = Marker()
            m.header.frame_id = self.robot.base_link_name
            m.header.stamp = timestamp
            m.ns = f"ghost_robot_{link_name}"
            m.action = Marker.DELETEALL
            markers.append(m)
        
        self.marker_pub.publish(MarkerArray(markers=markers))
        self.sock.send_json({"status": "ok"})

    def _handle_obstacles(self, hdr: Dict, payload: bytes | None) -> None:
        """Display obstacle cylinders and cuboids."""
        color: list[float] = hdr.get("color", [0.8, 0.5, 0.6]) # light-red default
        if payload is None:
            self.sock.send_json({"status": "error", "msg": "missing payload"})
            return

        # deserialize the payload
        end = hdr["cuboid_dims_bytesize"]
        cuboid_dims = np.frombuffer(
            payload[:end], hdr["dtype"]).reshape(hdr["cuboid_dims_shape"])
        start = end
        end = start + hdr["cuboid_centers_bytesize"]
        cuboid_centers = np.frombuffer(
            payload[start:end], hdr["dtype"]).reshape(hdr["cuboid_centers_shape"])
        start = end
        end = start + hdr["cuboid_quaternions_bytesize"]
        cuboid_quaternions = np.frombuffer(
            payload[start:end], hdr["dtype"]).reshape(hdr["cuboid_quaternions_shape"])
        start = end
        end = start + hdr["cylinder_radii_bytesize"]
        cylinder_radii = np.frombuffer(
            payload[start:end], hdr["dtype"]).reshape(hdr["cylinder_radii_shape"])
        start = end
        end = start + hdr["cylinder_heights_bytesize"]
        cylinder_heights = np.frombuffer(
            payload[start:end], hdr["dtype"]).reshape(hdr["cylinder_heights_shape"])
        start = end
        end = start + hdr["cylinder_centers_bytesize"]
        cylinder_centers = np.frombuffer(
            payload[start:end], hdr["dtype"]).reshape(hdr["cylinder_centers_shape"])
        start = end
        cylinder_quaternions = np.frombuffer(
            payload[start:], hdr["dtype"]).reshape(hdr["cylinder_quaternions_shape"])

        markers = []

        obstacle_idx = 0

        # process cuboids
        for dims, center, quat in zip(cuboid_dims, cuboid_centers, cuboid_quaternions):
            if len(dims) != 3 or len(center) != 3 or len(quat) != 4:
                self.sock.send_json({"status": "error", "msg": "Invalid cuboid parameters"})
                return
            
            if obstacle_idx >= MAX_OBSTACLES:
                self.sock.send_json({"status": "warning", "msg": f"Too many obstacles, more than {MAX_OBSTACLES}"})
                break

            m = Marker()
            m.header.frame_id = self.robot.base_link_name
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns   = "obstacles"
            m.id   = obstacle_idx; obstacle_idx += 1
            m.type = Marker.CUBE; m.action = Marker.ADD
            m.pose.position.x = float(center[0])
            m.pose.position.y = float(center[1])
            m.pose.position.z = float(center[2])
            m.pose.orientation.x = float(quat[1])
            m.pose.orientation.y = float(quat[2])
            m.pose.orientation.z = float(quat[3])
            m.pose.orientation.w = float(quat[0])
            m.scale.x = float(dims[0])
            m.scale.y = float(dims[1])
            m.scale.z = float(dims[2])
            m.color.r = float(color[0])
            m.color.g = float(color[1])
            m.color.b = float(color[2])
            m.color.a = 1.0
            
            markers.append(m)

        # process cylinders
        for radius, height, center, quat in zip(cylinder_radii, cylinder_heights, cylinder_centers, cylinder_quaternions):
            if len(center) != 3 or len(quat) != 4:
                self.sock.send_json({"status": "error", "msg": "Invalid cylinder parameters"})
                return

            if obstacle_idx >= MAX_OBSTACLES:
                self.sock.send_json({"status": "warning", "msg": f"Too many obstacles, more than {MAX_OBSTACLES}"})
                break

            m = Marker()
            m.header.frame_id = self.robot.base_link_name
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns   = "obstacles"
            m.id   = obstacle_idx; obstacle_idx += 1
            m.type = Marker.CYLINDER; m.action = Marker.ADD
            m.pose.position.x = float(center[0])
            m.pose.position.y = float(center[1])
            m.pose.position.z = float(center[2])
            m.pose.orientation.x = float(quat[1])
            m.pose.orientation.y = float(quat[2])
            m.pose.orientation.z = float(quat[3])
            m.pose.orientation.w = float(quat[0])
            m.scale.x = float(radius)
            m.scale.y = float(radius)
            m.scale.z = float(height)
            m.color.r = float(color[0])
            m.color.g = float(color[1])
            m.color.b = float(color[2])
            m.color.a = 1.0
            markers.append(m)

        self.marker_pub.publish(MarkerArray(markers=markers))
        self.sock.send_json({"status": "ok"})

    def _handle_clear_obstacles(self) -> None:
        """
        Clear all obstacle markers.
        
        Uses the knowledge that there are at most 40 obstacles to a given scene.
        """
        markers = []
        timestamp = self.get_clock().now().to_msg()

        for i in range(MAX_OBSTACLES): # at most 40 obstacles in Avoid Everything
            m = Marker()
            m.header.frame_id = self.robot.base_link_name
            m.header.stamp = timestamp
            m.ns = "obstacles"
            m.id = i
            m.action = Marker.DELETEALL
            markers.append(m)

        self.marker_pub.publish(MarkerArray(markers=markers))
        self.sock.send_json({"status": "ok"})

    def _handle_shutdown(self) -> None:
        """Handle shutdown command - clean up and exit."""
        cprint("Shutdown command received", "yellow")
        self.sock.send_json({"status": "ok"})

        # Give ZMQ time to send the response
        time.sleep(0.1)
        self._cleanup()
        import os
        os._exit(0)

    def _cleanup(self) -> None:
        """Clean up resources."""
        cprint("Cleaning up viz_server...", "yellow")
        # First, close ZMQ resources to stop incoming requests
        if hasattr(self, 'sock'):
            try:
                self.sock.close(0)
            except:
                pass
        if hasattr(self, 'zmq_ctx'):
            try:
                self.zmq_ctx.term()
            except:
                pass
        
        # handle the subprocess
        if hasattr(self, 'rsp_proc') and self.rsp_proc:
            cprint("Terminating robot_state_publisher...", "yellow")
            try:
                # Check if process is still running
                if self.rsp_proc.poll() is None:
                    # First try graceful termination
                    self.rsp_proc.terminate()
                    
                    # Wait up to 2 seconds for graceful shutdown
                    try:
                        self.rsp_proc.wait(timeout=2.0)
                        cprint("robot_state_publisher terminated gracefully", "green")
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't respond
                        cprint("Force killing robot_state_publisher...", "red")
                        self.rsp_proc.kill()
                        try:
                            self.rsp_proc.wait(timeout=1.0)
                            cprint("robot_state_publisher force killed", "yellow")
                        except subprocess.TimeoutExpired:
                            cprint("robot_state_publisher kill timed out", "red")
                            # Try system-level kill as last resort
                            try:
                                import os
                                os.kill(self.rsp_proc.pid, 9)
                                cprint("robot_state_publisher killed with SIGKILL", "red")
                            except:
                                cprint("Failed to kill robot_state_publisher", "red")
                else:
                    cprint("robot_state_publisher already terminated", "green")
                    
            except Exception as e:
                cprint(f"Error terminating robot_state_publisher: {e}", "red")
        
        # final cleanup of any remaining robot_state_publisher processes via pkill
        try:
            result = subprocess.run(["pkill", "-f", "robot_state_publisher"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                cprint("Killed remaining robot_state_publisher processes", "yellow")
        except:
            pass
                
        if hasattr(self, 'lock'):
            try:
                self.lock.release()
            except:
                pass

    def _publish_neutral_state(self) -> None:
        """Publish neutral joint state (all joints at 0 radians) after startup."""
        time.sleep(1.0)  # Wait for robot_state_publisher to be ready
        neutral_joints = {name: 0.0 for name in self.joint_names}
        self._publish_joints(neutral_joints)
        cprint("Published neutral joint state (all joints at 0 radians)", "green")

    def _publish_world_to_base_transform(self) -> None:
        """Publish static transform from world frame to robot base link."""
        transform = TransformStamped()
        
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "world"
        transform.child_frame_id = self.robot.base_link_name
        
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        
        self.static_tf_broadcaster.sendTransform(transform)
        cprint(f"Published static transform: world -> {self.robot.base_link_name}", "green")

    def _parse_mimic_joints(self) -> Dict[str, Dict]:
        """
        Parse mimic joint relationships from URDF.
        
        Returns dict: {mimic_joint_name: {parent: str, multiplier: float, offset: float}}
        """
        mimic_joints = {}
        for joint in self.urdf_robot.joints:
            if hasattr(joint, 'mimic') and joint.mimic:
                # Handle the case where multiplier might be None
                multiplier = getattr(joint.mimic, 'multiplier', None)
                if multiplier is None:
                    multiplier = 1.0
                offset = getattr(joint.mimic, 'offset', None)
                if offset is None:
                    offset = 0.0
                mimic_info = {
                    'parent': joint.mimic.joint,
                    'multiplier': multiplier,
                    'offset': offset
                }
                mimic_joints[joint.name] = mimic_info

        return mimic_joints

    def _resolve_mimic_joints(self, joints: Dict[str, float]) -> Dict[str, float]:
        """
        Apply mimic joint relationships to resolve all joint values.
        
        If both parent and mimic joint are specified, parent takes precedence.
        If only mimic joint is specified, we compute parent value (reverse mimic).
        """
        resolved = joints.copy()
        for mimic_joint, mimic_info in self.mimic_joints.items():
            parent_joint = mimic_info['parent']
            multiplier = mimic_info['multiplier']
            offset = mimic_info['offset']
            parent_in_joints = parent_joint in resolved
            mimic_in_joints = mimic_joint in resolved

            if parent_in_joints and mimic_in_joints:
                parent_val = resolved[parent_joint]
                mimic_val = resolved[mimic_joint]
                expected_mimic = parent_val * multiplier + offset
                
                if abs(mimic_val - expected_mimic) > 1e-6:
                    cprint(f"Warning: Inconsistent mimic joint values for {mimic_joint}. " +
                          f"Expected {expected_mimic:.4f} from {parent_joint}={parent_val:.4f}, " +
                          f"got {mimic_val:.4f}. Using parent value.", "yellow")

                resolved[mimic_joint] = expected_mimic

            elif parent_in_joints and not mimic_in_joints:
                resolved[mimic_joint] = resolved[parent_joint] * multiplier + offset

            elif not parent_in_joints and mimic_in_joints:
                if multiplier != 0:
                    resolved[parent_joint] = (resolved[mimic_joint] - offset) / multiplier
                else:
                    cprint(f"Warning: Cannot reverse mimic for {mimic_joint} (multiplier=0)", "yellow")

            elif not parent_in_joints and not mimic_in_joints:
                pass

        if set(resolved) != set(self.joint_names):
            missing = set(self.joint_names) - set(resolved)
            extra = set(resolved) - set(self.joint_names)
            raise ValueError(f"joint set mismatch after mimic resolution. Missing: {missing}, Extra: {extra}")

        return resolved

# ---------------------------------------------------------------------- #
# CLI entry-point
# ---------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", required=True, help="Path to robot URDF")
    p.add_argument("--hz",   type=float, default=30.0, help="Interpolator rate [Hz]")
    p.add_argument("--segment_duration", type=float, default=1.0,
                   help="Seconds per trajectory segment")
    args = p.parse_args()

    rclpy.init()
    
    # Create the server
    server = VizServer(args.urdf, args.hz, args.segment_duration)
    
    # Set up signal handling for graceful shutdown
    def signal_handler(signum, frame):
        cprint(f"Received signal {signum}, shutting down gracefully...", "yellow")
        server._cleanup()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command
    
    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        cprint("Keyboard interrupt received, shutting down...", "yellow")
        server._cleanup()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
