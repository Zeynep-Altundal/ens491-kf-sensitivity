"""ROS 2 node: simulate the flight once at start-up, then publish truth and sensors in real time.

Topics (see README.md, names in topics.py): /drone/truth/pose, TF world -> base_link, /drone/imu
(200 Hz), /drone/position (10 Hz), /drone/truth/path (2 Hz), /drone/marker (1 Hz per id) and
/drone/sim/info (latched JSON with the trajectory and the time origin t0).
"""

from __future__ import annotations

import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Pose, PoseStamped, TransformStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker

from . import flight
from .quadrotor import QuadParams
from . import reference as references
from . import topics


def _pose(p: np.ndarray, q_wxyz: np.ndarray) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in p)
    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = (float(v) for v in q_wxyz)
    return pose


def drone_markers(stamp, body_frame: str) -> list[Marker]:
    """Body, arms and rotor discs of the quadrotor in the body frame (frame-locked)."""
    params = QuadParams()
    rotors = params.rotor_positions()
    specs = []
    body = Marker(type=Marker.CUBE)
    body.scale.x, body.scale.y, body.scale.z = 0.12, 0.12, 0.05
    body.color.r, body.color.g, body.color.b, body.color.a = 0.2, 0.2, 0.2, 1.0
    specs.append(body)
    arms = Marker(type=Marker.LINE_LIST)
    arms.scale.x = 0.02
    arms.color.r, arms.color.g, arms.color.b, arms.color.a = 0.1, 0.1, 0.8, 1.0
    for i, j in ((0, 1), (2, 3)):
        for r in (rotors[i], rotors[j]):
            pt = Pose().position
            pt.x, pt.y, pt.z = (float(v) for v in r)
            arms.points.append(pt)
    specs.append(arms)
    props = Marker(type=Marker.SPHERE_LIST)
    props.scale.x, props.scale.y, props.scale.z = 0.18, 0.18, 0.01
    props.color.r, props.color.g, props.color.b, props.color.a = 0.9, 0.3, 0.1, 0.8
    for r in rotors:
        pt = Pose().position
        pt.x, pt.y, pt.z = float(r[0]), float(r[1]), 0.03
        props.points.append(pt)
    specs.append(props)
    for i, m in enumerate(specs):
        m.header.stamp, m.header.frame_id = stamp, body_frame
        m.ns, m.id, m.action, m.frame_locked = "drone", i, Marker.ADD, True
        m.pose.orientation.w = 1.0
    return specs


class QuadrotorSimNode(Node):
    def __init__(self) -> None:
        super().__init__(topics.SIM_NODE)
        self.declare_parameter("seed", 0)
        self.declare_parameter("trajectory", references.DEFAULT_TRAJECTORY)
        self.declare_parameter("duration", 0.0)  # 0: the whole trajectory
        self.declare_parameter("world_frame", topics.WORLD_FRAME)
        self.declare_parameter("body_frame", topics.BODY_FRAME)
        # Name of a node to wait for before publishing (e.g. the Isaac Sim renderer); "" = none.
        self.declare_parameter("wait_for_node", "")
        self.declare_parameter("wait_timeout_s", 300.0)
        seed = int(self.get_parameter("seed").value)
        trajectory = str(self.get_parameter("trajectory").value)
        duration = float(self.get_parameter("duration").value) or references.get(trajectory).duration
        self.world = str(self.get_parameter("world_frame").value)
        self.body = str(self.get_parameter("body_frame").value)

        t_start = time.monotonic()
        config = flight.default_config(seed=seed, duration=duration)
        self.data = flight.run(seed=seed, duration=duration, config=config, trajectory=trajectory)
        self.get_logger().info(
            f"simulated {duration:.1f} s of trajectory {trajectory} (seed {seed}) in "
            f"{time.monotonic() - t_start:.1f} s")

        self.imu_rate = config.imu_rate
        self.meas_step = int(round(config.imu_rate / config.meas_rate))
        dt = 1.0 / config.imu_rate
        self.gyro_var = (config.gyro_noise_density / np.sqrt(dt)) ** 2
        self.accel_var = (config.accel_noise_density / np.sqrt(dt)) ** 2

        def pub(msg_type, topic):
            return self.create_publisher(msg_type, topic.name, topics.qos_profile(topic))

        self.pub_pose = pub(PoseStamped, topics.TRUTH_POSE)
        self.pub_imu = pub(Imu, topics.IMU)
        self.pub_pos = pub(PointStamped, topics.POSITION)
        self.pub_path = pub(Path, topics.TRUTH_PATH)
        self.pub_marker = pub(Marker, topics.MARKER)
        self.pub_info = pub(String, topics.SIM_INFO)
        self.tf = TransformBroadcaster(self)

        self.path = Path()
        self.path.header.frame_id = self.world
        self.k = 0
        self.n = len(self.data["t"])
        self.info = {"trajectory": trajectory, "seed": seed, "duration_s": duration,
                     "imu_rate_hz": config.imu_rate, "pos_rate_hz": config.meas_rate}
        self.dt = dt
        self.wait_for = str(self.get_parameter("wait_for_node").value)
        self.wait_until = time.monotonic() + float(self.get_parameter("wait_timeout_s").value)
        self.timer = None
        if self.wait_for:
            self.get_logger().info(f"waiting for node {self.wait_for}")
            self.wait_timer = self.create_timer(0.5, self._wait)
        else:
            self._start()

    def _wait(self) -> None:
        if self.wait_for in self.get_node_names():
            self.get_logger().info(f"node {self.wait_for} found")
        elif time.monotonic() < self.wait_until:
            return
        else:
            self.get_logger().warning(f"node {self.wait_for} not found; publishing anyway")
        self.wait_timer.cancel()
        self._start()

    def _start(self) -> None:
        self.t0_ns = self.get_clock().now().nanoseconds
        self.pub_info.publish(String(data=json.dumps({**self.info, "t0_ns": self.t0_ns}, sort_keys=True)))
        self.get_logger().info(f"publishing the flight in real time (t0 = {self.t0_ns} ns)")
        self.timer = self.create_timer(self.dt, self._tick)

    def _stamp(self, t: float):
        return Time(nanoseconds=self.t0_ns + int(round(t * 1e9))).to_msg()

    def _tick(self) -> None:
        k = self.k
        if k >= self.n:
            self.get_logger().info("flight finished; no further messages are published")
            self.timer.cancel()
            return
        d = self.data
        t = float(d["t"][k])
        stamp = self._stamp(t)
        p, q = d["p"][k], d["q"][k]

        pose = PoseStamped()
        pose.header.stamp, pose.header.frame_id = stamp, self.world
        pose.pose = _pose(p, q)
        self.pub_pose.publish(pose)

        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, self.world, self.body
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (
            float(v) for v in p)
        tf.transform.rotation = pose.pose.orientation
        self.tf.sendTransform(tf)

        imu = Imu()
        imu.header.stamp, imu.header.frame_id = stamp, self.body
        imu.orientation_covariance[0] = -1.0  # orientation not provided
        g, a = d["gyro_meas"][k], d["accel_meas"][k]
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = (float(v) for v in g)
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = (
            float(v) for v in a)
        for i in range(3):
            imu.angular_velocity_covariance[4 * i] = self.gyro_var
            imu.linear_acceleration_covariance[4 * i] = self.accel_var
        self.pub_imu.publish(imu)

        if k % self.meas_step == 0:
            j = k // self.meas_step
            if d["meas_available"][j]:
                pt = PointStamped()
                pt.header.stamp, pt.header.frame_id = stamp, self.world
                pt.point.x, pt.point.y, pt.point.z = (float(v) for v in d["z_meas"][j])
                self.pub_pos.publish(pt)
            self.path.poses.append(pose)
        if k % (self.meas_step * 5) == 0:
            self.path.header.stamp = stamp
            self.pub_path.publish(self.path)
        if k % int(self.imu_rate) == 0:
            self._publish_markers(stamp)
        self.k += 1

    def _publish_markers(self, stamp) -> None:
        for m in drone_markers(stamp, self.body):
            self.pub_marker.publish(m)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = QuadrotorSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
