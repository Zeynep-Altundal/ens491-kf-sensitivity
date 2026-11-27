"""ROS 2 node: publish one estimator results run in real time on the reserved topics.

The same elements as the Isaac Sim estimator replay (``../../isaac_render/README.md``, "Estimator
replay"), from the same files through ``estimate_view.py``: truth pose, TF and path, drone marker,
and for KF, EKF and ESKF the pose with covariance, path and 3-sigma covariance ellipsoid; peak
markers; position-outage flag and the corruption banner (text marker at the drone). The spare slot
publishes nothing. Topic names, rates, QoS and colours: ``topics.py``.
"""

from __future__ import annotations

import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from . import estimate_view as ev
from . import topics
from .repo_path import repo_root
from .sim_node import drone_markers


def _pose(p, q_wxyz) -> Pose:
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in p)
    pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z = (float(v) for v in q_wxyz)
    return pose


def _rgba(marker: Marker, name: str, alpha: float = 1.0) -> None:
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = topics.rgba(name, alpha)


def covariance_marker(p, P, name: str, stamp, frame: str) -> Marker:
    """3-sigma ellipsoid: a sphere marker scaled to the full axes and turned to the eigenvectors."""
    semi, rot = ev.ellipsoid(P)
    m = Marker(type=Marker.SPHERE, action=Marker.ADD)
    m.header.stamp, m.header.frame_id = stamp, frame
    m.ns, m.id = f"covariance_{name}", 0
    m.pose = _pose(p, ev.rot_to_quat(rot))
    m.scale.x, m.scale.y, m.scale.z = (float(max(2.0 * a, 1e-3)) for a in semi)
    _rgba(m, name, topics.COVARIANCE_ALPHA)
    return m


def peak_markers(view: ev.RunView, time_s: float, stamp, frame: str) -> MarkerArray:
    """One pin (peak colour) with a head in the estimator colour per peak that has started."""
    arr = MarkerArray()
    clear = Marker(action=Marker.DELETEALL)
    clear.header.frame_id = frame
    arr.markers.append(clear)
    for name in ev.SHOWN:
        for i, (t_start, _t_peak, where) in enumerate(ev.visible_peaks(view, name, time_s)):
            stick = Marker(type=Marker.CYLINDER, action=Marker.ADD)
            stick.header.stamp, stick.header.frame_id = stamp, frame
            stick.ns, stick.id = f"peaks_{name}", 2 * i
            stick.pose = _pose(np.asarray(where) + [0.0, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0])
            stick.scale.x, stick.scale.y, stick.scale.z = 0.03, 0.03, 0.4
            _rgba(stick, "peak")
            head = Marker(type=Marker.SPHERE, action=Marker.ADD)
            head.header.stamp, head.header.frame_id = stamp, frame
            head.ns, head.id = f"peaks_{name}", 2 * i + 1
            head.pose = _pose(np.asarray(where) + [0.0, 0.0, 0.42], [1.0, 0.0, 0.0, 0.0])
            head.scale.x = head.scale.y = head.scale.z = 0.12
            _rgba(head, name)
            arr.markers += [stick, head]
    return arr


class ResultsNode(Node):
    def __init__(self) -> None:
        super().__init__("rviz_drone_results")
        self.declare_parameter("scenario", "nominal")
        self.declare_parameter("seed", 0)
        self.declare_parameter("loop", True)
        scenario = str(self.get_parameter("scenario").value)
        seed = int(self.get_parameter("seed").value)
        self.loop = bool(self.get_parameter("loop").value)
        t_load = time.monotonic()
        self.view = ev.load(repo_root(), scenario, seed)
        self.get_logger().info(f"loaded {scenario} seed {seed} in {time.monotonic() - t_load:.1f} s: "
                               f"{json.dumps(self.view.sources)}")
        self.world, self.body = topics.WORLD_FRAME, topics.BODY_FRAME

        def pub(msg_type, topic):
            return self.create_publisher(msg_type, topic.name, topics.qos_profile(topic))

        self.pub_truth = pub(PoseStamped, topics.TRUTH_POSE)
        self.pub_truth_path = pub(Path, topics.TRUTH_PATH)
        self.pub_marker = pub(Marker, topics.MARKER)
        self.pub_info = pub(String, topics.SIM_INFO)
        self.pub_est = {}
        for name in ev.SHOWN:
            pose_t, path_t, cov_t = topics.estimate_topics(name)
            self.pub_est[name] = (pub(PoseWithCovarianceStamped, pose_t), pub(Path, path_t), pub(Marker, cov_t))
        self.pub_peaks = pub(MarkerArray, topics.PEAKS)
        self.pub_outage = pub(Bool, topics.OUTAGE)
        self.pub_banner = pub(Marker, topics.OUTAGE_MARKER)
        self.tf = TransformBroadcaster(self)
        self.dt = float(self.view.t[1] - self.view.t[0])
        self.n = len(self.view.t)
        self._reset()
        self.pub_info.publish(String(data=json.dumps({
            "trajectory": ev.TRAJECTORY, "seed": seed, "scenario": scenario, "source": "results",
            "duration_s": float(self.view.t[-1]), "t0_ns": self.t0_ns}, sort_keys=True)))
        self.timer = self.create_timer(self.dt, self._tick)

    def _reset(self) -> None:
        self.k = 0
        self.t0_ns = self.get_clock().now().nanoseconds
        self.paths = {name: Path() for name in ("truth",) + ev.SHOWN}
        for p in self.paths.values():
            p.header.frame_id = self.world
        self.n_peaks = -1

    def _stamp(self, t: float):
        return Time(nanoseconds=self.t0_ns + int(round(t * 1e9))).to_msg()

    def _tick(self) -> None:
        if self.k >= self.n:
            if not self.loop:
                self.get_logger().info("run finished")
                self.timer.cancel()
                return
            self._reset()
        k, v = self.k, self.view
        t = float(v.t[k])
        stamp = self._stamp(t)
        truth = PoseStamped()
        truth.header.stamp, truth.header.frame_id = stamp, self.world
        truth.pose = _pose(v.truth_p[k], v.truth_q[k])
        self.pub_truth.publish(truth)
        tfs = []
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, self.world, self.body
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (
            float(x) for x in v.truth_p[k])
        tf.transform.rotation = truth.pose.orientation
        tfs.append(tf)
        every_10hz = k % int(round(0.1 / self.dt)) == 0
        every_2hz = k % int(round(0.5 / self.dt)) == 0
        for name in ev.SHOWN:
            pose_pub, path_pub, cov_pub = self.pub_est[name]
            p, P = v.est_p[name][k], v.est_P[name][k]
            msg = PoseWithCovarianceStamped()
            msg.header.stamp, msg.header.frame_id = stamp, self.world
            msg.pose.pose = _pose(p, [1.0, 0.0, 0.0, 0.0])  # position estimate; attitude not shown
            cov = np.zeros((6, 6))
            cov[:3, :3] = P
            msg.pose.covariance = [float(x) for x in cov.flatten()]
            pose_pub.publish(msg)
            etf = TransformStamped()
            etf.header.stamp, etf.header.frame_id = stamp, self.world
            etf.child_frame_id = topics.estimate_frame(name)
            etf.transform.translation.x, etf.transform.translation.y, etf.transform.translation.z = (
                float(x) for x in p)
            etf.transform.rotation.w = 1.0
            tfs.append(etf)
            if every_10hz:
                ps = PoseStamped()
                ps.header.stamp, ps.header.frame_id = stamp, self.world
                ps.pose = msg.pose.pose
                self.paths[name].poses.append(ps)
                cov_pub.publish(covariance_marker(p, P, name, stamp, self.world))
            if every_2hz:
                self.paths[name].header.stamp = stamp
                path_pub.publish(self.paths[name])
        self.tf.sendTransform(tfs)
        if every_10hz:
            self.paths["truth"].poses.append(truth)
            act = ev.active(v.windows, t)
            self.pub_outage.publish(Bool(data=any(w.kind == "outage" for w in act)))
            self.pub_banner.publish(self._banner(act, stamp))
            shown = sum(len(ev.visible_peaks(v, n, t)) for n in ev.SHOWN)
            if shown != self.n_peaks:
                self.pub_peaks.publish(peak_markers(v, t, stamp, self.world))
                self.n_peaks = shown
        if every_2hz:
            self.paths["truth"].header.stamp = stamp
            self.pub_truth_path.publish(self.paths["truth"])
        if k % int(round(1.0 / self.dt)) == 0:
            for m in drone_markers(stamp, self.body):
                self.pub_marker.publish(m)
        self.k += 1

    def _banner(self, act, stamp) -> Marker:
        m = Marker(type=Marker.TEXT_VIEW_FACING)
        m.header.stamp, m.header.frame_id = stamp, self.body
        m.ns, m.id, m.frame_locked = "corruption", 0, True
        if not act:
            m.action = Marker.DELETE
            return m
        m.action = Marker.ADD
        m.text = ev.banner(act)
        m.pose.position.z = 0.8
        m.pose.orientation.w = 1.0
        m.scale.z = 0.25
        _rgba(m, act[0].colour)
        return m


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ResultsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
