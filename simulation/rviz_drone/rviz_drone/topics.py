"""Single source of the ROS 2 names, rates, QoS settings and colours of this package.

Pure Python (no ROS import), so that it is used by the simulator node, the launch file, the Isaac
Sim subscriber (``../../isaac_render/render_flight.py``) and the tests alike. The tables in
``../README.md`` ("ROS 2 graph") describe the same values.
"""

from __future__ import annotations

from dataclasses import dataclass

WORLD_FRAME = "world"
BODY_FRAME = "base_link"

SIM_NODE = "rviz_drone_sim"
ISAAC_NODE = "isaac_drone_render"


@dataclass(frozen=True)
class Qos:
    reliable: bool = True
    transient_local: bool = False
    depth: int = 10


@dataclass(frozen=True)
class Topic:
    name: str
    type: str  # package/msg/Type
    frame_id: str | None
    rate_hz: float | None  # None: once or on events
    qos: Qos
    reserved: bool = False  # True: estimator and analysis topics, not published by the simulator node


TRUTH_POSE = Topic("/drone/truth/pose", "geometry_msgs/msg/PoseStamped", WORLD_FRAME, 200.0, Qos(depth=10))
IMU = Topic("/drone/imu", "sensor_msgs/msg/Imu", BODY_FRAME, 200.0, Qos(depth=50))
POSITION = Topic("/drone/position", "geometry_msgs/msg/PointStamped", WORLD_FRAME, 10.0, Qos(depth=10))
TRUTH_PATH = Topic("/drone/truth/path", "nav_msgs/msg/Path", WORLD_FRAME, 2.0, Qos(depth=1))
MARKER = Topic("/drone/marker", "visualization_msgs/msg/Marker", BODY_FRAME, 3.0, Qos(depth=10))
SIM_INFO = Topic("/drone/sim/info", "std_msgs/msg/String", None, None, Qos(transient_local=True, depth=1))
TF = Topic("/tf", "tf2_msgs/msg/TFMessage", WORLD_FRAME, 200.0, Qos(depth=100))

PUBLISHED = (TRUTH_POSE, TF, IMU, POSITION, TRUTH_PATH, MARKER, SIM_INFO)

# Estimators and analysis: published by the results replay node (results_node.py) only.
ESTIMATORS = ("kf", "ekf", "eskf", "spare")


def estimate_topics(name: str) -> tuple[Topic, Topic, Topic]:
    if name not in ESTIMATORS:
        raise ValueError(f"unknown estimator {name!r}; expected one of {ESTIMATORS}")
    base = f"/drone/estimate/{name}"
    return (Topic(f"{base}/pose", "geometry_msgs/msg/PoseWithCovarianceStamped", WORLD_FRAME, 200.0,
                  Qos(depth=50), reserved=True),
            Topic(f"{base}/path", "nav_msgs/msg/Path", WORLD_FRAME, 2.0, Qos(depth=1), reserved=True),
            Topic(f"{base}/covariance", "visualization_msgs/msg/Marker", WORLD_FRAME, 10.0, Qos(depth=10),
                  reserved=True))


def estimate_frame(name: str) -> str:
    if name not in ESTIMATORS:
        raise ValueError(f"unknown estimator {name!r}")
    return f"{BODY_FRAME}_{name}"


PEAKS = Topic("/drone/analysis/peaks", "visualization_msgs/msg/MarkerArray", WORLD_FRAME, None,
              Qos(transient_local=True, depth=1), reserved=True)
OUTAGE = Topic("/drone/analysis/outage", "std_msgs/msg/Bool", None, 10.0, Qos(depth=10), reserved=True)
OUTAGE_MARKER = Topic("/drone/analysis/outage_marker", "visualization_msgs/msg/Marker", BODY_FRAME, 10.0,
                      Qos(depth=10), reserved=True)

RESERVED = tuple(t for name in ESTIMATORS for t in estimate_topics(name)) + (PEAKS, OUTAGE, OUTAGE_MARKER)

# Colours, RGB 0-255.
COLOURS = {
    "truth": (25, 255, 0),
    "measurement": (204, 41, 204),
    "kf": (31, 119, 180),
    "ekf": (255, 127, 14),
    "eskf": (23, 190, 207),
    "spare": (255, 215, 0),
    "peak": (214, 39, 40),
    "outage": (150, 150, 150),
    "corruption": (148, 103, 189),  # any corruption window other than an outage (results replay)
}
COVARIANCE_ALPHA = 0.25  # covariance markers use the colour of their filter with this alpha


def rgba(name: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    r, g, b = COLOURS[name]
    return r / 255.0, g / 255.0, b / 255.0, alpha


def qos_profile(topic: Topic):
    """rclpy QoSProfile of a topic (imports rclpy only when called)."""
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST, depth=topic.qos.depth,
        reliability=ReliabilityPolicy.RELIABLE if topic.qos.reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL if topic.qos.transient_local else DurabilityPolicy.VOLATILE)
