"""Tests of the topic, frame and colour table (topics.py) and its use in the RViz configuration."""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import pytest

from rviz_drone import topics

PACKAGE = Path(__file__).resolve().parents[1]


def test_topic_names_are_unique_and_reserved_ones_are_not_published():
    names = [t.name for t in topics.PUBLISHED + topics.RESERVED]
    assert len(names) == len(set(names))
    assert all(not t.reserved for t in topics.PUBLISHED)
    assert all(t.reserved for t in topics.RESERVED)
    assert all(re.fullmatch(r"(/[a-z0-9_]+)+", n) for n in names)


def test_every_estimator_has_pose_path_and_covariance_topics_and_a_frame():
    for name in topics.ESTIMATORS:
        pose, path, cov = topics.estimate_topics(name)
        assert pose.name == f"/drone/estimate/{name}/pose"
        assert pose.type == "geometry_msgs/msg/PoseWithCovarianceStamped"
        assert path.type == "nav_msgs/msg/Path" and cov.type == "visualization_msgs/msg/Marker"
        assert topics.estimate_frame(name) == f"base_link_{name}"
    with pytest.raises(ValueError):
        topics.estimate_topics("ukf")


def test_colours_are_distinct():
    for (a, ca), (b, cb) in itertools.combinations(topics.COLOURS.items(), 2):
        assert sum((x - y) ** 2 for x, y in zip(ca, cb)) ** 0.5 > 60, (a, b)
    assert topics.rgba("kf", 0.25)[3] == 0.25


def test_rviz_configuration_uses_the_reserved_topics_and_colours():
    text = (PACKAGE / "rviz" / "rviz_drone.rviz").read_text()
    for name in topics.ESTIMATORS:
        r, g, b = topics.COLOURS[name]
        assert re.search(rf"Color: {r}; {g}; {b}\n\s+Topic:\n\s+Value: /drone/estimate/{name}/path", text), name
    r, g, b = topics.COLOURS["truth"]
    assert f"Color: {r}; {g}; {b}" in text
    r, g, b = topics.COLOURS["measurement"]
    assert f"Color: {r}; {g}; {b}" in text


def test_the_sim_node_uses_the_table():
    text = (PACKAGE / "rviz_drone" / "sim_node.py").read_text()
    assert '"/drone/' not in text
    for name in ("TRUTH_POSE", "IMU", "POSITION", "TRUTH_PATH", "MARKER", "SIM_INFO"):
        assert f"topics.{name}" in text
