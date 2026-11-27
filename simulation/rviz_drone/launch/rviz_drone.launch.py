"""Start the quadrotor simulation node and, optionally, Isaac Sim (isaac:=true) and RViz (rviz:=true).

With results:=SCENARIO the estimator replay starts instead: the node ``results_node`` publishes one
results run (truth, estimates, covariance, peaks, corruption) on the reserved topics, and Isaac Sim
(if started) shows the same run offline (``render_flight.py --results``), both in real time from
their own start.

Isaac Sim 6.0.1 runs from its own pip installation through ``simulation/isaac_render/isaac_python.sh``
(isolated per-user locations, bundled ROS 2 Jazzy libraries, see ``simulation/isaac_render/README.md``);
the repository root is taken from ``RVIZ_DRONE_REPO_ROOT`` (set by ``env.sh``). When Isaac Sim is
started, the simulation node waits until the Isaac Sim node is in the graph before it publishes.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from rviz_drone import topics


def _nodes(context):
    trajectory = LaunchConfiguration("trajectory").perform(context)
    seed = LaunchConfiguration("seed").perform(context)
    isaac = LaunchConfiguration("isaac").perform(context).lower() in ("true", "1")
    results = LaunchConfiguration("results").perform(context)
    share = get_package_share_directory("rviz_drone")
    repo = os.environ.get("RVIZ_DRONE_REPO_ROOT", "")
    ws_home = os.path.join(repo, "simulation", "rviz_drone_ws", "ros_home") if repo else ""

    if results:
        actions = [Node(
            package="rviz_drone",
            executable="results_node",
            name="rviz_drone_results",
            output="screen",
            parameters=[{"scenario": results, "seed": ParameterValue(int(seed), value_type=int)}],
        )]
    else:
        actions = [Node(
            package="rviz_drone",
            executable="sim_node",
            name=topics.SIM_NODE,
            output="screen",
            parameters=[{"seed": ParameterValue(int(seed), value_type=int),
                         "trajectory": trajectory,
                         "wait_for_node": topics.ISAAC_NODE if isaac else ""}],
        )]
    rviz_env = {"HOME": ws_home} if ws_home else {}  # RViz writes ~/.rviz2; keep it in the workspace
    actions.append(Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(share, "rviz", "rviz_drone.rviz")],
        condition=IfCondition(LaunchConfiguration("rviz")),
        additional_env=rviz_env,
    ))
    if isaac:
        if not repo:
            raise RuntimeError("RVIZ_DRONE_REPO_ROOT is not set; source simulation/rviz_drone/env.sh")
        render = os.path.join(repo, "simulation", "isaac_render")
        mode = (["--results", results, "--seed", seed] if results else
                ["--source", "ros2", "--trajectory", trajectory])
        actions.append(ExecuteProcess(
            cmd=[os.path.join(render, "isaac_python.sh"), os.path.join(render, "render_flight.py"),
                 *mode, "--gui", "--camera", LaunchConfiguration("isaac_camera").perform(context)],
            name="isaac_sim",
            output="screen",
            additional_env={"ISAACSIM_ROS2": "0" if results else "1",
                            "ISAACSIM601_ROOT": os.environ.get(
                                "ISAACSIM601_ROOT", os.path.expanduser("~/isaacsim-6.0.1-ens491")),
                            "OMNI_KIT_ACCEPT_EULA": "YES"},
            sigterm_timeout="20",
        ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true", description="start RViz"),
        DeclareLaunchArgument("isaac", default_value="true", description="start Isaac Sim 6.0.1 (GUI)"),
        DeclareLaunchArgument("isaac_camera", default_value="chase",
                              description="Isaac Sim camera: chase, overview or topdown"),
        DeclareLaunchArgument("seed", default_value="0", description="seed of the sensor errors"),
        DeclareLaunchArgument("results", default_value="",
                              description="estimator replay of this results scenario (e.g. outage_40s_calm)"),
        DeclareLaunchArgument("trajectory", default_value="mission",
                              description="reference trajectory: mission or figure_eights"),
        OpaqueFunction(function=_nodes),
    ])
