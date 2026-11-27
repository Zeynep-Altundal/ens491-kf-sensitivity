from glob import glob

from setuptools import setup

package_name = "rviz_drone"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Quadrotor simulation with synthetic IMU and position measurements, visualised in RViz.",
    license="OPEN",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "sim_node = rviz_drone.sim_node:main",
            "results_node = rviz_drone.results_node:main",
        ],
    },
)
