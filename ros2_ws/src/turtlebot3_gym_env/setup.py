from setuptools import find_packages, setup

package_name = "turtlebot3_gym_env"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "gymnasium", "stable-baselines3", "numpy"],
    zip_safe=True,
    maintainer="Yan Santos Leite",
    maintainer_email="santosleiteyan@icloud.com",
    description="Gymnasium environment over ROS2/Gazebo for TurtleBot3 Waffle",
    license="MIT",
    entry_points={
        "console_scripts": [
            "smoketest = turtlebot3_gym_env.gazebo_gym_env:_smoketest",
        ],
    },
)
