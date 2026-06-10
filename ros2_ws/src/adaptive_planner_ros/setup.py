from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'adaptive_planner_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yan Santos Leite',
    maintainer_email='leiteyan@discente.ufg.br',
    description='Adaptive planner switching — IC UFG 2026',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'adaptive_switcher_node = adaptive_planner_ros.adaptive_switcher_node:main',
            'rl_controller_node = adaptive_planner_ros.rl_controller_node:main',
            'density_estimator_node = adaptive_planner_ros.density_estimator:main',
        ],
    },
)
