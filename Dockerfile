FROM osrf/ros:humble-desktop-full

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-psutil \
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-gazebo \
    ros-humble-nav2-bringup \
    ros-humble-twist-mux \
    ros-humble-gazebo-ros \
    ros-humble-gazebo-ros-pkgs \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (SB3, gymnasium, analysis) ───────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ── Workspace ────────────────────────────────────────────────────────────────
WORKDIR /workspace
COPY ros2_ws/src/ ros2_ws/src/

RUN bash -c "source /opt/ros/humble/setup.bash && \
    cd ros2_ws && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1"

# ── Environment ──────────────────────────────────────────────────────────────
ENV TURTLEBOT3_MODEL=waffle
ENV ROS_DOMAIN_ID=0
ENV GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models

RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /workspace/ros2_ws/install/setup.bash" >> /root/.bashrc && \
    echo "export TURTLEBOT3_MODEL=waffle" >> /root/.bashrc

# Porta Gazebo master (comunicação intra-container)
EXPOSE 11345

# ── Entrypoints ──────────────────────────────────────────────────────────────
# Por padrão: abre bash interativo.
# Substitua CMD ao usar docker-compose para serviços específicos.
CMD ["/bin/bash"]
