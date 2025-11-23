FROM osrf/ros:humble-desktop-full-ubuntu22.04

# Instalar dependências básicas
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Configurar workspace
WORKDIR /workspace
RUN mkdir -p ros2_ws/src

# Configurar environment ROS
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Expor porta para possível Gazebo
EXPOSE 11345

CMD ["/bin/bash"]
