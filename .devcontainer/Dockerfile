FROM osrf/ros:humble-desktop

USER root

ARG USERNAME=ros_user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && usermod -aG video,dialout $USERNAME \
    && apt-get update && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install all your system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    v4l-utils \
    ros-humble-vision-opencv \
    ros-humble-cv-bridge \
    ros-humble-v4l2-camera \
    libcamera-dev \
    ros-humble-vision-msgs \
    ros-humble-image-transport \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# Install MediaPipe via pip
RUN pip3 install mediapipe

# Install Hardware Libraries
# Note: numpy<2 required for cv_bridge compatibility with ROS2 Humble
RUN pip3 install pigpio opencv-python
RUN pip3 uninstall -y numpy && pip3 install "numpy<2"

# Install x11-apps for GUI test
RUN apt-get update && apt-get install -y x11-apps && rm -rf /var/lib/apt/lists/*

# Set permissions for SSH key usage
RUN chmod 700 /root/.ssh && chmod 600 /root/.ssh/* || true

# Add bashrc aliases and ROS2 sourcing for root
RUN echo "\n# Custom FaceLock Aliases and ROS2 setup" >> /root/.bashrc \
    && echo "alias sbrc='source ~/.bashrc'" >> /root/.bashrc \
    && echo "alias cbrc='code ~/.bashrc'" >> /root/.bashrc \
    && echo "alias colconbuild='cd /workspaces/FaceLock && colcon build && source /workspaces/FaceLock/install/setup.bash'" >> /root/.bashrc \
    && echo "alias cleancolconbuild='cd /workspaces/FaceLock && rm -rf build/ install/ log/ && colcon build && source /workspaces/FaceLock/install/setup.bash'" >> /root/.bashrc \
    && echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "source /workspaces/FaceLock/install/setup.bash" >> /root/.bashrc

# Switch to the restricted user for development
USER $USERNAME
WORKDIR /workspaces/FaceLock

# Add bashrc aliases and ROS2 sourcing for development user
RUN echo "\n# Custom FaceLock Aliases and ROS2 setup" >> /home/$USERNAME/.bashrc \
    && echo "alias sbrc='source ~/.bashrc'" >> /home/$USERNAME/.bashrc \
    && echo "alias cbrc='code ~/.bashrc'" >> /home/$USERNAME/.bashrc \
    && echo "alias colconbuild='cd /workspaces/FaceLock && colcon build && source /workspaces/FaceLock/install/setup.bash'" >> /home/$USERNAME/.bashrc \
    && echo "alias cleancolconbuild='cd /workspaces/FaceLock && rm -rf build/ install/ log/ && colcon build && source /workspaces/FaceLock/install/setup.bash'" >> /home/$USERNAME/.bashrc \
    && echo "alias facelock='cd /workspaces/FaceLock && ros2 launch face_lock startup.launch.py'" >> /home/$USERNAME/.bashrc \
    && echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "source /workspaces/FaceLock/install/setup.bash" >> /home/$USERNAME/.bashrc
