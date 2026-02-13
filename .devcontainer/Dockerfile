FROM nvcr.io/nvidia/cuda:12.1.1-devel-ubuntu22.04
RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y --no-install-recommends \
        libxau6 libxau6:i386 \
        libxdmcp6 libxdmcp6:i386 \
        libxcb1 libxcb1:i386 \
        libxext6 libxext6:i386 \
        libx11-6 libx11-6:i386 && \
    rm -rf /var/lib/apt/lists/*

### Setup nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
        ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
        ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

COPY .devcontainer/NGC-DL-CONTAINER-LICENSE /

### Install ROS2 Humble
RUN apt-get update && apt-get install -y software-properties-common \
    && apt-add-repository universe \
    && apt-get update && apt-get install -y curl \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y ros-humble-desktop \
    && apt-get install -y python3-colcon-common-extensions python3-rosdep \
    && rosdep init \
    && rosdep update \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y ros-humble-joint-state-publisher-gui

# Required for non-glvnd setups.
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglvnd0 libglvnd0:i386 \
        libgl1 libgl1:i386 \
        libglx0 libglx0:i386 \
        libegl1 libegl1:i386 \
        libgles2 libgles2:i386 && \
    rm -rf /var/lib/apt/lists/*

COPY .devcontainer/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libglvnd-dev libglvnd-dev:i386 \
        libgl1-mesa-dev libgl1-mesa-dev:i386 \
        libegl1-mesa-dev libegl1-mesa-dev:i386 \
        libgles2-mesa-dev libgles2-mesa-dev:i386 && \
    rm -rf /var/lib/apt/lists/*

### Installing basic development tools
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections \
  && apt update \
  && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    git libsparsehash-dev build-essential wget curl ninja-build tmux \
    neovim nano software-properties-common python3-pip python3-dev python3-tk \
    freeglut3-dev \
  && apt-get clean && rm -rf /var/lib/apt


### Install specific version of OMPL, with python bindings
ENV OMPL_VERSION=1.6.0
RUN apt-get update \
    && apt-get install -y g++ pkg-config libboost-serialization-dev libboost-filesystem-dev libboost-system-dev libboost-program-options-dev libboost-test-dev libeigen3-dev libode-dev wget libyaml-cpp-dev \ 
    && apt-get install -y castxml libboost-python-dev libboost-numpy-dev python3-numpy pypy3 \ 
    && pip install pygccxml \
    && pip install pyplusplus
RUN wget https://github.com/ompl/ompl/archive/refs/tags/$OMPL_VERSION.tar.gz \
    && tar zxf $OMPL_VERSION.tar.gz \ 
    && cd ompl-$OMPL_VERSION \ 
    && mkdir -p build/Release && cd build/Release \
    && export CXX=g++ && export MAKEFLAGS="-j `nproc`" \ 
    && cmake ../.. -DOMPL_BUILD_PYBINDINGS=ON -DPYTHON_EXEC=/usr/bin/python3 \
    && make update_bindings \
    && make install


### Install Foxglove Studio
RUN curl -L -o /tmp/foxglove.deb https://get.foxglove.dev/desktop/latest/foxglove-studio-latest-linux-amd64.deb \
  && apt-get update && \
  apt-get install -y /tmp/foxglove.deb && \
  rm /tmp/foxglove.deb
RUN apt-get update && apt-get install -y ros-humble-foxglove-bridge

### Build Foam (collision sphere generation)
RUN git config --global url."https://github.com/".insteadOf git@github.com: \
    && git clone --recursive https://github.com/CoMMALab/foam.git /opt/foam
WORKDIR /opt/foam
RUN cmake -Bbuild -GNinja .
RUN cmake --build build
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

### Install Blender (Desktop GUI + Python module)
# Install Blender 4.2.0 LTS desktop application with GUI
ENV BLENDER_VERSION=4.2.0
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Download and install specific Blender 4.2.0 LTS version
RUN wget -q -O /tmp/blender.tar.xz \
    "https://download.blender.org/release/Blender4.2/blender-${BLENDER_VERSION}-linux-x64.tar.xz" \
    && tar -xf /tmp/blender.tar.xz -C /opt/ \
    && mv /opt/blender-${BLENDER_VERSION}-linux-x64 /opt/blender \
    && ln -s /opt/blender/blender /usr/local/bin/blender \
    && rm /tmp/blender.tar.xz

# Note: bpy (Blender as Python module) version 4.0.0 is installed via requirements.txt 
# (latest available on PyPI, compatible with desktop Blender 4.2 LTS installation)

# Install additional dependencies for Blender Python scripting and GUI
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libxi6 \
#     libxrandr2 \
#     libxxf86vm1 \
#     libxfixes3 \
#     libxcursor1 \
#     libxinerama1 \
#     libxss1 \
#     libgconf-2-4 \
#     libasound2 \
#     libgl1-mesa-glx \
#     libglu1-mesa \
#     && rm -rf /var/lib/apt/lists/*

### Install Python dependencies (overwriting those from Foam)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
  && pip install --no-cache-dir --upgrade -r /tmp/requirements.txt \
       -f https://data.pyg.org/whl/torch-2.1.2+cu121.html \
  && yes | splendor_asset_installer

### Add user
ARG USERNAME=USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


ENV SHELL /bin/bash
ENV GIT_EDITOR=vim
USER $USERNAME
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# MATLAB-like Bash history
RUN printf '# Include system wide settings which are ignored\n# by default if one has its own .inputrc\n$include /etc/inputrc\n\n# Matlab like Bash history\n"\\e[B": history-search-forward\n"\\e[A": history-search-backward\n' > ~/.inputrc && \
    echo -e '\n# Activate inputrc (matlab-like command history)\nexport INPUTRC=~/.inputrc' >> ~/.bashrc

WORKDIR /workspace

# Set Blender startup file (start in scripting mode)
RUN mkdir -p /home/$USERNAME/.config/blender/4.2/config
COPY --chown=$USERNAME:$USERNAME .devcontainer/blender_config/startup.blend /home/$USERNAME/.config/blender/4.2/config/startup.blend

# Copy RViz configuration file
RUN mkdir -p /home/$USERNAME/.rviz2
COPY --chown=$USERNAME:$USERNAME rviz_config/avoid_everything.rviz /home/$USERNAME/.rviz2/default.rviz
