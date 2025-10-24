# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# The image below is a pinned version of nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 (from Jan 2018)
# If updating the base image, be sure to test on GPU since it has broken in the past.
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]

##########################################################
### System dependencies
##########################################################

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    curl \
    git \
    # libav-tools \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    net-tools \
    software-properties-common \
    swig \
    unzip \
    vim \
    wget \
    xpra \
    xserver-xorg-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8

COPY ./azure/files/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./azure/files/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY ./environment.yml /opt/environment.yml

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

##########################################################
### gsutil
##########################################################
# RUN curl -sSL https://sdk.cloud.google.com | bash

# ENV PATH $PATH:/root/google-cloud-sdk/bin

##########################################################
### MuJoCo
##########################################################
# Note: ~ is an alias for /root
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN ln -s /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}
COPY ./azure/files/mjkey.txt /root/.mujoco

##########################################################
### Example Python Installation
##########################################################
# ENV PATH /opt/conda/bin:$PATH
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
#     /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
#     rm /tmp/miniconda.sh && \
#     ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#     echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

# RUN . /opt/conda/etc/profile.d/conda.sh && \
#     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
#     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
#     conda update -y --name base conda && \
#     conda clean --all -y
# PyTorch with CUDA 11.1 support
--find-links https://download.pytorch.org/whl/torch_stable.html

# Core dependencies
numpy
gymnasium>=0.28.0
mujoco-py==2.1.2.14
matplotlib>=3.5.0
torch==1.9.1+cu111
typed-argument-parser

# D4RL dataset
git+https://github.com/Farama-Foundation/d4rl@f2a05c0d66722499bf8031b094d9af3aea7c372b#egg=d4rl

# Image and video processing
scikit-image>=0.20.0
scikit-video>=1.1.11
pillow

# Development and utilities
gitpython
einops
tqdm
pandas
wandb
humanfriendly>=4.7

# JAX ecosystem
flax>=0.3.5
jax>=0.2.21

# Distributed computing
ray>=2.31.0

# Google Cloud utilities
crcmod
google-api-python-client
cryptography
gdown

# Custom package
git+https://github.com/JannerM/doodad.git@janner

# RUN git config --global url."https://".insteadOf git://
# RUN conda env create -f /opt/environment.yml
# ENV PATH /opt/conda/envs/diffusion/bin:$PATH

RUN apt -get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./azure/files/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-build-isolation -r /tmp/requirements.txt

##########################################################
### gym sometimes has this patchelf issue
##########################################################
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

# RUN apt-get install -y patchelf
# RUN pip install gym[all]==0.12.5

# RUN echo "source activate /opt/conda/envs/diffusion && export PYTHONPATH=$PYTHONPATH:/home/code && export CUDA_VISIBLE_DEVICES=0" >> ~/.bashrc
RUN source ~/.bashrc
