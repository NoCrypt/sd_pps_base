# syntax=docker/dockerfile:1
# Build the main image
FROM python:3.10-bookworm AS base

# Set shell
SHELL ["/bin/bash", "-ceuxo", "pipefail"]

# env stuff idk
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_PRIORITY=critical
ENV PIP_PREFER_BINARY=1

# Set args
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# make pip STFU about being root
ENV PIP_ROOT_USER_ACTION=ignore
ENV _PIP_LOCATIONS_NO_WARN_ON_MISMATCH=1

# torch architecture list for from-source builds
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# set up apt to cache packages
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get -y update

# Install dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update \
  && apt-get -y install --no-install-recommends \
    apt-transport-https \
    apt-utils \
    build-essential \
    ca-certificates \
    curl \
    fonts-dejavu-core \
    git \
    gnupg2 \
    jq \
    libgoogle-perftools-dev \
    moreutils \
    nano \
    netbase \
    pkg-config \
    procps \
    rsync \
    sudo \
    unzip \
    wget \
    aria2 \
    liblz4-tool \
    libunwind-dev \
    lz4 \
  && apt-get clean

# Get nVidia repo key and add to apt sources
ARG CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64"
ARG CUDA_REPO_KEY="https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/3bf863cc.pub"
RUN curl -fsSL ${CUDA_REPO_KEY} \
    | gpg --dearmor -o /etc/apt/trusted.gpg.d/cuda.gpg \
  && echo "deb ${CUDA_REPO_URL} /" >/etc/apt/sources.list.d/cuda.list

# enable contrib and non-free repos
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
  sed -i 's/Components: main$/Components: main contrib non-free/' /etc/apt/sources.list.d/debian.sources \
  && apt-get update

# add nVidia repo apt pin to prevent kernel driver installation
COPY cuda-repo-pin /etc/apt/preferences.d/cuda-repo-pin

# PATH
ENV PATH=$PATH:/usr/local/cuda/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Install CUDNN
ARG CUDA_VERSION="12.1"
ARG CUDNN_VERSION="8.9.3.28-1"
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update \
  && apt-get -y install --no-install-recommends \
    libcudnn8=${CUDNN_VERSION}*cuda${CUDA_VERSION} \
    libcudnn8-dev=${CUDNN_VERSION}*cuda${CUDA_VERSION} \
  && apt-get clean

# Install other CUDA libraries
ARG CUDA_RELEASE="12-1"
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update \
  && apt-get -y install --no-install-recommends \
    cuda-libraries-${CUDA_RELEASE} \
    cuda-compiler-${CUDA_RELEASE} \
    cuda-nvcc-${CUDA_RELEASE} \
    libgl1 \
    libgl-dev \
    libglx-dev \
  && apt-get clean

# Update pip and wheel, but *not* setuptools. debian setuptools has some
# modifications & replacing it with a pypi one breaks things
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install -U pip wheel

# Install PyTorch
ARG TORCH_VERSION="2.0.1+cu118"
ARG TORCH_INDEX="https://download.pytorch.org/whl/cu118"
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install torch==${TORCH_VERSION} torchvision --extra-index-url ${TORCH_INDEX}

# add the nVidia python index
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install nvidia-pyindex

# start of my nonesense ------------

# install more things
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --upgrade jupyterlab numpy matplotlib ipython ipykernel ipywidgets cython tqdm gdown pillow

# install jupyter stuff
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --upgrade jupyter_contrib_nbextensions jupyterlab-git

# install NodeJS 18.x
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    curl -sL https://deb.nodesource.com/setup_18.x | bash  \
  && apt-get install -y --no-install-recommends nodejs


# CUDA-related
ENV CUDA_MODULE_LOADING=LAZY
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
ENV PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.9,max_split_size_mb:512"
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV NVIDIA_REQUIRE_CUDA="cuda>=11.8 driver>=450"

# expose ports
EXPOSE 6006 8888

# start jupyter
CMD ["/bin/bash" "-c" "jupyter lab --allow-root --ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True"]