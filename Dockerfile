# syntax=docker/dockerfile:1
FROM ghcr.io/neggles/tensorpods/base:cu121-torch201

# update apt
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get -y update

# Install dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update \
  && apt-get -y install --no-install-recommends \
    aria2 \
    liblz4-tool \
    libunwind-dev \
    lz4 \
    zip \
  && apt-get clean

# Install Node.js - do this first since it's not likely to change much
ARG NODE_MAJOR=18
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/nodesource.gpg \
  && echo "deb [signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_${NODE_MAJOR}.x nodistro main" \
        | tee /etc/apt/sources.list.d/nodesource.list \
  && apt-get -y update \
  && apt-get install -y nodejs

# Add jupyter stuff for gradient
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --upgrade \
      jupyterlab \
      numpy \
      matplotlib \
      ipython \
      ipykernel \
      ipywidgets \
      cython \
      tqdm \
      gdown \
      pillow

# Add some jupyter extensions
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --upgrade \
    jupyter_contrib_nbextensions \
    jupyterlab-git \
    jupyterlab-widgets \
    jupyter-nbextensions-configurator

# Add some gradient stuff
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install --upgrade gradient-utils

# Add A1111 dependencies (cursed way to do this)
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    python3 -m pip install \
      absl-py==1.4.0 \
      accelerate==0.21.0 \
      addict==2.4.0 \
      aenum==3.1.15 \
      aiofiles==23.2.1 \
      aiohttp==3.8.5 \
      aiosignal==1.3.1 \
      altair==5.1.1 \
      antlr4-python3-runtime==4.9.3 \
      anyio==3.7.1 \
      astunparse==1.6.3 \
      async-timeout==4.0.3 \
      basicsr==1.4.2 \
      blendmodes==2022 \
      boltons==23.0.0 \
      cachetools==5.3.1 \
      clean-fid==0.1.35 \
      click==8.1.7 \
      coloredlogs==15.0.1 \
      cssselect2==0.7.0 \
      deepdanbooru==1.0.2 \
      deprecation==2.1.0 \
      diffusers==0.20.1 \
      dill==0.3.7 \
      einops==0.4.1 \
      facexlib==0.3.0 \
      fastapi==0.94.0 \
      ffmpy==0.3.1 \
      filterpy==1.4.5 \
      flatbuffers==23.5.26 \
      frozenlist==1.4.0 \
      fsspec==2023.9.0 \
      ftfy==6.1.1 \
      future==0.18.3 \
      fvcore==0.1.5.post20221221 \
      gast==0.4.0 \
      gfpgan==1.3.8 \
      GitPython==3.1.32 \
      google-auth==2.23.0 \
      google-auth-oauthlib==1.0.0 \
      google-pasta==0.2.0 \
      gradio==3.41.2 \
      gradio_client==0.5.0 \
      greenlet==2.0.2 \
      grpcio==1.58.0 \
      h11==0.12.0 \
      h5py==3.9.0 \
      httpcore==0.15.0 \
      httpx==0.24.1 \
      huggingface-hub==0.17.1 \
      humanfriendly==10.0 \
      imageio==2.31.3 \
      importlib-metadata==6.8.0 \
      importlib-resources==6.0.1 \
      inflection==0.5.1 \
      iopath==0.1.9 \
      joblib==1.3.2 \
      jsonmerge==1.8.0 \
      keras==2.13.1 \
      kornia==0.6.7 \
      lark==1.1.2 \
      lazy_loader==0.3 \
      libclang==16.0.6 \
      lightning-utilities==0.9.0 \
      llvmlite==0.40.1 \
      lmdb==1.4.1 \
      lpips==0.1.4 \
      Markdown==3.4.4 \
      markdown-it-py==3.0.0 \
      mdurl==0.1.2 \
      mediapipe==0.10.5 \
      multidict==6.0.4 \
      natsort==8.4.0 \
      numba==0.57.1 \
      numpy==1.23.5 \
      oauthlib==3.2.2 \
      omegaconf==2.2.3 \
      onnxruntime-gpu==1.15.1 \
      open-clip-torch==2.20.0 \
      opencv-contrib-python==4.8.0.76 \
      opencv-python==4.8.0.76 \
      opencv-python-headless==4.8.0.76 \
      opt-einsum==3.3.0 \
      orjson==3.9.7 \
      pandas==2.1.0 \
      piexif==1.1.3 \
      pilgram==1.2.1 \
      portalocker==2.7.0 \
      protobuf==3.20.3 \
      py-cpuinfo==9.0.0 \
      pyasn1==0.5.0 \
      pyasn1-modules==0.3.0 \
      pycloudflared==0.2.0 \
      pydantic==1.10.12 \
      pydub==0.25.1 \
      pyfunctional==1.4.3 \
      python-dotenv==1.0.0 \
      python-multipart==0.0.6 \
      pytorch-lightning==1.9.4 \
      pytz==2023.3.post1 \
      PyWavelets==1.4.1 \
      realesrgan==0.3.0 \
      regex==2023.8.8 \
      reportlab==4.0.4 \
      requests-oauthlib==1.3.1 \
      resize-right==0.0.2 \
      rich==13.5.2 \
      rsa==4.9 \
      safetensors==0.3.1 \
      scikit-image==0.21.0 \
      scikit-learn==1.3.0 \
      scipy==1.11.2 \
      seaborn==0.12.2 \
      semantic-version==2.10.0 \
      sentencepiece==0.1.99 \
      sounddevice==0.4.6 \
      SQLAlchemy==2.0.20 \
      starlette==0.26.1 \
      svglib==1.5.1 \
      tabulate==0.9.0 \
      tb-nightly==2.15.0a20230914 \
      tensorboard==2.13.0 \
      tensorboard-data-server==0.7.1 \
      tensorflow==2.13.0 \
      tensorflow-estimator==2.13.0 \
      tensorflow-io-gcs-filesystem==0.34.0 \
      termcolor==2.3.0 \
      threadpoolctl==3.2.0 \
      tifffile==2023.8.30 \
      timm==0.9.2 \
      tokenizers==0.13.3 \
      tomesd==0.1.3 \
      toolz==0.12.0 \
      torchdiffeq==0.2.3 \
      torchmetrics==1.1.2 \
      torchsde==0.2.5 \
      trampoline==0.1.2 \
      transformers==4.30.2 \
      typing_extensions==4.5.0 \
      tzdata==2023.3 \
      ultralytics==8.0.180 \
      urllib3==1.26.16 \
      uvicorn==0.23.2 \
      websockets==11.0.3 \
      Werkzeug==2.3.7 \
      wrapt==1.15.0 \
      yacs==0.1.8 \
      yapf==0.40.1 \
      yarl==1.9.2 \
      zipp==3.16.2

# CUDA-related
ENV PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.9,max_split_size_mb:512"
ENV TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
ENV CUDA_MODULE_LOADING=LAZY
ENV TCMALLOC_AGGRESSIVE_DECOMMIT=t

SHELL ["/usr/bin/env", "bash", "-l"]

# expose ports
EXPOSE 6006 8888
CMD ["/bin/bash", "-l"]
