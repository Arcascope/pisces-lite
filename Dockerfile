# syntax=docker/dockerfile:1
#
# Minimal pisces-lite image.
#
# Contents:
#   - NVIDIA JAX base (nvcr.io/nvidia/jax) for CUDA userspace + Python 3.12
#   - git + cmake + build-essential (required to build senpy from GitHub)
#   - pisces-lite itself, which pulls numpy/scipy/sklearn/pandas/matplotlib/
#     seaborn/tqdm and senpy as dependencies
#
# Explicitly NOT installed here:
#   - TensorFlow, Keras, YDF, JAX/flax/optax as pisces-lite deps, pisces2
#
# Framework consumers (e.g. autofish-jax) layer flax/optax or other trainer
# packages on top of this image. The base already ships JAX itself.
#
# Build:
#   bash build.sh              # tags pisces-lite:latest
#   bash build.sh v0.0.1       # tags pisces-lite:v0.0.1
#   NO_CACHE=true bash build.sh

FROM nvcr.io/nvidia/jax:26.03-py3

ARG USERNAME=ubuntu
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN if ! getent group ${USERNAME} > /dev/null; then \
      groupadd -g ${GROUP_ID} ${USERNAME}; \
    fi && \
    if ! getent passwd ${USERNAME} > /dev/null; then \
      useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USERNAME}; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /pisces-lite
COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir .

USER ${USERNAME}
WORKDIR /home/${USERNAME}/workspace

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

CMD ["/bin/bash"]

LABEL runtime.options.gpu="--gpus all"
LABEL runtime.options.ipc="--ipc=host"
LABEL runtime.example="docker run -it --gpus all --ipc=host pisces-lite:latest"
