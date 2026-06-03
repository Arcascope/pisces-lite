#!/bin/bash
set -e

CPU_BUILD=${CPU_BUILD:-false}
NO_CACHE=${NO_CACHE:-false}

if [ "$CPU_BUILD" = "true" ]; then
    echo "Building CPU-only version..."
    CPU_SUFFIX="-cpu"
    DOCKER_SUFFIX=".cpu"
else
    CPU_SUFFIX=""
    DOCKER_SUFFIX=""
fi

if [ "$NO_CACHE" = "true" ]; then
    echo "Building without cache..."
    CACHE_ARG="--no-cache"
else
    CACHE_ARG=""
fi


docker buildx build \
    $CACHE_ARG \
    -t "pisces-lite${CPU_SUFFIX}":"${1:-latest}" \
    -f ./"Dockerfile${DOCKER_SUFFIX}" \
    "$PWD"
