#!/bin/bash
set -e

NO_CACHE=${NO_CACHE:-false}

if [ "$NO_CACHE" = "true" ]; then
    echo "Building without cache..."
    CACHE_ARG="--no-cache"
else
    CACHE_ARG=""
fi

docker buildx build \
    $CACHE_ARG \
    -t pisces-lite:"${1:-latest}" \
    -f ./Dockerfile \
    "$PWD"
