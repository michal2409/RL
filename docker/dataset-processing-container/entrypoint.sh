#!/usr/bin/env bash
set -euo pipefail

printf '%s' "$DOCKER_TOKEN" \
    | docker login --username "$DOCKER_USER" --password-stdin "$DOCKER_REGISTRY"
exec "$@"
