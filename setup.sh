#!/usr/bin/env bash
# Create the Python environment for training or evaluation.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

export UV_CONCURRENT_DOWNLOADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

case "${1:-train}" in
    train)
        uv sync --locked --python 3.11
        ;;
    eval)
        uv sync --locked --python 3.11 --extra eval
        uv run --no-sync python scripts/proteins/download_models.py
        ;;
    *)
        echo "Usage: ./setup.sh [train|eval]" >&2
        exit 2
        ;;
esac
