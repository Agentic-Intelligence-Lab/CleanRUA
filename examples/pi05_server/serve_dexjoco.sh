#!/usr/bin/env bash
# Start DexJoCo's multi-task Pi0.5 policy server (OpenPI websocket) on a GPU host, listening on localhost only.
# Usage: serve_dexjoco.sh [DEXJOCO_ROOT] [CHECKPOINT_DIR]
#   GPU=<index> picks the GPU (default: the one with the most free memory); PORT defaults to 6001.
# Clients reach it through an SSH tunnel (tunnel.sh with PORT=6001 LOCAL_PORT=6001).
set -euo pipefail

ROOT=${1:-/export/code/dexjoco}
CKPT=${2:-/data/models/pi05_dexjoco_multi_task}
PORT=${PORT:-6001}
GPU=${GPU:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1 | cut -d, -f1)}

cd "$ROOT/openpi"  # The training-config loader reads config.yaml from here.
echo "DexJoCo Pi0.5 (multi_task) on GPU $GPU, port $PORT, checkpoint $CKPT"
exec env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 CUDA_VISIBLE_DEVICES="$GPU" \
  .venv/bin/python scripts/serve_policy.py --port="$PORT" policy:checkpoint --policy.config=multi_task --policy.dir="$CKPT"
