#!/usr/bin/env bash
# Start XPolicyLab's Pi0.5 policy server on a GPU host, listening on localhost only.
# Usage: serve.sh [ROBODOJO_ROOT] [CHECKPOINT_DIR]
#   GPU=<index> picks the GPU (default: the one with the most free memory); PORT defaults to 6000.
# Clients reach it through an SSH tunnel (see tunnel.sh), so the port is never exposed publicly.
set -euo pipefail

ROOT=${1:-/export/code/RoboDojo}
CKPT=${2:-$ROOT/checkpoints/RoboDojo-sim-arx_x5-joint-0}
PORT=${PORT:-6000}
GPU=${GPU:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1 | cut -d, -f1)}

POLICY_DIR=$ROOT/XPolicyLab/policy/Pi_05
cd "$POLICY_DIR"
source openpi/.venv/bin/activate

echo "Pi0.5 on GPU $GPU, port $PORT, checkpoint $CKPT"
# Allocate GPU memory on demand: XPolicyLab's own script reserves a fixed 30% of the GPU.
exec env XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$ROOT:$POLICY_DIR/openpi/src" python ../../setup_policy_server.py --config_path deploy.yml \
  --overrides port="$PORT" host=localhost bench_name=RoboDojo task_name=stack_bowls ckpt_name="$CKPT" \
  env_cfg_type=arx_x5 seed=0 policy_name=Pi_05 action_type=joint action_dim=14
