# Installation

CleanRUA runs inside the simulator's Python environment. Each agent file lists only what it needs.

## 1. Simulator

Install [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo#readme) and its assets, then use
its Python 3.11 interpreter from the CleanRUA directory:

```sh
pip install 'gymnasium>=1.2,<2' 'httpx>=0.28,<1' 'pydantic>=2.11,<3' 'tyro>=0.9,<1' \
  'scipy>=1.14' 'tensorboardX>=2.6' tensorboard moviepy
export OMNI_KIT_ACCEPT_EULA=YES   # Otherwise Isaac Sim waits for an interactive prompt.
```

## 2. Model API

Use a Chat Completions-compatible vision model with tool calling and token-usage reporting:

```sh
export CLEANRUA_BASE_URL=https://YOUR_PROVIDER/v1
export CLEANRUA_API_KEY=YOUR_API_KEY
```

Add `--provider aliyun` for Alibaba Cloud.

## 3. Direct control

```sh
python -m cleanrua.direct_robodojo --native-root /path/to/RoboDojo --model YOUR_VISION_MODEL
tensorboard --logdir runs
```

`--help` lists every option. Each run writes TensorBoard scalars, `trajectory.jsonl`, the images
shown to the model, videos and `result.json` to `runs/<exp_name>__<seed>__<time>/`.

## 4. Hybrid control with Pi0.5

Pi0.5 runs as XPolicyLab's policy server in its own environment. XPolicyLab ships inside RoboDojo.

```sh
cd /path/to/RoboDojo/XPolicyLab/policy/Pi_05
bash install.sh

# Inference weights and normalizer only (about 12.4 GB; the 32 GB train_state is not needed).
hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
  --revision 269457a3b42d8d0ae69898a601f2c1487c2f826e --local-dir /tmp/robodojo-hf \
  --include "ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/params/*" \
            "ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/assets/*" \
            "ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/_CHECKPOINT_METADATA"
CKPT=/tmp/robodojo-hf/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0   # Or wherever you keep it.

# Allocate GPU memory on demand: Pi0.5 uses about 8.5 GB, so Isaac Sim fits on the same 16 GB GPU.
source openpi/.venv/bin/activate
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/path/to/RoboDojo:$PWD/openpi/src python ../../setup_policy_server.py --config_path deploy.yml \
  --overrides port=6000 host=localhost bench_name=RoboDojo task_name=stack_bowls ckpt_name=$CKPT \
  env_cfg_type=arx_x5 seed=0 policy_name=Pi_05 action_type=joint action_dim=14
```

The server is ready when it listens on port 6000, after about 15 seconds; the first prediction
compiles the model and takes about 10 seconds. XPolicyLab's `setup_eval_policy_server.sh` fixes
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.3`, which fits only on GPUs with at least 32 GB.

Keep the server running. In another terminal, from the CleanRUA directory:

```sh
python -m cleanrua.hybrid_robodojo --native-root /path/to/RoboDojo --model YOUR_VISION_MODEL
```

`--policy-url` selects another server address (default `ws://localhost:6000`).

## 5. Tests

```sh
python -m unittest discover -s tests -q                            # CPU: no simulator or model
python tests/smoke_robodojo.py --native-root /path/to/RoboDojo     # GPU: native control, no model
```

The GPU test scripts capture, hold, a 1 cm lift, gripper close/open, return and finish through the
agents' wrapper. It checks control, not task success.
