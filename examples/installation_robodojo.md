# Installation: RoboDojo

## 1. RoboDojo

Install [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo#readme) and its assets. In its Python 3.11
environment, from the CleanRUA directory:

```sh
pip install 'gymnasium>=1.2,<2' 'httpx>=0.28,<1' 'pydantic>=2.11,<3' 'tyro>=0.9,<1' \
  'scipy>=1.14' 'tensorboardX>=2.6' tensorboard moviepy
export OMNI_KIT_ACCEPT_EULA=YES   # Otherwise Isaac Sim waits for an interactive prompt.
```

## 2. Model API

A Chat Completions model with vision, tool calling and usage reporting (add `--provider aliyun` for Alibaba Cloud):

```sh
export CLEANRUA_MODEL_NAME=YOUR_VISION_MODEL
export CLEANRUA_BASE_URL=https://YOUR_PROVIDER/v1
export CLEANRUA_API_KEY=YOUR_API_KEY
```

## 3. Pi0.5

Delegate, Monitor and `cap_delegate` need XPolicyLab's Pi0.5 server, which ships inside RoboDojo. One checkpoint
serves every task:

```sh
cd /path/to/RoboDojo/XPolicyLab/policy/Pi_05 && bash install.sh
hf download RoboDojo-Benchmark/RoboDojo --repo-type dataset \
  --revision 269457a3b42d8d0ae69898a601f2c1487c2f826e --local-dir /tmp/robodojo-hf \
  --include "ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/params/*" \
            "ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/assets/*" \
            "ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/_CHECKPOINT_METADATA"
CKPT=/tmp/robodojo-hf/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0
source openpi/.venv/bin/activate
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
PYTHONPATH=/path/to/RoboDojo:$PWD/openpi/src python ../../setup_policy_server.py --config_path deploy.yml \
  --overrides port=6000 host=localhost bench_name=RoboDojo task_name=stack_bowls ckpt_name=$CKPT \
  env_cfg_type=arx_x5 seed=0 policy_name=Pi_05 action_type=joint action_dim=14
```

It listens on port 6000 after about 15 seconds and uses about 8.5 GB of GPU memory; on a 16 GB GPU, stop it before
running Direct or CaP. To serve it from another GPU host, run
[`examples/pi05_server/serve.sh`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/examples/pi05_server/serve.sh) there and [`examples/pi05_server/tunnel.sh`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/examples/pi05_server/tunnel.sh)
here.

## 4. Run

```sh
python -m cleanrua.direct_robodojo --native-root /path/to/RoboDojo --task stack_bowls
python -m cleanrua.delegate_robodojo --native-root /path/to/RoboDojo --task stack_bowls
tensorboard --logdir runs
```

`--task` takes any task in `RoboDojo/task/RoboDojo/config`, and `--help` lists every option. Each run writes
TensorBoard scalars, `trajectory.jsonl`, the images shown to the model, videos and `result.json` to
`runs/<task>__<exp_name>__<seed>__<time>/`. CaP agents run the model's programs without a sandbox.

## 5. Tests

```sh
python -m unittest discover -s tests -q                            # CPU: no simulator or model
python tests/smoke_robodojo.py --native-root /path/to/RoboDojo     # GPU: native control, no model
```
