# Installation: DexJoCo

## 1. DexJoCo

Install [DexJoCo](https://github.com/brave-eai/dexjoco#installation) and its `dexjoco` conda environment, then add
CleanRUA's requirements:

```sh
conda activate dexjoco
pip install 'httpx>=0.28,<1' 'pydantic>=2.11,<3' 'tyro>=0.9,<1' 'tensorboardX>=2.6' moviepy
```

Images render with EGL unless `MUJOCO_GL` is set.

## 2. Model API

A Chat Completions model with vision, tool calling and usage reporting (add `--provider aliyun` for Alibaba Cloud):

```sh
export CLEANRUA_MODEL_NAME=YOUR_VISION_MODEL
export CLEANRUA_BASE_URL=https://YOUR_PROVIDER/v1
export CLEANRUA_API_KEY=YOUR_API_KEY
```

## 3. Pi0.5

Every DexJoCo agent needs the multi-task Pi0.5 checkpoint from [DexJoCo-Pi05](https://huggingface.co/DexJoCo/DexJoCo-Pi05),
served by OpenPI in DexJoCo's `openpi` environment. One checkpoint serves every task:

```sh
cd /path/to/dexjoco/openpi && bash install.bash && conda activate openpi
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/serve_policy.py --port=6001 policy:checkpoint \
  --policy.config=multi_task --policy.dir=/path/to/pi05_dexjoco_multi_task
```

It uses about 8.6 GB of GPU memory and listens on all interfaces, so keep port 6001 closed to other machines. To
serve it from another GPU host, run [`examples/pi05_server/serve_dexjoco.sh`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/examples/pi05_server/serve_dexjoco.sh)
there and forward the port with `PORT=6001 LOCAL_PORT=6001 examples/pi05_server/tunnel.sh` here.

## 4. Run

```sh
python -m cleanrua.delegate_dexjoco --native-root /path/to/dexjoco --task pick_bucket
python -m cleanrua.monitor_dexjoco --native-root /path/to/dexjoco --task pick_bucket
python -m cleanrua.cap_delegate_dexjoco --native-root /path/to/dexjoco --task pick_bucket
```

`--task` takes any task in `dexjoco/configs/multi_task`, `--seed` picks the scene, and `--policy-url` defaults to
`ws://localhost:6001`. Runs are written as for RoboDojo, to `runs/<task>__<exp_name>__<seed>__<time>/`.

## 5. Tests

```sh
python -m unittest discover -s tests -q   # CPU: no simulator or model
```
