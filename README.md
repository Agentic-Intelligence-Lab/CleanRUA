# <img alt="" src="docs/logo.jpg" height="44" align="top" /> CleanRUA

CleanRUA provides clean, single-file implementations of robot-use agents for research.
Inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl), each file keeps an agent's
prompt, tools and interaction loop together, on top of a plain Gymnasium environment.

---
### Quick start

Prepare the environment named in an agent's file header, then from this directory:

```sh
pip install 'gymnasium>=1.2,<2' 'httpx>=0.28,<1' 'pydantic>=2.11,<3' 'tyro>=0.9,<1' \
  'scipy>=1.14' 'tensorboardX>=2.6' tensorboard moviepy

export CLEANRUA_BASE_URL=https://YOUR_PROVIDER/v1 CLEANRUA_API_KEY=...
python -m cleanrua.direct_robodojo --help
tensorboard --logdir runs
```

Each run saves TensorBoard scalars, a JSONL trajectory, the images shown to the model,
videos and the official result under `runs/`.

| Agent | Idea |
|---|---|
| [direct_robodojo.py](cleanrua/direct_robodojo.py) | A vision-language model controls the robot through motion tools. |
| [hybrid_robodojo.py](cleanrua/hybrid_robodojo.py) | The model also delegates local phases to a vision-language-action model. |

> [!TIP]
> 1. To try your own idea, add a `@Tool` method to an agent's wrapper or change its `RUAgent`.
> 2. Environments are plain `gym.Env`s without tools or models; agents add them with a `gym.Wrapper`.
> 3. A finished tool call is not task success; results report the environment's own verdict.

### Tests

```sh
python -m unittest discover -s tests -q    # CPU only, no simulator or model
```

### Acknowledgements

CleanRUA is inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl) and
[RoboSimStudio](https://github.com/robosimstudio/robosimstudio), and follows the tool design of
[uni-robot-agent](https://github.com/Open-RUA/uni-robot-agent). Environments and action models come from
[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) and
[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab). Thanks to their authors.

### License

CleanRUA is released under the [MIT License](LICENSE).
