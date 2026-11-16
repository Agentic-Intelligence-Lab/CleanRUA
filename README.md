# <img alt="" src="docs/logo.jpg" height="44" align="top" /> CleanRUA

[Agent implementation](direct_robodojo.py) | [RoboDojo wrapper](cleanrua_utils/robodojo_wrappers.py)

CleanRUA is a clean, readable implementation of robot-use agents for research
and experimentation. Vision-language models observe a scene, call robot tools,
and use execution feedback to choose the next action.

Inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl), each agent implementation
keeps its prompts, tools, and interaction loop together, with small wrappers for
the underlying simulation environment.

---
### Quick start

Start with [Direct control on RoboDojo](direct_robodojo.py). Follow the
[official RoboDojo installation guide](https://github.com/RoboDojo-Benchmark/RoboDojo#readme)
to prepare its Python 3.11 environment and assets, then use that same interpreter
from the CleanRUA directory:

```sh
python -m pip install 'httpx>=0.28,<1' 'jsonschema>=4.23,<5' 'scipy>=1.14'

# Check native capture and motion without a model.
python direct_robodojo.py --native-root /path/to/RoboDojo --smoke-test --output-dir ./runs

# Set CLEANRUA_API_KEY in your shell before running an agent.
python direct_robodojo.py --native-root /path/to/RoboDojo \
  --base-url https://YOUR_PROVIDER/v1 --model YOUR_VISION_MODEL --output-dir ./runs
```

Use a Chat Completions-compatible vision model with tool calling and token usage
reporting. For Alibaba Cloud's compatible API, add `--provider aliyun`.
Trajectories, images, videos, and results are saved beneath the launch directory's
`runs/`; use `--output-dir` to choose another location.

> [!TIP]
> Start with `--smoke-test`, then inspect the prompts and main loop to try your
> own agent ideas. Rotation conversions and interpolation use SciPy, not custom
> quaternion formulas. No OpenRUA environment service is required.

> [!NOTE]
> The current native baseline supports `stack_bowls`, dual ARX X5, and seed 0.
> Actions target **link6** poses; `gripper` ranges from 0 (closed) to 1 (open).
> Simulation only. A passing smoke test is not evidence of task success.

### Acknowledgements

CleanRUA is inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl) and
[RoboSimStudio](https://github.com/robosimstudio/robosimstudio), and uses
[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) for simulation.
Thanks to their authors and contributors. Upstream code and assets retain their
respective licenses.

### License

CleanRUA is released under the [MIT License](LICENSE).
