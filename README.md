# CleanRUA (Clean Implementations of Robot-Use Agents)

Following the [CleanRL](https://github.com/vwxyzjn/cleanrl) design philosophy, CleanRUA provides single-file, high-quality implementations of robot-use agents for reproducible research. Each code file contains a whole agent, including the prompt, tools, and the interaction loop. The environments are plain Gymnasium environments.

<img alt="Successful episodes. Top, RoboDojo head camera: delegate stacking bowls, direct stacking blocks and direct lifting scissors. Bottom, DexJoCo ego camera: delegate clicking the mouse on its pad, delegate hammering the nail and monitor lifting the bucket with the boxed food inside" src="docs/demo.gif" width="100%" />

### News
- **[2026/11/23]** Each paradigm has a prototype of about 100 lines without an environment: [direct.py](cleanrua/direct.py), [cap.py](cleanrua/cap.py), [delegate.py](cleanrua/delegate.py) and [monitor.py](cleanrua/monitor.py). Experimental variants move to [cleanrua/experimental/](cleanrua/experimental).
- **[2026/11/21]** Add [delegate_memory_dexjoco.py](cleanrua/experimental/delegate_memory_dexjoco.py), which keeps a playbook of lessons per task across episodes. With playbooks learned on other seeds, it succeeded in 10 of 30 test episodes, against 17 for Delegate. Add [cap_explore_dexjoco.py](cleanrua/cap_explore_dexjoco.py), which learns a task over episodes and compiles it into one program, following RAX. After 16 learning episodes, its program solved `click_mouse` on all 10 test seeds without the MLLM, against 9 for Delegate and 5 for Pi0.5 alone. The Code paradigm is now CaP (Code as Policies), and its files are `cap_*.py`.
- **[2026/11/20]** Add [DexJoCo](https://github.com/brave-eai/dexjoco), with dexterous Allegro hands, and three agents for it: [delegate_dexjoco.py](cleanrua/delegate_dexjoco.py), [monitor_dexjoco.py](cleanrua/monitor_dexjoco.py) and [cap_delegate_dexjoco.py](cleanrua/cap_delegate_dexjoco.py). Over 50 episodes on five tasks, Delegate succeeded in 36 and Pi0.5 alone in 28 ([evaluation](https://cleanrua.agentic-intelligence-lab.org/get-started/evaluation/)). [monitor_check_dexjoco.py](cleanrua/experimental/monitor_check_dexjoco.py) makes Monitor judge every stretch and correct stalls; it succeeded less often than Monitor.
- **[2026/11/19]** Add two variants of [cap_robodojo.py](cleanrua/cap_robodojo.py): [cap_delegate_robodojo.py](cleanrua/cap_delegate_robodojo.py), whose programs can also hand phases to Pi0.5, and [cap_jev_robodojo.py](cleanrua/experimental/cap_jev_robodojo.py), which keeps its whole history and lets Jev drop the parts no longer needed. [direct_jev_robodojo.py](cleanrua/experimental/direct_jev_robodojo.py) does the same for Direct.
- **[2026/11/18]** Add [cap_robodojo.py](cleanrua/cap_robodojo.py) and [monitor_robodojo.py](cleanrua/monitor_robodojo.py), and rename `hybrid_robodojo.py` to [delegate_robodojo.py](cleanrua/delegate_robodojo.py).
- **[2026/11/17]** Add [hybrid_robodojo.py](cleanrua/delegate_robodojo.py) that uses a MLLM moves the robot with motion tools and hands local phases to Pi0.5.
- **[2026/11/16]** Add [direct_robodojo.py](cleanrua/direct_robodojo.py) that uses a MLLM moves the robot with motion tools.

### Get started

Install the dependencies by following the installation guide for [RoboDojo](examples/installation_robodojo.md) or [DexJoCo](examples/installation_dexjoco.md). Then set the model and run an agent:

```sh
export CLEANRUA_MODEL_NAME=YOUR_MODEL
export CLEANRUA_BASE_URL=https://YOUR_PROVIDER/v1
export CLEANRUA_API_KEY=YOUR_API_KEY
python -m cleanrua.direct_robodojo --native-root /path/to/RoboDojo --task stack_bowls
```

Tips:
1. To add a tool, write a `@Tool` method in the agent's wrapper. To change how the agent decides, edit its `RUAgent`.
2. The environment has no tools and no models. Each agent adds them with a `gym.Wrapper`.
3. `finish` only ends the episode. The environment's own check decides success.
4. A variant adds a word to its algorithm's name, such as `cap_delegate_robodojo.py`, and is still one self-contained file.

### Algorithms implemented

Each prototype shows its algorithm in about 100 lines, without an environment; read it first. In the diagrams, the solid arrows show what the MLLM sends each turn, and the dashed arrow brings images and state back.

| Algorithm | Description | Prototype | Variants implemented |
|---|---|---|---|
| Direct | <img src="docs/paradigms/direct.svg" width="360" alt="Direct: the MLLM sends a target pose to inverse kinematics, which moves the robot." /><br>An MLLM moves the arms with motion tools. | [`direct.py`](cleanrua/direct.py) | [`direct_robodojo.py`](cleanrua/direct_robodojo.py) |
| CaP | <img src="docs/paradigms/cap.svg" width="360" alt="CaP: the MLLM sends a program to a Python robot API, which moves the robot." /><br>An MLLM writes Python programs that move the arms through a robot API. | [`cap.py`](cleanrua/cap.py) | [`cap_robodojo.py`](cleanrua/cap_robodojo.py)<br>[`cap_delegate_robodojo.py`](cleanrua/cap_delegate_robodojo.py)<br>[`cap_delegate_dexjoco.py`](cleanrua/cap_delegate_dexjoco.py)<br>[`cap_explore_dexjoco.py`](cleanrua/cap_explore_dexjoco.py) |
| Delegate | <img src="docs/paradigms/delegate.svg" width="360" alt="Delegate: the MLLM sends an instruction to a VLA, Pi0.5, which moves the robot." /><br>An MLLM gives sub-task instructions to Pi0.5 and can also move the arms itself. | [`delegate.py`](cleanrua/delegate.py) | [`delegate_robodojo.py`](cleanrua/delegate_robodojo.py)<br>[`delegate_dexjoco.py`](cleanrua/delegate_dexjoco.py) |
| Monitor | <img src="docs/paradigms/monitor.svg" width="360" alt="Monitor: a VLA works on the task instruction while the MLLM lets it go on or corrects the robot." /><br>Pi0.5 does the task; an MLLM watches it and corrects failures with motion tools. | [`monitor.py`](cleanrua/monitor.py) | [`monitor_robodojo.py`](cleanrua/monitor_robodojo.py)<br>[`monitor_dexjoco.py`](cleanrua/monitor_dexjoco.py) |

Experimental variants, whose benefit is not established, are in [`cleanrua/experimental/`](cleanrua/experimental): [`direct_jev_robodojo.py`](cleanrua/experimental/direct_jev_robodojo.py) and [`cap_jev_robodojo.py`](cleanrua/experimental/cap_jev_robodojo.py) compact the history with Jev, [`delegate_memory_dexjoco.py`](cleanrua/experimental/delegate_memory_dexjoco.py) keeps a playbook across episodes, and [`monitor_check_dexjoco.py`](cleanrua/experimental/monitor_check_dexjoco.py) makes Monitor judge every stretch.

### Cite Us

```bibtex
@misc{cleanrua2026yuan,
  title        = {CleanRUA: Clean Implementations of Robot-Use Agents},
  author       = {Mingqi Yuan, Xudong Wu, Jiayu Chen},
  year         = {2026},
  howpublished = {\url{https://github.com/Agentic-Intelligence-Lab/CleanRUA}},
}
```

### Acknowledgements

CleanRUA is inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl),
[RoboSimStudio](https://github.com/robosimstudio/robosimstudio) and
[uni-robot-agent](https://github.com/Open-RUA/uni-robot-agent). It uses
[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) and [DexJoCo](https://github.com/brave-eai/dexjoco)
for simulation, and [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) and DexJoCo's OpenPI server for action
models. We thank their authors.

### License

CleanRUA is released under the [MIT License](LICENSE).
