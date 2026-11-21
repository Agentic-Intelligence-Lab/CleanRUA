# Code

## Overview

The MLLM writes Python programs. Each turn it sends one program through `run_code`. The program calls a small
`robot` API, can compute targets with NumPy, and prints whatever the model wants to read. When it ends, the model
sees the printed output with fresh images and state.

Based on [Code as Policies](https://arxiv.org/abs/2209.07753) and its multi-turn successors such as
[CaP-X](https://arxiv.org/abs/2603.22435).

| Variant | Description |
|---|---|
| [`prototypes/code.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/prototypes/code.py) | The algorithm alone, without an environment; read this first |
| [`code_robodojo.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/code_robodojo.py) | RoboDojo, two ARX X5 arms |
| [`code_delegate_robodojo.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/code_delegate_robodojo.py) | Programs can also hand short phases to Pi0.5 with `robot.vla` ([below](#code_delegate)) |
| [`code_jev_robodojo.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/code_jev_robodojo.py) | Keeps the whole history and compacts it with Jev ([below](#code_jev)) |
| [`code_delegate_dexjoco.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/code_delegate_dexjoco.py) | `code_delegate` on DexJoCo, Panda arms with Allegro hands ([below](#code_delegate)) |
| [`code_explore_dexjoco.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/code_explore_dexjoco.py) | Learns a task over episodes and compiles it into one program that runs alone ([below](#code_explore)) |

## Usage

```sh
python -m cleanrua.code_robodojo --native-root /path/to/RoboDojo --task stack_blocks
```

!!! warning
    Programs run in the agent's own process without a sandbox. Run this agent only in an environment you trust.

## Tools and robot API

| Tool | What it does |
|---|---|
| `run_code(code)` | Runs a Python program and returns its printed output |
| `finish(status, summary)` | Ends the episode when the model judges the task done |

Inside a program:

| Name | What it does |
|---|---|
| `robot.state()` | Measured `pose`, `joints`, `gripper` and `gripper_command` of each arm, as NumPy arrays |
| `robot.move(targets, profile, max_steps)` | The same motion as [Direct](direct.md)'s `move`; returns `targets_reached`, `stop_reason` and `errors` |
| `robot.hold(steps)` | Advances physics while the arms hold |
| `np` | NumPy |

## Implementation details

1. **One namespace per episode.** Variables defined by one program are available to the next, so the model can
   keep measured targets and helper functions.
2. **Threads.** A program runs in a background thread. Each `robot` call sends a request and waits; `run_code`
   executes the motion in the main thread, where the simulator lives, and replies with the result. The
   Gymnasium loop therefore stays the same as in the other agents.
3. **Errors.** An uncaught exception stops the program and is reported with its line number; motions already
   executed remain. A motion that fails, for example without an inverse-kinematics solution, raises
   `ValueError` inside the program, which can catch it and try something else.
4. **Limits.** A program that commands no motion for 60 seconds is abandoned. When the native episode ends, the
   running program stops quietly. Only the last 4,000 characters of output are returned.
5. **Calibration in code.** The observation includes each camera's intrinsics and extrinsics, so a program can
   triangulate an object from two views. The model did this on its own in the checks below.

## Functional checks

!!! note
    One episode per task on RoboDojo seed 0, with GPT-6 Astra through an OpenAI-compatible API and a 400k-token
    cap, run on 2026-11-18. These check that the agent works; they are not success rates.

| Task | Score | Decisions | Tokens |
|---|---|---|---|
| general_pickup | **100** (success) | 9 | 62k |
| stack_blocks | **100** (success) | 24 | 164k |

All 24 programs in `stack_blocks` ran without errors; most were 2 to 9 lines long.

## `code_delegate`

Based on [PyRUA-Lean](https://github.com/DAGroup-PKU/PyRUA-Lean), which composes robot primitives and frozen VLA
policies in persistent Python cells. Programs get one more call:

| Name | What it does |
|---|---|
| `robot.vla(instruction, arms, chunks=1, actions_per_chunk=25)` | Runs Pi0.5 on `arms` with a short instruction, such as "Pick up the white block"; the other arm holds |

A program can mix it with `robot.move`: run the action model for a chunk, check the measured gripper opening,
then correct or retry. It needs the Pi0.5 server of [Delegate](delegate.md). On `put_bottles_into_dustbin` it
succeeded in 10 decisions and 56k tokens, where Code scored 25 with 386k, but every program there only called
`robot.vla`.

On DexJoCo the call is `robot.vla(instruction, arms=None, steps=60)`, executed as in
[Delegate on DexJoCo](delegate.md#on-dexjoco), and `robot.move` takes flange poses. There is no plain Code agent
for DexJoCo yet: without Pi0.5, a program would have to shape 16 finger joints per hand itself. In the first round
(2026-11-20, five single-arm tasks, seeds 0 to 9) it succeeded in 28 of 50 episodes, as often as Pi0.5 alone and
against 36 for Delegate: only 16 of its 288 programs called `robot.move`, and 5 of its 8 `finish` calls claimed
successes the official check rejected, such as a bucket lifted less than the required 15 cm.

## `code_jev`

Keeps the whole history (`--history-size None` is the default). Once the history text is estimated to exceed
`--jev-budget-tokens` (2,500), it asks [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), a
model that answers typed questions with probabilities, two questions about every older tool call: is its result
still needed verbatim, and is the call still worth remembering. A pair whose result is needed stays; a pair whose
call alone is needed keeps the call with its result cut to 300 characters; the rest is removed. The newest four
pairs are never touched, and nothing kept is rewritten or summarized. It needs `TYPESAFE_API_KEY`.

In one episode per task (2026-11-19), against Code with the last four pairs (window) and Code with the whole
history:

| Task | Window | Whole history | `code_jev` |
|---|---|---|---|
| stack_blocks | **100** · 102k tokens | **100** · 502k | **100** · 95k |
| arrange_largest_number | 30 · 297k | 30 · 787k | **100** · 141k |
| classify_objects | 40 · 610k | stopped by a provider timeout | **100** · 443k |

Jev itself cost about $0.01 per episode and 0.5 seconds per compaction. One episode per cell cannot separate the
method from sampling variance; repeats are planned.

## `code_explore`

Learns a DexJoCo task by exploring it and compiles what worked into one robot program, following
[RAX](https://github.com/robot-agent-exploration/rax). There is no action model: programs command the flange poses
and all 16 finger joints, and see each camera's depth.

| Name | What it does |
|---|---|
| `robot.move(targets, max_steps)` | As above, and `hand` takes 16 finger joint targets, at most 0.1 rad per step within the joint limits |
| `robot.camera(name)` | The camera's image, depth in metres and calibration |
| `robot.point(name, u, v)` | World position of the surface seen at a pixel |

Each learning episode first rehearses the current `controller.py` without the MLLM. If the task is not done, the
MLLM explores from there with programs. Afterwards it rewrites `notes.md` and `controller.py` from the episode, its
final images and the official outcome. A program that succeeds in two rehearsals in a row is copied to
`verified.py`. A test episode runs only the frozen program, so it uses no tokens.

```sh
python -m cleanrua_utils.benchmark run --benchmark dexjoco --algorithms code_explore --tasks click_mouse \
  --seeds 20-25 --memory runs/memory --learn --native-root /path/to/dexjoco --output-dir runs/train
python -m cleanrua_utils.benchmark run --benchmark dexjoco --algorithms code_explore --tasks click_mouse \
  --seeds 0-9 --memory runs/memory --native-root /path/to/dexjoco --output-dir runs/test
```

While exploring, the MLLM can restart its scene with `reset_scene`, up to `--attempts` times, and every tool result
reports `task_success`, the official check, as RAX's agents have in simulation. Goal images of the finished task in
other scenes, in `<memory-dir>/<task>/goal/`, are shown with the task and to the compile step.

On `click_mouse` (2026-11-21) it learned a program that solved all ten test seeds without the MLLM:

| | Pi0.5 alone | Delegate | `code_explore` |
|---|---|---|---|
| Successes on seeds 0 to 9 | 5 | 9 | **10** |
| Steps per episode | 872 | 661 | **487** |
| Tokens per episode | 0 | 27k | **0** |

Learning took 16 episodes on seeds 20 to 35 and 809k tokens. The MLLM first solved the task in the fifth episode:
it pushed the mouse about 15 cm with the side of a curled hand and pressed down once the mouse overlapped the pad.
The program it compiled from that episode, 314 lines, found the pad by colour and the mouse in the depth image,
checked the result after each stage, and took the blue monitor as the sign of a click. It passed the next two
rehearsals and was verified, then succeeded in every later learning and test episode. The goal images came from
successful Pi0.5 rollouts on seeds 40, 42 and 44. A first version without restarts, the success check or goal
images did not learn the task in six episodes.
