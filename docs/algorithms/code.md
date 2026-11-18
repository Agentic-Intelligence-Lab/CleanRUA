# Code

## Overview

The MLLM writes Python programs. Each turn it sends one program through `run_code`. The program calls a small
`robot` API, can compute targets with NumPy, and prints whatever the model wants to read. When it ends, the model
sees the printed output with fresh images and state.

Based on [Code as Policies](https://arxiv.org/abs/2209.07753) and its multi-turn successors such as
[CaP-X](https://arxiv.org/abs/2603.22435).

| Variant | Description |
|---|---|
| [`code_robodojo.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/code_robodojo.py) | RoboDojo, two ARX X5 arms |

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
