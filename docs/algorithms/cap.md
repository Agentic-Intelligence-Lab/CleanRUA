# CaP

## Overview

The MLLM writes Python programs. Each turn it sends one program through `run_code`. The program calls a small
`robot` API, can compute targets with NumPy, and prints whatever the model wants to read. When it ends, the model
sees the printed output with fresh images and state.

Based on [Code as Policies](https://arxiv.org/abs/2209.07753) and its multi-turn successors such as
[CaP-X](https://arxiv.org/abs/2603.22435).

| Variant | Description |
|---|---|
| [`cap.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/cap.py) | The algorithm alone, without an environment; read this first |
| [`cap_robodojo.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/cap_robodojo.py) | RoboDojo, two ARX X5 arms |
| [`cap_delegate_robodojo.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/cap_delegate_robodojo.py) | Programs can also hand short phases to Pi0.5 with `robot.vla` ([below](#cap_delegate)) |
| [`cap_jev_robodojo.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/experimental/cap_jev_robodojo.py) | Experimental: keeps the whole history and compacts it with Jev ([below](#cap_jev)) |
| [`cap_tools_robodojo.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/experimental/cap_tools_robodojo.py) | Experimental: programs get geometry tools for calibrated RGB cameras (pixel rays, triangulation, projection) and `robot.reachable`; `move` turns a grasp half a turn about its approach axis when that keeps the wrist roll off its joint limit |
| [`cap_delegate_dexjoco.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/cap_delegate_dexjoco.py) | `cap_delegate` on DexJoCo, Panda arms with Allegro hands ([below](#cap_delegate)) |
| [`cap_explore_dexjoco.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/cap_explore_dexjoco.py) | Learns a task over episodes and compiles it into one program that runs alone ([below](#cap_explore)) |

## Usage

```sh
python -m cleanrua.cap_robodojo --native-root /path/to/RoboDojo --task stack_blocks
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

`cap_tools` succeeded on `general_pickup` with Qwen4-Max (2026-11-24, one episode): 15 decisions, 146k tokens, 14 minutes. It triangulated the scissors from two cameras and grasped them on the fourth attempt; the half-turn grasp did not trigger. With Qwen3.8-Flash the same task failed: the model closed the gripper above the scissors and could not tell that it held nothing.

## `cap_delegate`

Based on [PyRUA-Lean](https://github.com/DAGroup-PKU/PyRUA-Lean), which composes robot primitives and frozen VLA
policies in persistent Python cells. Programs get one more call:

| Name | What it does |
|---|---|
| `robot.vla(instruction, arms, chunks=1, actions_per_chunk=25)` | Runs Pi0.5 on `arms` with a short instruction, such as "Pick up the white block"; the other arm holds |

A program can mix it with `robot.move`: run the action model for a chunk, check the measured gripper opening,
then correct or retry. It needs the Pi0.5 server of [Delegate](delegate.md). On `put_bottles_into_dustbin` it
succeeded in 10 decisions and 56k tokens, where CaP scored 25 with 386k, but every program there only called
`robot.vla`.

On DexJoCo the call is `robot.vla(instruction, arms=None, steps=60)`, executed as in
[Delegate on DexJoCo](delegate.md#on-dexjoco), and `robot.move` takes flange poses. There is no plain CaP agent
for DexJoCo yet: without Pi0.5, a program would have to shape 16 finger joints per hand itself. In the first round
(2026-11-20, five single-arm tasks, seeds 0 to 9) it succeeded in 28 of 50 episodes, as often as Pi0.5 alone and
against 36 for Delegate: only 16 of its 288 programs called `robot.move`, and 5 of its 8 `finish` calls claimed
successes the official check rejected, such as a bucket lifted less than the required 15 cm.

## `cap_jev`

Keeps the whole history (`--history-size None` is the default). Once the history text is estimated to exceed
`--jev-budget-tokens` (2,500), it asks [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), a
model that answers typed questions with probabilities, two questions about every older tool call: is its result
still needed verbatim, and is the call still worth remembering. A pair whose result is needed stays; a pair whose
call alone is needed keeps the call with its result cut to 300 characters; the rest is removed. The newest four
pairs are never touched, and nothing kept is rewritten or summarized. It needs `TYPESAFE_API_KEY`.

In one episode per task (2026-11-19), against CaP with the last four pairs (window) and CaP with the whole
history:

| Task | Window | Whole history | `cap_jev` |
|---|---|---|---|
| stack_blocks | **100** · 102k tokens | **100** · 502k | **100** · 95k |
| arrange_largest_number | 30 · 297k | 30 · 787k | **100** · 141k |
| classify_objects | 40 · 610k | stopped by a provider timeout | **100** · 443k |

Jev itself cost about $0.01 per episode and 0.5 seconds per compaction. One episode per cell cannot separate the
method from sampling variance; repeats are planned.

## `cap_explore`

Learns a DexJoCo task over several episodes and compiles what worked into one robot program, which then runs alone,
following [RAX](https://github.com/robot-agent-exploration/rax). There is no action model: programs command the
flange poses and all 16 finger joints of each hand, and read each camera's depth. Learning costs tokens; a test
episode runs only the frozen program and costs none.

### How it learns

1. **Rehearse.** A learning episode first runs the task's current `controller.py` alone, without the MLLM, as a test
   episode would.
2. **Explore.** If the rehearsal failed, the scene starts again from its initial state with the full step budget, and
   the MLLM explores it with programs through `run_code`. It sees its notes, the current program and how the
   rehearsal went, and goal images of the finished task in other scenes. Every tool result reports `task_success`,
   the official check, as RAX's agents have in simulation. `reset_scene` plays the scene from its start again, up to
   `--attempts` times.
3. **Compile.** After the episode, one more request rewrites `notes.md` and `controller.py` from the episode's
   programs and results, its final images, the goal images and the outcome of each attempt.
4. **Verify.** A program that succeeds in `--verify-runs` (2) rehearsals in a row is copied to `verified.py`. A
   successful rehearsal ends the episode without the MLLM, so a working program costs no more tokens.
5. **Test.** With `--no-learn`, an episode runs `verified.py` (or `controller.py` if nothing was verified) and
   nothing else.

Inside a program:

| Name | What it does |
|---|---|
| `robot.state()` | Measured `pose`, `hand` and `hand_command` of each arm |
| `robot.move(targets, max_steps)` | Moves arms toward flange poses `[x, y, z, roll, pitch, yaw]` and, with `hand`, 16 finger joint targets |
| `robot.hold(steps)` | Advances physics while the arms hold |
| `robot.camera(name)` | The camera's image, depth in metres and calibration |
| `robot.point(name, u, v)` | World position of the surface seen at a pixel |

### Memory

Everything learned lives in `<memory-dir>/<task>/`:

| File | Contents |
|---|---|
| `notes.md` | Lessons for the exploring MLLM, at most 4,000 characters, each marked with its evidence |
| `controller.py` | The latest program, `main(robot)` |
| `verified.py` | The last program that passed verification; testing runs this one |
| `status.json` | Program version, consecutive successful rehearsals and the verified version |
| `goal/*.png` | Goal images, which you provide |

### Usage

Learn on some seeds and test on others with the benchmark runner, which logs every learning episode to
`<memory>/learning.jsonl`, freezes a hashed copy of the memory for testing and refuses test seeds the task learned
from (see [Evaluation](../get-started/evaluation.md)):

```sh
python -m cleanrua_utils.benchmark run --benchmark dexjoco --algorithms cap_explore --tasks click_mouse \
  --seeds 20-35 --memory runs/memory --learn --native-root /path/to/dexjoco --output-dir runs/train \
  -- --max-total-tokens 150000
python -m cleanrua_utils.benchmark run --benchmark dexjoco --algorithms cap_explore --tasks click_mouse \
  --seeds 0-9 --memory runs/memory --native-root /path/to/dexjoco --output-dir runs/test
python -m cleanrua_utils.benchmark summary runs/test   # Test results and the learning cost behind them.
```

### Implementation details

1. **Fingers.** `hand` targets are clipped to the Allegro joint limits and approached at most 0.1 rad per control
   step. `move` reports `targets_reached` once the pose is within 0.01 m and 0.05 rad and the finger command has
   arrived; fingers that stop short of it may be touching an object.
2. **Depth.** `DexJoCoEnv.depth(camera)` renders the camera's depth buffer on demand and converts it to metres along
   the optical axis. Checked against the simulator's ground truth, flat surfaces back-project within 1 to 2 cm.
3. **Compiling.** The reply must contain a ```` ```markdown ```` block with the notes and a ```` ```python ```` block
   defining `main(robot)` that compiles; otherwise the previous notes and program stay. The episode transcript sent
   to it is cut to its last 30,000 characters.
4. **Outcomes.** A test episode succeeds if the program does. A learning episode succeeds if its rehearsal or any of
   its attempts does, and its steps are those of all of them. The tokens of the compile request count towards the
   episode.
5. **Verification is not undone.** A later failed rehearsal resets the count and starts exploring again, but
   `verified.py` stays until a new program is verified.

### Results on `click_mouse`

!!! note
    One learning run on seeds 20 to 35, then the frozen program on seeds 0 to 9, with GPT-6 Astra, 2026-11-21. Pi0.5
    alone and Delegate ran on the same test seeds in the [DexJoCo round](../get-started/evaluation.md#dexjoco). The
    goal images are the final ego-camera frames of successful Pi0.5 rollouts on seeds 40, 42 and 44, which belong to
    neither set.

| | Pi0.5 alone | Delegate | `cap_explore` |
|---|---|---|---|
| Successes on seeds 0 to 9 | 5 | 9 | **10** (95% CI 72–100%) |
| Control steps per episode | 872 | 661 | **487** |
| Tokens per episode | 0 | 27k | **0** |
| Seconds per episode | – | 79 | **32** |

On the same seeds, the program solved every seed that Pi0.5 alone failed (0, 3, 6, 8 and 9; exact McNemar
p = 0.06) and the one that Delegate failed (4), and lost none.

**Learning.** Sixteen episodes used 809k tokens, 13,580 control steps and 30 minutes, about as many tokens as 30
Delegate episodes.

| Seeds | What happened | Tokens |
|---|---|---|
| 20–23 | Each exploration stopped at the 150k-token cap after 16 to 23 decisions without success; programs v1 to v4 | 164k–172k each |
| 24 | v4's rehearsal failed. In its first attempt the MLLM pushed the mouse about 15 cm with the side of a curled hand and pressed down once the mouse overlapped the pad; the check passed at decision 11. The compile step wrote v5 | 137k |
| 25–26 | v5's rehearsals succeeded; v5 was verified | 0 |
| 27–35 | v5's rehearsals all succeeded | 0 |

**The learned program.** `verified.py` has 314 lines and uses only NumPy and the robot API above:

1. Locate the purple pad by colour and depth, with the hand raised, and keep the least occluded estimate.
2. Find the mouse as a dark cluster 1 to 7.5 cm above the pad in the depth image.
3. Curl the hand into a fist, place it behind the mouse on the side away from the pad, lower it, and push towards
   the pad. If the push stalls, lift slightly and push again; after each push, retreat and locate the mouse again.
4. Once the mouse overlaps the pad, open the hand slightly and press the left button, whose position it estimates
   from the mouse's long axis.
5. Treat a blue monitor as the sign of a click, as in the goal images.

In every test episode one push moved the mouse 11 to 12.5 cm onto the pad. In six, the program then pressed the button
and saw the monitor turn blue. In the other four, the check passed before the program's press step: the button had
already been pressed, with the mouse on the pad, during the push or the retreat after it. DexJoCo counts a success
once the mouse's centre is on the pad and the monitor has stayed blue for 10 control steps.

**A first version did not learn.** Without scene restarts, the success check or goal images, and with exploration
continuing in the rehearsed scene, six learning episodes on seeds 20 to 25 (579k tokens) produced no success, and
the last program solved none of the test seeds. From the third episode on, the rehearsal used 600 to 870 of the
1,170 steps and left the MLLM 5 to 10 decisions.

**Limits.**

- One task and one learning run; learning itself is stochastic, and other runs may take longer or fail.
- No exploration used `reset_scene`: each stopped at the token cap or succeeded within its first attempt. The gains
  came from the fresh scene after the rehearsal, the success check and the goal images; which mattered most was not
  tested separately.
- The success check exists only in simulation, and the goal images came from Pi0.5's successes.
- Each exploration decision sent about 9k prompt tokens, mostly the two current images, the three goal images sent
  every turn and eight rounds of history.
- The program is specific to its task by design: it relies on the pad's colour, the `ego_right` camera and the
  blue monitor.
