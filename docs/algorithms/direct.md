# Direct

## Overview

The MLLM controls the robot one motion at a time. Each turn it calls a motion tool, usually `move` with an
absolute grasp-site pose for one or both arms. Inverse kinematics turns the pose into joint targets, and the
model sees the resulting images and state before its next call.

Based on uni-robot-agent's [Clean mode](https://github.com/Open-RUA/uni-robot-agent), a four-tool baseline, and
on direct LLM control as evaluated in [GPT-6 Astra on RoboDojo](https://arxiv.org/abs/2609.24170).

| Variant | Description |
|---|---|
| [`direct_robodojo.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/direct_robodojo.py) | RoboDojo, two ARX X5 arms |
| [`direct_jev_robodojo.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/direct_jev_robodojo.py) | Keeps the whole history and compacts it with Jev, as [`code_jev`](code.md#code_jev) does |

## Usage

```sh
python -m cleanrua.direct_robodojo --native-root /path/to/RoboDojo --task stack_blocks
```

## Tools

| Tool | What it does |
|---|---|
| `move(targets, profile, max_steps)` | Moves one or both arms toward absolute grasp-site poses `[x, y, z, roll, pitch, yaw]`, with an optional gripper opening per arm |
| `hold(steps)` | Advances physics for 1 to 40 control steps while the arms hold |
| `capture_scene()` | Returns fresh images and state without moving |
| `finish(status, summary)` | Ends the episode when the model judges the task done |

## Implementation details

1. **Control point.** Poses refer to each arm's grasp site, the centre between the finger pads, 14.93 cm in
   front of link6. The offset comes from uni-robot-agent's `x5_grasp_site.urdf`. Orientations are fixed-axis
   XYZ Euler angles. The observation adds this `pose` to each arm.
2. **Profiles.** `standard` moves 4 cm per control step and stops within 3 cm and 0.1 rad; `approach` 4 cm,
   2 cm and 0.05 rad; `precision` 1 cm, 5 mm and 0.05 rad. Rotation uses part of each step's displacement,
   because rotating link6 also moves the grasp site.
3. **Stopping.** `move` stops when every pose is within tolerance and requested fingers have stopped moving
   (`targets_reached`), when an arm improves by less than a tenth of the tolerance over three steps
   (`stalled`), or after `max_steps` (at most 24). Feedback gives each arm's position and orientation error.
4. **Inverse kinematics.** RoboDojo's native solver targets link6. A waypoint whose solution would jump a joint
   by more than 1 rad is rejected, and `move` returns the error as feedback.
5. **Gripper.** Openings are normalized: 0 closed, 1 open. The observation reports both the measured opening and
   the last command, because the fingers can stop on an object before reaching the command.

## Functional checks

!!! note
    One episode per task on RoboDojo seed 0, with GPT-6 Astra through an OpenAI-compatible API, run on
    2026-11-17. These check that the agent works; they are not success rates.

| Task | Score | Decisions | Tokens |
|---|---|---|---|
| general_pickup | **100** (success) | 6 | 35k |
| stack_blocks | **100** (success) | 46 | 265k |
| stack_bowls | 15 | 60 (budget) | 339k |
| push_T | 0 | 60 (budget) | 363k |

In `stack_bowls`, 11 of the 58 `move` calls failed on inverse kinematics and 5 stalled.

`direct_jev` gave no clear gain (2026-11-19, one episode each): `stack_blocks_by_language` still succeeded, with
329k tokens instead of 207k, and `put_bottles_into_dustbin` scored 25 instead of 10 but still ran out of steps.
Direct's failures come from inverse kinematics and step limits more than from its context.
