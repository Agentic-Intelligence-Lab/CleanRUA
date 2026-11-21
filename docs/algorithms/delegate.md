# Delegate

## Overview

The MLLM splits the task into sub-tasks and gives each one to a vision-language-action model (VLA) through
`actor_act`, for example "Pick up the left bowl and nest it into the middle bowl" with the left arm. Pi0.5 carries
out the sub-task with fresh observations every chunk. The MLLM can also move the arms itself with the
[Direct](direct.md) tools, for example to return them to their initial poses.

Based on uni-robot-agent's [hybrid mode](https://github.com/Open-RUA/uni-robot-agent) and hierarchical VLM–VLA
systems such as [Hi Robot](https://arxiv.org/abs/2502.19417).

| Variant | Description |
|---|---|
| [`prototypes/delegate.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/prototypes/delegate.py) | The algorithm alone, without an environment; read this first |
| [`delegate_robodojo.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/delegate_robodojo.py) | RoboDojo, two ARX X5 arms, Pi0.5 through XPolicyLab |
| [`delegate_dexjoco.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/delegate_dexjoco.py) | DexJoCo, Panda arms with Allegro hands, DexJoCo's multi-task Pi0.5 ([below](#on-dexjoco)) |
| [`delegate_memory_dexjoco.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/delegate_memory_dexjoco.py) | Keeps a playbook per task across episodes ([below](#delegate_memory)) |

## Usage

Start the Pi0.5 policy server first; see the installation guide for [RoboDojo](../get-started/installation_robodojo.md) or
[DexJoCo](../get-started/installation_dexjoco.md). Then:

```sh
python -m cleanrua.delegate_robodojo --native-root /path/to/RoboDojo --task stack_bowls
```

## Tools

| Tool | What it does |
|---|---|
| `actor_act(instruction, active_groups, actions_per_chunk, max_chunks)` | Runs Pi0.5 on the given arms with a short instruction |
| `move`, `hold`, `capture_scene`, `finish` | As in [Direct](direct.md) |

## Implementation details

1. **Chunks.** Each chunk sends the current camera images and joint state with the instruction to Pi0.5, which
   predicts 50 control steps. The first `actions_per_chunk` (default 25, at most 40) run and the rest are
   discarded. A call runs `max_chunks` chunks (default 6) unless the native episode ends.
2. **Arms.** Arms outside `active_groups` hold their joints and gripper. Predicted gripper openings are clipped to
   [0, 1].
3. **Server.** Pi0.5 runs as XPolicyLab's websocket policy server in its own environment. Its checkpoint was
   trained on all RoboDojo tasks, so one server serves every task. It needs about 8.5 GB of GPU memory.
4. **Differences from uni-robot-agent.** There is no chunk reviewer, no `task_progress` and no gripper
   constraints, and openings are normalized rather than metric. A reviewer that checked every chunk before it ran
   stopped Pi0.5 too often and used 10 to 40 times more tokens in our comparison, so it was removed.

## Functional checks

!!! note
    One episode per task on RoboDojo seed 0, with GPT-6 Astra through an OpenAI-compatible API. These check that
    the agent works; they are not success rates.

| Task | Score | Decisions | Tokens | Run |
|---|---|---|---|---|
| stack_bowls | **100** (success) | 4 | 25k | 2026-11-18 |
| stack_bowls | **100** (success) | 3 | 15k | 2026-11-17, earlier version |
| stack_blocks | **100** (success) | 3 | 22k | 2026-11-17, earlier version |
| put_bottles_into_dustbin | **100** (success) | 5 | 35k | 2026-11-17, earlier version |
| push_T | 0 | 5 | 35k | 2026-11-17, earlier version |

The earlier version had the same `actor_act` behaviour but two extra, unused tool arguments.

## On DexJoCo

```sh
python -m cleanrua.delegate_dexjoco --native-root /path/to/dexjoco --task pick_bucket
```

`actor_act(instruction, active_groups, steps)` runs Pi0.5 for a number of control steps rather than chunks, and
executes it as DexJoCo's official evaluation does: a prediction covers 30 steps, Pi0.5 is asked again once fewer
than 24 remain, so every 7 steps, and the overlap of the old and new predictions is blended. Executing a prefix of
each prediction instead, as on RoboDojo, failed every `pick_bucket` episode we tried. `move` takes flange poses and
an optional `hand='open'`; only Pi0.5 shapes the fingers.

The MLLM usually starts with the task instruction and then gives step instructions, such as "Lower the held
boxed food into the red bucket and release it", although this Pi0.5 was trained only with the tasks' own
instructions. In the first round on five single-arm tasks (2026-11-20, seeds 0 to 9) it succeeded in 36 of 50
episodes, against 28 of 50 for Pi0.5 alone with the task instruction on the same seeds (exact McNemar p = 0.10).

## `delegate_memory`

Delegate on DexJoCo with a playbook that carries lessons across episodes, following
[Reflexion](https://arxiv.org/abs/2303.11366) and the playbooks of [RobotUse](https://github.com/robotuse-team/RobotUse).
Each task has its own playbook, `<memory-dir>/<task>.md`, which is added to the task text at the start of an
episode. After the episode, the MLLM rewrites it from the episode's tool calls and results, the final images and
the official outcome, which the agent itself never sees. Lessons must say how much evidence they have and contain
no coordinates. `--no-learn` evaluates a frozen playbook; the reflection's tokens count towards the episode.
To learn on some seeds and test on others, use the benchmark runner, which keeps the two apart
(see [Evaluation](../get-started/evaluation.md)):

```sh
python -m cleanrua_utils.benchmark run --benchmark dexjoco --algorithms delegate_memory --tasks pick_bucket \
  --seeds 10-14 --memory runs/memory --learn --native-root /path/to/dexjoco --output-dir runs/train
python -m cleanrua_utils.benchmark run --benchmark dexjoco --algorithms delegate_memory --tasks pick_bucket \
  --seeds 0-9 --memory runs/memory --native-root /path/to/dexjoco --output-dir runs/test
```

It made Delegate worse (2026-11-21). Playbooks were learned on seeds 10 to 14 of `click_mouse`, `pick_bucket` and
`fold_glasses`, then frozen and tested on seeds 0 to 9:

| | Pi0.5 alone | Delegate | `delegate_memory` |
|---|---|---|---|
| Successes | 11 of 30 | **17 of 30** | 10 of 30 |
| Tokens per episode | 0 | **33k** | 48k |

On the same seeds, `delegate_memory` succeeded 3 times where Delegate failed and failed 10 times where Delegate
succeeded (exact McNemar p = 0.09). The playbooks came from 6 successes in 15 training episodes, yet the agent
followed them as recipes: on `click_mouse` every test episode began with the playbook's four instructions and step
counts, and lost three seeds that Delegate solved by varying its instructions. Because Pi0.5 samples a different
motion every time, a sequence that worked once is weak evidence, and the playbook mostly took away the agent's
variety. Each decision also carried about 900 more prompt tokens.
