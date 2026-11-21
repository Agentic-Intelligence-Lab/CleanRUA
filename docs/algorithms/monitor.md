# Monitor

## Overview

Pi0.5 does the task, and the MLLM supervises it. The MLLM calls `run_policy` to let Pi0.5 continue with the
native task instruction on both arms, then checks the returned images and state. When Pi0.5 fails in a way it
does not recover from, such as a dropped object or an arm stuck against something, the MLLM corrects the robot
with the [Direct](direct.md) motion tools and hands control back.

Based on VLA-led supervision such as [Spotter](https://arxiv.org/abs/2609.36808) and
[GPT 6 Astra as an Embodied Policy](https://github.com/anonymous-report-421/GPT-as-Policy).

| Variant | Description |
|---|---|
| [`monitor.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/monitor.py) | The algorithm alone, without an environment; read this first |
| [`monitor_robodojo.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/monitor_robodojo.py) | RoboDojo, two ARX X5 arms, Pi0.5 through XPolicyLab |
| [`monitor_dexjoco.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/monitor_dexjoco.py) | DexJoCo, Panda arms with Allegro hands, DexJoCo's multi-task Pi0.5 |
| [`monitor_check_dexjoco.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/monitor_check_dexjoco.py) | Every `run_policy` call first judges progress; a negative judgment requires a correction ([below](#monitor_check)) |

## Usage

Start the Pi0.5 policy server first; see the installation guide for [RoboDojo](../get-started/installation_robodojo.md) or
[DexJoCo](../get-started/installation_dexjoco.md). Then:

```sh
python -m cleanrua.monitor_robodojo --native-root /path/to/RoboDojo --task stack_bowls
```

## Tools

| Tool | What it does |
|---|---|
| `run_policy(chunks)` | Lets Pi0.5 continue the task for 1 to 8 chunks on both arms |
| `move`, `hold`, `capture_scene`, `finish` | As in [Direct](direct.md), for corrections |

## Implementation details

1. **The VLA leads.** Pi0.5 always receives the native task instruction and controls both arms. The MLLM never
   rewrites the instruction or chooses arms; that is the difference from [Delegate](delegate.md).
2. **Chunks.** Each chunk sends fresh images and joint state to Pi0.5 and runs the first `--actions-per-chunk`
   steps (default 25, one second) of its prediction. `run_policy` defaults to 2 chunks per call.
3. **After a correction.** Pi0.5 continues from whatever state the correction left, because every chunk starts
   from a fresh observation.
4. **Supervision after execution.** The MLLM judges what has already happened. Unlike a reviewer that approves
   each chunk before it runs, it cannot stop a chunk in advance, so it never blocks Pi0.5.

## Functional checks

!!! note
    One episode per task on RoboDojo seed 0, with GPT-6 Astra through an OpenAI-compatible API, run on
    2026-11-18. These check that the agent works; they are not success rates.

| Task | Score | Decisions | Tokens |
|---|---|---|---|
| stack_bowls | **100** (success) | 2 | 12k |
| stack_blocks | 15 | 3 | 18k |

In both episodes the MLLM called `run_policy` with 8 chunks each time and never corrected the robot. In
`stack_blocks`, Pi0.5 had not finished when the 550-step native limit ran out.

## On DexJoCo

```sh
python -m cleanrua.monitor_dexjoco --native-root /path/to/dexjoco --task pick_bucket
```

`run_policy(steps)` runs Pi0.5 for 1 to 300 control steps (default 60), replanning and blending as described for
[Delegate on DexJoCo](delegate.md#on-dexjoco). In the first round (2026-11-20, five single-arm tasks, seeds 0
to 9) it succeeded in 31 of 50 episodes, close to Pi0.5 alone (28 of 50), with the fewest tokens: in 175 calls the
MLLM never corrected the robot and only let Pi0.5 continue.

## `monitor_check`

Monitor's MLLM never corrected the robot on DexJoCo, so this variant makes it judge. `run_policy` takes two more
required arguments, `progress` (`started`, `progressing`, `stalled` or `failed`) and `evidence`, the judgment of the
stretch since the previous call. After `stalled` or `failed`, `run_policy` is refused until `move` has corrected
the robot, for example by raising the open hand above the object so Pi0.5 can retry.

```sh
python -m cleanrua.monitor_check_dexjoco --native-root /path/to/dexjoco --task pick_bucket
```

It made things worse (2026-11-20, `click_mouse`, `pick_bucket` and `fold_glasses`, seeds 0 to 4):

| | Monitor | `monitor_check` |
|---|---|---|
| Successes | **9 of 15** | 3 of 15 |
| Tokens per episode | **18k** | 46k |
| Decisions per episode | 4.3 | 9.9 |

On the same seeds, `monitor_check` succeeded 3 times where Monitor failed and failed 9 times where Monitor
succeeded (exact McNemar p = 0.15). The MLLM judged 31 of 105 stretches stalled and made 48 corrections, often
also after judging progress. Each correction moved the hand away from where Pi0.5 was working, and Pi0.5 then had to
start the grasp again; on `pick_bucket` it never recovered. Two `finish` calls also claimed successes the official
check rejected. With an action model that only this MLLM can steer by moving the wrist, frequent corrections cost
more than they save.
