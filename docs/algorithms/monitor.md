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
| [`monitor_robodojo.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/monitor_robodojo.py) | RoboDojo, two ARX X5 arms, Pi0.5 through XPolicyLab |
| [`monitor_dexjoco.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua/monitor_dexjoco.py) | DexJoCo, Panda arms with Allegro hands, DexJoCo's multi-task Pi0.5 |

## Usage

Start the Pi0.5 policy server first; see the [installation guide](../get-started/installation.md). Then:

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
to 2) it succeeded in 9 of 15 episodes, as often as Pi0.5 alone, with the fewest tokens: in all 18 episodes,
including `bimanual_assembly`, the MLLM only called `run_policy` and never corrected the robot.
