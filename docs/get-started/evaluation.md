---
hide:
  - toc
---

# Evaluation

Every algorithm runs on the same tasks with the same model and budget. The environment's own step limit ends each
episode, and its official check decides success.

??? note "Learning across episodes"
    Agents that learn from earlier episodes keep what they learn in a memory directory and are evaluated in two
    runs of `cleanrua_utils.benchmark`. The learning run, with `--memory DIR --learn`, logs every learning episode's
    seed, outcome and cost to `DIR/learning.jsonl`. The test run, with `--memory DIR`, freezes a copy of the
    directory with its file hashes, refuses test seeds that a task learned from, and stops if the copy changes.
    `summary` then reports the learning cost behind the frozen memory next to the test results.

## RoboDojo

8 tasks, one episode each, seed 0, Pi0.5 for Delegate and Monitor, with two MLLMs:

- **GPT-6 Astra** (2026-11-18, commit `64a2d92`): each episode may take 1 hour.
- **Qwen4-Max** (2026-11-25, RoboDojo `fbfbbcb`), reasoning effort xhigh: each episode may take 3 hours, since
  Qwen4-Max takes about 50 seconds per decision.

<div class="rua-row">
<div class="rua-legend"><span><i></i>GPT-6 Astra (solid)</span><span><i class="rua-hatched"></i>Qwen4-Max (hatched)</span></div>
<div class="rua-chart" data-src="../../assets/results/robodojo-round1.json ../../assets/results/robodojo-qwen4max.json" data-chart="success"></div>
<div class="rua-chart" data-src="../../assets/results/robodojo-round1.json ../../assets/results/robodojo-qwen4max.json" data-chart="score"></div>
<div class="rua-chart" data-src="../../assets/results/robodojo-round1.json ../../assets/results/robodojo-qwen4max.json" data-chart="tokens"></div>
<div class="rua-chart" data-src="../../assets/results/robodojo-round1.json ../../assets/results/robodojo-qwen4max.json" data-chart="tokens_per_success"></div>
</div>

??? note "Observations"
    Tasks: `stack_bowls`, `stack_blocks`, `general_pickup`, `put_bottles_into_dustbin`, `arrange_largest_number`,
    `classify_objects`, `stack_blocks_by_language` and `press_by_number`. One episode per task, algorithm and model
    shows where each paradigm works and what it costs; it is not a success rate. With GPT-6 Astra, CaP on
    `classify_objects` stopped on a provider error and is excluded, and CaP and Direct were not run on
    `press_by_number`. With Qwen4-Max all 32 episodes ran; two `put_bottles_into_dustbin` episodes were run again,
    CaP's after a provider error (HTTP 500) and Delegate's after the provider's content filter refused a request
    (HTTP 400).

    - **The Pi0.5 paradigms are the most robust to the model.** Monitor and Delegate solved the long
      pick-and-place tasks Pi0.5 was trained on (`put_bottles_into_dustbin`, `classify_objects`, `stack_bowls`)
      with both models. Monitor did best with Qwen4-Max, 4 of 8, adding `general_pickup`.
    - **The model-controlled paradigms depend on the model.** With GPT-6 Astra, CaP and Direct did better where the
      right object or precise alignment matters (`general_pickup`, `stack_blocks`, `stack_blocks_by_language`),
      and CaP had the highest mean score. With Qwen4-Max each solved only `general_pickup`; CaP reached partial
      credit on 5 tasks but finished none of them.
    - **Direct is the most expensive and the slowest** with both models: 61% of all tokens with GPT-6 Astra,
      53% with Qwen4-Max, where 4 of its 8 episodes ran out of the 3-hour budget after about 100 decisions each.
      Most of its 171 tool errors with GPT-6 Astra had no inverse-kinematics solution.
    - **Qwen4-Max reasons more per decision.** Every algorithm used several times more tokens per episode; overall
      8 of 32 episodes succeeded, against 11 of 29 with GPT-6 Astra.
    - **Monitor never corrected Pi0.5** with GPT-6 Astra; the model only let the policy continue.
    - **Single episodes vary a lot.** The only GPT-6 Astra `finish` call claimed a success the official check
      rejected, and an earlier Qwen4-Max CaP episode on `stack_blocks` succeeded where the one here scored 15.

    The episodes behind these charts are in [`robodojo-round1.json`](../assets/results/robodojo-round1.json)
    (GPT-6 Astra) and [`robodojo-qwen4max.json`](../assets/results/robodojo-qwen4max.json) (Qwen4-Max).

## DexJoCo

5 single-arm tasks, seeds 0 to 9, DexJoCo's multi-task Pi0.5 for every algorithm, with two MLLMs:

- **GPT-6 Astra** (2026-11-20, commit `c64fe56`): each episode may take 1 hour.
- **Qwen4-Max** (2026-11-25, commit `26d1b3f`), reasoning effort xhigh: each episode may take 3 hours.

<div class="rua-row">
<div class="rua-legend"><span><i></i>GPT-6 Astra (solid)</span><span><i class="rua-hatched"></i>Qwen4-Max (hatched)</span></div>
<div class="rua-chart" data-src="../../assets/results/dexjoco-round1.json ../../assets/results/dexjoco-qwen4max.json" data-chart="success"></div>
<div class="rua-chart" data-src="../../assets/results/dexjoco-round1.json ../../assets/results/dexjoco-qwen4max.json" data-chart="tokens"></div>
<div class="rua-chart" data-src="../../assets/results/dexjoco-round1.json ../../assets/results/dexjoco-qwen4max.json" data-chart="tokens_per_success"></div>
</div>

??? note "Observations"
    Tasks: `click_mouse`, `hammer_nail`, `water_plant`, `pick_bucket` and `fold_glasses`; 50 episodes per
    algorithm and model on the same seeds. Six GPT-6 Astra episodes stopped by provider errors were run again; no
    Qwen4-Max episode was. DexJoCo has no partial credit, so there is no score chart.

    | | GPT-6 Astra | Qwen4-Max |
    |---|---|---|
    | Delegate | **36** of 50 | 31 of 50 |
    | Monitor | 31 of 50 | **33** of 50 |
    | `cap_delegate` | 28 of 50 | 30 of 50 |
    | Pi0.5 alone | 28 of 50 | 28 of 50 |

    - **Pi0.5 alone is the reference.** With the task instruction and no MLLM, it succeeded in 28 of 50 episodes
      on the same seeds. It already solves most `water_plant` and `hammer_nail` scenes, so most differences
      come from `click_mouse`, `pick_bucket` and `fold_glasses`.
    - **Delegate with GPT-6 Astra did best**, and helped most on `click_mouse` (9 against 5). On the same seeds it
      succeeded 13 times where Pi0.5 alone failed and failed 5 times where Pi0.5 alone succeeded (exact McNemar
      p = 0.10): suggestive but not significant. The MLLM usually starts with the task instruction and then gives
      step instructions such as "Lower the held boxed food into the red bucket and release it", even though this
      Pi0.5 was trained only with whole task instructions.
    - **No other algorithm and model differs significantly from Pi0.5 alone.** With Qwen4-Max, Monitor succeeded
      12 times where Pi0.5 alone failed and failed 7 times where it succeeded (p = 0.36); Delegate 11 and 8,
      `cap_delegate` 10 and 8.
    - **Monitor is the cheapest with both models** (15k and 24k tokens per episode) and never corrected the robot
      with GPT-6 Astra: in 175 calls it only let Pi0.5 continue.
    - **`cap_delegate` used the most tokens per success.** With GPT-6 Astra only 16 of its 288 programs called
      `robot.move`.
    - **`finish` was rare and wrong.** 5 of `cap_delegate`'s 8 `finish` calls with GPT-6 Astra, and each
      algorithm's one call with Qwen4-Max, claimed successes the official check rejected, for example a bucket
      lifted less than the required 15 cm.
    - **`fold_glasses` is the hardest single-arm task**, and no algorithm solved `bimanual_assembly` in three seeds
      each with either model; Pi0.5 alone fails it too. It is not in the charts.

    The episodes behind these charts are in [`dexjoco-round1.json`](../assets/results/dexjoco-round1.json)
    (GPT-6 Astra) and [`dexjoco-qwen4max.json`](../assets/results/dexjoco-qwen4max.json) (Qwen4-Max).
