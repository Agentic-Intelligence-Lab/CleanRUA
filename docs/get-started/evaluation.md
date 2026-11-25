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

Round 1 (2026-11-18, commit `64a2d92`): 8 tasks, one episode each, seed 0, GPT-6 Astra, Pi0.5 for Delegate
and Monitor.

<div class="rua-row">
<div class="rua-chart" data-src="../../assets/results/robodojo-round1.json" data-chart="success"></div>
<div class="rua-chart" data-src="../../assets/results/robodojo-round1.json" data-chart="score"></div>
<div class="rua-chart" data-src="../../assets/results/robodojo-round1.json" data-chart="tokens"></div>
<div class="rua-chart" data-src="../../assets/results/robodojo-round1.json" data-chart="tokens_per_success"></div>
</div>

??? note "Observations"
    Tasks: `stack_bowls`, `stack_blocks`, `general_pickup`, `put_bottles_into_dustbin`, `arrange_largest_number`,
    `classify_objects`, `stack_blocks_by_language` and `press_by_number`. One episode per task–algorithm pair
    shows where each paradigm works and what it costs; it is not a success rate. CaP on `classify_objects`
    stopped on a provider error and is excluded, and CaP and Direct were not run on `press_by_number`.

    - **The paradigms succeed on different tasks.** Monitor and Delegate solved the long pick-and-place tasks
      Pi0.5 was trained on (`put_bottles_into_dustbin`, `classify_objects`, `stack_bowls`) with 10k–50k tokens.
      CaP and Direct did better where the right object or precise alignment matters (`general_pickup`,
      `stack_blocks`, `stack_blocks_by_language`).
    - **CaP is the stronger model-controlled paradigm.** It has the highest mean score, and in each task it
      usually needs about half the decisions and tokens of Direct.
    - **Direct is the most expensive.** It used 61% of all tokens; most of its 171 tool errors had no
      inverse-kinematics solution.
    - **Monitor never corrected Pi0.5.** The model only let the policy continue, so Monitor performs like Pi0.5
      on its own.
    - **Single episodes vary a lot**, and the only `finish` call claimed a success the official check rejected.

    The episodes behind these charts are in
    [`robodojo-round1.json`](../assets/results/robodojo-round1.json).

Round 2 (2026-11-25): the same 8 tasks and settings with Qwen4-Max (reasoning effort xhigh) instead of GPT-6 Astra,
on RoboDojo `fbfbbcb`. Each episode may take 3 hours instead of 1, since Qwen4-Max takes about 50 seconds per
decision.

<div class="rua-row">
<div class="rua-chart" data-src="../../assets/results/robodojo-qwen4max.json" data-chart="success"></div>
<div class="rua-chart" data-src="../../assets/results/robodojo-qwen4max.json" data-chart="score"></div>
<div class="rua-chart" data-src="../../assets/results/robodojo-qwen4max.json" data-chart="tokens"></div>
<div class="rua-chart" data-src="../../assets/results/robodojo-qwen4max.json" data-chart="tokens_per_success"></div>
</div>

??? note "Observations"
    All four algorithms ran on all 8 tasks, 32 episodes. Two `put_bottles_into_dustbin` episodes were run again:
    CaP's after a provider error (HTTP 500) and Delegate's after the provider's content filter refused a request
    (HTTP 400, "inappropriate content"). The reruns finished normally.

    - **The Pi0.5 paradigms lead again.** Monitor succeeded in 4 of 8 episodes (`general_pickup`, `stack_bowls`,
      `classify_objects`, `put_bottles_into_dustbin`) with about 100k tokens and 10 minutes of model time per
      episode; Delegate in 2 (`stack_bowls`, `classify_objects`).
    - **CaP and Direct each solved only `general_pickup`.** Round 1 with GPT-6 Astra solved 3 of 6 CaP episodes;
      here CaP reached partial credit on 5 tasks but finished none of them.
    - **Direct is the most expensive and the slowest.** It used 53% of all tokens, and 4 of its 8 episodes ran
      out of the 3-hour wall-clock budget after about 100 decisions each.
    - **Qwen4-Max reasons more per decision.** Every algorithm used several times more tokens per episode than in
      round 1; overall 8 of 32 episodes succeeded, against 11 of 29 with GPT-6 Astra.
    - **Single episodes vary a lot.** An earlier CaP episode on `stack_blocks` with the same model succeeded; the one
      here scored 15.

    The episodes behind these charts are in
    [`robodojo-qwen4max.json`](../assets/results/robodojo-qwen4max.json).

## DexJoCo

Round 1 (2026-11-20, commit `c64fe56`): 5 single-arm tasks, seeds 0 to 9, GPT-6 Astra, DexJoCo's multi-task Pi0.5
for every algorithm.

<div class="rua-row">
<div class="rua-chart" data-src="../../assets/results/dexjoco-round1.json" data-chart="success"></div>
<div class="rua-chart" data-src="../../assets/results/dexjoco-round1.json" data-chart="tokens"></div>
<div class="rua-chart" data-src="../../assets/results/dexjoco-round1.json" data-chart="tokens_per_success"></div>
</div>

??? note "Observations"
    Tasks: `click_mouse`, `hammer_nail`, `water_plant`, `pick_bucket` and `fold_glasses`; 50 episodes per
    algorithm on the same seeds. Six episodes stopped by provider errors were run again. DexJoCo has no partial
    credit, so there is no score chart.

    - **Pi0.5 alone is the reference.** With the task instruction and no MLLM, it succeeded in 28 of 50 episodes
      on the same seeds. It already solves most `water_plant` and `hammer_nail` scenes, so most differences
      come from `click_mouse`, `pick_bucket` and `fold_glasses`.
    - **Delegate did best**, 36 of 50, and helped most on `click_mouse` (9 against 5). On the same seeds it
      succeeded 13 times where Pi0.5 alone failed and failed 5 times where Pi0.5 alone succeeded; with 50
      episodes this is suggestive but not significant (exact McNemar p = 0.10). The MLLM usually starts with the
      task instruction and then gives step instructions such as "Lower the held boxed food into the red bucket
      and release it", even though this Pi0.5 was trained only with whole task instructions.
    - **Monitor performed like Pi0.5 alone** (31 of 50) with the fewest tokens. In 175 calls it never corrected
      the robot; it only let Pi0.5 continue.
    - **`cap_delegate` did not beat Pi0.5 alone** (28 of 50) and used the most tokens per success. Only 16 of
      its 288 programs called `robot.move`, and 5 of its 8 `finish` calls claimed successes the official check
      rejected, for example a bucket lifted less than the required 15 cm.
    - **No algorithm solved `bimanual_assembly`** in three seeds each, and Pi0.5 alone fails it too. It is not
      in the charts.

    The episodes behind these charts are in
    [`dexjoco-round1.json`](../assets/results/dexjoco-round1.json).
