---
hide:
  - toc
---

# Evaluation

Every algorithm runs on the same tasks with the same model and budget. The environment's own step limit ends each
episode, and its official check decides success.

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
    shows where each paradigm works and what it costs; it is not a success rate. Code on `classify_objects`
    stopped on a provider error and is excluded, and Code and Direct were not run on `press_by_number`.

    - **The paradigms succeed on different tasks.** Monitor and Delegate solved the long pick-and-place tasks
      Pi0.5 was trained on (`put_bottles_into_dustbin`, `classify_objects`, `stack_bowls`) with 10k–50k tokens.
      Code and Direct did better where the right object or precise alignment matters (`general_pickup`,
      `stack_blocks`, `stack_blocks_by_language`).
    - **Code is the stronger model-controlled paradigm.** It has the highest mean score, and in each task it
      usually needs about half the decisions and tokens of Direct.
    - **Direct is the most expensive.** It used 61% of all tokens; most of its 171 tool errors had no
      inverse-kinematics solution.
    - **Monitor never corrected Pi0.5.** The model only let the policy continue, so Monitor performs like Pi0.5
      on its own.
    - **Single episodes vary a lot**, and the only `finish` call claimed a success the official check rejected.

    The episodes behind these charts are in
    [`robodojo-round1.json`](../assets/results/robodojo-round1.json).

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
    - **`code_delegate` did not beat Pi0.5 alone** (28 of 50) and used the most tokens per success. Only 16 of
      its 288 programs called `robot.move`, and 5 of its 8 `finish` calls claimed successes the official check
      rejected, for example a bucket lifted less than the required 15 cm.
    - **No algorithm solved `bimanual_assembly`** in three seeds each, and Pi0.5 alone fails it too. It is not
      in the charts.

    The episodes behind these charts are in
    [`dexjoco-round1.json`](../assets/results/dexjoco-round1.json).
