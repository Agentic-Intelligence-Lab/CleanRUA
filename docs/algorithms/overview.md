# Overview

CleanRUA classifies robot-use agents by two questions: **what does the MLLM send each turn**, and **what turns
it into robot motion**. The answers decide the shape of the interaction loop, so each paradigm is one algorithm
and one file.

![Four paradigms. Direct: the MLLM sends a target pose to inverse kinematics. Code: it sends a program to a Python robot API. Delegate: it sends a sub-task instruction to a VLA. Monitor: a VLA works on the task while the MLLM lets it continue or corrects the robot directly. In all four, camera images and robot state return to the MLLM.](../paradigms.svg){ width="100%" }

| Algorithm | The MLLM sends | Motion comes from | Based on |
|---|---|---|---|
| [Direct](direct.md) | One motion tool call, such as a target pose | Inverse kinematics | [uni-robot-agent](https://github.com/Open-RUA/uni-robot-agent) Clean mode; [GPT-6 Astra on RoboDojo](https://arxiv.org/abs/2609.24170) |
| [Code](code.md) | A Python program | A robot API called by the program | [Code as Policies](https://arxiv.org/abs/2209.07753); [CaP-X](https://arxiv.org/abs/2603.22435) |
| [Delegate](delegate.md) | A sub-task instruction for a VLA | Pi0.5 | [uni-robot-agent](https://github.com/Open-RUA/uni-robot-agent) hybrid mode; [Hi Robot](https://arxiv.org/abs/2502.19417) |
| [Monitor](monitor.md) | "Continue", or a correction | Pi0.5 with the task instruction, and the MLLM's motion tools | [Spotter](https://arxiv.org/abs/2609.36808); [GPT 6 Astra as an Embodied Policy](https://github.com/anonymous-report-421/GPT-as-Policy) |

Cross-episode learning, such as playbooks, memories and skill libraries, works on top of any of these
paradigms. It will be added as variants, for example `direct_memory_robodojo.py`.

## What every agent shares

Each agent file has the same layout, top to bottom:

1. `Args`: every hyperparameter, parsed with [tyro](https://github.com/brentyi/tyro). `--help` lists them.
2. `RoboDojoWrapper`: a `gym.Wrapper` that turns the joint-level environment into a tool-calling one. Tools are
   `@Tool` methods: the signature becomes the argument schema and the docstring the description shown to the
   model. A tool runs as a generator of joint-level actions, and `step()` executes them one by one with
   `for motion in self.parse_action(action)()`.
3. `SYSTEM_PROMPT`: the robot conventions and the paradigm's rules.
4. `RUAgent`: the policy. It sends the current images and state to the model and returns one tool call.
5. The main loop: reset, ask the agent, step the environment, log, and finally record the environment's verdict.

The interaction itself follows the Gymnasium API: the action is `{'name': tool, 'arguments': JSON}`, and the
observation is the environment observation plus `feedback`, a JSON string describing how the tool call went. A tool
that raises `ValueError` returns the error as feedback instead of crashing.

Defaults shared by all agents:

| Setting | Default |
|---|---|
| Context | System prompt, task text, the last 4 tool calls with their results (text only), and the current observation with all three camera images |
| Task text | The native instruction, per-task requirements from uni-robot-agent (for example returning the arms), and the initial grasp-site poses |
| Budgets | 60 decisions, 1,000,000 tokens and 1,800 seconds per episode |
| Model API | Chat Completions with `tool_choice=required`, one tool call per turn and `stream=False` |

## Logged outputs

Each run writes to `runs/<task>__<exp_name>__<seed>__<time>/`:

| Output | Content |
|---|---|
| `result.json` | Stop reason, official success and score, decisions, tokens, total reward and elapsed time |
| `trajectory.jsonl` | Every model reply, tool call, feedback and arm state |
| `images/` | The camera images shown to the model at each decision |
| `videos/` | All three cameras side by side, when `--capture-video` is on |
| TensorBoard | `charts/total_tokens`, `reward`, `native_steps`, `tool_ok`, `success` and `score` |

`finish` only ends the episode. Success and score always come from the environment's own check, which the agent
never sees.
