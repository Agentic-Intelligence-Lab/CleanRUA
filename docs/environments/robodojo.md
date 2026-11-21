# RoboDojo

[`cleanrua_utils/robodojo_env.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua_utils/robodojo_env.py)
wraps [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)'s official evaluator as a `gym.Env` with two
ARX X5 arms. Scene setup, control and success checks are RoboDojo's own. The environment knows nothing about tools
or language models; each agent adds those with a `gym.Wrapper`.

```python
from cleanrua_utils.robodojo_env import RoboDojoEnv

env = RoboDojoEnv('/path/to/RoboDojo', task='stack_bowls')
observation, info = env.reset(seed=0)   # info['instruction'] is the native task instruction.
```

## Spaces

**Action**: one 25 Hz control step, which RoboDojo interpolates over 10 physics ticks.

```python
{'joints':   {'left': (6,), 'right': (6,)},   # joint targets in radians
 'grippers': {'left': opening, 'right': opening}}   # 0 closed .. 1 open
```

**Observation**:

```python
{'arms':    {arm: {'joints': (6,),
                   'ee_pose': (7,),            # link6 [x, y, z, qw, qx, qy, qz]
                   'gripper': (),              # measured opening
                   'gripper_command': ()}},    # last commanded opening
 'cameras': {name: {'rgb': (H, W, 3),
                    'intrinsic_K': (3, 3),
                    'T_world_from_optical': (4, 4)}},   # x right, y down, z forward
 'native_steps': int}
```

Poses are in the environment-local world frame, in metres. The three cameras are `head`, `left_wrist` and
`right_wrist`. Reward is 1 on task success.

## Methods

| Method | What it does |
|---|---|
| `solve_ik(arm, position, quaternion)` | Native inverse kinematics for link6; returns joint positions or `None` |
| `native_observation()` | The latest observation in RoboDojo's own format, as consumed by policies trained on RoboDojo data |
| `evaluate()` | The official verdict: `success`, and `score` (100 on success, otherwise the task's partial credit, or 0 if it has none) |

## Tasks

`task` takes any name in `RoboDojo/task/RoboDojo/config`, such as `stack_bowls`, `stack_blocks`,
`general_pickup` or `put_bottles_into_dustbin`. Only seed 0, the official evaluation layout, is supported. Each
task has a native step limit, for example 800 control steps for `stack_bowls`.

## Practical notes

- Set `OMNI_KIT_ACCEPT_EULA=YES`; otherwise Isaac Sim waits for an interactive prompt.
- The first launch takes about 100 seconds.
- Isaac Sim uses about 7 GB of GPU memory and the Pi0.5 server about 8.5 GB. On a 16 GB GPU, some scenes, such
  as `general_pickup` and `pour_liquid_into_cup`, do not fit next to the server. Stop the server before running
  Direct or CaP.
