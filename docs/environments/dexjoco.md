# DexJoCo

[`cleanrua_utils/dexjoco_env.py`](https://github.com/Agentic-Intelligence-Lab/CleanRUA/blob/main/cleanrua_utils/dexjoco_env.py)
wraps the task environments of [DexJoCo](https://github.com/brave-eai/dexjoco), a MuJoCo benchmark for dexterous
manipulation, as a `gym.Env`. Each arm is a Franka Panda with a 16-joint Allegro hand. Scenes, control and success
checks are DexJoCo's own.

```python
from cleanrua_utils.dexjoco_env import DexJoCoEnv

env = DexJoCoEnv('/path/to/dexjoco', task='pick_bucket')
observation, info = env.reset(seed=0)   # info['instruction'] is the task instruction.
```

## Spaces

**Action**: one 50 Hz control step, ten physics substeps of DexJoCo's operational-space controller.

```python
{'arms': {arm: {'pose': (7,),     # target flange [x, y, z, qw, qx, qy, qz]
                'hand': (16,)}}}  # Allegro joint targets in radians
```

**Observation**:

```python
{'arms':    {arm: {'pose': (7,),            # measured flange pose
                   'hand': (16,),           # measured hand joints
                   'hand_command': (16,)}}, # last commanded hand joints
 'cameras': {name: {'rgb': (640, 640, 3),
                    'intrinsic_K': (3, 3),
                    'T_world_from_optical': (4, 4)}},   # x right, y down, z forward
 'native_steps': int}
```

Arms are `right` for single-arm tasks and `right`, `left` for bimanual ones. Hand joints are ordered index, middle,
ring and thumb, four each. The cameras are those the multi-task Pi0.5 policy was trained with, for example `front`
and `wrist` for `pick_bucket`. Reward is 1 on task success.

## Methods

| Method | What it does |
|---|---|
| `native_observation()` | Images and the 46-dimensional state in the format of DexJoCo's multi-task Pi0.5 policy |
| `evaluate()` | The official verdict: `success`, and `score` 100 or 0, since DexJoCo has no partial credit |

## Tasks

Single arm: `pick_bucket`, `click_mouse`, `fold_glasses`, `hammer_nail`, `pinch_tongs`, `water_plant`. Bimanual:
`bimanual_assembly`, `bimanual_hanoi`, `bimanual_microwave_cook`, `bimanual_photograph`, `bimanual_unlock_ipad`.
Instructions and cameras come from `configs/multi_task/<task>.yaml`; step limits range from 1,000 to 1,500 control
steps. Any seed works, and the same seed gives the same scene.

## Practical notes

- Images render with EGL unless `MUJOCO_GL` is set.
- DexJoCo paces its steps to 30 Hz of wall-clock time; the wrapper turns that off, so episodes run as fast as the
  simulator allows.
- For `click_mouse` the wrapper first holds the recorded start pose for 30 steps, as DexJoCo's evaluation does.
