"""A DexJoCo task with Franka Panda arms and Allegro hands, as a Gymnasium environment.

A thin layer over DexJoCo's own task environments, so scenes, control and success checks are the
official ones. Nothing here knows about tools or language models.

Action, one 50 Hz control step (ten 2 ms physics substeps of DexJoCo's operational-space controller):
    {'arms': {arm: {'pose': (7,) target flange [x, y, z, qw, qx, qy, qz], 'hand': (16,) Allegro joint targets}}}
Observation:
    {'arms':    {arm: {'pose' (7,) measured flange pose, 'hand' (16,) measured joints, 'hand_command' (16,)}},
     'cameras': {name: {'rgb' (H, W, 3), 'intrinsic_K' (3, 3), 'T_world_from_optical' (4, 4)}},
     'native_steps': control steps taken}
Arms are 'right' for single-arm tasks and 'right', 'left' for bimanual ones. Poses are in the world frame, in
metres; camera optical frames are x right, y down, z forward. Hand joints are in radians, ordered index, middle,
ring and thumb, four joints each. Reward is 1 on task success. Images are rendered with EGL unless MUJOCO_GL is set.
"""
import os
import random
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from scipy.spatial.transform import Rotation

# Native episode length of each task, in control steps.
STEP_LIMITS = {'bimanual_assembly': 1500, 'bimanual_hanoi': 1500, 'bimanual_microwave_cook': 1100,
               'bimanual_photograph': 1000, 'bimanual_unlock_ipad': 1200, 'click_mouse': 1200, 'fold_glasses': 1200,
               'hammer_nail': 1000, 'pick_bucket': 1200, 'pinch_tongs': 1000, 'water_plant': 1000}


class DexJoCoEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 50}

    def __init__(self, native_root: Path, task: str = 'pick_bucket', seed: int = 0, render_mode: str | None = None):
        from dexjoco.tasks import CONFIG_MAPPING
        self.render_mode, self.task, self.step_limit = render_mode, task, STEP_LIMITS[task]
        # The multi-task configs name the instruction and the cameras the released Pi0.5 policy was trained with.
        self.config = yaml.safe_load((Path(native_root) / 'configs/multi_task' / f'{task}.yaml').read_text())
        self.arms = ('right', 'left') if self.config['robot_type'] == 'dual_arm' else ('right',)
        self.cameras = tuple(dict.fromkeys(self.config['camera_mapping'].values()))
        self._seed = seed
        self._seed_rngs(seed)  # DexJoCo samples its scenes from the global NumPy and Python generators.
        self.native = CONFIG_MAPPING[task]().get_environment(policy_mode=True, render_mode='rgb_array', seed=seed)
        self.native.unwrapped.hz = 1e9  # Native stepping sleeps to keep 30 Hz wall-clock time; run as fast as possible.

        observation, _ = self.reset(seed=seed)
        box = lambda shape, low=-np.inf, high=np.inf, dtype=np.float64: spaces.Box(low, high, shape, dtype)
        arm = lambda keys: spaces.Dict({key: box((7,) if key == 'pose' else (16,)) for key in keys})
        self.action_space = spaces.Dict({'arms': spaces.Dict({name: arm(('pose', 'hand')) for name in self.arms})})
        self.observation_space = spaces.Dict({
            'arms': spaces.Dict({name: arm(('pose', 'hand', 'hand_command')) for name in self.arms}),
            'cameras': spaces.Dict({name: spaces.Dict({
                'rgb': box(camera['rgb'].shape, 0, 255, np.uint8), 'intrinsic_K': box((3, 3)),
                'T_world_from_optical': box((4, 4))}) for name, camera in observation['cameras'].items()}),
            'native_steps': spaces.Discrete(self.step_limit + 1)})

    @staticmethod
    def _seed_rngs(seed: int) -> None:
        np.random.seed(seed)
        random.seed(seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:  # The same seed always gives the same scene.
            self._seed_rngs(seed)
        native, _ = self.native.reset()
        if self.task == 'click_mouse':  # As in the official evaluation: first raise the hand to the recorded start pose.
            start = [-.44294, 1.3729e-06, 1.517, *Rotation.from_rotvec([-3.14156462, -6.91584035e-05, -1.40317984e-03])
                     .as_quat(scalar_first=True), *np.eye(16)[12] * .263]
            for _ in range(30):
                native, *_ = self.native.step(np.array(start))
        self.steps, self.success = 0, False
        state = native['state']
        self._hand_command = {arm: state[self._hand_slice(arm)].copy() for arm in self.arms}
        return self._observe(native), {'instruction': self.config['prompt'], 'step_limit': self.step_limit}

    def step(self, action):
        arms = action['arms']
        if len(self.arms) == 1:
            native_action = np.concatenate([arms['right']['pose'], arms['right']['hand']])
        else:  # [right pose, left pose, right hand, left hand]
            native_action = np.concatenate([arms['right']['pose'], arms['left']['pose'],
                                            arms['right']['hand'], arms['left']['hand']])
        native, _, native_done, _, info = self.native.step(native_action.astype(np.float64))
        self._hand_command = {arm: np.asarray(arms[arm]['hand'], np.float64).copy() for arm in self.arms}
        self.steps += 1
        self.success = self.success or bool(info.get('succeed', False))
        truncated = not self.success and self.steps >= self.step_limit
        terminated = bool(native_done) and not truncated
        return self._observe(native), float(self.success), terminated, truncated, {'success': self.success}

    def native_observation(self) -> dict:
        """The latest observation in the format of DexJoCo's multi-task Pi0.5 policy, without the prompt:
        images named as in the task's multi-task config, and the 46-dimensional dual-arm state, zero-padded for
        single-arm tasks. Images are full resolution; the policy client resizes them."""
        state = self._native['state'][:23 * len(self.arms)]
        return {**{key: self._native[name] for key, name in self.config['camera_mapping'].items()},
                'state': np.concatenate([state, np.zeros(46 - len(state))])}

    def evaluate(self) -> dict:
        """Official verdict: DexJoCo tasks have no partial credit, so the score is 100 or 0."""
        return {'success': self.success, 'score': 100. if self.success else 0.}

    def render(self):
        if self.render_mode == 'rgb_array':
            return np.concatenate([self._native[name] for name in self.cameras], axis=1)

    def close(self):
        self.native.close()

    def _hand_slice(self, arm: str) -> slice:
        # Native state: single-arm [pose 7, hand 16]; bimanual [right pose 7, left pose 7, right hand 16, left hand 16].
        if len(self.arms) == 1:
            return slice(7, 23)
        return slice(14, 30) if arm == 'right' else slice(30, 46)

    def _observe(self, native: dict) -> dict:
        self._native, state = native, native['state']
        arms = {}
        for i, arm in enumerate(self.arms):
            arms[arm] = {'pose': np.asarray(state[7 * i:7 * i + 7], np.float64),
                         'hand': np.asarray(state[self._hand_slice(arm)], np.float64),
                         'hand_command': self._hand_command[arm].copy()}
        model, data, cameras = self.native.unwrapped.model, self.native.unwrapped.data, {}
        height, width = model.vis.global_.offheight, model.vis.global_.offwidth
        names = {'ego': 'back', 'ego_left': 'left', 'ego_right': 'right', 'wrist': 'handcam_rgb',
                 'wrist_left': 'handcam_rgb_left', 'wrist_right': 'handcam_rgb_right'}
        for name in self.cameras:  # Calibration from the MuJoCo cameras: their frames look along -z with y up.
            camera = model.camera(names.get(name, name)).id
            focal = height / 2 / np.tan(np.deg2rad(model.cam_fovy[camera]) / 2)
            pose = np.eye(4)
            pose[:3, :3] = data.cam_xmat[camera].reshape(3, 3) @ np.diag([1., -1., -1.])
            pose[:3, 3] = data.cam_xpos[camera]
            cameras[name] = {'rgb': np.asarray(native[name], np.uint8),
                             'intrinsic_K': np.array([[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]]),
                             'T_world_from_optical': pose}
        return {'arms': arms, 'cameras': cameras, 'native_steps': self.steps}
