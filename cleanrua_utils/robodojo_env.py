"""A RoboDojo task with two ARX X5 arms, as a Gymnasium environment.

A thin layer over RoboDojo's own evaluator, so control, scene setup and success checks are
the official ones. Nothing here knows about tools or language models.

Action, one 25 Hz control step (10 physics ticks, interpolated natively):
    {'joints': {arm: (6,) joint targets}, 'grippers': {arm: opening, 0 closed .. 1 open}}
Observation:
    {'arms':    {arm: {'joints' (6,), 'ee_pose' (7,) link6 [x, y, z, qw, qx, qy, qz],
                       'gripper' measured opening, 'gripper_command' last commanded opening}},
     'cameras': {name: {'rgb' (H, W, 3), 'intrinsic_K' (3, 3), 'T_world_from_optical' (4, 4)}},
     'native_steps': control steps taken}
Poses are in the environment-local world frame, in metres; camera optical frames are x right,
y down, z forward. Reward is 1 on task success.
"""
import atexit
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation

ARMS = ('left', 'right')


class RoboDojoEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 25}

    def __init__(self, native_root: Path, task: str = 'stack_bowls', cameras=('head', 'left_wrist', 'right_wrist'),
                 layout: Path | None = None, render_mode: str | None = None):
        self.render_mode, self.cameras, root = render_mode, tuple(cameras), Path(native_root).resolve()
        sys.path[:0] = [str(root), str(root / 'XPolicyLab')]  # RoboDojo's `utils` must win over XPolicyLab's.
        from isaaclab.app import AppLauncher
        from env.camera_manager.capture.render_sync import zero_delay_kit_args
        self.app = AppLauncher(headless=True, enable_cameras=True, kit_args=zero_delay_kit_args()).app
        atexit.register(self.app.close)  # Closing Isaac Sim ends the process, so it runs after everything else.
        from omegaconf import OmegaConf
        from src.eval_client import eval_env
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization

        evaluation = load_yaml(str(root / 'env_cfg/arx_x5.yml'))
        evaluation.update(task_name=task, policy_name='cleanrua', num_envs=1, seed=0, eval_num=1)
        config = OmegaConf.create({
            **{key: load_yaml(str(root / 'env_cfg' / key / f'{name}.yml')) for key, name in evaluation['config'].items()},
            'task_env': load_yaml(str(root / f'task/RoboDojo/config/{task}.yml')),
            'eval_cfg': evaluation, 'deploy_cfg': {'port': 0}})
        config, _ = process_config(process_randomization(config), task_name=task)
        config.sim.scene.num_envs, config.sim.seed, config.camera.default_frequency = 1, [0], 25
        # The evaluator normally queries a remote policy; here actions arrive through step().
        eval_env.WsModelClient = lambda **_: SimpleNamespace(call=lambda **_: None)
        self.sim = eval_env.create_eval_env(config, self.app)
        self._videos = tempfile.TemporaryDirectory()  # Native per-episode recordings, which we do not keep.
        self.sim._stream_dir = self._videos.name
        seeds = self.sim.seed_manager
        layout = layout or seeds.seed_info[seeds.seed_list[0]]['scene_layout']  # Official layout of seed 0.
        seeds.seed_info, self._started = {0: {'scene_layout': str(layout)}}, False
        self._chains = {arm: self._urdf_chain(arm) for arm in ARMS}

        observation, _ = self.reset()  # Robots and cameras exist only once a scene is loaded.
        box = lambda shape, low=-np.inf, high=np.inf, dtype=np.float64: spaces.Box(low, high, shape, dtype)
        arm = spaces.Dict({'joints': box((6,)), 'ee_pose': box((7,)), 'gripper': box((), 0, 1),
                           'gripper_command': box((), 0, 1)})
        self.action_space = spaces.Dict({'joints': spaces.Dict({name: box((6,)) for name in ARMS}),
                                         'grippers': spaces.Dict({name: box((), 0, 1) for name in ARMS})})
        self.observation_space = spaces.Dict({
            'arms': spaces.Dict({name: arm for name in ARMS}),
            'cameras': spaces.Dict({name: spaces.Dict({
                'rgb': box(camera['rgb'].shape, 0, 255, np.uint8), 'intrinsic_K': box((3, 3)),
                'T_world_from_optical': box((4, 4))}) for name, camera in observation['cameras'].items()}),
            'native_steps': spaces.Discrete(self.sim.step_lim + 1)})

    def reset(self, *, seed=None, options=None):
        if seed not in (None, 0):
            raise ValueError('Only seed 0, the official evaluation layout, is supported')
        super().reset(seed=seed)
        if self._started:
            self.sim.close()  # As in native evaluation: tear the scene down between episodes.
        self.sim.reset(seed=[0])  # Loads the layout, lets objects settle and records the arms' start poses.
        self.sim.run_reward()  # Register the task's success checks ...
        if hasattr(self.sim, 'get_score'):
            self.sim.get_score()  # ... and its partial-credit score, which not every task has.
        self._started, self._gripper_command = True, dict.fromkeys(ARMS, 1.)  # Native reset opens both grippers.
        observation = self._observe()
        return observation, {'instruction': self.sim.obs_manager.instruction[0]}

    def step(self, action):
        self.sim.take_action({  # Interpolates natively, advances physics and checks success and failure.
            **{f'{arm}_arm_joint_state': np.asarray(action['joints'][arm]).tolist() for arm in ARMS},
            **{f'{arm}_ee_joint_state': [float(action['grippers'][arm])] for arm in ARMS}})
        # Keep the commanded opening: native state reports a ramped value while fingers move slowly.
        self._gripper_command = {arm: float(action['grippers'][arm]) for arm in ARMS}
        success = bool(self.sim.end_flag[0] and self.sim.success[0])
        truncated = not success and self.sim.take_action_cnt[0] >= self.sim.step_lim
        terminated = bool(self.sim.end_flag[0]) and not truncated
        return self._observe(), float(success), terminated, truncated, {'success': success}

    def solve_ik(self, arm: str, position, quaternion) -> np.ndarray | None:
        """Native inverse kinematics: joints placing `arm`'s link6 at a pose (wxyz quaternion), or None."""
        robot = self.sim.robot_manager.get_robot_by_arm_name(f'{arm}_arm')
        result = self.sim.robot_manager.solve_ik([*position, *quaternion], 0, robot, trans='world')
        return np.asarray(result['joint_value'], np.float64) if result['status'] == 'Success' else None

    def forward_kinematics(self, arm: str, joints) -> np.ndarray:
        """Robot-only kinematics: `arm`'s link6 pose [x, y, z, qw, qx, qy, qz] for joint positions."""
        robot = self.sim.robot_manager.get_robot_by_arm_name(f'{arm}_arm')
        base = self.sim.robot_manager.get_link_pose(robot, robot.base_link, is_relative=True)[0]
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = Rotation.from_quat(base[3:], scalar_first=True).as_matrix(), base[:3]
        values = dict(zip(robot.arm_joints_name, joints))
        for name, origin, axis in self._chains[arm]:
            motion = np.eye(4)
            if axis is not None:
                motion[:3, :3] = Rotation.from_rotvec(axis * values[name]).as_matrix()
            pose = pose @ origin @ motion
        return np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)]

    def native_observation(self) -> dict:
        """The latest observation in RoboDojo's own format, as consumed by policies trained on RoboDojo data."""
        return self._native

    def evaluate(self) -> dict:
        """Official verdict for the current state, including end-of-episode checks. As in native evaluation,
        success scores 100; otherwise tasks with partial credit report it and the others score 0."""
        rewards = self.sim.reward_manager
        reward = rewards.get_reward(final_check=True)[0]
        success = bool(reward > 1 - 1e-3)
        partial = float(rewards.get_score(reward_lst=[reward])[0]) if hasattr(self.sim, 'get_score') else 0.
        return {'success': success, 'score': 100. if success else partial}

    def render(self):
        if self.render_mode == 'rgb_array':
            return np.concatenate([camera['rgb'] for camera in self._observe()['cameras'].values()], axis=1)

    def close(self):
        if self._started:
            self.sim.close()
            self._videos.cleanup()
            self._started = False

    def _urdf_chain(self, arm: str) -> list:
        """Joints from base_link to link6 in the robot's URDF: (name, origin transform, axis or None if fixed)."""
        robot = self.sim.robot_manager.get_robot_by_arm_name(f'{arm}_arm')
        joints = {joint.find('child').get('link'): joint for joint in ElementTree.parse(robot.urdf_path).iter('joint')}
        chain, link = [], robot.ee_link_name
        while link != robot.base_link:
            joint, origin = joints[link], np.eye(4)
            element = joint.find('origin') if joint.find('origin') is not None else ElementTree.Element('origin')
            origin[:3, :3] = Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            origin[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
            axis = None if joint.get('type') == 'fixed' else np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            chain.insert(0, (joint.get('name'), origin, axis))
            link = joint.find('parent').get('link')
        return chain

    def _observe(self) -> dict:
        self.sim.obs_manager.render_for_capture()
        native = self._native = self.sim.obs_manager.get_obs(env_idx_list=[0])[0]
        state, rm, arms = native['state'], self.sim.robot_manager, {}
        for arm in ARMS:
            robot = rm.get_robot_by_arm_name(f'{arm}_arm')
            (low, high), finger = robot.gripper_scale, rm.get_end_effector_real_val(robot, [0])[0][0]
            opening = (finger - low) / (high - low)  # Native normalization, as for gripper_command.
            arms[arm] = {'joints': np.asarray(state[f'{arm}_arm_joint_state'], np.float64),
                         'ee_pose': np.asarray(state[f'{arm}_ee_pose'], np.float64),
                         'gripper': np.array(np.clip(opening if robot.gripper_move['sign'] == 1 else 1 - opening, 0, 1)),
                         'gripper_command': np.array(self._gripper_command[arm])}
        cameras, manager = {}, self.sim.camera_manager
        for name in self.cameras:  # Calibration from the rendering cameras themselves.
            camera = manager.cameras[0][manager.camera_names[0].index(f'cam_{name}')]
            position, quaternion = camera.get_world_pose(camera_axes='ros')
            pose = np.eye(4)
            pose[:3, :3], pose[:3, 3] = Rotation.from_quat(quaternion, scalar_first=True).as_matrix(), position
            cameras[name] = {'rgb': np.asarray(native['vision'][f'cam_{name}']['color'], np.uint8),
                             'intrinsic_K': np.asarray(camera.get_intrinsics_matrix(), np.float64),
                             'T_world_from_optical': pose}
        return {'arms': arms, 'cameras': cameras, 'native_steps': int(self.sim.take_action_cnt[0])}
