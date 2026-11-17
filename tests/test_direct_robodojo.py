"""CPU tests for the tool wrapper and the agent. No simulator, no network."""
import json
import os
import unittest
import warnings
from unittest import mock

import gymnasium as gym
import httpx
import numpy as np
from gymnasium.utils.env_checker import check_env

from cleanrua import direct_robodojo as direct

ARMS = ('left', 'right')


class FakeRoboDojo(gym.Env):
    """Same spaces as RoboDojoEnv; solve_ik remembers the requested pose and step() reaches it.
    Like the real X5, a closed empty gripper measures above 0 (here 0.2)."""

    def __init__(self, ik_fails_after: int | None = None, stuck: bool = False):
        box = lambda shape, low=-np.inf, high=np.inf, dtype=np.float64: gym.spaces.Box(low, high, shape, dtype)
        arm = gym.spaces.Dict({'joints': box((6,)), 'ee_pose': box((7,)), 'gripper': box((), 0, 1),
                               'gripper_command': box((), 0, 1)})
        self.action_space = gym.spaces.Dict({'joints': gym.spaces.Dict({a: box((6,)) for a in ARMS}),
                                             'grippers': gym.spaces.Dict({a: box((), 0, 1) for a in ARMS})})
        self.observation_space = gym.spaces.Dict({
            'arms': gym.spaces.Dict({a: arm for a in ARMS}),
            'cameras': gym.spaces.Dict({'head': gym.spaces.Dict({
                'rgb': box((4, 4, 3), 0, 255, np.uint8), 'intrinsic_K': box((3, 3)), 'T_world_from_optical': box((4, 4))})}),
            'native_steps': gym.spaces.Discrete(801)})
        self.ik_fails_after, self.ik_calls, self.stuck = ik_fails_after, 0, stuck

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps, self.pending = 0, {}
        self.arms = {a: {'joints': np.zeros(6), 'ee_pose': np.array([x, 0., .9, 1., 0., 0., 0.]),
                         'gripper': np.array(1.), 'gripper_command': np.array(1.)} for a, x in zip(ARMS, (-.3, .3))}
        return self._obs(), {'instruction': 'Stack the bowls.'}

    def _obs(self):
        camera = {'rgb': np.zeros((4, 4, 3), np.uint8), 'intrinsic_K': np.eye(3), 'T_world_from_optical': np.eye(4)}
        return {'arms': {a: {k: np.array(v) for k, v in arm.items()} for a, arm in self.arms.items()},
                'cameras': {'head': camera}, 'native_steps': self.steps}

    def solve_ik(self, arm, position, quaternion):
        self.ik_calls += 1
        if self.ik_fails_after is not None and self.ik_calls > self.ik_fails_after:
            return None
        self.pending[arm] = np.concatenate([position, quaternion])
        return self.arms[arm]['joints'] + .01

    def step(self, action):
        assert self.action_space.contains(action), 'wrapper must emit joint-level actions'
        for a in ARMS:
            command = np.array(action['grippers'][a], np.float64)
            self.arms[a].update(joints=action['joints'][a].copy(), gripper=np.clip(command, .2, 1.), gripper_command=command)
            if a in self.pending and not self.stuck:  # A stuck arm never reaches the requested link6 pose.
                self.arms[a]['ee_pose'] = self.pending.pop(a)
        self.steps += 1
        return self._obs(), 0., False, False, {'success': False}

    def native_observation(self):
        return {'state': {f'{a}_arm_joint_state': self.arms[a]['joints'].tolist() for a in ARMS}, 'instruction': 'native'}

    def evaluate(self):
        return {'success': False, 'score': 0.}


def call(name, **arguments):
    return {'name': name, 'arguments': json.dumps(arguments)}


class WrapperTest(unittest.TestCase):
    def setUp(self):
        self.env = direct.RoboDojoWrapper(FakeRoboDojo())
        self.env.reset(seed=0)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, json.loads(obs['feedback']), terminated

    def test_gymnasium_api(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # Expected: "env is wrapped" and unbounded Box warnings.
            check_env(direct.RoboDojoWrapper(FakeRoboDojo()), skip_render_check=True)

    def test_schemas_come_from_signatures_and_docstrings(self):
        schemas = {t['function']['name']: t['function'] for t in self.env.schemas()}
        self.assertEqual(set(schemas), {'move', 'hold', 'capture_scene', 'finish'})
        move = schemas['move']['parameters']
        self.assertEqual(move['required'], ['targets'])
        target = move['properties']['targets']['items']
        self.assertEqual((target['properties']['arm']['enum'], set(target['required'])), (['left', 'right'], {'arm', 'pose'}))
        self.assertFalse(move['additionalProperties'] or target['additionalProperties'])
        self.assertNotIn('$ref', json.dumps(schemas))  # Nested models are inlined for providers without $ref support.
        self.assertEqual(move['properties']['profile']['enum'], ['standard', 'approach', 'precision'])
        self.assertEqual(set(schemas['finish']['parameters']['required']), {'status', 'summary'})
        self.assertTrue(all(schema['description'] for schema in schemas.values()))

    def test_hold_executes_each_yielded_motion(self):
        obs, feedback, terminated = self.step(call('hold', steps=3))
        self.assertEqual(feedback, {'tool': 'hold', 'ok': True, 'steps': 3})
        self.assertEqual(obs['native_steps'], 3)
        self.assertFalse(terminated)

    def test_observation_adds_grasp_site_pose(self):
        obs, _ = self.env.reset()
        np.testing.assert_allclose(obs['arms']['left']['pose'], [-.3 + .149335, -.000002, .9 - .000854, 0., 0., 0.])

    def test_move_stops_once_pose_reached_and_fingers_rest(self):
        pose = (self.env.observation['arms']['left']['pose'] + [0., 0., .05, 0., 0., 0.]).tolist()
        # Precision: 1 cm increments and 5 mm tolerance, so 5 cm takes five steps; fingers rest after the first.
        obs, feedback, _ = self.step(call('move', targets=[{'arm': 'left', 'pose': pose, 'gripper': 0.}], profile='precision'))
        self.assertEqual({k: feedback[k] for k in ('ok', 'steps', 'targets_reached', 'stop_reason')},
                         {'ok': True, 'steps': 5, 'targets_reached': True, 'stop_reason': 'targets_reached'})
        self.assertEqual(set(feedback['errors']), {'left'})
        left, right = obs['arms']['left'], obs['arms']['right']
        np.testing.assert_allclose(left['pose'], pose, atol=1e-9)
        np.testing.assert_allclose(left['ee_pose'][:3], np.array(pose[:3]) - direct.GRASP_SITE, atol=1e-9)  # IK on link6.
        self.assertEqual((left['gripper_command'], left['gripper']), (0., .2))
        self.assertEqual(right['gripper_command'], 1.)  # The other arm holds.
        np.testing.assert_allclose(right['joints'], 0.)

    def test_move_coordinates_both_arms(self):
        arms = self.env.observation['arms']
        targets = [{'arm': arm, 'pose': (arms[arm]['pose'] + [0., .02, 0., 0., 0., 0.]).tolist()} for arm in ('left', 'right')]
        obs, feedback, _ = self.step(call('move', targets=targets, profile='precision'))
        self.assertEqual((feedback['steps'], feedback['targets_reached'], set(feedback['errors'])), (2, True, {'left', 'right'}))
        for target in targets:
            np.testing.assert_allclose(obs['arms'][target['arm']]['pose'], target['pose'], atol=1e-9)

    def test_move_stops_when_progress_stalls(self):
        env = direct.RoboDojoWrapper(FakeRoboDojo(stuck=True))
        obs, _ = env.reset()
        goal = (obs['arms']['left']['pose'] + [0., 0., .1, 0., 0., 0.]).tolist()
        obs, *_ = env.step(call('move', targets=[{'arm': 'left', 'pose': goal}]))
        feedback = json.loads(obs['feedback'])
        self.assertEqual((feedback['ok'], feedback['steps'], feedback['stop_reason']), (True, 4, 'stalled'))

    def test_move_reports_ik_failure_after_partial_execution(self):
        env = direct.RoboDojoWrapper(FakeRoboDojo(ik_fails_after=2))
        obs, _ = env.reset()
        goal = (obs['arms']['left']['pose'] + [0., 0., .1, 0., 0., 0.]).tolist()
        obs, *_ = env.step(call('move', targets=[{'arm': 'left', 'pose': goal}], profile='precision'))
        feedback = json.loads(obs['feedback'])
        self.assertEqual((feedback['ok'], feedback['steps'], feedback['targets_reached']), (False, 2, False))
        self.assertIn('inverse-kinematics', feedback['error'])

    def test_invalid_calls_become_feedback_without_motion(self):
        pose = [0.] * 6
        for action in [call('fly'), {'name': 'hold', 'arguments': '{bad json'}, call('hold', steps=0),
                       call('hold', speed=1), call('move', targets=[]), call('move', targets=[{'arm': 'middle', 'pose': pose}]),
                       call('move', targets=[{'arm': 'left', 'pose': pose}, {'arm': 'left', 'pose': pose}]),
                       call('move', targets=[{'arm': 'left', 'pose': pose}], profile='fast'),
                       call('finish'), call('finish', status='failure', summary='x'), call('finish', status='success', summary='')]:
            obs, feedback, terminated = self.step(action)
            self.assertFalse(feedback['ok'], action)
            self.assertEqual(feedback['steps'], 0)
            self.assertEqual(obs['native_steps'], 0)
            self.assertFalse(terminated)

    def test_capture_scene_and_finish_do_not_step_physics(self):
        obs, feedback, terminated = self.step(call('capture_scene'))
        self.assertEqual((feedback['ok'], feedback['steps'], terminated), (True, 0, False))
        obs, feedback, terminated = self.step(call('finish', status='success', summary='Bowls stacked, arms home.'))
        self.assertEqual((feedback['ok'], obs['native_steps'], terminated), (True, 0, True))
        self.assertTrue(self.env.finished)


class AgentTest(unittest.TestCase):
    def test_tool_calls_and_results_are_paired_in_history(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            n = len(requests)
            return httpx.Response(200, json={'usage': {'total_tokens': 10}, 'choices': [{'message': {
                'role': 'assistant', 'content': 'look', 'tool_calls': [{'id': f'call-{n}', 'type': 'function',
                    'function': {'name': 'capture_scene', 'arguments': '{}'}}]}}]})

        variables = {'CLEANRUA_API_KEY': 'test-key', 'CLEANRUA_BASE_URL': 'https://model.invalid/v1',
                     'CLEANRUA_MODEL_NAME': 'test-model'}
        with mock.patch.dict(os.environ, variables):
            agent = direct.RUAgent(direct.Args(), tools=[])
        agent.client = httpx.Client(base_url=agent.client.base_url, transport=httpx.MockTransport(handler))
        env = direct.RoboDojoWrapper(FakeRoboDojo())
        obs, _ = env.reset()
        agent.reset('Task: stack the bowls.')
        action = agent.get_action(obs)
        self.assertEqual(action, {'name': 'capture_scene', 'arguments': '{}'})
        obs, *_ = env.step(action)
        agent.get_action(obs)

        first, second = requests
        self.assertEqual((first['model'], first['stream']), ('test-model', False))  # Some providers stream by default.
        self.assertEqual([m['role'] for m in first['messages']], ['system', 'user', 'user'])
        self.assertEqual([m['role'] for m in second['messages']], ['system', 'user', 'assistant', 'tool', 'user'])
        self.assertEqual(second['messages'][3], {'role': 'tool', 'tool_call_id': 'call-1', 'content': obs['feedback']})
        observation = json.loads(second['messages'][-1]['content'][0]['text'].removeprefix('Current observation: '))
        np.testing.assert_allclose(observation['arms']['left']['pose'],  # Grasp site as [xyz, rpy], rounded.
                                   np.round([-.3 + .149335, -.000002, .9 - .000854, 0., 0., 0.], 4))
        images = [part for part in second['messages'][-1]['content'] if part['type'] == 'image_url']
        self.assertEqual(len(images), 1)  # Only the current observation carries images.
        self.assertTrue(images[0]['image_url']['url'].startswith('data:image/png;base64,'))
        self.assertEqual(agent.total_tokens, 20)


if __name__ == '__main__':
    unittest.main()
