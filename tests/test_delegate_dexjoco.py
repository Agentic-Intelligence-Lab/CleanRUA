"""CPU tests for the DexJoCo delegate wrapper, with a fake DexJoCo environment and a fake Pi0.5 server."""
import json
import unittest

import gymnasium as gym
import numpy as np

from cleanrua import delegate_dexjoco as delegate

OPEN_HAND = np.r_[np.zeros(12), .263, 0., 0., 0.]


class FakeDexJoCo(gym.Env):
    """Arms reach every commanded pose and hand posture within one control step."""

    def __init__(self, arms=('right',)):
        self.arms = arms
        box = lambda size: gym.spaces.Box(-np.inf, np.inf, (size,), np.float64)
        self.observation_space = gym.spaces.Dict({
            'arms': gym.spaces.Dict({arm: gym.spaces.Dict({'pose': box(7), 'hand': box(16), 'hand_command': box(16)})
                                     for arm in arms}),
            'cameras': gym.spaces.Dict({}), 'native_steps': gym.spaces.Discrete(101)})

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.state = {arm: (np.array([-.4, .4 * i, 1.5, 0., 1., 0., 0.]), OPEN_HAND.copy())
                      for i, arm in enumerate(self.arms)}
        return self._observe(), {'instruction': 'Place the boxed food into the bucket.', 'step_limit': 100}

    def step(self, action):
        self.state = {arm: (np.array(action['arms'][arm]['pose']), np.array(action['arms'][arm]['hand']))
                      for arm in self.arms}
        self.steps += 1
        return self._observe(), 0., False, self.steps >= 100, {}

    def native_observation(self):
        return {'base': np.zeros((640, 640, 3), np.uint8), 'state': np.zeros(46)}

    def _observe(self):
        return {'arms': {arm: {'pose': pose.copy(), 'hand': hand.copy(), 'hand_command': hand.copy()}
                         for arm, (pose, hand) in self.state.items()}, 'cameras': {}, 'native_steps': self.steps}


class FakePolicy:
    """Predicts 30 steps; the n-th prediction puts the right hand at x=y=z=n and the left at -n, rotations zero."""

    def __init__(self):
        self.requests = []

    def reset(self):
        pass

    def infer(self, observation):
        self.requests.append(observation)
        actions = np.zeros((30, 44))
        actions[:, :3], actions[:, 22:25], actions[:, 6:22] = len(self.requests), -len(self.requests), .5
        actions.setflags(write=False)  # Like the arrays the real client decodes.
        return {'actions': actions}


def call(env, name, **arguments):
    obs, *_ = env.step({'name': name, 'arguments': json.dumps(arguments)})
    return obs, json.loads(obs['feedback'])


class ActorActTest(unittest.TestCase):
    def setUp(self):
        self.policy = FakePolicy()
        self.env = delegate.DexJoCoWrapper(FakeDexJoCo(('right', 'left')), self.policy)
        self.env.reset()

    def test_replans_below_24_planned_steps_and_blends_the_overlap(self):
        obs, feedback = call(self.env, 'actor_act', instruction='grasp the box', active_groups=['right'], steps=8)
        self.assertEqual(feedback, {'tool': 'actor_act', 'ok': True, 'steps': 8, 'inferences': 2})
        np.testing.assert_allclose(obs['arms']['right']['pose'][:3], 1 + 1 / 24)  # First blended step, after 7 steps.
        np.testing.assert_allclose(obs['arms']['right']['hand'], .5)
        np.testing.assert_allclose(obs['arms']['left']['pose'][:3], [-.4, .4, 1.5])  # Inactive arm holds.
        np.testing.assert_allclose(obs['arms']['left']['hand'], OPEN_HAND)
        request = self.policy.requests[0]
        self.assertEqual(request['prompt'], 'grasp the box')
        self.assertEqual(request['base'].shape, (224, 224, 3))

    def test_invalid_arguments_do_not_query_the_policy(self):
        for arguments in [{'instruction': ''}, {'instruction': 'x', 'active_groups': ['middle']},
                          {'instruction': 'x', 'active_groups': ['left', 'left']}, {'instruction': 'x', 'steps': 0}]:
            obs, feedback = call(self.env, 'actor_act', **arguments)
            self.assertFalse(feedback['ok'], arguments)
        self.assertEqual(self.policy.requests, [])


class MoveTest(unittest.TestCase):
    def setUp(self):
        self.env = delegate.DexJoCoWrapper(FakeDexJoCo(), FakePolicy())
        self.obs, _ = self.env.reset()

    def test_advances_one_centimetre_per_step_until_reached(self):
        call(self.env, 'hold', steps=1)
        self.env.unwrapped.state['right'][1][:] = 1.  # A closed hand, to see 'open' restore it.
        self.env.observation = self.env._with_poses(self.env.unwrapped._observe())
        goal = (self.obs['arms']['right']['pose'] + [0, 0, -.05, 0, 0, 0]).tolist()
        obs, feedback = call(self.env, 'move', targets=[{'arm': 'right', 'pose': goal, 'hand': 'open'}])
        self.assertEqual((feedback['steps'], feedback['stop_reason']), (5, 'targets_reached'))
        np.testing.assert_allclose(obs['arms']['right']['pose'], goal, atol=1e-9)
        np.testing.assert_allclose(obs['arms']['right']['hand'], OPEN_HAND)

    def test_missing_arm_is_feedback(self):
        obs, feedback = call(self.env, 'move', targets=[{'arm': 'left', 'pose': [0.] * 6}])
        self.assertEqual((feedback['ok'], feedback['steps']), (False, 0))
        self.assertIn('no left arm', feedback['error'])


if __name__ == '__main__':
    unittest.main()
