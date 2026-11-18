"""CPU tests for the delegate wrapper's actor_act tool, with a fake action-model server."""
import json
import unittest

import numpy as np

from cleanrua import delegate_robodojo as delegate
from test_direct_robodojo import FakeRoboDojo


class FakePolicy:
    """Returns chunks of native RoboDojo actions: the left arm's joints count up by 0.1 per predicted step."""

    def __init__(self, horizon=50):
        self.horizon, self.calls = horizon, []

    def call(self, func_name, obs=None):
        self.calls.append((func_name, obs))
        if func_name == 'get_action':
            return [{'left_arm_joint_state': [.1 * (i + 1)] * 6, 'left_ee_joint_state': [1.5],  # Out-of-range opening.
                     'right_arm_joint_state': [-1.] * 6, 'right_ee_joint_state': [0.]} for i in range(self.horizon)]


class ActorActTest(unittest.TestCase):
    def setUp(self):
        self.policy = FakePolicy()
        self.env = delegate.RoboDojoWrapper(FakeRoboDojo(), self.policy)
        self.env.reset()

    def act(self, **arguments):
        obs, reward, terminated, truncated, info = self.env.step({'name': 'actor_act', 'arguments': json.dumps(arguments)})
        return obs, json.loads(obs['feedback'])

    def test_runs_a_prefix_of_each_fresh_chunk_on_active_arms_only(self):
        obs, feedback = self.act(instruction='pick up the bowl', active_groups=['left'], actions_per_chunk=3, max_chunks=2)
        self.assertEqual(feedback, {'tool': 'actor_act', 'ok': True, 'steps': 6, 'chunks': 2})
        self.assertEqual(obs['native_steps'], 6)
        np.testing.assert_allclose(obs['arms']['left']['joints'], .3)  # Third step of the second chunk.
        self.assertEqual(obs['arms']['left']['gripper_command'], 1.)  # Clipped to [0, 1].
        np.testing.assert_allclose(obs['arms']['right']['joints'], 0.)  # Inactive arm holds.
        self.assertEqual(obs['arms']['right']['gripper_command'], 1.)
        names = [name for name, _ in self.policy.calls]
        self.assertEqual(names, ['reset', 'update_obs', 'get_action', 'update_obs', 'get_action'])
        sent = self.policy.calls[3][1]  # The second chunk sees the state after the first chunk.
        self.assertEqual(sent['instruction'], 'pick up the bowl')
        np.testing.assert_allclose(sent['state']['left_arm_joint_state'], .3)

    def test_invalid_arguments_do_not_query_the_policy(self):
        for arguments in [{'instruction': '', 'active_groups': ['left']}, {'instruction': 'x', 'active_groups': ['middle']},
                          {'instruction': 'x', 'active_groups': ['left', 'left']}]:
            obs, feedback = self.act(**arguments)
            self.assertFalse(feedback['ok'], arguments)
        self.assertEqual([name for name, _ in self.policy.calls], ['reset'])

    def test_schema_matches_tool_contract(self):
        schema = {t['function']['name']: t['function']['parameters'] for t in self.env.schemas()}['actor_act']
        self.assertEqual(set(schema['required']), {'instruction', 'active_groups'})
        self.assertEqual(set(schema['properties']), {'instruction', 'active_groups', 'actions_per_chunk', 'max_chunks'})
        self.assertEqual(schema['properties']['actions_per_chunk']['default'], 25)
        self.assertEqual(schema['properties']['max_chunks']['default'], 6)
        self.assertNotIn('$ref', json.dumps(schema))


if __name__ == '__main__':
    unittest.main()
