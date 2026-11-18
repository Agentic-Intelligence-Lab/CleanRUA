"""CPU tests for the monitor wrapper's run_policy tool, with a fake action-model server."""
import json
import unittest

import numpy as np

from cleanrua import monitor_robodojo as monitor
from test_delegate_robodojo import FakePolicy
from test_direct_robodojo import FakeRoboDojo


class RunPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = FakePolicy()
        self.env = monitor.RoboDojoWrapper(FakeRoboDojo(), self.policy, actions_per_chunk=3)
        self.env.reset()

    def test_runs_both_arms_with_the_task_instruction(self):
        obs, *_ = self.env.step({'name': 'run_policy', 'arguments': json.dumps({'chunks': 2})})
        feedback = json.loads(obs['feedback'])
        self.assertEqual(feedback, {'tool': 'run_policy', 'ok': True, 'steps': 6, 'chunks': 2})
        np.testing.assert_allclose(obs['arms']['left']['joints'], .3)  # Third step of the second chunk.
        np.testing.assert_allclose(obs['arms']['right']['joints'], -1.)  # Both arms follow the policy.
        self.assertEqual((obs['arms']['left']['gripper_command'], obs['arms']['right']['gripper_command']), (1., 0.))
        sent = [obs for name, obs in self.policy.calls if name == 'update_obs']
        self.assertEqual([obs['instruction'] for obs in sent], ['Stack the bowls.'] * 2)

    def test_corrections_and_schemas(self):
        schemas = {t['function']['name']: t['function']['parameters'] for t in self.env.schemas()}
        self.assertEqual(set(schemas), {'run_policy', 'move', 'hold', 'capture_scene', 'finish'})
        self.assertEqual(set(schemas['run_policy']['properties']), {'chunks'})
        obs, *_ = self.env.step({'name': 'run_policy', 'arguments': json.dumps({'chunks': 9})})
        self.assertFalse(json.loads(obs['feedback'])['ok'])
        self.assertEqual([name for name, _ in self.policy.calls], ['reset'])


if __name__ == '__main__':
    unittest.main()
