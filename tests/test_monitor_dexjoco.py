"""CPU tests for the DexJoCo monitor wrapper's run_policy tool."""
import json
import unittest

import numpy as np

from cleanrua import monitor_dexjoco as monitor
from test_delegate_dexjoco import FakeDexJoCo, FakePolicy


class RunPolicyTest(unittest.TestCase):
    def test_runs_every_arm_with_the_task_instruction(self):
        policy = FakePolicy()
        env = monitor.DexJoCoWrapper(FakeDexJoCo(('right', 'left')), policy)
        env.reset()
        obs, *_ = env.step({'name': 'run_policy', 'arguments': json.dumps({'steps': 7})})
        self.assertEqual(json.loads(obs['feedback']), {'tool': 'run_policy', 'ok': True, 'steps': 7, 'inferences': 1})
        np.testing.assert_allclose(obs['arms']['right']['pose'][:3], 1.)
        np.testing.assert_allclose(obs['arms']['left']['pose'][:3], -1.)
        self.assertEqual(policy.requests[0]['prompt'], 'Place the boxed food into the bucket.')
        self.assertNotIn('actor_act', [tool['function']['name'] for tool in env.schemas()])


if __name__ == '__main__':
    unittest.main()
