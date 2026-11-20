"""CPU tests for the DexJoCo monitor_check wrapper: run_policy needs a progress judgment and a correction after
judging the task stalled or failed."""
import json
import unittest

from cleanrua import monitor_check_dexjoco as monitor_check
from test_delegate_dexjoco import FakeDexJoCo, FakePolicy


class CheckedRunPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = FakePolicy()
        self.env = monitor_check.DexJoCoWrapper(FakeDexJoCo(), self.policy)
        self.obs, _ = self.env.reset()

    def call(self, name, **arguments):
        obs, *_ = self.env.step({'name': name, 'arguments': json.dumps(arguments)})
        return json.loads(obs['feedback'])

    def test_negative_judgment_needs_a_move_first(self):
        self.assertTrue(self.call('run_policy', progress='started', evidence='initial scene', steps=3)['ok'])
        refused = self.call('run_policy', progress='stalled', evidence='hand hovering', steps=3)
        self.assertEqual((refused['ok'], refused['steps']), (False, 0))
        self.assertIn('correct the robot with `move`', refused['error'])
        self.assertTrue(self.call('hold', steps=2)['ok'])  # hold runs but is not a correction.
        self.assertFalse(self.call('run_policy', progress='failed', evidence='box dropped', steps=3)['ok'])
        pose = self.env.observation['arms']['right']['pose'].tolist()
        self.assertTrue(self.call('move', targets=[{'arm': 'right', 'pose': pose, 'hand': 'open'}])['ok'])
        self.assertTrue(self.call('run_policy', progress='failed', evidence='box dropped', steps=3)['ok'])
        self.assertFalse(self.call('run_policy', progress='stalled', evidence='still hovering', steps=3)['ok'])
        self.assertEqual(len(self.policy.requests), 2)

    def test_progress_is_required(self):
        feedback = self.call('run_policy', steps=3)
        self.assertFalse(feedback['ok'])
        schema = {t['function']['name']: t['function']['parameters'] for t in self.env.schemas()}['run_policy']
        self.assertEqual(set(schema['required']), {'progress', 'evidence'})


if __name__ == '__main__':
    unittest.main()
