"""CPU tests for robot.vla in the code-with-action-model wrapper, with a fake action-model server."""
import json
import unittest

import numpy as np

from cleanrua import code_vla_robodojo as code_vla
from test_delegate_robodojo import FakePolicy
from test_direct_robodojo import FakeRoboDojo


class VlaTest(unittest.TestCase):
    def setUp(self):
        self.policy = FakePolicy()
        self.env = code_vla.RoboDojoWrapper(FakeRoboDojo(), self.policy)
        self.env.reset()

    def run_code(self, program):
        obs, *_ = self.env.step({'name': 'run_code', 'arguments': json.dumps({'code': program})})
        return obs, json.loads(obs['feedback'])

    def test_program_runs_the_action_model_then_moves(self):
        obs, feedback = self.run_code(
            "print(robot.vla('pick up the bowl', ['left'], chunks=2, actions_per_chunk=3))\n"
            "print(np.round(robot.state()['left']['joints'][0], 2), robot.state()['right']['gripper_command'])\n"
            "robot.hold(1)")
        self.assertEqual((feedback['ok'], feedback['steps']), (True, 7))  # Two chunks of three, then one hold.
        self.assertEqual(feedback['output'], "{'chunks': 2}\n0.3 1.0\n")  # The right arm held.
        sent = [obs for name, obs in self.policy.calls if name == 'update_obs']
        self.assertEqual([obs['instruction'] for obs in sent], ['pick up the bowl'] * 2)
        np.testing.assert_allclose(sent[1]['state']['left_arm_joint_state'], .3)  # Fresh state per chunk.
        self.assertGreaterEqual(self.env.policy_s, 0.)

    def test_invalid_calls_raise_inside_the_program(self):
        for call in ["robot.vla('', ['left'])", "robot.vla('x', ['middle'])", "robot.vla('x', ['left'], chunks=9)",
                     "robot.vla('x', ['left', 'left'])"]:
            obs, feedback = self.run_code(call)
            self.assertFalse(feedback['ok'], call)
            self.assertEqual(feedback['steps'], 0)
        self.assertEqual([name for name, _ in self.policy.calls], ['reset'])

    def test_reset_resets_the_policy_and_schemas_stay_two_tools(self):
        self.env.reset()
        self.assertEqual([name for name, _ in self.policy.calls], ['reset', 'reset'])
        self.assertEqual({t['function']['name'] for t in self.env.schemas()}, {'run_code', 'finish'})
        self.assertIn('robot.vla(', {t['function']['name']: t['function'] for t in self.env.schemas()}[
            'run_code']['description'])


if __name__ == '__main__':
    unittest.main()
