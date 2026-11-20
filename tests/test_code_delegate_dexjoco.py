"""CPU tests for the DexJoCo code_delegate wrapper: programs that mix robot.vla and robot.move."""
import json
import unittest

import numpy as np

from cleanrua import code_delegate_dexjoco as code_delegate
from test_delegate_dexjoco import FakeDexJoCo, FakePolicy


class RunCodeTest(unittest.TestCase):
    def setUp(self):
        self.policy = FakePolicy()
        self.env = code_delegate.DexJoCoWrapper(FakeDexJoCo(), self.policy)
        self.env.reset()

    def run_code(self, code):
        obs, *_ = self.env.step({'name': 'run_code', 'arguments': json.dumps({'code': code})})
        return obs, json.loads(obs['feedback'])

    def test_program_mixes_vla_and_move(self):
        obs, feedback = self.run_code(
            "print(robot.vla('grasp the box', steps=3))\n"
            "pose = robot.state()['right']['pose']\n"
            "result = robot.move([{'arm': 'right', 'pose': (pose + [0, 0, .02, 0, 0, 0]).tolist()}])\n"
            "print(result['stop_reason'])")
        self.assertTrue(feedback['ok'], feedback)
        self.assertEqual(feedback['output'], "{'inferences': 1}\ntargets_reached\n")
        self.assertEqual(feedback['steps'], 5)
        np.testing.assert_allclose(obs['arms']['right']['pose'][:3], [1., 1., 1.02])
        self.assertEqual(self.policy.requests[0]['prompt'], 'grasp the box')

    def test_errors_reach_the_program_and_the_namespace_persists(self):
        obs, feedback = self.run_code(
            "try:\n    robot.vla('x', arms=['left'])\nexcept ValueError as exc:\n    print('caught')\nkept = 3")
        self.assertEqual((feedback['ok'], feedback['output'], feedback['steps']), (True, 'caught\n', 0))
        obs, feedback = self.run_code('robot.hold(kept)\nprint(1 / 0)')
        self.assertEqual(feedback['steps'], 3)
        self.assertEqual(feedback['error'], 'ZeroDivisionError at line 2: division by zero')
        self.assertEqual(self.policy.requests, [])


if __name__ == '__main__':
    unittest.main()
