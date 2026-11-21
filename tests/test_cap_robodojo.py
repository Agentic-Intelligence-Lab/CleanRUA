"""CPU tests for the code wrapper's run_code tool. No simulator, no network."""
import json
import unittest

import numpy as np

from cleanrua import cap_robodojo as code
from test_direct_robodojo import FakeRoboDojo


class RunCodeTest(unittest.TestCase):
    def setUp(self):
        self.env = code.RoboDojoWrapper(FakeRoboDojo())
        self.env.reset()

    def run_code(self, program, env=None):
        obs, reward, terminated, truncated, info = (env or self.env).step(
            {'name': 'run_code', 'arguments': json.dumps({'code': program})})
        return obs, json.loads(obs['feedback'])

    def test_program_moves_the_robot_and_prints(self):
        obs, feedback = self.run_code(
            "pose = robot.state()['left']['pose']\n"
            "result = robot.move([{'arm': 'left', 'pose': list(pose + [0, 0, .05, 0, 0, 0]), 'gripper': 0.}],"
            " profile='precision')\n"
            "print(result['stop_reason'], np.round(robot.state()['left']['pose'][2], 3))\n"
            "robot.hold(2)")
        # Precision: 1 cm increments, so 5 cm takes five steps; then two hold steps.
        self.assertEqual((feedback['ok'], feedback['steps']), (True, 7))
        self.assertEqual(feedback['output'], 'targets_reached 0.949\n')
        self.assertEqual(obs['arms']['left']['gripper_command'], 0.)
        np.testing.assert_allclose(obs['arms']['right']['joints'], 0.)  # The unlisted arm holds.

    def test_namespace_persists_between_programs(self):
        self.run_code('start = robot.state()["left"]["pose"].copy()')
        obs, feedback = self.run_code('print(np.round(start[0], 4))')
        self.assertEqual((feedback['ok'], feedback['output'], feedback['steps']), (True, '-0.1507\n', 0))

    def test_errors_report_the_line_and_keep_earlier_motions(self):
        obs, feedback = self.run_code('robot.hold(3)\nprint("before")\nraise RuntimeError("boom")')
        self.assertEqual((feedback['ok'], feedback['steps'], feedback['output']), (False, 3, 'before\n'))
        self.assertEqual(feedback['error'], 'RuntimeError at line 3: boom')
        obs, feedback = self.run_code("robot.move([{'arm': 'middle', 'pose': [0] * 6}])")
        self.assertFalse(feedback['ok'])
        self.assertIn('ValidationError at line 1', feedback['error'])

    def test_programs_can_catch_motion_failures(self):
        env = code.RoboDojoWrapper(FakeRoboDojo(ik_fails_after=2))
        env.reset()
        obs, feedback = self.run_code(
            "pose = robot.state()['left']['pose']\n"
            "try:\n"
            "    robot.move([{'arm': 'left', 'pose': list(pose + [0, 0, .1, 0, 0, 0])}], profile='precision')\n"
            "except ValueError as exc:\n"
            "    print('caught', 'inverse-kinematics' in str(exc))", env)
        self.assertEqual((feedback['ok'], feedback['steps'], feedback['output']), (True, 2, 'caught True\n'))

    def test_a_program_without_motion_is_abandoned(self):
        timeout, code.PROGRAM_TIMEOUT = code.PROGRAM_TIMEOUT, .2
        try:
            obs, feedback = self.run_code('import time\ntime.sleep(1)\nrobot.hold(1)')
        finally:
            code.PROGRAM_TIMEOUT = timeout
        self.assertEqual((feedback['ok'], feedback['steps']), (False, 0))
        self.assertIn('abandoned', feedback['error'])

    def test_schemas(self):
        schemas = {t['function']['name']: t['function'] for t in self.env.schemas()}
        self.assertEqual(set(schemas), {'run_code', 'finish'})
        self.assertEqual(schemas['run_code']['parameters']['required'], ['code'])


if __name__ == '__main__':
    unittest.main()
