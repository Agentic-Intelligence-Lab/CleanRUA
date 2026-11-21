"""CPU tests for the four algorithm prototypes, with a scripted MLLM, a fake robot and a fake VLA."""
import json
import unittest
from unittest import mock

import numpy as np

from cleanrua.prototypes import code, delegate, direct, monitor


class FakeRobot:
    """Episodes end after `limit` control steps; poses with x > 1 are unreachable; ik paths take three steps."""

    def __init__(self, limit=1000):
        self.limit, self.actions = limit, []

    def observe(self):
        return {'images': {'front': np.zeros((4, 4, 3), np.uint8)}, 'state': {'steps': len(self.actions)}}

    def ik(self, pose, gripper):
        return [] if pose[0] > 1 else [('ik', pose[0], gripper)] * 3

    def step(self, action):
        self.actions.append(action)
        return len(self.actions) >= self.limit


def scripted(*calls):
    """An mllm() that replies with these (name, arguments) tool calls in order and records what it was sent."""
    sent = []

    def mllm(messages):
        sent.append(messages)
        name, arguments = calls[len(sent) - 1]
        return {'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'call-{len(sent)}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(arguments)}}]}
    return mllm, sent


class PrototypeTest(unittest.TestCase):
    def test_direct_follows_ik_and_reports_unreachable_poses(self):
        robot, (mllm, sent) = FakeRobot(), scripted(('move', {'pose': [.5] + [0] * 5, 'gripper': 1}),
                                                    ('move', {'pose': [2] + [0] * 5, 'gripper': 0}), ('finish', {}))
        with mock.patch.object(direct, 'mllm', mllm):
            self.assertEqual(direct.direct(robot, 'stack the blocks'), 'finish')
        self.assertEqual(robot.actions, [('ik', .5, 1)] * 3)
        self.assertEqual([m['role'] for m in sent[2]],
                         ['system', 'user', 'assistant', 'tool', 'assistant', 'tool', 'user'])
        self.assertEqual(json.loads(sent[2][-2]['content']), {'error': 'No inverse-kinematics solution'})
        self.assertEqual(sent[0][-1]['content'][-1]['type'], 'image_url')

    def test_code_keeps_variables_reports_errors_and_stops_with_the_episode(self):
        robot = FakeRobot(limit=5)
        mllm, sent = scripted(('run_code', {'code': 'target = [.2] + [0] * 5\nprint(robot.move(target, 0))'}),
                              ('run_code', {'code': 'robot.move([2] + [0] * 5, 0)'}),
                              ('run_code', {'code': 'robot.move(target, 1)\nprint("not reached")'}))
        with mock.patch.object(code, 'mllm', mllm):
            self.assertEqual(code.code(robot, 'stack the blocks'), 'episode_ended')
        self.assertEqual(json.loads(sent[1][-2]['content']), {'output': '3\n'})
        self.assertEqual(json.loads(sent[2][-2]['content'])['error'], 'ValueError: No inverse-kinematics solution')
        self.assertEqual(len(robot.actions), 5)

    def test_delegate_and_monitor_differ_in_who_writes_the_instruction(self):
        for module, run, call in ((delegate, delegate.delegate,
                                   ('vla', {'instruction': 'grasp the bowl', 'steps': 60})),
                                  (monitor, monitor.monitor, ('run_policy', {'steps': 60}))):
            with self.subTest(module=module.__name__):
                robot, instructions = FakeRobot(), []

                def policy(observation, instruction):
                    instructions.append((observation['state']['steps'], instruction))
                    return [('vla', i) for i in range(50)]
                mllm, sent = scripted(call, ('move', {'pose': [.1] + [0] * 5, 'gripper': 0}), ('finish', {}))
                with mock.patch.object(module, 'mllm', mllm):
                    self.assertEqual(run(robot, policy, 'stack the bowls'), 'finish')
                expected = 'grasp the bowl' if module is delegate else 'stack the bowls'
                self.assertEqual(instructions, [(0, expected), (25, expected), (50, expected)])  # 25 actions per chunk.
                self.assertEqual(len(robot.actions), 63)


if __name__ == '__main__':
    unittest.main()
