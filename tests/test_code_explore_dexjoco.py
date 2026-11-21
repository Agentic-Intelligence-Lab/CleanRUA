"""CPU tests for the DexJoCo code_explore agent: finger targets and depth in programs, and the compile step."""
import json
import os
import unittest
from unittest import mock

import httpx
import numpy as np

from cleanrua import code_explore_dexjoco as explore
from test_delegate_dexjoco import FakeDexJoCo


class DepthDexJoCo(FakeDexJoCo):
    """Adds one camera that sees a flat surface 0.5 m in front of it, looking down from 1.5 m."""
    success = False

    def _observe(self):
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = np.diag([1., -1., -1.]), [0., 0., 1.5]  # Optical z points down.
        camera = {'rgb': np.zeros((4, 4, 3), np.uint8), 'intrinsic_K': np.array([[2., 0, 2], [0, 2., 2], [0, 0, 1]]),
                  'T_world_from_optical': pose}
        return {**super()._observe(), 'cameras': {'front': camera}}

    def depth(self, camera):
        return np.full((4, 4), .5, np.float32)


class RunCodeTest(unittest.TestCase):
    def setUp(self):
        self.env = explore.DexJoCoWrapper(DepthDexJoCo())
        self.env.reset()

    def run_code(self, code):
        obs, *_ = self.env.step({'name': 'run_code', 'arguments': json.dumps({'code': code})})
        return obs, json.loads(obs['feedback'])

    def test_fingers_move_at_most_a_tenth_of_a_radian_per_step_within_limits(self):
        obs, feedback = self.run_code(
            "pose = robot.state()['right']['pose'].tolist()\n"
            "print(robot.move([{'arm': 'right', 'pose': pose, 'hand': [0.3] * 12 + [5.0, 0.3, 0.3, 0.3]}]))")
        self.assertTrue(feedback['ok'], feedback)
        self.assertIn("'stop_reason': 'targets_reached'", feedback['output'])
        self.assertEqual(feedback['steps'], 12)  # The thumb's first joint travels 1.133 rad, 0.263 -> 1.396 (its limit).
        np.testing.assert_allclose(obs['arms']['right']['hand'][:12], .3)
        np.testing.assert_allclose(obs['arms']['right']['hand'][12], 1.396)

    def test_programs_read_depth_and_back_project_pixels_without_moving(self):
        obs, feedback = self.run_code(
            "camera = robot.camera('front')\n"
            "print(camera['depth'].shape, robot.point('front', 2, 2).round(3).tolist(), "
            "robot.point('front', 3, 2).round(3).tolist())")
        self.assertTrue(feedback['ok'], feedback)
        self.assertEqual(feedback['output'], '(4, 4) [0.0, 0.0, 1.0] [0.25, 0.0, 1.0]\n')
        self.assertEqual(feedback['steps'], 0)

    def test_the_action_model_is_gone(self):
        obs, feedback = self.run_code("robot.vla('grasp')")
        self.assertIn("AttributeError", feedback['error'])


class AttemptTest(unittest.TestCase):
    def test_an_ended_attempt_needs_reset_scene_and_attempts_are_limited(self):
        env = explore.DexJoCoWrapper(DepthDexJoCo(), attempts=2)
        env.reset(seed=3)
        step = lambda name, **arguments: json.loads(env.step({'name': name, 'arguments': json.dumps(arguments)})[0]['feedback'])
        feedback = step('run_code', code='robot.hold(150)')  # The fake episode ends after 100 steps.
        self.assertEqual((feedback['steps'], feedback['attempt_over'], feedback['task_success'], feedback['attempts_left']),
                         (100, True, False, 1))
        self.assertIn('call reset_scene', step('run_code', code='robot.hold(1)')['error'])
        env.namespace['kept'] = 1
        feedback = step('reset_scene')
        self.assertEqual((feedback['ok'], feedback['attempt_over'], feedback['attempts_left']), (True, False, 0))
        self.assertEqual((env.observation['native_steps'], env.namespace['kept']), (0, 1))
        self.assertEqual(step('run_code', code='robot.hold(2)')['steps'], 2)
        self.assertIn('No attempts left', step('reset_scene')['error'])
        self.assertEqual((env.outcomes, env.total_steps), ([False], 102))

    def test_testing_allows_no_restarts(self):
        env = explore.DexJoCoWrapper(DepthDexJoCo())
        env.reset(seed=0)
        feedback = json.loads(env.step({'name': 'reset_scene', 'arguments': '{}'})[0]['feedback'])
        self.assertFalse(feedback['ok'])


class CompileTest(unittest.TestCase):
    def agent(self, reply):
        def handler(request):
            self.request = json.loads(request.content)
            return httpx.Response(200, json={'usage': {'total_tokens': 50, 'prompt_tokens': 40, 'completion_tokens': 10},
                                             'choices': [{'message': {'role': 'assistant', 'content': reply}}]})
        variables = {'CLEANRUA_API_KEY': 'k', 'CLEANRUA_BASE_URL': 'https://model.invalid/v1', 'CLEANRUA_MODEL_NAME': 'm'}
        with mock.patch.dict(os.environ, variables):
            agent = explore.RUAgent(explore.Args(notes_chars=20), tools=[])
        agent.client = httpx.Client(base_url=agent.client.base_url, transport=httpx.MockTransport(handler))
        agent.reset('Task: click.', step_limit=100)
        return agent

    def test_reply_gives_notes_and_a_program(self):
        agent = self.agent('Here.\n```markdown\n- Approach from above, 5 cm.\n```\n'
                           '```python\ndef main(robot):\n    robot.hold(1)\n```')
        notes, program = agent.compile(['Task: click.', 'This episode: ...'], {'cameras': {}})
        self.assertEqual(notes, '- Approach from abov')  # Cut to notes_chars.
        self.assertEqual(program, 'def main(robot):\n    robot.hold(1)\n')
        self.assertIn('at most 20 characters', self.request['messages'][0]['content'])
        self.assertEqual((agent.usage['total_tokens'], agent.usage['compile_requests']), (50, 1))

    def test_a_reply_without_a_valid_program_is_rejected(self):
        for reply in ('```markdown\nnotes\n```', '```python\ndef run(robot):\n    pass\n```',
                      '```python\ndef main(robot):\n    robot.hold(\n```'):
            with self.subTest(reply=reply), self.assertRaises((ValueError, SyntaxError)):
                self.agent(reply).compile(['Task'], {'cameras': {}})


if __name__ == '__main__':
    unittest.main()
