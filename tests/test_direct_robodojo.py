"""Offline native-entry lifecycle tests; no services, simulator or live model."""
import ast
from dataclasses import asdict
import inspect
import json
import math
import os
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import httpx

import direct_robodojo as m
from scipy.spatial.transform import Rotation
from cleanrua_utils.robodojo_wrappers import waypoint
from direct_robodojo import Action, Observation
from offline_helpers import OfflineTestCase, ROOT, fake_observation, model_reply


class MainTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.env = Mock()
        self.env.arms = {'left': 'left_arm', 'right': 'right_arm'}
        self.env.robot = {'control_link': 'link6'}
        self.env.instruction = 'Stack the bowls.'
        self.env.initial_state = fake_observation().state
        self.env.reset.return_value = (asdict(fake_observation()), {})
        self.env.finalize.return_value = {'task_success': False}
        self.env.blocked = False
        self.env.step.side_effect = self.step
        self.steps = 0
        self.agent = Mock(total_tokens=0)
        self.agent.get_action.return_value = Action('finish', {'summary': 'Observed completion'}, 'finish-ID')

    def step(self, tool, arguments, call_id):
        obs = fake_observation()
        count = {'hold': 2, 'move': 1}.get(tool, 0)
        self.steps += count
        obs.state['native_steps'] = self.steps
        obs.feedback = {'call_id': call_id, 'call_ok': True,
                        'execution': 'confirmed' if count else 'none', 'steps_executed': count,
                        'targets_reached': tool == 'move',
                        'stop_reason': 'requested_step_budget' if tool == 'hold' else 'targets_reached'}
        return asdict(obs), 0., False, False, {'task_success': False}

    def run_main(self, *extra, smoke=False, agent_factory=None, expected_error=None, cleanup='completed'):
        argv = [str(ROOT / 'direct_robodojo.py'), '--native-root', str(self.args.native_root),
                '--output-dir', str(self.root / 'runs'), '--model', 'offline-model',
                '--base-url', 'https://model.invalid/v1', '--api-key-env', self.args.api_key_env,
                '--no-capture-video', *extra]
        if smoke:
            argv.append('--smoke-test')
        if agent_factory is None:
            agent_factory = Mock(return_value=self.agent)
        with patch.object(sys, 'argv', argv), patch.object(m, 'RoboDojoEnv', return_value=self.env) as factory, \
                patch.object(m, 'RUAgent', agent_factory), patch('sys.stdout', new=Mock()):
            code = 0
            if expected_error is not None:
                error_type = expected_error if isinstance(expected_error, type) else type(expected_error)
                with self.assertRaises(error_type) as raised:
                    m.main()
                if isinstance(expected_error, BaseException):
                    self.assertIs(raised.exception, expected_error)
                code = 1
            else:
                try:
                    m.main()
                except SystemExit as exc:
                    code = exc.code
        run_dirs = list((self.root / 'runs').iterdir())
        self.assertEqual(len(run_dirs), 1)
        run_dir = run_dirs[0]
        self.assertEqual(factory.call_args.args[1], run_dir)
        self.assertEqual(len(factory.call_args.args), 3)
        self.assertTrue(callable(factory.call_args.args[2]))
        result = json.loads((run_dir / 'result.json').read_text())
        trajectory = [json.loads(line) for line in (run_dir / 'trajectory.jsonl').read_text().splitlines()]
        self.assertEqual(result['cleanup'], cleanup)
        self.assertEqual(trajectory[-1]['event'], 'result' if cleanup == 'completed' else 'result_before_cleanup')
        self.assertEqual(trajectory[-1]['stop_reason'], result['stop_reason'])
        self.assertNotIn('offline-key', json.dumps(trajectory))
        self.env.close.assert_called_once_with()
        return code, result, trajectory

    def test_smoke_is_three_current_actions_without_a_model(self):
        no_agent = Mock(side_effect=AssertionError('Smoke must not construct an agent'))
        code, result, _ = self.run_main(smoke=True, agent_factory=no_agent)
        self.assertEqual(code, 0)
        self.assertEqual(result['stop_reason'], 'smoke_complete')
        self.assertEqual(result['decisions'], 3)
        self.assertEqual(result['reported_tokens'], 0)
        self.assertFalse(result['agent_finished'])
        actions = [call.kwargs for call in self.env.step.call_args_list]
        self.assertEqual([a['tool'] for a in actions], ['capture_scene', 'hold', 'move'])
        self.assertEqual(actions[1]['arguments']['steps'], 2)
        self.assertEqual(actions[2]['arguments']['arm'], 'left')
        self.assertAlmostEqual(actions[2]['arguments']['pose'][2], .91)
        self.assertNotIn('gripper', actions[2]['arguments'])
        no_agent.assert_not_called()
        self.env.finalize.assert_called_once_with('smoke_complete')

    def test_finish_claim_does_not_override_official_false_verdict(self):
        code, result, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertTrue(result['agent_finished'])
        self.assertIs(result['task_success'], False)
        self.assertEqual(result['stop_reason'], 'completion_claimed')
        self.assertEqual(self.env.step.call_args.kwargs['call_id'], 'finish-ID')
        self.assertIn('return both arms', self.agent.reset.call_args.args[0])
        self.agent.close.assert_called_once_with()

    def test_actual_agent_token_budget_blocks_returned_action_before_execution(self):
        real_agent = m.RUAgent
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json=model_reply('over-budget', tool='hold', tokens=100))

        client = httpx.Client(transport=httpx.MockTransport(respond), trust_env=False)
        self.addCleanup(client.close)
        with patch.object(m.httpx, 'Client', return_value=client):
            code, result, events = self.run_main('--max-total-tokens', '100', agent_factory=real_agent)
        self.assertEqual((code, result['stop_reason'], result['reported_tokens']), (1, 'token_budget', 100))
        self.assertEqual(result['decisions'], 1)
        self.assertEqual(len(requests), 1)
        self.assertTrue(client.is_closed)
        self.env.step.assert_not_called()
        self.env.finalize.assert_called_once_with('token_budget')
        action_event = next(event for event in events if event['event'] == 'action')
        self.assertEqual(action_event['action']['call_id'], 'over-budget')

    def test_expired_budget_prevents_first_model_request(self):
        clock = iter([0., 0., 2., 2., 2.])  # Start, deadline, check, two result writes.
        with patch.object(m.time, 'monotonic', side_effect=lambda: next(clock)):
            code, result, _ = self.run_main('--episode-timeout', '1')
        self.assertEqual((code, result['stop_reason'], result['decisions']), (1, 'wall_time_limit', 0))
        self.agent.get_action.assert_not_called()
        self.env.step.assert_not_called()
        self.agent.close.assert_called_once_with()

    def test_request_overrunning_deadline_cannot_execute(self):
        clock = iter([0., 0., .5, 2., 2., 2.])
        with patch.object(m.time, 'monotonic', side_effect=lambda: next(clock)):
            code, result, _ = self.run_main('--episode-timeout', '1')
        self.assertEqual((code, result['stop_reason'], result['decisions']), (1, 'wall_time_limit', 1))
        self.agent.get_action.assert_called_once()
        self.env.step.assert_not_called()

    def test_decision_limit_finalizes_and_closes(self):
        self.agent.get_action.return_value = Action('capture_scene', {'summary': 'Look'})
        code, result, _ = self.run_main('--max-decisions', '1')
        self.assertEqual((code, result['stop_reason']), (1, 'max_decisions'))
        self.env.step.assert_called_once()
        self.agent.close.assert_called_once_with()

    def test_rejected_hold_fails_smoke(self):
        def reject_hold(**action):
            response = self.step(**action)
            if action['tool'] == 'hold':
                response[0]['feedback'].update(call_ok=False, steps_executed=0)
            return response

        self.env.step.side_effect = reject_hold
        code, result, _ = self.run_main(smoke=True, expected_error=RuntimeError)
        self.assertEqual((code, result['stop_reason'], result['decisions']), (1, 'error', 2))
        self.assertIn('smoke test', result['error'])

    def test_mismatched_feedback_blocks_environment_and_preserves_no_replay(self):
        def mismatch(**action):
            response = self.step(**action)
            response[0]['feedback']['call_id'] = 'another-action'
            return response

        self.env.step.side_effect = mismatch
        code, result, _ = self.run_main(expected_error=RuntimeError)
        self.assertEqual((code, result['stop_reason']), (1, 'execution_unknown'))
        self.assertTrue(self.env.blocked)
        self.env.step.assert_called_once()
        self.env.finalize.assert_called_once_with('execution_unknown')
        self.agent.close.assert_called_once_with()

    def test_native_unknown_execution_is_not_retried(self):
        failure = RuntimeError('Unconfirmed tick')

        def fail(**action):
            self.env.blocked = True  # Native queue/physics owns this state.
            raise failure

        self.env.step.side_effect = fail
        code, result, _ = self.run_main(expected_error=failure)
        self.assertEqual((code, result['stop_reason']), (1, 'execution_unknown'))
        self.assertTrue(self.env.blocked)
        self.env.step.assert_called_once()
        self.agent.close.assert_called_once_with()

    def test_model_transport_error_is_redacted_and_cleanup_runs(self):
        failure = httpx.ReadTimeout('sensitive URL/header offline-key')
        self.agent.get_action.side_effect = failure
        code, result, _ = self.run_main(expected_error=failure)
        self.assertEqual((code, result['stop_reason'], result['error']), (1, 'error', 'ReadTimeout'))
        self.env.step.assert_not_called()
        self.agent.close.assert_called_once_with()

    def test_reset_failure_finalizes_and_closes_without_constructing_agent(self):
        failure = RuntimeError('Invalid native layout')
        self.env.reset.side_effect = failure
        no_agent = Mock(side_effect=AssertionError('Reset failed'))
        code, result, _ = self.run_main(agent_factory=no_agent, expected_error=failure)
        self.assertEqual((code, result['stop_reason']), (1, 'error'))
        self.assertIn('Invalid native layout', result['error'])
        self.env.finalize.assert_called_once_with('error')
        no_agent.assert_not_called()

    def test_keyboard_interrupt_finalizes_and_closes(self):
        failure = KeyboardInterrupt('Interrupted request')
        self.agent.get_action.side_effect = failure
        code, result, _ = self.run_main(expected_error=failure)
        self.assertEqual((code, result['stop_reason']), (1, 'interrupted'))
        self.agent.close.assert_called_once_with()
        self.env.step.assert_not_called()

    def test_finalization_failure_still_closes_both_resources(self):
        failure = RuntimeError('Offline finalization failure')
        self.env.finalize.side_effect = failure
        code, result, _ = self.run_main(expected_error=failure, cleanup='pending')
        self.assertEqual(code, 1)
        self.assertIsNone(result['error'])  # Persisted before finalization/cleanup completes.
        self.assertIsNone(result['task_success'])
        self.agent.close.assert_called_once_with()

    def test_environment_close_failure_still_closes_agent_and_writes_result(self):
        """Environment cleanup errors must not bypass agent cleanup or results."""
        failure = RuntimeError('Offline close failure')
        self.env.close.side_effect = failure
        code, result, _ = self.run_main(expected_error=failure, cleanup='pending')
        self.agent.close.assert_called_once_with()
        self.assertEqual(code, 1)
        self.assertIs(result['task_success'], False)

    def test_unexpected_step_failure_rethrows_original_without_retry_or_feedback(self):
        failure = ValueError('Unexpected capture calibration error')
        self.env.step.side_effect = failure
        _, result, events = self.run_main(expected_error=failure)
        self.assertEqual(result['stop_reason'], 'error')
        self.assertIn('Unexpected capture calibration error', result['error'])
        self.assertFalse(self.env.blocked)
        self.env.step.assert_called_once()
        self.agent.get_action.assert_called_once()
        self.agent.close.assert_called_once_with()
        self.env.finalize.assert_called_once_with('error')
        self.assertFalse(any(event['event'] == 'step' for event in events))

    def test_agent_close_error_still_closes_environment_and_chains_original_failure(self):
        original, cleanup = RuntimeError('Original step failure'), RuntimeError('Agent close failure')
        self.env.step.side_effect = original
        self.agent.close.side_effect = cleanup
        _, result, events = self.run_main(expected_error=cleanup, cleanup='pending')
        self.assertEqual(result['error'], 'RuntimeError: Original step failure')
        self.assertEqual(events[-1]['event'], 'result_before_cleanup')
        self.agent.close.assert_called_once_with()
        self.env.step.assert_called_once()
        self.assertIs(cleanup.__context__, original)

    def test_finalization_failure_chains_original_and_still_persists_result(self):
        original, finalization = RuntimeError('Original request failure'), RuntimeError('Finalization failed')
        self.agent.get_action.side_effect = original
        self.env.finalize.side_effect = finalization
        _, result, _ = self.run_main(expected_error=finalization, cleanup='pending')
        self.assertIs(finalization.__context__, original)
        self.assertEqual(result['error'], 'RuntimeError: Original request failure')
        self.agent.close.assert_called_once_with()
        self.env.step.assert_not_called()

    def test_token_budget_before_request_does_not_call_policy(self):
        self.agent.total_tokens = 100
        code, result, _ = self.run_main('--max-total-tokens', '100')
        self.assertEqual((code, result['stop_reason'], result['decisions']), (1, 'token_budget', 0))
        self.agent.get_action.assert_not_called()
        self.env.step.assert_not_called()

    def test_invalid_schemas_return_exact_feedback_without_native_execution(self):
        real_agent, requests = m.RUAgent, []
        move = {'summary': 'Measured target', 'arm': 'left', 'pose': [0., 0., .9, 0., 0., 0.]}
        invalid = [('hold', {'summary': 'Bad steps', 'steps': True}),
                   ('hold', {'summary': 'Too many', 'steps': 25}),
                   ('move', {**move, 'opening_m': .04}), ('move', {**move, 'gripper': 1.1}),
                   ('move', {**move, 'gripper': True}), ('move', {**move, 'arm': 'unknown'}),
                   ('move', {**move, 'pose': [0, 0, .9]}), ('unknown', {'summary': 'Bad tool'}),
                   ('capture_scene', {'summary': ''}), ('finish', {'summary': 'Done', 'extra': 1})]

        def respond(request):
            self.env.step.assert_not_called()  # No wrapper calls for any rejected action.
            body = json.loads(request.content)
            requests.append(body)
            index = len(requests) - 1
            if index:
                receipt = body['messages'][-2]
                feedback = json.loads(receipt['content'])
                self.assertEqual(receipt['tool_call_id'], f'schema-{index - 1}')
                self.assertEqual(feedback['call_id'], f'schema-{index - 1}')
                self.assertEqual(feedback['stop_reason'], 'invalid_arguments')
                self.assertFalse(feedback['call_ok'])
                self.assertEqual((feedback['execution'], feedback['steps_executed']), ('none', 0))
            tool, arguments = invalid[index] if index < len(invalid) else ('finish', {'summary': 'Done'})
            reply = model_reply(f'schema-{index}', tool=tool)
            reply['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = json.dumps(arguments)
            return httpx.Response(200, json=reply)

        client = httpx.Client(transport=httpx.MockTransport(respond), trust_env=False)
        self.addCleanup(client.close)
        with patch.object(m.httpx, 'Client', return_value=client):
            code, result, events = self.run_main(agent_factory=real_agent)
        self.assertEqual(code, 0)
        self.assertEqual(len(requests), len(invalid) + 1)
        self.assertEqual(result['decisions'], len(invalid) + 1)
        self.env.step.assert_called_once_with(tool='finish', arguments={'summary': 'Done'},
                                              call_id=f'schema-{len(invalid)}')
        self.assertEqual(self.steps, 0)
        self.env.reset.assert_called_once_with()
        self.env.finalize.assert_called_once_with('completion_claimed')
        rejected = [event for event in events if event['event'] == 'step'][:-1]
        initial = asdict(fake_observation())
        for event in rejected:
            self.assertEqual(event['observation']['state'], initial['state'])
            self.assertEqual(event['observation']['snapshot_id'], initial['snapshot_id'])
            self.assertEqual(event['observation']['cameras'], initial['cameras'])
            self.assertEqual((event['reward'], event['terminated'], event['truncated'], event['info']),
                             (0., False, False, {}))
        self.assertTrue(client.is_closed)

    def test_nonfinite_action_cannot_be_logged_or_executed(self):
        # The real policy rejects this before constructing Action. Inject a corrupt
        # action to check strict artifact serialization also fails closed.
        self.agent.get_action.return_value = Action(
            'move', {'summary': 'Bad number', 'arm': 'left', 'pose': [0., 0., .9, float('nan'), 0., 0.]}, 'bad')
        _, result, events = self.run_main(expected_error=ValueError)
        self.assertEqual(result['stop_reason'], 'error')
        self.assertEqual(result['decisions'], 1)
        self.env.step.assert_not_called()
        self.assertFalse(any(e['event'] in ('action', 'step') for e in events))
        self.agent.close.assert_called_once_with()


class PureDirectTests(unittest.TestCase):
    def test_native_tool_contract_and_injected_arm_names(self):
        tools = m.tool_schemas(['test-arm'])
        self.assertEqual([t['function']['name'] for t in tools], ['capture_scene', 'move', 'hold', 'finish'])
        move = tools[1]['function']
        properties = move['parameters']['properties']
        self.assertEqual(properties['arm']['enum'], ['test-arm'])
        self.assertEqual(properties['gripper'], {'type': 'number', 'minimum': 0, 'maximum': 1})
        self.assertNotIn('opening_m', properties)
        self.assertNotIn('gripper', move['parameters']['required'])
        self.assertIn('world link6', move['description'])
        self.assertIn('not metres', move['description'])
        self.assertIs(m.Action, Action)
        self.assertIs(m.Observation, Observation)
        self.assertEqual(list(inspect.signature(m.RoboDojoEnv).parameters), ['args', 'run_dir', 'record'])
        self.assertEqual(list(inspect.signature(m.RoboDojoEnv.step).parameters),
                 ['self', 'tool', 'arguments', 'call_id'])

    def test_rotation_roundtrip_and_native_link6_waypoint_bound(self):
        for angles in ([.3, .7, -.2], [0., math.pi / 2, 0.], [.5, -math.pi / 2, .8]):
            rotation = Rotation.from_euler('xyz', angles)
            obs = fake_observation()
            obs.state['arms']['left']['control_pose']['quaternion'] = rotation.as_quat(scalar_first=True).tolist()
            # At gimbal lock SciPy chooses a different, equivalent Euler representation.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                action = m.smoke_actions(obs, 'left')[-1]
            reconstructed = Rotation.from_euler('xyz', action.arguments['pose'][3:])
            self.assertLess((rotation.inv() * reconstructed).magnitude(), 1e-6)
        pose = {'position': [0., 0., 1.], 'quaternion': [1., 0., 0., 0.]}
        goal = {'position': [.4, .2, 1.],
                'quaternion': Rotation.from_euler('xyz', [0., 1., 2.]).as_quat(scalar_first=True).tolist()}
        step = waypoint(pose, goal)
        self.assertLessEqual(math.dist(pose['position'], step['position']), .0200001)
        self.assertLessEqual(Rotation.from_quat(step['quaternion'], scalar_first=True).magnitude(), .2000001)

    def test_waypoint_shortest_path_and_quaternion_sign(self):
        start = Rotation.from_euler('z', 170, degrees=True)
        target = Rotation.from_euler('z', -170, degrees=True)
        current = {'position': [0., 0., 1.], 'quaternion': start.as_quat(scalar_first=True).tolist()}
        goal = {'position': [.01, 0., 1.], 'quaternion': target.as_quat(scalar_first=True).tolist()}
        step = waypoint(current, goal)
        reached = Rotation.from_quat(step['quaternion'], scalar_first=True)
        self.assertAlmostEqual((start.inv() * reached).as_rotvec()[2], .2)
        self.assertEqual(step['position'], goal['position'])
        equivalent = waypoint(current, {**goal, 'quaternion': [-v for v in goal['quaternion']]})
        self.assertLess((reached.inv() * Rotation.from_quat(equivalent['quaternion'], scalar_first=True)).magnitude(), 1e-12)

    def test_waypoint_same_and_tiny_rotations_and_nonunit_input(self):
        current = {'position': [0., 0., 1.], 'quaternion': [2., 0., 0., 0.]}
        for theta in (0., 1e-10, .1, math.pi):
            goal = {'position': [0., 0., 1.],
                    'quaternion': Rotation.from_rotvec([theta, 0., 0.]).as_quat(scalar_first=True).tolist()}
            result = waypoint(current, goal)
            self.assertAlmostEqual(sum(v*v for v in result['quaternion']), 1.)
            self.assertAlmostEqual(Rotation.from_quat(result['quaternion'], scalar_first=True).magnitude(), min(theta, .2))
            self.assertEqual(result['position'], current['position'])

    def test_budget_boundaries_and_stop_precedence(self):
        with patch.object(m.time, 'monotonic', return_value=10.):
            self.assertIsNone(m.budget_stop(11., 99, 100))
            self.assertEqual(m.budget_stop(11., 100, 100), 'token_budget')
            self.assertEqual(m.budget_stop(10., 0, 100), 'wall_time_limit')
            self.assertEqual(m.budget_stop(10., 100, 100), 'wall_time_limit')
        self.assertEqual(m.episode_stop(True, True, True, {}), 'completion_claimed')
        self.assertEqual(m.episode_stop(False, True, True, {}), 'environment_truncated')
        self.assertEqual(m.episode_stop(False, True, False, {}), 'environment_ended')
        self.assertEqual(m.episode_stop(True, True, True, {'stop_reason': 'wall_time_limit'}), 'wall_time_limit')


class ImportBoundaryTests(OfflineTestCase):
    def test_import_and_help_without_native_runtime_or_network(self):
        # Fresh interpreter: a cached native import cannot accidentally pass this test.
        program = '''
import importlib.abc
import runpy
import socket
import sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'uni_robot_agent', 'isaaclab', 'isaacsim', 'omni', 'pxr', 'torch',
            'curobo', 'env', 'task', 'utils', 'omegaconf', 'transforms3d', 'PIL'
        }:
            raise AssertionError('Forbidden runtime import: ' + fullname)
sys.meta_path.insert(0, BlockNative())
def offline(*args, **kwargs):
    raise AssertionError('Network is forbidden')
socket.socket.connect = offline
socket.create_connection = offline
import httpx
httpx.Client = offline
import cleanrua_utils.robodojo_wrappers as wrapper
assert 'direct_robodojo' not in sys.modules
assert not hasattr(wrapper, 'Action') and not hasattr(wrapper, 'Observation')
import direct_robodojo
assert not any(n == 'uni_robot_agent' or n.startswith('uni_robot_agent.') for n in sys.modules)
sys.argv = ['direct_robodojo.py', '--help']
runpy.run_module('direct_robodojo', run_name='__main__')
'''
        environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(ROOT),
                       'HOME': str(self.root), 'XDG_CACHE_HOME': str(self.root / 'cache')}
        result = subprocess.run([sys.executable, '-B', '-c', program], cwd=self.root,
                                env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ('--native-root', '--layout', '--seed', '--no-capture-video'):
            self.assertIn(flag, result.stdout)
        self.assertNotIn('--env-url', result.stdout)
        self.assertFalse((self.root / 'runs').exists())

    def test_native_entry_has_no_service_routes_or_harness_imports(self):
        paths = [ROOT / 'direct_robodojo.py', *sorted((ROOT / 'cleanrua_utils').rglob('*.py'))]
        for path in paths:
            with self.subTest(path=path.name):
                tree = ast.parse(path.read_text())
                imports, strings = [], []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        imports.append(node.module or '')
                    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                        strings.append(node.value)
                self.assertFalse(any(name.split('.')[0] == 'uni_robot_agent' for name in imports))
                allowed_local = {'cleanrua_utils.robodojo_wrappers'}
                self.assertTrue(all(name in allowed_local for name in imports if name.startswith('cleanrua_')))
                # The wrapper uses SciPy, never entry-owned data classes or local math helpers.
                if path.name == 'robodojo_wrappers.py':
                    self.assertNotIn('direct_robodojo', imports)
                    self.assertNotIn('jsonschema', imports)
                    self.assertEqual([name for name in imports if name.startswith('cleanrua_')], [])
                self.assertFalse({'/steps', '/captures', '/finish', '/info', '/requests/', '/artifacts/'} & set(strings))
        self.assertNotIn('env_url', m.Args.__dataclass_fields__)


if __name__ == '__main__':
    unittest.main()