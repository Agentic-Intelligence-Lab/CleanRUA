"""Native adapter logic against a CPU fake manager, not an HTTP service.

No reset launches Isaac. The real _macro, _read_state, step and reward paths run;
only native objects and image acquisition are replaced by small test doubles.
"""
import copy
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch
from uuid import uuid4

import httpx

import direct_robodojo as direct
from cleanrua_utils.robodojo_wrappers import RoboDojoEnv
from offline_helpers import OfflineTestCase, fake_observation, model_reply


class ControlQueue:
    def __init__(self, trace):
        self.trace = trace
        self.pending = []
        self.commands = []

    def push(self, env_ids, commands):
        assert env_ids == [0] and len(commands) == len(commands[0]) == 1
        command = copy.deepcopy(commands[0][0])
        self.commands.append(command)
        self.pending.append(command)
        self.trace.append('push')

    def pop(self, env_ids):
        assert env_ids == [0] and len(self.pending) == 1
        self.trace.append('pop')
        return self.pending.pop(0)


class NativeTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.adapter = RoboDojoEnv(self.args, self.root, self.record)
        self.addCleanup(self.adapter.close)
        self.trace = []
        self.queue = ControlQueue(self.trace)
        self.measured = fake_observation().state['arms']
        self.robots = {}
        for name, sign in [('left', 1), ('right', -1)]:
            self.measured[name]['finger_joints'] = [.2, -.15]
            self.robots[name] = SimpleNamespace(arm_name=name + '_arm', gripper_name=name + '_gripper',
                                               gripper_scale=[.1, .5], gripper_move={'sign': sign, 'mimic': [0, -1, .05]})
        self.rm = Mock()
        self.rm.control_manager = self.queue
        self.rm.process_name.side_effect = lambda name: name
        self.rm.get_joint.side_effect = lambda robot, ids: [self.data(robot)['joints'][:]]
        self.rm.get_real_endpose.side_effect = self.endpose
        self.rm.get_end_effector_real_val.side_effect = lambda robot, ids: [self.data(robot)['finger_joints'][:]]
        self.rm.solve_ik.return_value = {'status': 'Success', 'joint_value': [.4, .1]}
        self.native = Mock()
        self.native.robot_manager = self.rm
        self.native.take_action_cnt = [0]
        self.native.success = [True]
        self.native.end_flag = [False]
        self.native.step_lim = 100
        self.native.step.side_effect = lambda command: self.trace.append('step')
        self.native.sim_step.side_effect = lambda **kwargs: self.trace.append('sim_step')
        self.native.reward_manager.step.side_effect = lambda ids: self.trace.append('reward')
        self.native.reward_manager.get_reward.return_value = [.2]
        self.native.reward_manager.get_score.return_value = [.4]
        # Simulate acquisition: close() owns registered callbacks, not attributes.
        self.adapter.app = Mock()
        self.adapter._cleanup.callback(self.adapter.app.close)
        self.adapter.env = self.native
        self.adapter._cleanup.callback(self.native.close)
        self.adapter._robots = self.robots
        self.adapter._grippers = {'left': .8, 'right': .3}
        self.adapter._finger_limits = {name: [[.1, .5], [-.45, -.05]] for name in self.robots}
        self.adapter._joint_limits = {name: [[-1., 1.], [-.5, .5]] for name in self.robots}
        self.adapter.robot = {'workspace_bounds': [[-.65, -.65, .70], [.65, .35, 1.50]]}
        self.capture = self.enterContext(patch.object(self.adapter, '_capture', side_effect=self.capture_state))
        self.capture_state()

    def data(self, robot):
        return self.measured[robot.arm_name.removesuffix('_arm')]

    def endpose(self, robot, ids, *, is_relative):
        self.assertEqual(ids, [0])
        self.assertTrue(is_relative)
        pose = self.data(robot)['control_pose']
        return [pose['position'] + pose['quaternion']]

    def capture_state(self, feedback=None):
        self.adapter.state = self.adapter._read_state()
        self.adapter.observation = dict(snapshot_id='cpu-snapshot', state=copy.deepcopy(self.adapter.state),
                                        cameras={}, feedback=feedback or {})
        return self.adapter.observation

    def move(self, **updates):
        return dict(tool='move', arguments={'summary': 'Measured link6 target', 'arm': 'left',
                    'pose': [-.3, 0., .9, 0., 0., 0.], 'steps': 1, **updates}, call_id=uuid4().hex)

    def test_macro_has_ten_physics_ticks_but_one_control_step(self):
        commands = self.adapter._grippers.copy()
        before = copy.deepcopy(self.measured)
        goal = {'position': [-.3, 0., 1.], 'quaternion': [1., 0., 0., 0.]}
        self.adapter._macro('left', goal, commands)
        self.assertEqual(self.trace, ['push', 'pop', 'step', 'sim_step'] * 10 + ['reward'])
        self.assertEqual(len(self.queue.commands), 10)
        self.assertEqual(self.native.step.call_count, 10)
        self.assertEqual(self.native.sim_step.call_args_list, [call(render=False)] * 10)
        self.assertEqual(self.native.take_action_cnt, [1])
        self.assertEqual(self.adapter.state['native_steps'], 1)
        self.assertEqual(self.adapter.state['state_version'], 1)
        self.native.reward_manager.step.assert_called_once_with([0])
        self.native.reward_manager.get_reward.assert_called_once_with(final_check=False)
        self.native.reward_manager.get_score.assert_called_once_with(reward_lst=[.2])
        self.capture.assert_called_once_with()
        self.assertFalse(self.adapter.terminated)
        self.assertFalse(self.adapter.truncated)
        self.assertEqual(self.measured, before)  # No fabricated native measurements.
        self.assertIsNot(self.adapter._grippers, commands)
        self.assertEqual(self.adapter._grippers, commands)
        ik_args, ik_kwargs = self.rm.solve_ik.call_args
        self.assertEqual(ik_args[1:], (0, self.robots['left']))
        self.assertEqual(ik_kwargs, {'trans': 'world'})
        self.assertAlmostEqual(ik_args[0][2], .92)  # 2 cm waypoint, no TCP offset.
        self.assertEqual(ik_args[0][3:], [1., 0., 0., 0.])
        for tick, command in enumerate(self.queue.commands):
            alpha = min((tick + 1) / 9, 1.)
            for actual, source, target in zip(command['left_arm']['position'], [.1, -.2], [.4, .1]):
                self.assertAlmostEqual(actual, (1 - alpha) * source + alpha * target)
            for actual, expected in zip(command['right_arm']['position'], [.1, -.2]):
                self.assertAlmostEqual(actual, expected)
            for name in ('left', 'right'):
                self.assertEqual(command[name + '_arm']['velocity'], [0., 0.])
                self.assertEqual(command[name + '_gripper']['velocity'], [0., 0.])
        self.assertEqual(self.native.step.call_args_list, [call(command) for command in self.queue.commands])
        event = next(e for e in self.events if e['event'] == 'native_command')
        self.assertEqual(event['next_step'], 1)
        self.assertEqual(event['commands'], self.queue.commands)

    def test_hold_preserves_joint_measurements_and_gripper_intent(self):
        obs, reward, terminated, truncated, _ = self.adapter.step('hold', {'summary': 'Hold', 'steps': 2}, 'hold')
        self.rm.solve_ik.assert_not_called()
        self.assertEqual(self.native.take_action_cnt, [2])
        self.assertEqual(len(self.queue.commands), 20)
        self.assertEqual(obs['feedback']['steps_executed'], 2)
        self.assertEqual(obs['feedback']['stop_reason'], 'requested_step_budget')
        self.assertEqual(reward, .2)
        self.assertFalse(terminated or truncated)
        for name, expected in [('left', .42), ('right', .38)]:
            for tick in (8, 9, 18, 19):
                fingers = self.queue.commands[tick][name + '_gripper']['position']
                self.assertAlmostEqual(fingers[0], expected)
                self.assertAlmostEqual(fingers[1], -expected + .05)
        self.assertEqual(self.adapter._grippers, {'left': .8, 'right': .3})

    def test_raw_gripper_measurements_are_normalized_with_native_sign_and_clamped(self):
        for raw, left, right in [(.2, .25, .75), (.1, 0., 1.), (.5, 1., 0.), (-.1, 0., 1.), (.8, 1., 0.)]:
            with self.subTest(raw=raw):
                for data in self.measured.values():
                    data['finger_joints'][0] = raw
                state = self.adapter._read_state()
                self.assertAlmostEqual(state['arms']['left']['gripper'], left)
                self.assertAlmostEqual(state['arms']['right']['gripper'], right)
                self.assertEqual(state['arms']['left']['finger_joints'][0], raw)
                self.assertEqual(state['arms']['left']['gripper_command'], .8)
        self.measured['left']['finger_joints'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'proprioception'):
            self.adapter._read_state()

    def test_move_omission_retains_command_and_explicit_gripper_requires_measured_arrival(self):
        obs, *_ = self.adapter.step(**self.move())
        self.assertTrue(obs['feedback']['targets_reached'])
        self.assertEqual(self.adapter._grippers, {'left': .8, 'right': .3})
        obs, *_ = self.adapter.step(**self.move(gripper=1.))
        self.assertEqual(self.adapter._grippers, {'left': 1., 'right': .3})
        self.assertFalse(obs['feedback']['targets_reached'])  # Measured .25 is not commanded 1.
        self.assertEqual(obs['feedback']['stop_reason'], 'max_steps')
        self.assertAlmostEqual(self.queue.commands[-1]['left_gripper']['position'][0], .5)
        self.assertAlmostEqual(self.queue.commands[-1]['right_gripper']['position'][0], .38)

    def test_failed_ik_never_queues_or_executes(self):
        self.rm.solve_ik.return_value = {'status': 'Failure'}
        action = self.move(gripper=0.)
        obs, reward, terminated, truncated, _ = self.adapter.step(**action)
        self.assertEqual(obs['feedback']['call_id'], action['call_id'])
        self.assertFalse(obs['feedback']['call_ok'])
        self.assertEqual(obs['feedback']['execution'], 'none')
        self.assertIn('IK', obs['feedback']['message'])
        self.assertEqual(obs['feedback']['steps_executed'], 0)
        self.assertFalse(self.adapter.blocked or terminated or truncated)
        self.assertEqual(reward, 0.)
        self.assertEqual(self.queue.commands, [])
        self.native.step.assert_not_called()
        self.native.sim_step.assert_not_called()
        self.native.reward_manager.step.assert_not_called()
        self.capture.assert_not_called()
        self.assertEqual(self.adapter._grippers, {'left': .8, 'right': .3})

    def test_invalid_ik_joint_targets_are_rejected_before_queue_admission(self):
        for target in ([.4], [.4, .1, 0.], [1.2, -.2], [float('nan'), .1], [float('inf'), .1], [.4, .6]):
            with self.subTest(target=target):
                self.rm.solve_ik.return_value = {'status': 'Success', 'joint_value': target}
                obs, *_ = self.adapter.step(**self.move())
                self.assertFalse(obs['feedback']['call_ok'])
                self.assertEqual(obs['feedback']['stop_reason'], 'planning_rejected')
                self.assertFalse(self.adapter.blocked)
        self.assertEqual(self.queue.commands, [])
        self.native.step.assert_not_called()

    def test_reused_identity_never_executes_even_after_planning_rejection(self):
        action = self.move(pose=[0, 0, float('nan'), 0, 0, 0])
        obs, *_ = self.adapter.step(**action)
        self.assertFalse(obs['feedback']['call_ok'])
        self.assertEqual(obs['feedback']['steps_executed'], 0)
        with self.assertRaisesRegex(RuntimeError, 'replay'):
            self.adapter.step(**action)
        self.assertEqual(self.queue.commands, [])
        self.rm.solve_ik.assert_not_called()

    def test_expired_motion_deadline_executes_nothing(self):
        self.adapter.deadline = 0.
        obs, *_ = self.adapter.step('hold', {'summary': 'No time'}, 'expired')
        self.assertEqual(obs['feedback']['stop_reason'], 'wall_time_limit')
        self.assertEqual(obs['feedback']['steps_executed'], 0)
        self.assertEqual(obs['feedback']['execution'], 'none')
        self.native.step.assert_not_called()

    def test_queue_admission_failure_blocks_all_further_actions_without_replay(self):
        failure = RuntimeError('Queue acknowledgement lost')
        with patch.object(self.queue, 'push', side_effect=failure) as push:
            with self.assertRaises(RuntimeError) as raised:
                self.adapter.step('hold', {'summary': 'Hold'}, 'lost')
            self.assertIs(raised.exception, failure)
            self.assertEqual(push.call_count, 1)
        self.assertTrue(self.adapter.blocked)
        self.native.step.assert_not_called()
        for tool in ('hold', 'finish', 'capture_scene'):
            with self.assertRaisesRegex(RuntimeError, 'no replay'):
                self.adapter.step(tool, {'summary': 'Do not retry'}, tool)
        outcome = self.adapter.finalize('execution_unknown')
        self.assertIsNone(outcome['task_success'])
        self.assertTrue(outcome['execution_unknown'])
        self.native.reward_manager.get_reward.assert_not_called()

    def test_partial_physics_failure_blocks_further_actions_and_does_not_count_macro(self):
        failure = RuntimeError('Second physics tick failed')
        self.native.sim_step.side_effect = [None, failure]
        with self.assertRaises(RuntimeError) as raised:
            self.adapter.step('hold', {'summary': 'Hold'}, 'partial')
        self.assertIs(raised.exception, failure)
        self.assertTrue(self.adapter.blocked)
        self.assertEqual(self.native.step.call_count, 2)
        self.assertEqual(self.native.take_action_cnt, [0])
        self.native.reward_manager.step.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'no replay'):
            self.adapter.step('hold', {'summary': 'Do not replay'}, 'retry')
        self.assertEqual(self.native.step.call_count, 2)

    def test_success_and_truncation_come_from_reward_manager_not_tool_success(self):
        self.native.step_lim = 1
        obs, _, terminated, truncated, info = self.adapter.step('hold', {'summary': 'Hold', 'steps': 4}, 'limit')
        self.assertTrue(obs['feedback']['call_ok'])
        self.assertFalse(info['task_success'] or terminated)
        self.assertTrue(truncated)
        self.assertEqual(obs['feedback']['steps_executed'], 1)
        self.assertEqual(obs['feedback']['stop_reason'], 'native_end')
        self.native.reward_manager.get_reward.assert_called_once_with(final_check=True)
        self.native.reward_manager.get_reward.return_value = [1.]
        verdict = self.adapter._evaluate(final=True)
        self.assertTrue(verdict['task_success'])
        self.assertTrue(self.adapter.terminated)
        self.assertFalse(self.adapter.truncated)  # Success wins on the last allowed macro.
        self.native.success[0] = False
        verdict = self.adapter._evaluate(final=True)
        self.assertFalse(verdict['task_success'])
        self.assertTrue(verdict['native_failed'])

    def test_finish_is_idempotent_and_false_verdict_is_not_a_failed_submission(self):
        obs, _, terminated, _, info = self.adapter.step('finish', {'summary': 'Claim completion'}, 'finish-ID')
        self.assertTrue(obs['feedback']['call_ok'])
        self.assertEqual(obs['feedback']['call_id'], 'finish-ID')
        self.assertTrue(info['agent_finished'])
        self.assertFalse(info['task_success'])
        self.assertNotIn('task_success', obs['feedback'])
        self.assertTrue(terminated)
        self.assertIsNotNone(self.adapter.outcome)
        outcome = self.adapter.finalize('another reason')
        self.assertIs(outcome, self.adapter.outcome)
        self.assertEqual(outcome['reason'], 'completion_claimed')
        self.native.reward_manager.step.assert_called_once_with([0])
        self.native.reward_manager.get_reward.assert_called_once_with(final_check=True)
        self.native.step.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'active episode'):
            self.adapter.step('hold', {'summary': 'Too late'}, 'late')

    def test_capture_is_read_only_for_physics(self):
        obs, reward, *_ = self.adapter.step('capture_scene', {'summary': 'Look'}, 'capture')
        self.assertIsInstance(obs, dict)
        self.assertTrue(obs['feedback']['call_ok'])
        self.assertEqual(obs['feedback']['execution'], 'none')
        self.assertEqual(obs['state']['native_steps'], 0)
        self.assertEqual(reward, 0.)
        self.capture.assert_called_once_with()
        self.native.step.assert_not_called()
        self.native.sim_step.assert_not_called()

    def test_preload_closure_uses_reachable_measured_joint_limits(self):
        robot = self.robots['left']
        robot.gripper_scale = [-.01, .044]
        robot.gripper_move = {'sign': 1, 'mimic': [0, 1, 0]}
        self.adapter._finger_limits['left'] = [[0., .044], [0., .044]]
        self.measured['left']['finger_joints'] = [0., 0.]
        self.capture_state()
        self.assertTrue(self.adapter._gripper_arrived('left', 0.))
        # Native command normalization can differ from reachable physical opening.
        self.assertGreater(self.adapter.state['arms']['left']['gripper'], 0.)
        self.measured['left']['finger_joints'] = [.02, .02]
        self.capture_state()
        self.assertFalse(self.adapter._gripper_arrived('left', 0.))
        self.measured['left']['finger_joints'] = [.044, .044]
        self.capture_state()
        self.assertTrue(self.adapter._gripper_arrived('left', 1.))

    def test_rejection_after_completed_macro_is_not_unknown_or_replayed(self):
        self.rm.solve_ik.side_effect = [
            {'status': 'Success', 'joint_value': [.4, .1]}, {'status': 'Fail'}]
        obs, *_ = self.adapter.step(**self.move(pose=[-.3, 0., 1., 0., 0., 0.], steps=4))
        self.assertFalse(obs['feedback']['call_ok'])
        self.assertEqual(obs['feedback']['stop_reason'], 'planning_rejected')
        self.assertEqual(obs['feedback']['execution'], 'confirmed')
        self.assertEqual(obs['feedback']['steps_executed'], 1)
        self.assertEqual(self.native.sim_step.call_count, 10)
        self.assertEqual(self.rm.solve_ik.call_count, 2)
        self.assertFalse(self.adapter.blocked)

    def test_close_releases_both_resources_even_after_native_close_error(self):
        failure = RuntimeError('Environment close failed')
        self.native.close.side_effect = failure
        writer = Mock()
        self.adapter._cleanup.callback(writer.close)
        with self.assertRaises(RuntimeError) as raised:
            self.adapter.close()
        self.assertIs(raised.exception, failure)
        self.adapter.close()
        writer.close.assert_called_once_with()
        self.native.close.assert_called_once_with()
        self.adapter.app.close.assert_called_once_with()
        self.assertTrue(self.adapter.closed)

    def test_failed_reset_closes_and_cannot_be_retried(self):
        adapter = RoboDojoEnv(self.args, self.root, self.record)
        app = Mock()

        def partial_launch():
            adapter.app = app
            adapter._cleanup.callback(app.close)
            raise RuntimeError('Startup failed')

        with patch.object(adapter, '_launch', side_effect=partial_launch) as launch:
            with self.assertRaisesRegex(RuntimeError, 'Startup failed'):
                adapter.reset()
            self.assertTrue(adapter.closed)
            app.close.assert_called_once_with()
            with self.assertRaisesRegex(RuntimeError, 'new adapter'):
                adapter.reset()
            launch.assert_called_once_with()

    def test_workspace_checks_both_limits_of_each_position_axis(self):
        lower, upper = self.adapter.robot['workspace_bounds']
        for axis in range(3):
            for bound, offset in ((lower, -.001), (upper, .001)):
                with self.subTest(axis=axis, boundary=bound[axis]):
                    pose = [0., 0., .9, -3., 2., 3.]
                    pose[axis] = bound[axis]
                    obs, *_ = self.adapter.step(**self.move(pose=pose))
                    self.assertTrue(obs['feedback']['call_ok'])
                    count = self.native.sim_step.call_count
                    pose[axis] += offset
                    obs, *_ = self.adapter.step(**self.move(pose=pose))
                    self.assertEqual(obs['feedback']['stop_reason'], 'planning_rejected')
                    self.assertEqual(obs['feedback']['steps_executed'], 0)
                    self.assertEqual(self.native.sim_step.call_count, count)

    def test_joint_span_guard_is_independent_of_native_bounds(self):
        self.adapter._joint_limits['left'] = [[-3., 3.], [-3., 3.]]
        self.rm.solve_ik.return_value = {'status': 'Success', 'joint_value': [1.2, -.2]}
        obs, *_ = self.adapter.step(**self.move())
        self.assertEqual(obs['feedback']['stop_reason'], 'planning_rejected')
        self.assertEqual(self.queue.commands, [])

    def test_held_arm_uses_its_own_native_joint_bounds(self):
        self.adapter._joint_limits['right'] = [[-.05, .05], [-.5, .5]]
        obs, *_ = self.adapter.step(**self.move())
        self.assertEqual(obs['feedback']['stop_reason'], 'planning_rejected')
        self.assertEqual(self.queue.commands, [])
        self.native.step.assert_not_called()

    def test_keyboard_interrupt_during_physics_is_original_and_blocks_replay(self):
        failure = KeyboardInterrupt('Interrupted physics')
        self.native.sim_step.side_effect = failure
        with self.assertRaises(KeyboardInterrupt) as raised:
            self.adapter.step('hold', {'summary': 'Hold'}, 'interrupted')
        self.assertIs(raised.exception, failure)
        self.assertTrue(self.adapter.blocked)
        self.assertEqual(self.native.take_action_cnt, [0])
        with self.assertRaisesRegex(RuntimeError, 'no replay'):
            self.adapter.step('hold', {'summary': 'Retry'}, 'interrupted')
        self.native.sim_step.assert_called_once_with(render=False)
        self.native.reward_manager.step.assert_not_called()

    def test_unexpected_capture_error_propagates_without_tool_feedback(self):
        for failure in (RuntimeError('Camera failed'), ValueError('Bad calibration'), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                before = copy.deepcopy(self.adapter.observation)
                self.capture.side_effect = failure
                identity = uuid4().hex
                with self.assertRaises(type(failure)) as raised:
                    self.adapter.step('capture_scene', {'summary': 'Look'}, identity)
                self.assertIs(raised.exception, failure)
                self.assertEqual(self.adapter.observation, before)
                with self.assertRaisesRegex(RuntimeError, 'replay'):
                    self.adapter.step('capture_scene', {'summary': 'Look'}, identity)
        self.assertEqual(self.capture.call_count, 3)
        self.native.step.assert_not_called()
        self.native.reward_manager.step.assert_not_called()

    def test_capture_failure_after_motion_preserves_acknowledged_macro(self):
        failure = RuntimeError('Capture failed after physics')
        self.capture.side_effect = failure
        with self.assertRaises(RuntimeError) as raised:
            self.adapter.step('hold', {'summary': 'Hold'}, 'acknowledged')
        self.assertIs(raised.exception, failure)
        self.assertEqual(self.native.take_action_cnt, [1])
        self.assertEqual(self.native.sim_step.call_count, 10)
        self.assertFalse(self.adapter.blocked)  # Physics was acknowledged; this is a capture failure.
        with self.assertRaisesRegex(RuntimeError, 'replay'):
            self.adapter.step('hold', {'summary': 'Hold'}, 'acknowledged')
        self.assertEqual(self.native.sim_step.call_count, 10)
        self.capture.assert_called_once_with()

    def test_unexpected_ik_exception_is_not_planning_feedback(self):
        failure = ValueError('Unexpected IK implementation error')
        self.rm.solve_ik.side_effect = failure
        with self.assertRaises(ValueError) as raised:
            self.adapter.step(**self.move())
        self.assertIs(raised.exception, failure)
        self.rm.solve_ik.assert_called_once()
        self.assertEqual(self.queue.commands, [])
        self.assertEqual(self.adapter.observation['feedback'], {})

    def test_close_runs_all_callbacks_and_propagates_last_cleanup_failure(self):
        first, second = RuntimeError('Writer failed'), RuntimeError('Native close failed')
        order = []

        def close_writer():
            order.append('writer')
            raise first

        def close_native():
            order.append('native')
            raise second

        self.native.close.side_effect = close_native
        self.adapter.app.close.side_effect = lambda: order.append('app')
        self.adapter._cleanup.callback(close_writer)
        with self.assertRaises(RuntimeError) as raised:
            self.adapter.close()
        self.assertIs(raised.exception, second)
        self.assertEqual(order, ['writer', 'native', 'app'])
        self.adapter.close()
        self.assertEqual(order, ['writer', 'native', 'app'])
        # Standard ExitStack reports the last cleanup failure; no custom error aggregation.

    def test_complete_main_agent_wrapper_workflow_with_mock_model_and_native(self):
        requests = []
        actions = [('capture_scene', {'summary': 'Inspect'}),
                   ('hold', {'summary': 'Settle', 'steps': 2}),
                   ('move', self.move()['arguments']),
                   ('finish', {'summary': 'Observed completion'})]
        image = self.root / 'current.png'
        image.write_bytes(b'offline-image-bytes')
        self.adapter.instruction = 'Stack the bowls.'
        self.adapter.initial_state = copy.deepcopy(self.adapter.state)

        def capture(feedback=None):
            obs = self.capture_state(feedback)
            obs['cameras'] = {'head': {'path': str(image), 'image_size': [2, 3],
                                      'intrinsic_K': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                      'T_world_from_optical': [[1, 0, 0, 0], [0, 1, 0, 0],
                                                               [0, 0, 1, 0], [0, 0, 0, 1]]}}
            return obs

        self.capture.side_effect = capture
        initial = copy.deepcopy(capture())

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            tool, arguments = actions[len(requests) - 1]
            reply = model_reply(f'workflow-{len(requests)}', tool=tool, tokens=11)
            reply['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = json.dumps(arguments)
            return httpx.Response(200, json=reply)

        def environment(args, run_dir, record):
            self.adapter.run_dir, self.adapter.record = run_dir, record
            return self.adapter

        client = httpx.Client(transport=httpx.MockTransport(respond), trust_env=False)
        self.addCleanup(client.close)
        with patch.object(direct, 'parse_args', return_value=self.args), \
                patch.object(direct, 'RoboDojoEnv', side_effect=environment), \
                patch.object(self.adapter, 'reset', return_value=(initial, {})), \
                patch.object(direct.httpx, 'Client', return_value=client), patch('sys.stdout', new=Mock()):
            direct.main()
        self.assertTrue(client.is_closed)
        self.native.close.assert_called_once_with()
        self.adapter.app.close.assert_called_once_with()
        self.assertEqual(len(requests), 4)
        self.assertEqual(self.native.take_action_cnt, [3])
        self.assertEqual(self.native.sim_step.call_count, 30)
        for index, body in enumerate(requests):
            messages = body['messages']
            self.assertIn('return both arms', messages[1]['content'])
            self.assertIn('link6', messages[0]['content'])
            self.assertFalse(body['parallel_tool_calls'])
            pairs = messages[2:-1]
            self.assertEqual([p['role'] for p in pairs], ['assistant', 'tool'] * index)
            for number, (action, receipt) in enumerate(zip(pairs[::2], pairs[1::2]), 1):
                self.assertEqual(action['tool_calls'][0]['id'], f'workflow-{number}')
                self.assertEqual(receipt['tool_call_id'], f'workflow-{number}')
                self.assertEqual(json.loads(receipt['content'])['call_id'], f'workflow-{number}')
            self.assertEqual(sum(p['type'] == 'image_url' for p in messages[-1]['content']), 1)
            self.assertNotIn('image_url', json.dumps(messages[:-1]))
            self.assertNotIn('task_success', json.dumps(body))
        run_dir, = self.args.output_dir.iterdir()
        result = json.loads((run_dir / 'result.json').read_text())
        self.assertEqual((result['decisions'], result['reported_tokens'], result['cleanup']), (4, 44, 'completed'))
        self.assertTrue(result['agent_finished'])
        self.assertIs(result['task_success'], False)
        events = [json.loads(line) for line in (run_dir / 'trajectory.jsonl').read_text().splitlines()]
        self.assertEqual([e['action']['tool'] for e in events if e['event'] == 'action'], [a[0] for a in actions])
        self.assertEqual(len([e for e in events if e['event'] == 'native_command']), 3)
        self.assertNotIn('offline-key', json.dumps(events))
        self.assertEqual(events[-2]['cleanup'], 'pending')
        self.assertEqual(events[-1]['cleanup'], 'completed')


class CameraTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            self.skipTest(f'Optional CPU image dependencies unavailable: {exc}')
        self.Image = Image

        # Patch only optional/native modules; keep real PNG encoding and adapter math.
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.orientation = [math.sqrt(.5), 0., 0., math.sqrt(.5)]
        self.quat2mat = Mock(return_value=rotation)
        self.wait = Mock()
        self.writer = Mock(side_effect=AssertionError('Video disabled; no encoder should start'))
        modules = {}
        for name in ('env', 'env.camera_manager', 'env.camera_manager.capture',
                     'env.camera_manager.capture.render_sync', 'utils', 'utils.save_file',
                     'transforms3d', 'transforms3d.quaternions'):
            module = ModuleType(name)
            module.__path__ = []
            modules[name] = module
        modules['env.camera_manager.capture.render_sync'].wait_for_latest_cameras = self.wait
        modules['utils.save_file'].VideoStreamWriter = self.writer
        modules['transforms3d.quaternions'].quat2mat = self.quat2mat
        adapter = RoboDojoEnv(self.args, self.root, self.record)
        self.addCleanup(adapter.close)
        adapter.cameras = ['head']
        self.camera = Mock()
        self.camera.get_world_pose.return_value = ([10.5, 20.25, 30.75], self.orientation)
        self.intrinsic = [[400., 0., 1.], [0., 500., 1.], [0., 0., 1.]]
        self.camera.get_intrinsics_matrix.return_value = self.intrinsic
        rgba = np.zeros((2, 3, 4), dtype=np.uint8)
        rgba[:, :, :] = [12, 34, 56, 255]
        origin = Mock()
        origin.detach.return_value.cpu.return_value.numpy.return_value = np.array([10., 20., 30.])
        native = Mock()
        native.env_origins = [origin]
        native.camera_manager = SimpleNamespace(camera_names=[['cam_head']], cameras=[[self.camera]])
        native.capture_manager.step.return_value = [{'rgb': [{'data': rgba}]}]
        adapter.env = native
        adapter._cleanup.callback(native.close)
        self.adapter, self.native, self.state = adapter, native, fake_observation().state
        self.enterContext(patch.dict(sys.modules, modules))
        self.enterContext(patch.object(adapter, '_read_state', return_value=self.state))

    def test_capture_uses_ros_optical_world_from_camera_and_environment_local_origin(self):
        obs = self.adapter._capture({'call_id': 'capture-ID'})
        self.camera.get_world_pose.assert_called_once_with(camera_axes='ros')
        self.quat2mat.assert_called_once_with(self.orientation)
        self.wait.assert_called_once_with(self.native, capture_manager=self.native.capture_manager)
        self.native.capture_manager.step.assert_called_once_with(env_ids=[0])
        calibration = obs['cameras']['head']
        self.assertEqual(calibration['image_size'], [2, 3])
        self.assertEqual(calibration['intrinsic_K'], self.intrinsic)
        self.assertEqual(calibration['T_world_from_optical'],
                         [[0., -1., 0., .5], [1., 0., 0., .25], [0., 0., 1., .75], [0., 0., 0., 1.]])
        path = Path(calibration['path'])
        self.assertTrue(path.is_relative_to(self.root / 'images'))
        self.assertIn(obs['snapshot_id'], path.name)
        with self.Image.open(path) as image:
            self.assertEqual((image.mode, image.size), ('RGB', (3, 2)))
            self.assertEqual(image.getpixel((0, 0)), (12, 34, 56))
        self.assertEqual(obs['feedback'], {'call_id': 'capture-ID'})
        self.state['native_steps'] = 999
        self.assertEqual(obs['state']['native_steps'], 0)  # Immutable captured state.
        self.native.step.assert_not_called()
        self.native.sim_step.assert_not_called()
        self.writer.assert_not_called()

    def test_camera_selection_is_nonempty_distinct_and_native_supported(self):
        # Validate against native camera names, not an entry-level fixed allowlist.
        self.assertEqual(list(self.adapter._capture()['cameras']), ['head'])
        accepted = []
        for selection in ([], ['missing'], ['head', 'head']):
            self.adapter.cameras = selection
            try:
                self.adapter._capture()
            except ValueError:
                pass
            else:
                accepted.append(selection)
        self.native.step.assert_not_called()
        self.native.sim_step.assert_not_called()
        self.assertEqual(accepted, [], 'Empty, duplicate and unsupported camera selections must be rejected')


if __name__ == '__main__':
    unittest.main()