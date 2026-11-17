"""CPU tests for the hybrid wrapper's actor_act tool, with a fake action-model server and reviewer."""
import json
import os
import unittest
from unittest import mock

import httpx
import numpy as np

from cleanrua import hybrid_robodojo as hybrid
from test_direct_robodojo import FakeRoboDojo


class FakePolicy:
    """Returns chunks of native RoboDojo actions: the left arm's joints count up by 0.1 per predicted step."""

    def __init__(self, horizon=50):
        self.horizon, self.calls = horizon, []

    def call(self, func_name, obs=None):
        self.calls.append((func_name, obs))
        if func_name == 'get_action':
            return [{'left_arm_joint_state': [.1 * (i + 1)] * 6, 'left_ee_joint_state': [1.5],  # Out-of-range opening.
                     'right_arm_joint_state': [-1.] * 6, 'right_ee_joint_state': [0.]} for i in range(self.horizon)]


class FakeReviewer:
    """Answers each request with the next scripted decision, checked like the real reviewer."""

    def __init__(self, *decisions):
        self.decisions, self.requests = list(decisions), []

    def __call__(self, context, images, check):
        self.requests.append((context, images))
        decision = self.decisions.pop(0)
        review = hybrid.Review(request_id=context['request_id'], reason='Evidence.', assessment=assessment(
            'progressing' if context['episode_started'] else 'not_started', decision.pop('intent', 'aligned')), **decision)
        check(review)
        return review


def assessment(status='not_started', intent='aligned'):
    return {'task_progress': {'verified_completed': [], 'currently_attempting': 'grasp', 'remaining': ['place']},
            'current_subgoal': 'grasp', 'execution_status': status, 'execution_evidence': 'Seen.',
            'expected_next_intent': 'approach', 'predicted_next_intent': 'approach', 'intent_status': intent,
            'intent_evidence': 'Preview.'}


class ActorActTest(unittest.TestCase):
    def make(self, reviewer=None, env=None):
        self.policy = FakePolicy()
        self.env = hybrid.RoboDojoWrapper(env or FakeRoboDojo(), self.policy, reviewer)
        self.env.reset()

    def act(self, **arguments):
        obs, reward, terminated, truncated, info = self.env.step({'name': 'actor_act', 'arguments': json.dumps(arguments)})
        return obs, json.loads(obs['feedback'])

    def test_unreviewed_runs_a_prefix_of_each_fresh_chunk_on_active_arms_only(self):
        self.make()
        obs, feedback = self.act(instruction='pick up the bowl', active_groups=['left'], actions_per_chunk=3, max_chunks=2)
        self.assertEqual((feedback['ok'], feedback['steps'], feedback['chunks'], feedback['stop_reason']),
                         (True, 6, 2, 'max_chunks'))
        self.assertEqual(obs['native_steps'], 6)
        np.testing.assert_allclose(obs['arms']['left']['joints'], .3)  # Third step of the second chunk.
        self.assertEqual(obs['arms']['left']['gripper_command'], 1.)  # Clipped to [0, 1].
        np.testing.assert_allclose(obs['arms']['right']['joints'], 0.)  # Inactive arm holds.
        self.assertEqual(obs['arms']['right']['gripper_command'], 1.)
        names = [name for name, _ in self.policy.calls]
        self.assertEqual(names, ['reset', 'update_obs', 'get_action', 'update_obs', 'get_action'])
        sent = self.policy.calls[3][1]  # The second chunk sees the state after the first chunk.
        self.assertEqual(sent['instruction'], 'pick up the bowl')
        np.testing.assert_allclose(sent['state']['left_arm_joint_state'], .3)

    def test_reviewer_sees_preview_and_previous_chunk_and_can_stop(self):
        reviewer = FakeReviewer({'mode': 'student', 'steps': 2, 'continuation': 'continue'},
                                {'mode': 'stop', 'steps': 0, 'continuation': 'phase_complete'})
        self.make(reviewer)
        progress = {'verified_completed': ['bowl located'], 'currently_attempting': 'grasp', 'remaining': ['place']}
        obs, feedback = self.act(instruction='pick up the bowl', active_groups=['left'], actions_per_chunk=4,
                                 task_progress=progress)
        self.assertEqual((feedback['ok'], feedback['steps'], feedback['chunks'], feedback['stop_reason'],
                          feedback['stop_evidence']), (True, 2, 2, 'phase_complete', 'Evidence.'))
        self.assertEqual([r['mode'] for r in feedback['reviews']], ['student', 'stop'])
        (first, first_images), (second, second_images) = reviewer.requests
        self.assertEqual((first['task_progress'], first['episode_started'], len(first['preview'])), (progress, False, 4))
        self.assertEqual(set(first['preview'][0]), {'left'})  # Active arms only: [grasp-site pose, opening].
        np.testing.assert_allclose(first['preview'][1]['left'][2] - first['current_state']['left']['pose'][2], .002, atol=1e-4)
        self.assertEqual(first['preview'][0]['left'][6], 1.)
        self.assertEqual(set(first_images), {'current/head'})
        self.assertEqual(set(second_images), {'current/head', 'previous/head'})
        self.assertEqual((second['previous_chunk']['mode'], second['episode_started']), ('student', True))
        self.assertEqual(second['task_progress'], assessment()['task_progress'])  # Carried from the previous review.
        self.assertEqual(obs['native_steps'], 2)

    def test_eef_correction_holds_an_absolute_target_under_constraints(self):
        env = FakeRoboDojo()
        reviewer = FakeReviewer({'mode': 'eef', 'steps': 3, 'continuation': 'continue', 'intent': 'misaligned'})
        self.make(reviewer, env)
        pose = (self.env.observation['arms']['left']['pose'] + [0., 0., -.03, 0., 0., 0.]).tolist()
        reviewer.decisions[0]['target'] = {'left': {'pose': pose, 'gripper_closed': True}}
        obs, feedback = self.act(instruction='grasp', active_groups=['left'], max_chunks=1,
                                 constraints={'gripper_targets': {'left': .4}})
        self.assertEqual((feedback['ok'], feedback['steps']), (True, 3))
        np.testing.assert_allclose(obs['arms']['left']['pose'], pose, atol=1e-9)
        self.assertEqual(obs['arms']['left']['gripper_command'], .4)  # The constraint overrides gripper_closed.
        np.testing.assert_allclose(obs['arms']['right']['joints'], 0.)

    def test_edit_correction_ramps_an_offset_over_the_preview(self):
        reviewer = FakeReviewer({'mode': 'edit', 'steps': 2, 'continuation': 'continue', 'intent': 'misaligned',
                                 'edit': {'left': {'delta_position': [.02, 0., 0.], 'delta_rotation_vector': [0., 0., 0.],
                                                   'gripper': 'closed'}}})
        self.make(reviewer)
        obs, feedback = self.act(instruction='grasp', active_groups=['left'], max_chunks=1)
        self.assertEqual((feedback['ok'], feedback['steps']), (True, 2))
        context = reviewer.requests[0][0]
        np.testing.assert_allclose(obs['arms']['left']['pose'][:3],  # Second preview pose plus the full offset.
                                   np.array(context['preview'][1]['left'][:3]) + [.02, 0., 0.], atol=1e-4)
        self.assertEqual(obs['arms']['left']['gripper_command'], 0.)

    def test_rejected_decisions_raise_without_motion(self):
        for decision in [{'mode': 'stop', 'steps': 3, 'continuation': 'stop'},  # stop runs no steps.
                         {'mode': 'student', 'steps': 26, 'continuation': 'continue'},  # Beyond actions_per_chunk.
                         {'mode': 'eef', 'steps': 2, 'continuation': 'continue',  # Corrections need failure/misalignment.
                          'target': {'left': {'pose': [0.] * 6, 'gripper_closed': False}}},
                         {'mode': 'eef', 'steps': 2, 'continuation': 'continue', 'intent': 'misaligned',  # Too far.
                          'target': {'left': {'pose': [0.] * 6, 'gripper_closed': False}}}]:
            self.make(FakeReviewer(decision))
            obs, feedback = self.act(instruction='grasp', active_groups=['left'])
            self.assertFalse(feedback['ok'], decision)
            self.assertEqual(obs['native_steps'], 0)

    def test_invalid_arguments_do_not_query_the_policy(self):
        self.make()
        for arguments in [{'instruction': '', 'active_groups': ['left']}, {'instruction': 'x', 'active_groups': ['middle']},
                          {'instruction': 'x', 'active_groups': ['left'], 'constraints': {'gripper_targets': {'right': 0.}}}]:
            obs, feedback = self.act(**arguments)
            self.assertFalse(feedback['ok'])
        self.assertEqual([name for name, _ in self.policy.calls], ['reset'])

    def test_schema_matches_tool_contract(self):
        self.make()
        schema = {t['function']['name']: t['function']['parameters'] for t in self.env.schemas()}['actor_act']
        self.assertEqual(set(schema['required']), {'instruction', 'active_groups'})
        self.assertEqual(schema['properties']['actions_per_chunk']['default'], 25)
        self.assertEqual(schema['properties']['max_chunks']['default'], 6)
        self.assertEqual(set(schema['properties']), {'instruction', 'active_groups', 'constraints', 'actions_per_chunk',
                                                     'max_chunks', 'task_progress'})
        self.assertNotIn('$ref', json.dumps(schema))


class ReviewerTest(unittest.TestCase):
    def test_one_repair_after_a_rejected_decision(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            context = json.loads(requests[0]['messages'][1]['content'][0]['text'])
            review = {'request_id': 'stale' if len(requests) == 1 else context['request_id'], 'mode': 'student',
                      'steps': 1, 'reason': 'Aligned.', 'continuation': 'continue', 'assessment': assessment()}
            return httpx.Response(200, json={'usage': {'total_tokens': 7}, 'choices': [{'message': {'role': 'assistant',
                'tool_calls': [{'id': 'r', 'type': 'function', 'function': {'name': 'submit_review',
                                                                            'arguments': json.dumps(review)}}]}}]})

        variables = {'CLEANRUA_API_KEY': 'k', 'CLEANRUA_BASE_URL': 'https://model.invalid/v1', 'CLEANRUA_MODEL_NAME': 'm'}
        with mock.patch.dict(os.environ, variables):
            reviewer = hybrid.Reviewer(hybrid.Args())
        reviewer.client = httpx.Client(base_url=reviewer.client.base_url, transport=httpx.MockTransport(handler))
        reviewer.reset('Task: stack the bowls.')

        def check(review):
            if review.request_id != 'abc':
                raise ValueError('request_id must echo the current request')

        review = reviewer({'request_id': 'abc'}, {'current/head': np.zeros((4, 4, 3), np.uint8)}, check)
        self.assertEqual((review.request_id, reviewer.total_tokens, len(requests)), ('abc', 14, 2))
        first, second = requests
        self.assertEqual(first['tool_choice'], {'type': 'function', 'function': {'name': 'submit_review'}})
        self.assertEqual(json.loads(first['messages'][1]['content'][0]['text'])['global_instruction'], 'Task: stack the bowls.')
        self.assertIn('rejected: request_id must echo', second['messages'][-1]['content'])


if __name__ == '__main__':
    unittest.main()
