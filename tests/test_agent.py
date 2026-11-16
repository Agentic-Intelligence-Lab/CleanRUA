"""Exercise the actual RUAgent with fake HTTP, never a live provider."""
import base64
import json
import os
import unittest
from unittest.mock import Mock, patch

import httpx

import direct_robodojo as m
from offline_helpers import OfflineTestCase, fake_observation, model_reply


class AgentTests(OfflineTestCase):
    def make_agent(self, responder):
        # Replace construction, not a live client's transport after construction.
        client = httpx.Client(transport=httpx.MockTransport(responder), trust_env=False)
        self.addCleanup(client.close)
        with patch.object(m.httpx, 'Client', return_value=client) as factory:
            agent = m.RUAgent(self.args, m.tool_schemas(['left', 'right']),
                               {'control_link': 'link6'}, self.record)
        self.assertFalse(factory.call_args.kwargs['trust_env'])
        self.assertEqual(factory.call_args.kwargs['headers'], {'Authorization': 'Bearer offline-key'})
        self.assertEqual(factory.call_args.kwargs['timeout'].read, self.args.request_timeout)
        agent.reset('Stack the bowls.')
        self.addCleanup(agent.close)
        return agent

    def test_bounded_history_exact_ids_current_images_and_usage(self):
        requests = []

        def responder(request):
            self.assertEqual(str(request.url), 'https://model.invalid/v1/chat/completions')
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=model_reply(f'opaque:{len(requests)}', tokens=17))

        self.args.history_size = 2
        agent = self.make_agent(responder)
        previous_id = None
        for index in range(5):
            obs = fake_observation()
            obs.snapshot_id = f'snapshot-{index}'
            obs.feedback = {'call_id': previous_id, 'call_ok': True, 'execution': 'none'}
            for name in ('head', 'left_wrist', 'right_wrist'):
                path = self.root / f'{index}-{name}.png'
                # The policy transports bytes; PNG decoding belongs to capture tests.
                path.write_bytes(f'current-{index}-{name}'.encode())
                obs.cameras[name] = {'path': str(path), 'intrinsic_K': [[index, 0, 0]],
                                     'T_world_from_optical': [[1, 0, 0, index]], 'image_size': [2, 3]}
            action = agent.get_action(obs)
            self.assertEqual(action.call_id, f'opaque:{index + 1}')
            self.assertEqual(action.tool, 'capture_scene')
            self.assertEqual(action.arguments, {'summary': 'Look'})
            previous_id = action.call_id

        self.assertEqual(agent.total_tokens, 85)
        self.assertEqual(agent.used_ids, {f'opaque:{i}' for i in range(1, 6)})
        self.assertEqual(len(agent.history), 2)
        body = requests[-1]
        self.assertEqual(body['model'], 'offline-model')
        self.assertEqual(body['tool_choice'], 'required')
        self.assertFalse(body['parallel_tool_calls'])
        self.assertFalse(body['stream'])
        self.assertEqual(body['max_tokens'], self.args.max_output_tokens)
        self.assertNotIn('enable_thinking', body)
        pairs = body['messages'][2:-1]
        self.assertEqual([part['role'] for part in pairs], ['assistant', 'tool'] * 2)
        self.assertEqual([part['tool_call_id'] for part in pairs[1::2]], ['opaque:3', 'opaque:4'])
        for call, reply in zip(pairs[::2], pairs[1::2]):
            identity = call['tool_calls'][0]['id']
            self.assertEqual(identity, reply['tool_call_id'])
            self.assertEqual(identity, json.loads(reply['content'])['call_id'])
        self.assertNotIn('image_url', json.dumps(body['messages'][:-1]))
        content = body['messages'][-1]['content']
        images = [part['image_url']['url'] for part in content if part['type'] == 'image_url']
        self.assertEqual([base64.b64decode(image.split(',', 1)[1]) for image in images],
                         [f'current-4-{name}'.encode() for name in obs.cameras])
        current = json.loads(content[0]['text'].removeprefix('Current observation: '))
        self.assertEqual(current['snapshot_id'], 'snapshot-4')
        self.assertNotIn('path', current['cameras']['head'])
        self.assertEqual(current['cameras']['head']['T_world_from_optical'], [[1, 0, 0, 4]])
        logged = json.dumps(self.events)
        self.assertNotIn('offline-key', logged)
        self.assertNotIn('data:image/png;base64,', logged)
        self.assertEqual(self.events[-2]['images'], {n: c['path'] for n, c in obs.cameras.items()})
        self.assertNotIn('task_success', json.dumps(body))

    def test_feedback_mismatch_stops_before_another_http_request(self):
        responder = Mock(return_value=httpx.Response(200, json=model_reply('exact-ID')))
        agent = self.make_agent(responder)
        obs = fake_observation()
        agent.get_action(obs)
        for feedback in ({}, {'call_id': 'exact-id'}):
            with self.subTest(feedback=feedback), self.assertRaisesRegex(RuntimeError, 'feedback'):
                obs.feedback = feedback
                agent.get_action(obs)
        self.assertEqual(responder.call_count, 1)
        self.assertIsNotNone(agent.pending)
        self.assertEqual(len(agent.history), 0)

    def test_repeated_id_is_rejected_even_after_history_eviction(self):
        replies = iter([model_reply('first'), model_reply('second'), model_reply('third'), model_reply('first')])
        self.args.history_size = 1
        agent = self.make_agent(lambda request: httpx.Response(200, json=next(replies)))
        obs = fake_observation()
        for _ in range(3):
            action = agent.get_action(obs)
            obs.feedback = {'call_id': action.call_id, 'call_ok': True}
        with self.assertRaisesRegex(ValueError, 'replay'):
            agent.get_action(obs)
        self.assertIsNone(agent.pending)
        self.assertEqual(agent.total_tokens, 40)  # Rejected replies still consumed tokens.

    def test_reset_clears_history_pending_ids_and_usage(self):
        agent = self.make_agent(lambda request: httpx.Response(200, json=model_reply('reusable-after-reset')))
        obs = fake_observation()
        action = agent.get_action(obs)
        obs.feedback = {'call_id': action.call_id}
        agent._remember_feedback(obs)
        agent.reset('New task')
        self.assertEqual((agent.task, agent.total_tokens, len(agent.history)), ('New task', 0, 0))
        self.assertIsNone(agent.pending)
        self.assertFalse(agent.used_ids)
        self.assertEqual(agent.get_action(fake_observation()).call_id, action.call_id)

    def test_aliyun_is_the_only_provider_specific_request(self):
        self.args.provider = 'aliyun'

        def responder(request):
            self.assertIs(json.loads(request.content)['enable_thinking'], False)
            return httpx.Response(200, json=model_reply('aliyun'))

        self.make_agent(responder).get_action(fake_observation())

    def test_invalid_responses_never_create_pending_actions(self):
        invalid = []
        call = model_reply()['choices'][0]['message']['tool_calls'][0]
        for calls in ([], [None], [call, call]):
            raw = model_reply()
            raw['choices'][0]['message']['tool_calls'] = calls
            invalid.append((raw, ValueError))
        for arguments in ('[]', '{', '{"gripper": NaN}'):
            raw = model_reply()
            raw['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = arguments
            invalid.append((raw, ValueError))
        for usage in ({}, {'total_tokens': True}, {'total_tokens': -1}, {'total_tokens': 1.5}):
            raw = model_reply()
            raw['usage'] = usage
            invalid.append((raw, RuntimeError))
        for raw, error in invalid:
            with self.subTest(raw=raw):
                agent = self.make_agent(lambda request: httpx.Response(200, json=raw))
                with self.assertRaises(error):
                    agent.get_action(fake_observation())
                self.assertIsNone(agent.pending)
                self.assertFalse(agent.used_ids)

    def test_http_failure_and_timeout_are_not_retried(self):
        for failure in (httpx.Response(429), httpx.ReadTimeout('offline timeout')):
            with self.subTest(failure=failure):
                responder = Mock()
                if isinstance(failure, Exception):
                    responder.side_effect = failure
                else:
                    responder.return_value = failure
                agent = self.make_agent(responder)
                with self.assertRaises((RuntimeError, httpx.ReadTimeout)):
                    agent.get_action(fake_observation())
                self.assertEqual(responder.call_count, 1)
                self.assertEqual(agent.total_tokens, 0)
                self.assertIsNone(agent.pending)

    def test_missing_credential_does_not_create_http_client(self):
        with patch.dict(os.environ, {self.args.api_key_env: ''}), patch.object(m.httpx, 'Client') as client:
            with self.assertRaisesRegex(ValueError, 'credential'):
                m.RUAgent(self.args, [], {}, self.record)
            client.assert_not_called()

    def test_usage_requires_nonnegative_integer_including_zero(self):
        for tokens in (0, 20):
            with self.subTest(tokens=tokens):
                agent = self.make_agent(lambda request: httpx.Response(200, json=model_reply(tokens=tokens)))
                agent.get_action(fake_observation())
                self.assertEqual(agent.total_tokens, tokens)
        for usage in ({}, {'total_tokens': -1}, {'total_tokens': True}, {'total_tokens': '20'},
                      {'total_tokens': 2.5}, {'total_tokens': None}, None, []):
            with self.subTest(usage=usage):
                raw = model_reply()
                raw['usage'] = usage
                agent = self.make_agent(lambda request: httpx.Response(200, json=raw))
                with self.assertRaises(RuntimeError):
                    agent.get_action(fake_observation())
                self.assertIsNone(agent.pending)
                self.assertFalse(agent.used_ids)

    def test_http_failure_preserves_status_without_response_body_or_retries(self):
        responder = Mock(return_value=httpx.Response(429, text='secret response body'))
        agent = self.make_agent(responder)
        with self.assertRaisesRegex(RuntimeError, 'Model HTTP 429') as raised:
            agent.get_action(fake_observation())
        self.assertNotIn('secret', str(raised.exception))
        responder.assert_called_once()

    def test_malformed_model_calls_do_not_consume_identity(self):
        call = model_reply()['choices'][0]['message']['tool_calls'][0]
        invalid = [None, [], [call, call], [None], [{**call, 'type': 'custom'}],
                   [{**call, 'function': None}], [{**call, 'id': ''}], [{**call, 'id': '   '}],
                   [{**call, 'id': None}], [{**call, 'id': 42}],
                   [{**call, 'function': {'name': '', 'arguments': '{}'}}],
                   [{**call, 'function': {'name': 'hold', 'arguments': {}}}]]
        for calls in invalid:
            with self.subTest(calls=calls):
                raw = model_reply()
                raw['choices'][0]['message']['tool_calls'] = calls
                agent = self.make_agent(lambda request: httpx.Response(200, json=raw))
                with self.assertRaises(ValueError):
                    agent.get_action(fake_observation())
                self.assertIsNone(agent.pending)
                self.assertFalse(agent.used_ids)
                self.assertEqual(agent.total_tokens, 10)

    def test_json_arguments_require_objects_finite_values_and_preserve_unicode(self):
        for arguments in ('[]', 'null', '42', '{', '{"pose":[NaN]}', '{"gripper":Infinity}',
                          '{"gripper":-Infinity}'):
            with self.subTest(arguments=arguments):
                raw = model_reply()
                raw['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = arguments
                agent = self.make_agent(lambda request: httpx.Response(200, json=raw))
                with self.assertRaises(ValueError):
                    agent.get_action(fake_observation())
                self.assertFalse(agent.used_ids)
                self.assertIsNone(agent.pending)
        raw = model_reply()
        raw['choices'][0]['message']['tool_calls'][0]['function']['arguments'] = '{"summary":" café"}'
        agent = self.make_agent(lambda request: httpx.Response(200, json=raw))
        self.assertEqual(agent.get_action(fake_observation()).arguments, {'summary': ' café'})


if __name__ == '__main__':
    unittest.main()