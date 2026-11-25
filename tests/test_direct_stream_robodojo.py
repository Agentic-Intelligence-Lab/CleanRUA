"""CPU tests for the streaming Direct agent: replies assembled from server-sent events. No simulator, no network."""
import json
import os
import unittest
from unittest import mock

import httpx

from cleanrua.experimental import direct_stream_robodojo as stream
from test_direct_robodojo import FakeRoboDojo


def events(*chunks) -> bytes:
    """A streamed reply: one `data:` line per chunk, then `data: [DONE]`."""
    return b''.join(f'data: {json.dumps(chunk)}\n\n'.encode() for chunk in chunks) + b'data: [DONE]\n\n'


def delta(**fields) -> dict:
    return {'choices': [{'index': 0, 'delta': fields}]}


class StreamTest(unittest.TestCase):
    def agent(self, *replies):
        self.requests = []

        def handler(request):
            self.requests.append(json.loads(request.content))
            status, content = replies[len(self.requests) - 1]
            return httpx.Response(status, content=content, headers={'content-type': 'text/event-stream'})

        variables = {'CLEANRUA_API_KEY': 'test-key', 'CLEANRUA_BASE_URL': 'https://model.invalid/v1',
                     'CLEANRUA_MODEL_NAME': 'test-model'}
        with mock.patch.dict(os.environ, variables):
            agent = stream.RUAgent(stream.Args(tool_choice='auto'), tools=[])
        agent.client = httpx.Client(base_url=agent.client.base_url, transport=httpx.MockTransport(handler))
        agent.reset('Task: stack the bowls.', step_limit=800)
        return agent

    def test_a_tool_call_split_over_chunks_is_reassembled_with_its_usage(self):
        usage = {'total_tokens': 30, 'prompt_tokens': 10, 'completion_tokens': 20,
                 'completion_tokens_details': {'reasoning_tokens': 15}, 'prompt_tokens_details': {'cached_tokens': 4}}
        agent = self.agent((200, events(
            delta(role='assistant'), delta(reasoning_content='The bowl is '), delta(reasoning_content='to the left.'),
            delta(tool_calls=[{'index': 0, 'id': 'call-1', 'type': 'function',
                               'function': {'name': 'move', 'arguments': '{"targets": '}}]),
            delta(tool_calls=[{'index': 0, 'function': {'arguments': '[]}'}}]),
            {'choices': [], 'usage': usage})))
        obs, _ = stream.RoboDojoWrapper(FakeRoboDojo()).reset()
        action = agent.get_action(obs)

        self.assertEqual(action, {'name': 'move', 'arguments': '{"targets": []}'})
        request = self.requests[0]
        self.assertEqual((request['stream'], request['stream_options']), (True, {'include_usage': True}))
        self.assertEqual(agent.pending, {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call-1', 'type': 'function', 'function': {'name': 'move', 'arguments': '{"targets": []}'}}]})
        self.assertEqual({key: agent.last_usage[key] for key in ('total_tokens', 'reasoning_tokens', 'cached_tokens')},
                         {'total_tokens': 30, 'reasoning_tokens': 15, 'cached_tokens': 4})
        self.assertTrue(0 <= agent.last_usage['wait_s'] and 0 <= agent.last_usage['reasoning_s'])

    def test_a_text_reply_is_asked_again_and_an_error_keeps_its_text(self):
        call = [{'index': 0, 'id': 'call-1', 'function': {'name': 'capture_scene', 'arguments': '{}'}}]
        agent = self.agent((200, events(delta(content='Looking.'), {'choices': [], 'usage': {'total_tokens': 5}})),
                           (200, events(delta(tool_calls=call), {'choices': [], 'usage': {'total_tokens': 7}})))
        obs, _ = stream.RoboDojoWrapper(FakeRoboDojo()).reset()
        self.assertEqual(agent.get_action(obs), {'name': 'capture_scene', 'arguments': '{}'})
        self.assertEqual(self.requests[1]['messages'][-2:],
                         [{'role': 'assistant', 'content': 'Looking.'},
                          {'role': 'user', 'content': 'Reply with exactly one tool call.'}])
        self.assertEqual((agent.last_usage['total_tokens'], agent.usage['requests']), (12, 2))

        agent = self.agent((400, b'{"error": {"message": "bad tool_choice"}}'))
        with self.assertRaises(httpx.HTTPStatusError) as error:
            agent.get_action(obs)
        self.assertIn('bad tool_choice', error.exception.response.text)


if __name__ == '__main__':
    unittest.main()
