"""CPU tests for Jev context compaction in the direct agent. No simulator, no network."""
import json
import os
import unittest
from unittest import mock

import httpx

from cleanrua.experimental import direct_jev_robodojo as direct_jev
from test_direct_robodojo import FakeRoboDojo


class CompactionTest(unittest.TestCase):
    def setUp(self):
        self.chat, self.jev = [], []

        def chat(request):
            self.chat.append(json.loads(request.content))
            n = len(self.chat)
            arguments = json.dumps({'steps': 1, 'note': f'{n} ' + 'x' * 400})
            return httpx.Response(200, json={'usage': {'total_tokens': 10}, 'choices': [{'message': {
                'role': 'assistant', 'content': None, 'tool_calls': [{'id': f'call-{n}', 'type': 'function',
                    'function': {'name': 'hold', 'arguments': arguments}}]}}]})

        def jev(request):
            body = json.loads(request.content)
            self.jev.append(body)
            # t0: drop the pair; t1: keep the program, drop the result; everything else: keep.
            values = {'result_t0': .1, 'call_t0': .1, 'result_t1': .2, 'call_t1': .9}
            return httpx.Response(200, json={'answers': {name: {'noul': values.get(name, .8)}
                                                         for name in body['questions']},
                                             'usage': {'input_tokens': 123}})

        variables = {'CLEANRUA_API_KEY': 'k', 'CLEANRUA_BASE_URL': 'https://model.invalid/v1',
                     'CLEANRUA_MODEL_NAME': 'm', 'TYPESAFE_API_KEY': 'j'}
        args = direct_jev.Args(jev_budget_tokens=900, jev_keep_recent=2, jev_truncate_chars=20)
        with mock.patch.dict(os.environ, variables):
            self.agent = direct_jev.RUAgent(args, tools=[])
        self.agent.client = httpx.Client(base_url=self.agent.client.base_url, transport=httpx.MockTransport(chat))
        self.agent.jev = httpx.Client(transport=httpx.MockTransport(jev))
        self.env = direct_jev.RoboDojoWrapper(FakeRoboDojo())

    def play(self, turns):
        obs, _ = self.env.reset()
        self.agent.reset('Task: stack the bowls.', step_limit=800)
        for _ in range(turns):
            action = self.agent.get_action(obs)
            obs, *_ = self.env.step(action)
            obs = {**obs, 'feedback': json.dumps({'ok': True, 'output': 'y' * 300})}

    def test_compaction_drops_and_truncates_older_pairs_only(self):
        self.play(5)  # Four past pairs (~240 tokens each) first exceed the 900-token budget on the fifth request.
        self.assertEqual(len(self.jev), 1)
        request = self.jev[0]
        self.assertEqual((request['model'], request['state']['goal']), ('jev-latest', 'Task: stack the bowls.'))
        self.assertEqual([h['id'] for h in request['state']['history']], ['t0', 't1', 't2', 't3'])
        self.assertEqual(set(request['questions']), {'result_t0', 'call_t0', 'result_t1', 'call_t1'})
        messages = self.chat[-1]['messages']
        ids = [m['tool_call_id'] for m in messages if m['role'] == 'tool']
        self.assertEqual(ids, ['call-2', 'call-3', 'call-4'])  # call-1 dropped, the newest two pinned.
        self.assertTrue(messages[3]['content'].endswith('chars dropped by compaction]'))
        self.assertEqual(self.agent.compaction, {'pairs': 4, 'dropped': 1, 'truncated': 1, 'requests': 1,
                                                 'seconds': self.agent.compaction['seconds']})
        self.assertEqual({k: self.agent.usage[k] for k in ('compactions', 'jev_requests', 'jev_input_tokens',
                                                            'history_dropped', 'history_truncated')},
                         {'compactions': 1, 'jev_requests': 1, 'jev_input_tokens': 123, 'history_dropped': 1,
                          'history_truncated': 1})

    def test_a_failed_jev_request_keeps_the_history(self):
        self.agent.jev = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(429, text='slow')))
        self.play(5)
        self.assertEqual(len([m for m in self.chat[-1]['messages'] if m['role'] == 'tool']), 4)
        self.assertEqual(self.agent.usage['jev_errors'], 1)
        self.assertTrue(self.agent.compaction['error'].startswith('HTTPStatusError'))

    def test_default_keeps_the_whole_history(self):
        self.assertIsNone(direct_jev.Args().history_size)


if __name__ == '__main__':
    unittest.main()
