"""CPU tests for the DexJoCo delegate_memory agent: the reflection that rewrites a task's playbook."""
import json
import os
import unittest
from unittest import mock

import httpx
import numpy as np

from cleanrua import delegate_memory_dexjoco as memory


class ReflectTest(unittest.TestCase):
    def test_reflection_sees_episode_outcome_and_images_and_counts_its_usage(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                'usage': {'total_tokens': 30, 'prompt_tokens': 25, 'completion_tokens': 5},
                'choices': [{'message': {'role': 'assistant', 'content': '  - Lift the bucket higher.  ' + 'x' * 50}}]})

        variables = {'CLEANRUA_API_KEY': 'test-key', 'CLEANRUA_BASE_URL': 'https://model.invalid/v1',
                     'CLEANRUA_MODEL_NAME': 'test-model'}
        with mock.patch.dict(os.environ, variables):
            agent = memory.RUAgent(memory.Args(playbook_chars=40), tools=[])
        agent.client = httpx.Client(base_url=agent.client.base_url, transport=httpx.MockTransport(handler))
        agent.reset('Task: pick the bucket.', step_limit=1200)
        observation = {'cameras': {'front': {'rgb': np.zeros((8, 8, 3), np.uint8)}}}
        playbook = agent.reflect('Pick the bucket.', '- Old lesson.', '1. actor_act(...) -> {}\nOfficial outcome: failure.',
                                 observation)

        self.assertEqual(playbook, ('- Lift the bucket higher.  ' + 'x' * 50)[:40])  # Stripped, then cut to playbook_chars.
        request = requests[0]
        self.assertNotIn('tools', request)  # A plain text reply, not a tool call.
        self.assertIn('at most 40 characters', request['messages'][0]['content'])
        text = request['messages'][1]['content'][0]['text']
        for part in ('Pick the bucket.', '- Old lesson.', 'Official outcome: failure.'):
            self.assertIn(part, text)
        self.assertEqual([p['type'] for p in request['messages'][1]['content']], ['text', 'text', 'image_url'])
        self.assertEqual((agent.usage['total_tokens'], agent.usage['reflection_requests']), (30, 1))


if __name__ == '__main__':
    unittest.main()
