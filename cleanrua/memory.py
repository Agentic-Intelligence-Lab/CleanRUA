"""Memory, as a prototype: an agent carries lessons from one episode of a task into the next.

Memory is not a paradigm of its own: it wraps any of them (`direct`, `cap`, `delegate`, `monitor`). Each task keeps
a short text of lessons, its memory. An episode starts with the memory added to the task instruction; afterwards the
MLLM rewrites the memory from the episode's tool calls and results, its final images and the official outcome, as in
Reflexion. Learning and testing use different seeds: learning episodes rewrite the memory, and testing reads a frozen
copy with `learn=False`, so the test seeds never shape what the agent remembers.

A lesson from `delegate_memory`: a memory written as a recipe makes the agent repeat it step by step, and with a
stochastic VLA it then loses the variety that won other episodes. The reflection below therefore keeps what success
looks like, the mistakes to avoid and how much evidence backs each lesson, rather than a fixed sequence of steps.

This shows the algorithm, not a complete agent. `robot` stands for the paradigm's environment wrapper, with one
more method:
    robot.success() -> True if the official check counts the task as done
See experimental/delegate_memory_dexjoco.py for a complete agent, and cap_explore_dexjoco.py for an agent that
remembers a verified program instead of text.
"""
import json
import os
from pathlib import Path

import httpx

from cleanrua.direct import look

REFLECTION = ('You keep the memory of a robot agent for one task: short markdown bullets of lessons, read at the start '
              'of every episode of the task. Rewrite the memory from the episode below. Keep what success looks like '
              'and the mistakes to avoid, not a fixed sequence of steps; give each lesson its evidence, such as "(2 '
              'successes, 1 failure)", and drop lessons the outcomes contradict. Do not record coordinates: every '
              'episode starts from a different scene. Reply with the memory only, at most {limit} characters.')


def chat(messages: list[dict]) -> str:
    """The text reply of an OpenAI-compatible vision model."""
    response = httpx.post(os.environ['CLEANRUA_BASE_URL'].rstrip('/') + '/chat/completions', timeout=300.,
                          headers={'Authorization': f'Bearer {os.environ["CLEANRUA_API_KEY"]}'},
                          json={'model': os.environ['CLEANRUA_MODEL_NAME'], 'messages': messages, 'stream': False})
    response.raise_for_status()
    return (response.json()['choices'][0]['message'].get('content') or '').strip()


def reflect(memory: str, instruction: str, history: list[dict], outcome: dict, observation: dict,
            limit: int = 3000) -> str:
    """The memory rewritten from one finished episode."""
    steps = [json.dumps(m['tool_calls'][0]['function']) if m['role'] == 'assistant' else 'Result: ' + m['content']
             for m in history]
    episode = {'role': 'user', 'content': f'Task: {instruction}\n\nMemory so far:\n{memory or "(empty)"}\n\nTool calls '
               'and results of this episode:\n' + '\n'.join(steps) + f'\n\nOfficial outcome: {json.dumps(outcome)}\n'
               'The final images follow.'}
    return chat([{'role': 'system', 'content': REFLECTION.format(limit=limit)}, episode, look(observation)])[:limit]


def memory(paradigm, robot, instruction: str, path: Path, learn: bool = True, **kwargs) -> str:
    """Run one episode of `paradigm` with the task's memory in `path`, rewrite it if learning, and return why the
    episode stopped. Other arguments, such as a VLA `policy`, go to the paradigm."""
    lessons = path.read_text() if path.exists() else ''
    history = []
    reason = paradigm(robot, instruction=instruction + (f'\n\nLessons from earlier episodes of this task:\n{lessons}'
                                                        if lessons else ''), history=history, **kwargs)
    if learn:  # The official outcome is the learning signal; the agent's own `finish` can be wrong.
        outcome = {'success': robot.success(), 'stop_reason': reason, 'decisions': len(history) // 2}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(reflect(lessons, instruction, history, outcome, robot.observe()))
    return reason
