"""Monitor, as a prototype: a VLA does the task, and the MLLM supervises it.

The vision-language-action model (VLA) always gets the task's own instruction. Each turn the MLLM sees the task, its
recent tool calls with their results, and the current images and state. It lets the VLA continue with `run_policy`,
or, after a failure the VLA does not recover from, corrects the robot with `move` and then hands control back. The
MLLM never rewrites the instruction; that is the difference from Delegate.

This shows the algorithm, not a complete agent. `robot` and `policy` stand for an environment wrapper and a VLA with
    robot.observe() -> {'images': {camera: (H, W, 3) uint8 array}, 'state': {name: number or list}}
    robot.ik(pose, gripper) -> list of low-level actions reaching an end-effector pose; empty if unreachable
    robot.step(action) -> True once the episode has ended
    policy(observation, instruction) -> list of low-level actions, a chunk predicted by a VLA
See monitor_robodojo.py for a complete agent.
"""
import json
import os

import httpx
from cleanrua_utils.agent import look, request_tool

ACTIONS_PER_CHUNK = 25  # Actions run from each predicted chunk before the VLA sees a fresh observation.
SYSTEM = ('A vision-language-action model (VLA) does the task on a robot arm; you supervise it. Reply with exactly one '
          'tool call each turn. Let it continue with `run_policy` and check its progress in the images. A slow or '
          'imperfect stretch is not a failure. Only for a failure it does not recover from, correct the robot with a '
          'short `move`, then hand control back. Poses are [x, y, z, roll, pitch, yaw] of the end effector, in metres '
          'and radians.')
TOOLS = [
    {'type': 'function', 'function': {
        'name': 'run_policy', 'description': 'Let the VLA continue the task for a number of control steps.',
        'parameters': {'type': 'object', 'required': ['steps'], 'properties': {
            'steps': {'type': 'integer', 'minimum': 1, 'maximum': 300}}}}},
    {'type': 'function', 'function': {
        'name': 'move', 'description': 'Move the end effector to an absolute pose, with a gripper opening from 0 '
                                       '(closed) to 1 (open).',
        'parameters': {'type': 'object', 'required': ['pose', 'gripper'], 'properties': {
            'pose': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 6, 'maxItems': 6},
            'gripper': {'type': 'number', 'minimum': 0, 'maximum': 1}}}}},
    {'type': 'function', 'function': {
        'name': 'finish', 'description': 'End the episode once the task is done.',
        'parameters': {'type': 'object', 'properties': {}}}},
]


def mllm(messages: list[dict]) -> dict:
    """The reply of an OpenAI-compatible vision model, keeping its first tool call."""
    response = httpx.post(os.environ['CLEANRUA_BASE_URL'].rstrip('/') + '/chat/completions', timeout=300.,
                          headers={'Authorization': f'Bearer {os.environ["CLEANRUA_API_KEY"]}'},
                          json={'model': os.environ['CLEANRUA_MODEL_NAME'], 'messages': messages, 'tools': TOOLS,
                                'tool_choice': 'required', 'stream': False})
    response.raise_for_status()
    message = response.json()['choices'][0]['message']
    return {'role': 'assistant', 'content': message.get('content'), 'tool_calls': message['tool_calls'][:1]}


def monitor(robot, policy, instruction: str, max_decisions: int = 50, history_size: int = 4) -> str:
    """Run one episode and return why it stopped."""
    history = []
    for _ in range(max_decisions):
        message, call = request_tool(mllm, SYSTEM, f'Task: {instruction}', history, history_size, robot.observe())
        name, arguments = call['function']['name'], json.loads(call['function']['arguments'])
        if name == 'finish':
            return 'finish'
        if name == 'run_policy':  # The executor is the VLA, always with the task's own instruction.
            chunk = []
            for _ in range(arguments['steps']):
                if not chunk:
                    chunk = list(policy(robot.observe(), instruction))[:ACTIONS_PER_CHUNK]
                if robot.step(chunk.pop(0)):
                    return 'episode_ended'
            result = {'steps': arguments['steps']}
        else:  # move: a correction by inverse kinematics, as in Direct.
            path = robot.ik(arguments['pose'], arguments['gripper'])
            for action in path:
                if robot.step(action):
                    return 'episode_ended'
            result = {'steps': len(path)} if path else {'error': 'No inverse-kinematics solution'}
        history += [message, {'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result)}]
    return 'max_decisions'
