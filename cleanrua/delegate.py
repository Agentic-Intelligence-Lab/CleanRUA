"""Delegate, as a prototype: the MLLM gives a VLA short instructions, and can also move the robot itself.

Each turn the MLLM sees the task, its recent tool calls with their results, and the current images and state. It
splits the task into sub-tasks and hands each to a vision-language-action model (VLA) with `vla`, for example "Pick
up the left bowl". The VLA predicts a chunk of actions from a fresh observation and the instruction, and the robot
runs it; a new chunk is predicted when the last one is used up. For a segment whose target is clear, the MLLM can
`move` the robot itself.

This shows the algorithm, not a complete agent. `robot` and `policy` stand for an environment wrapper and a VLA with
    robot.observe() -> {'images': {camera: (H, W, 3) uint8 array}, 'state': {name: number or list}}
    robot.ik(pose, gripper) -> list of low-level actions reaching an end-effector pose; empty if unreachable
    robot.step(action) -> True once the episode has ended
    policy(observation, instruction) -> list of low-level actions, a chunk predicted by a VLA
See delegate_robodojo.py for a complete agent.
"""
import base64
import io
import json
import os

import httpx
from PIL import Image

from cleanrua.direct import move_path

ACTIONS_PER_CHUNK = 25  # Actions run from each predicted chunk before the VLA sees a fresh observation.
SYSTEM = ('You control a robot arm through a vision-language-action model (VLA). Reply with exactly one tool call each '
          'turn. Give the VLA one short instruction per phase, such as grasping or placing an object, and check the '
          'result in the images before the next phase. Poses for `move` are [x, y, z, roll, pitch, yaw] of the end '
          'effector, in metres and radians.')
TOOLS = [
    {'type': 'function', 'function': {
        'name': 'vla', 'description': 'Run the VLA with a short instruction for a number of control steps.',
        'parameters': {'type': 'object', 'required': ['instruction', 'steps'], 'properties': {
            'instruction': {'type': 'string'}, 'steps': {'type': 'integer', 'minimum': 1, 'maximum': 300}}}}},
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


def look(observation: dict) -> dict:
    """The current state and images as one user message."""
    content = [{'type': 'text', 'text': 'State: ' + json.dumps(observation['state'])}]
    for camera, image in observation['images'].items():
        png = io.BytesIO()
        Image.fromarray(image).save(png, format='PNG')
        content += [{'type': 'text', 'text': f'Camera {camera}:'}, {'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(png.getvalue()).decode()}}]
    return {'role': 'user', 'content': content}


def delegate(robot, policy, instruction: str, max_decisions: int = 50, history_size: int = 4,
             history: list | None = None, *, decide=None) -> str:
    """Run one episode and return why it stopped; pass `history` to keep its tool calls and results."""
    decide = mllm if decide is None else decide
    history = [] if history is None else history
    for _ in range(max_decisions):
        message = decide([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': f'Task: {instruction}'},
                        *history[-2 * history_size:], look(robot.observe())])
        call = message['tool_calls'][0]
        name, arguments = call['function']['name'], json.loads(call['function']['arguments'])
        if name == 'finish':
            return 'finish'
        if name == 'vla':  # The executor is the VLA, following the MLLM's instruction.
            chunk = []
            for _ in range(arguments['steps']):
                if not chunk:
                    chunk = list(policy(robot.observe(), arguments['instruction']))[:ACTIONS_PER_CHUNK]
                if robot.step(chunk.pop(0)):
                    return 'episode_ended'
            result = {'steps': arguments['steps']}
        else:  # move: inverse kinematics, as in Direct.
            path = move_path(robot, arguments)
            for action in path:
                if robot.step(action):
                    return 'episode_ended'
            result = {'steps': len(path)} if path else {'error': 'No inverse-kinematics solution'}
        history += [message, {'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result)}]
    return 'max_decisions'
