"""Direct, as a prototype: the MLLM moves the robot one motion at a time.

Each turn the MLLM sees the task, its recent tool calls with their results, and the current images and state. It
replies with one tool call: `move` to an end-effector pose, or `finish`. Inverse kinematics turns the pose into a
joint-space path, which the robot follows one control step at a time.

This shows the algorithm, not a complete agent. `robot` stands for an environment wrapper with
    robot.observe() -> {'images': {camera: (H, W, 3) uint8 array}, 'state': {name: number or list}}
    robot.ik(pose, gripper) -> list of low-level actions reaching an end-effector pose; empty if unreachable
    robot.step(action) -> True once the episode has ended
See direct_robodojo.py for a complete agent.
"""
from functools import partial
import base64
import io
import json
import os

import httpx
from PIL import Image

from cleanrua_utils.chat_tools import request_tool_call

SYSTEM = ('You control a robot arm. Reply with exactly one tool call each turn. Poses are [x, y, z, roll, pitch, '
          'yaw] of the end effector, in metres and radians. Use the images and state, and check each motion before '
          'the next.')
TOOLS = [
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


def mllm(messages: list[dict], *, tools=TOOLS, model=None, reasoning_effort=None, tool_choice='required') -> dict:
    """Select one tool through an OpenAI-compatible vision endpoint."""
    post = partial(httpx.post, os.environ['CLEANRUA_BASE_URL'].rstrip('/') + '/chat/completions', timeout=300.,
                   headers={'Authorization': f'Bearer {os.environ["CLEANRUA_API_KEY"]}'})
    body = {'model': model or os.environ['CLEANRUA_MODEL_NAME'], 'messages': messages, 'tools': tools,
            'tool_choice': tool_choice, 'parallel_tool_calls': False, 'stream': False}
    if reasoning_effort:
        body['reasoning_effort'] = reasoning_effort
    return request_tool_call(post, body)


def look(observation: dict) -> dict:
    """The current state and images as one user message."""
    content = [{'type': 'text', 'text': 'State: ' + json.dumps(observation['state'])}]
    for camera, image in observation['images'].items():
        png = io.BytesIO()
        Image.fromarray(image).save(png, format='PNG')
        content += [{'type': 'text', 'text': f'Camera {camera}:'}, {'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(png.getvalue()).decode()}}]
    return {'role': 'user', 'content': content}


def move_path(robot, arguments):
    if 'targets' in arguments:
        return robot.ik(**arguments)
    return robot.ik(arguments['pose'], arguments['gripper'])


def direct(robot, instruction: str, max_decisions: int = 50, history_size: int = 4,
           history: list | None = None, *, decide=None) -> str:
    """Run one episode and return why it stopped; pass `history` to keep its tool calls and results."""
    # Past tool calls and their results, as text: only the current observation carries images.
    decide = mllm if decide is None else decide
    history = [] if history is None else history
    for _ in range(max_decisions):
        message = decide([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': f'Task: {instruction}'},
                        *history[-2 * history_size:], look(robot.observe())])
        call = message['tool_calls'][0]
        if call['function']['name'] == 'finish':
            return 'finish'
        arguments = json.loads(call['function']['arguments'])
        path = move_path(robot, arguments)  # The executor is inverse kinematics.
        for action in path:
            if robot.step(action):
                return 'episode_ended'
        result = {'steps': len(path)} if path else {'error': 'No inverse-kinematics solution'}
        history += [message, {'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result)}]
    return 'max_decisions'
