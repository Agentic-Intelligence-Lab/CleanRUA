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
import base64
import io
import json
import os

import httpx
from PIL import Image

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


def direct(robot, instruction: str, max_decisions: int = 50, history_size: int = 4) -> str:
    """Run one episode and return why it stopped."""
    history = []  # Past tool calls and their results, as text: only the current observation carries images.
    for _ in range(max_decisions):
        message = mllm([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': f'Task: {instruction}'},
                        *history[-2 * history_size:], look(robot.observe())])
        call = message['tool_calls'][0]
        if call['function']['name'] == 'finish':
            return 'finish'
        arguments = json.loads(call['function']['arguments'])
        path = robot.ik(arguments['pose'], arguments['gripper'])  # The executor is inverse kinematics.
        for action in path:
            if robot.step(action):
                return 'episode_ended'
        result = {'steps': len(path)} if path else {'error': 'No inverse-kinematics solution'}
        history += [message, {'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result)}]
    return 'max_decisions'
