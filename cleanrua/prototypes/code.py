"""Code, as a prototype: the MLLM writes Python programs that move the robot.

Each turn the MLLM sends one program through `run_code`. The program calls a small robot API, can compute targets
with NumPy, and prints what the model wants to read; its variables persist into later programs. When it ends, the
model sees the printed output with fresh images and state. A program can do many motions, so one decision can cover
a whole stage of the task.

This shows the algorithm, not a complete agent. `robot` stands for an environment wrapper with
    robot.observe() -> {'images': {camera: (H, W, 3) uint8 array}, 'state': {name: number or list}}
    robot.ik(pose, gripper) -> list of low-level actions reaching an end-effector pose; empty if unreachable
    robot.step(action) -> True once the episode has ended
Programs run in this process without a sandbox. See code_robodojo.py for a complete agent.
"""
import base64
import contextlib
import io
import json
import os

import httpx
import numpy as np
from PIL import Image

SYSTEM = ('You control a robot arm by writing Python programs for `run_code`. Reply with exactly one tool call each '
          'turn. A program can use `np`, `robot.state()` and `robot.move(pose, gripper)`, where poses are '
          '[x, y, z, roll, pitch, yaw] of the end effector, in metres and radians, and the gripper opening runs from 0 '
          '(closed) to 1 (open). Keep each program to one stage you can verify, and print the values you need.')
TOOLS = [
    {'type': 'function', 'function': {
        'name': 'run_code', 'description': 'Run a Python program and return its printed output.',
        'parameters': {'type': 'object', 'required': ['code'], 'properties': {'code': {'type': 'string'}}}}},
    {'type': 'function', 'function': {
        'name': 'finish', 'description': 'End the episode once the task is done.',
        'parameters': {'type': 'object', 'properties': {}}}},
]


class RobotAPI:
    """The `robot` that programs see: the measured state and one motion primitive."""

    def __init__(self, robot):
        self.robot, self.ended = robot, False

    def state(self) -> dict:
        return self.robot.observe()['state']

    def move(self, pose, gripper: float) -> int:
        """Follow the inverse-kinematics path to a pose; returns the control steps taken."""
        path = self.robot.ik(pose, gripper)
        if not path:
            raise ValueError('No inverse-kinematics solution')  # The program may catch this and try something else.
        for action in path:
            if self.robot.step(action):
                self.ended = True
                raise SystemExit  # The episode ended: stop the program.
        return len(path)


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


def code(robot, instruction: str, max_decisions: int = 50, history_size: int = 4) -> str:
    """Run one episode and return why it stopped."""
    history, api = [], RobotAPI(robot)
    namespace = {'np': np, 'robot': api}  # Shared by every program of the episode.
    for _ in range(max_decisions):
        message = mllm([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': f'Task: {instruction}'},
                        *history[-2 * history_size:], look(robot.observe())])
        call = message['tool_calls'][0]
        if call['function']['name'] == 'finish':
            return 'finish'
        output, result = io.StringIO(), {}
        try:
            with contextlib.redirect_stdout(output):
                exec(json.loads(call['function']['arguments'])['code'], namespace)  # The executor is Python.
        except SystemExit:
            pass
        except Exception as exc:  # An error stops the program; motions already made remain.
            result['error'] = f'{type(exc).__name__}: {exc}'
        if api.ended:
            return 'episode_ended'
        result['output'] = output.getvalue()[-4000:]
        history += [message, {'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(result)}]
    return 'max_decisions'
