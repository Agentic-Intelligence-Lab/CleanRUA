"""Shared message construction for the simulation prototypes and real workstations."""
import base64
import io
import json

from PIL import Image


def look(observation: dict) -> dict:
    """The current state and images as one user message."""
    content = [{'type': 'text', 'text': 'State: ' + json.dumps(observation['state'])}]
    for camera, image in observation['images'].items():
        png = io.BytesIO()
        Image.fromarray(image).save(png, format='PNG')
        content += [{'type': 'text', 'text': f'Camera {camera}:'}, {'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(png.getvalue()).decode()}}]
    return {'role': 'user', 'content': content}


def request_tool(decide, system, task, history, history_size, observation):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task},
                *history[-2 * history_size:], look(observation)]
    message = decide(messages)
    return message, message["tool_calls"][0]
