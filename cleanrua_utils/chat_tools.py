"""Chat Completions tool selection shared by the RoboDojo and real-world agents."""
import time


class NoToolCall(RuntimeError):
    """All tool selection attempts returned text only."""


def request_tool_call(post, body, *, on_reply=None):
    """Ask again after a text-only answer, keeping the first tool call for execution."""
    for _ in range(3):
        start = time.monotonic()
        response = post(json=body)
        response.raise_for_status()
        reply = response.json()
        if on_reply is not None:
            on_reply(reply, time.monotonic() - start)
        message = reply['choices'][0]['message']
        assistant = {'role': 'assistant', 'content': message.get('content')}
        if message.get('reasoning_content'):
            assistant['reasoning_content'] = message['reasoning_content']
        if message.get('tool_calls'):
            return {**assistant, 'tool_calls': message['tool_calls'][:1]}
        assistant['content'] = assistant['content'] or ''
        body = {**body, 'messages': [*body['messages'], assistant,
                                     {'role': 'user', 'content': 'Reply with exactly one tool call.'}]}
    raise NoToolCall(f'The model replied without a tool call: {message.get("content")!r}')
