"""Real-workstation model selection, HTTP transport and local CLI backends."""
import base64
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import tempfile
from uuid import uuid4

import httpx


MODEL_KEYS = {"api": "CLEANRUA_MODEL_NAME", "local": "CLEANRUA_LOCAL_MODEL", "codex": "CLEANRUA_CODEX_MODEL"}


# Only use a provider's ambient key for that provider's HTTPS origin.
PROVIDER_KEYS = {
    "api.openai.com": "OPENAI_API_KEY",
    "dashscope.aliyuncs.com": "DASHSCOPE_API_KEY",
    "dashscope-intl.aliyuncs.com": "DASHSCOPE_API_KEY",
    "dashscope-us.aliyuncs.com": "DASHSCOPE_API_KEY",
    "openrouter.ai": "OPENROUTER_API_KEY",
}


def api_credentials():
    url = os.getenv("CLEANRUA_BASE_URL")
    if not url:
        raise ValueError("Set CLEANRUA_BASE_URL or pass --base-url for your provider")
    key = os.getenv("CLEANRUA_API_KEY")
    if key:
        return url, key
    endpoint = httpx.URL(url)
    source = PROVIDER_KEYS.get(endpoint.host) if endpoint.scheme == "https" and endpoint.port in (None, 443) else None
    if source:
        key = os.getenv(source)
        if key:
            return url, key
        raise ValueError(f"Set {source} or CLEANRUA_API_KEY for this API endpoint")
    raise ValueError("Set CLEANRUA_API_KEY for this API endpoint")


def decision_settings():
    backend = os.getenv("CLEANRUA_DECISION_BACKEND") or "api"
    return {"decision_backend": backend, "model": os.getenv(MODEL_KEYS.get(backend, "")),
            "reasoning_effort": os.getenv("CLEANRUA_REASONING_EFFORT", "default")}


def configure_decision(backend, model, reasoning_effort, base_url):
    backend = backend or decision_settings()["decision_backend"]
    if backend not in MODEL_KEYS and ":" not in backend:
        raise ValueError("Choose api, local, codex or module:factory")
    if model and backend not in MODEL_KEYS:
        raise ValueError("Configure custom models through their backend factory")
    if base_url and backend not in ("api", "local"):
        raise ValueError("--base-url applies to api and local backends")
    if reasoning_effort and backend not in MODEL_KEYS:
        raise ValueError("Configure reasoning effort through the custom backend factory")
    os.environ["CLEANRUA_DECISION_BACKEND"] = backend
    if model:
        os.environ[MODEL_KEYS[backend]] = model
    if reasoning_effort:
        os.environ["CLEANRUA_REASONING_EFFORT"] = reasoning_effort
    if base_url:
        os.environ["CLEANRUA_LOCAL_BASE_URL" if backend == "local" else "CLEANRUA_BASE_URL"] = base_url


def responses_decision(messages, tools, settings, key):
    items = []
    for msg in messages:
        if msg["role"] == "assistant":
            items.extend(msg["_response_items"])
        elif msg["role"] == "tool":
            items.append({"type": "function_call_output", "call_id": msg["tool_call_id"], "output": msg["content"]})
        elif isinstance(msg["content"], str):
            items.append(msg)
        else:
            content = []
            for part in msg["content"]:
                if part["type"] == "text":
                    content.append({"type": "input_text", "text": part["text"]})
                elif part["type"] == "image_url":
                    content.append({"type": "input_image", "image_url": part["image_url"]["url"]})
                else:
                    raise ValueError(f"Unsupported input part: {part['type']}")
            items.append({"role": msg["role"], "content": content})
    body = {"model": settings["model"], "input": items, "store": False, "tool_choice": "required",
            "parallel_tool_calls": False, "tools": [{"type": "function", **t["function"], "strict": False}
                                                    for t in tools]}
    if settings["reasoning_effort"] != "default":
        body["reasoning"] = {"effort": settings["reasoning_effort"]}
    response = httpx.post("https://api.openai.com/v1/responses", timeout=300,
                          headers={"Authorization": f"Bearer {key}"}, json=body)
    response.raise_for_status()
    reply = response.json()
    calls = [item for item in reply["output"] if item["type"] == "function_call"]
    if reply["status"] != "completed" or len(calls) != 1:
        raise ValueError("Expected one completed Responses function call")
    call = calls[0]
    return {"role": "assistant", "content": None, "_response_items": reply["output"],
            "tool_calls": [{"type": "function", "id": call["call_id"],
                            "function": {"name": call["name"], "arguments": call["arguments"]}}]}


def mllm(messages, tools):
    settings = decision_settings()
    if settings["decision_backend"] == "local":
        url = os.getenv("CLEANRUA_LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")
        key = os.getenv("CLEANRUA_LOCAL_API_KEY")
    else:
        url, key = api_credentials()
    if not settings["model"]:
        raise ValueError("Configure the served model ID or pass --model")
    if url.rstrip("/") == "https://api.openai.com/v1" and settings["model"] == "gpt-6.2-astra":
        return responses_decision(messages, tools, settings, key)
    body = {"model": settings["model"], "messages": messages, "tools": tools,
            "tool_choice": "required", "stream": False}
    if settings["reasoning_effort"] != "default":
        body["reasoning_effort"] = settings["reasoning_effort"]
    response = httpx.post(url.rstrip("/") + "/chat/completions", timeout=300,
                          headers={"Authorization": f"Bearer {key}"} if key else {}, json=body)
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    return {**message, "tool_calls": message["tool_calls"][:1]}


def configured_decider(tools):
    name = os.getenv("CLEANRUA_DECISION_BACKEND", "api")
    if name in ("api", "local"):
        return None
    if name == "codex":
        return CodexDecision(tools)
    module, sep, factory = name.partition(":")
    if not sep:
        raise ValueError("Decision backend must be api, local, codex or a trusted module:factory")
    decide = getattr(import_module(module), factory)(tools)
    if not callable(decide):
        raise TypeError("Decision backend factory must return a callable")
    return decide


class CodexDecision:
    def __init__(self, tools):
        self.tools = tools
        self.names = [tool["function"]["name"] for tool in tools]
        self.env = {k: v for k, v in os.environ.items() if k in {
            "PATH", "HOME", "CODEX_HOME", "USER", "TMPDIR", "LANG", "LC_ALL",
            "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY",
            "https_proxy", "http_proxy", "all_proxy", "no_proxy", "SSL_CERT_FILE"}}
        status = subprocess.run(["codex", "login", "status"], capture_output=True, text=True,
                                env=self.env, timeout=15)
        if status.returncode or "Logged in using ChatGPT" not in (status.stdout + status.stderr).splitlines():
            raise RuntimeError("Codex backend requires 'codex login' with a ChatGPT account")

    def __call__(self, messages):
        return self.complete(messages)["choices"][0]["message"]

    def complete(self, messages, timeout=180):
        schema = {"type": "object", "additionalProperties": False, "required": ["name", "arguments"],
                  "properties": {"name": {"type": "string", "enum": self.names},
                                 "arguments": {"type": "string", "description": "Tool arguments as a JSON object string"}}}
        if not self.tools:
            schema = {"type": "object", "additionalProperties": False, "required": ["content"],
                      "properties": {"content": {"type": "string"}}}
        with tempfile.TemporaryDirectory(prefix="cleanrua-codex-") as tmp:
            folder = Path(tmp)
            (folder / "action.schema.json").write_text(json.dumps(schema))
            args = ["codex", "exec", "--ignore-user-config", "--skip-git-repo-check", "--ephemeral",
                    "--sandbox", "read-only", "-c", 'forced_login_method="chatgpt"',
                    "-c", 'model_provider="openai"', "-c", 'web_search="disabled"',
                    "--disable", "shell_tool", "--disable", "apps", "--disable", "plugins",
                    "--json", "--output-schema", "action.schema.json", "--output-last-message", "action.json"]
            model = os.getenv("CLEANRUA_CODEX_MODEL")
            if model:
                args.extend(["--model", model])
            effort = os.getenv("CLEANRUA_REASONING_EFFORT", "default")
            if effort != "default":
                args.extend(["-c", "model_reasoning_effort=" + json.dumps(effort)])
            transcript = []
            index = 0
            for msg in messages:
                if not isinstance(msg.get("content"), list):
                    transcript.append(msg)
                    continue
                parts = []
                for part in msg["content"]:
                    if part["type"] == "text":
                        parts.append(part["text"])
                    elif part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        prefix = "data:image/png;base64,"
                        if not url.startswith(prefix):
                            raise ValueError("Expected a PNG data URL from the CleanRUA observation")
                        index += 1
                        path = folder / f"view-{index}.png"
                        path.write_bytes(base64.b64decode(url[len(prefix):], validate=True))
                        args.extend(["--image", str(path)])
                        parts.append(f"[attached image {index}]")
                    else:
                        raise ValueError(f"Unsupported message part: {part['type']}")
                transcript.append({**msg, "content": parts})
            prompt = ("Select exactly one CleanRUA tool using the attached observations and history. "
                      "Return its name and JSON-encoded arguments. The runner executes the action.\n"
                      + json.dumps({"tools": self.tools, "messages": transcript}, ensure_ascii=False))
            if not self.tools:
                prompt = "Answer the transcript in the content field, preserving the requested formatting.\n" + json.dumps(transcript, ensure_ascii=False)
            result = subprocess.run([*args, "-"], cwd=folder, env=self.env, input=prompt, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
            if result.returncode:
                # CLI stderr can echo the prompt. Surface the error without logging the full observation.
                errors = [line for line in result.stderr.splitlines() if line.startswith("ERROR:")]
                raise RuntimeError(f"Codex exited {result.returncode}. " + " ".join(errors[-1:])[:500])
            action = json.loads((folder / "action.json").read_text())
            if self.tools:
                if action["name"] not in self.names or not isinstance(json.loads(action["arguments"]), dict):
                    raise ValueError("CLI returned an unknown tool or invalid arguments")
                message = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": uuid4().hex, "type": "function", "function": action}]}
            else:
                message = {"role": "assistant", "content": action["content"]}
            usage = {}
            for line in (result.stdout or "").splitlines():
                event = json.loads(line)
                if event.get("type") == "turn.completed" and event.get("usage"):
                    u = event["usage"]
                    usage = {"prompt_tokens": u["input_tokens"], "completion_tokens": u["output_tokens"],
                             "total_tokens": u["input_tokens"] + u["output_tokens"],
                             "prompt_tokens_details": {"cached_tokens": u.get("cached_input_tokens", 0)}}
            return {"choices": [{"message": message}], "usage": usage}
