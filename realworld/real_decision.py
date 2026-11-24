"""Model selection and compatible HTTP transport for real-workstation runs."""
import os

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
