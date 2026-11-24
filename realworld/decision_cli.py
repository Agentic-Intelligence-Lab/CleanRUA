"""Decision backends for CleanRUA agents. Each invocation proposes one wrapper tool call."""
import base64
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import tempfile
from uuid import uuid4


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
