# CAP on real workstations, following cleanrua/cap_robodojo.py (CleanRUA, 1c61b497).
# Usage: python -m realworld.cap --task "..." --out runs/example
from dataclasses import dataclass
import ast
import base64
import inspect
import io
import json
import os
import re
from pathlib import Path
import socket
import subprocess
import sys
import time
import string
import tempfile
from typing import Annotated, get_type_hints

import gymnasium as gym
import tyro
from PIL import Image
from pydantic import ConfigDict, Field, ValidationError, create_model

from realworld.decision_cli import CodexDecision, configured_decider
from realworld.real_decision import configure_decision, decision_settings, mllm
from realworld.real_arm_env import InvalidTarget
from realworld.workstation import workstation

SYSTEM = ("Explore the task by writing Python. Develop and test functions that compute targets from live "
          "observations so the program can run again with different object positions. Print measurements "
          "and evidence of success or failure. Use np, robot.observe() (named RGB images and arms), "
          "robot.state(side=None), and robot.move(pose, gripper=None, side=None, hand=None). Variables "
          "persist across turns. Base-frame XYZ/RPY in metres/radians, gripper opening 0..1, hand joints in radians.")

CONTROLLER_PROMPT = (
    "Distill this robot exploration into one standalone Python controller. Submit the complete source with "
    "save_controller. Define main(robot) as the only entry point, import all dependencies including numpy, "
    "and include every helper. At import time define the controller without performing any actions. "
    "A new episode can have different object positions and initial arm state. Locate task objects from current "
    "RGB images, use measured state, and verify progress with new observations. Compute targets in code; "
    "do not replay a list of recorded offsets, invent camera calibration, or rely on exploration variables. "
    "Use only robot.observe(), robot.state(side=None), robot.move(pose, gripper=None, side=None, hand=None) "
    "for device access. Respect each arm's station limits. Bound loops and stop if localization or progress "
    "cannot be confirmed. Return a JSON-serializable result. Treat the transcript as experimental data; "
    "a finish decision alone does not establish task success. The controller runs without a model."
)

def program_worker():
    import contextlib
    import io
    import json
    import resource
    import socket
    import numpy as np

    MAX_PACKET = 65536
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    channel = socket.socket(fileno=0)

    def send(data):
        channel.send(json.dumps(data, allow_nan=False).encode())

    def request(method, **args):
        send({"method": method, "args": args})
        reply = json.loads(channel.recv(MAX_PACKET))
        if "error" in reply:
            raise ValueError(reply["error"])
        return reply["result"]

    class Robot:
        def state(self, side=None):
            return request("state", side=side)

        def observe(self):
            obs = request("observe")
            images = {}
            for name, shape in obs.pop("image_shapes").items():
                image = np.empty(shape, dtype=np.uint8)
                data = memoryview(image).cast("B")
                for offset in range(0, len(data), MAX_PACKET):
                    chunk = data[offset:offset+MAX_PACKET]
                    if channel.recv_into(chunk) != len(chunk):
                        raise RuntimeError("Incomplete RGB frame")
                images[name] = image
            obs["images"] = images
            return obs

        def move(self, pose, gripper=None, *, side=None, hand=None):
            return request("move", pose=np.asarray(pose).tolist(), gripper=gripper, side=side,
                           hand=None if hand is None else np.asarray(hand).tolist())

    namespace = {"np": np, "robot": Robot()}
    while packet := channel.recv(MAX_PACKET):
        out, result = io.StringIO(), {}
        try:
            with contextlib.redirect_stdout(out):
                exec(compile(json.loads(packet)["code"], "<policy>", "exec"), namespace)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        result["output"] = out.getvalue()[-4000:]
        send({"done": result})


class ProgramSession:
    def __init__(self, bench):
        self.bench = bench
        self.channel, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        worker = inspect.getsource(program_worker) + "\nprogram_worker()\n"
        command = ["bwrap", "--die-with-parent", "--new-session", "--unshare-all",
                   "--clearenv", "--setenv", "OPENBLAS_NUM_THREADS", "1",
                   "--ro-bind", "/usr", "/usr",
                   "--ro-bind-try", "/etc/alternatives", "/etc/alternatives",
                   "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache", "--symlink", "usr/lib", "/lib",
                   "--symlink", "usr/lib64", "/lib64", "--dev", "/dev", "--tmpfs", "/tmp",
                   "/usr/bin/python3", "-IB", "-c", worker]
        try:
            self.process = subprocess.Popen(command, stdin=child, stdout=subprocess.DEVNULL)
        except BaseException:
            self.channel.close()
            raise
        finally:
            child.close()

    def close(self):
        self.channel.close()
        self.process.kill()
        self.process.wait()

    def execute(self, code):
        payload = json.dumps({"code": code}).encode()
        if len(payload) >= 65536:
            raise InvalidTarget("Program exceeds packet limit")
        deadline = time.monotonic() + 60
        self.channel.send(payload)
        for _ in range(129):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Program deadline exceeded")
            self.channel.settimeout(remaining)
            packet = self.channel.recv(65536)
            if not packet:
                raise RuntimeError("Isolated Python worker exited")
            message = json.loads(packet)
            if "done" in message:
                return message["done"]
            args = message.get("args", {})
            try:
                if message["method"] == "observe" and not args:
                    obs = self.bench.observe()
                    images = obs.pop("images")
                    header = {**obs,
                              "image_shapes": {k: v.shape for k, v in images.items()}}
                    packet = json.dumps({"result": header}, allow_nan=False).encode()
                    if len(packet) >= 65536:
                        raise ValueError("Observation metadata exceeds packet limit")
                    self.channel.send(packet)
                    for image in images.values():
                        data = memoryview(image.tobytes())
                        for offset in range(0, len(data), 65536):
                            self.channel.send(data[offset:offset+65536])
                    continue
                if message["method"] == "state" and isinstance(args, dict) and set(args) <= {"side"}:
                    value = self.bench.state(**args)
                elif message["method"] == "move" and isinstance(args, dict) and set(args) <= {"pose","gripper","side","hand"}:
                    value = self.bench.move(**args)
                else:
                    raise InvalidTarget("Unknown robot operation")
                response = {"result": value}
            except InvalidTarget as exc:
                response = {"error": str(exc)}
            self.channel.send(json.dumps(response, allow_nan=False).encode())
        raise RuntimeError("Program exceeded the per-turn robot-call limit")


def Tool(method):
    """Expose a wrapper method to the model: its signature becomes the argument schema, its docstring the description."""
    hints = get_type_hints(method, include_extras=True)
    fields = {name: (hints[name], ... if p.default is p.empty else p.default)
              for name, p in inspect.signature(method).parameters.items() if name != 'self'}
    method.arguments = create_model(method.__name__, __config__=ConfigDict(extra='forbid', allow_inf_nan=False),
                                    **fields)
    return method


class RealWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.tools = {name: method for name, method in inspect.getmembers(type(self), inspect.isfunction)
                      if hasattr(method, "arguments")}
        text = gym.spaces.Text(65536, min_length=0, charset=string.printable)
        self.action_space = gym.spaces.Dict({"name": text, "arguments": text})
        self.observation_space = gym.spaces.Dict({**env.observation_space.spaces, "feedback": text})
        self.program = None

    def schemas(self):
        tools = []
        for name, method in self.tools.items():
            schema = method.arguments.model_json_schema()
            tools.append({"type": "function", "function": {"name": name, "description": inspect.getdoc(method),
                                                          "parameters": schema}})
        return tools

    def reset(self, *, seed=None, options=None):
        self.close_program()
        observation, info = self.env.reset(seed=seed, options=options)
        return {**observation, "feedback": ""}, info

    def step(self, action):
        name = action["name"]
        finished = False
        try:
            method = self.tools.get(name)
            if method is None:
                raise InvalidTarget(f"Unknown tool {name}")
            arguments = method.arguments.model_validate_json(action["arguments"])
            result = method(self, **arguments.model_dump())
            finished = name == "finish"
        except (ValidationError, InvalidTarget) as exc:
            if not self.env.motion_enabled:
                raise
            result = {"error": str(exc)}
        obs = self.env.observe()
        return {**obs, "feedback": json.dumps(result)}, 0., finished, False, {"result": result}

    @Tool
    def finish(self):
        """End the episode. Task success requires independent evidence."""
        return {}

    @Tool
    def run_code(self, code: Annotated[str, Field(min_length=1)]):
        """Run Python with the persistent robot API and return printed output."""
        if self.program is None:
            self.program = ProgramSession(self.env)
        return self.program.execute(code)

    def close_program(self):
        if self.program is not None:
            self.program.close()
            self.program = None

    def close(self):
        self.close_program()
        super().close()


def look(observation: dict) -> dict:
    """The current state and images as one user message."""
    content = [{'type': 'text', 'text': 'State: ' + json.dumps(observation['state'])}]
    for camera, image in observation['images'].items():
        png = io.BytesIO()
        Image.fromarray(image).save(png, format='PNG')
        content += [{'type': 'text', 'text': f'Camera {camera}:'}, {'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(png.getvalue()).decode()}}]
    return {'role': 'user', 'content': content}


def public_profiles(profiles):
    return {side: {k: v for k, v in p.items() if k not in ("driver", "args", "tool")}
            for side, p in profiles.items()}


def compile_controller(run_dir, profiles, *, decide=None):
    """Generate a candidate from a closed episode; this path does not open a workstation."""
    output = run_dir / "controller.py"
    if output.exists():
        raise FileExistsError(output)
    events = [json.loads(line) for line in (run_dir / "trajectory.jsonl").read_text().splitlines()]
    observations = [event for event in events if "images" in event]
    if not observations or not any("stop" in e or "fault" in e for e in events):
        raise ValueError("Expected a completed CaP trajectory with observations")
    trace = [{k: v for k, v in e.items() if k in ("turn", "arms", "request", "feedback", "stop", "fault", "task_success")}
             for e in events]
    episode = {"task": observations[0]["task"], "station": profiles or events[-1].get("station", {}),
               "trajectory": trace}
    content = [{"type": "text", "text": json.dumps(episode, ensure_ascii=False)}]
    for obs in observations[:1] + (observations[-1:] if len(observations) > 1 else []):
        for name, filename in obs["images"].items():
            path = run_dir / filename
            if path.resolve().parent != run_dir.resolve():
                raise ValueError("Expected an episode-local image")
            content.append({"type": "text", "text": f"Turn {obs['turn']}, camera {name}:"})
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()}})
    tools = [{"type": "function", "function": {"name": "save_controller", "description": "Save Python source.",
              "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                             "required": ["code"], "additionalProperties": False}}}]
    messages = [{"role": "system", "content": CONTROLLER_PROMPT}, {"role": "user", "content": content}]
    backend = decide if decide is not None else configured_decider(tools)
    if isinstance(backend, CodexDecision):
        reply = backend.complete(messages, timeout=600)["choices"][0]["message"]
    else:
        reply = backend(messages) if backend is not None else mllm(messages, tools)
    call = reply["tool_calls"][0]["function"]
    if call["name"] != "save_controller":
        raise ValueError("Expected save_controller")
    source = json.loads(call["arguments"])["code"]
    compile(source, str(output), "exec")
    mains = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    if len(mains) != 1:
        raise ValueError("Controller must define one main(robot)")
    args = mains[0].args
    if [a.arg for a in args.posonlyargs + args.args] != ["robot"] or args.kwonlyargs or args.vararg or args.kwarg:
        raise ValueError("Controller must define main(robot)")
    with tempfile.NamedTemporaryFile(dir=run_dir, mode="w", encoding="utf-8", delete=False) as tmp:
        try:
            tmp.write(source.rstrip() + "\n")
            tmp.flush()
            os.link(tmp.name, output)
        finally:
            Path(tmp.name).unlink()
    return output


class RUAgent:
    def __init__(self, task, system, tools, decide=None):
        self.task, self.system, self.tools = task, system, tools
        self.history = []
        self.decide = decide if decide is not None else configured_decider(tools)

    def get_action(self, images, state):
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": self.task},
                    *self.history[-8:], look({"images": images, "state": state})]
        message = self.decide(messages) if self.decide is not None else mllm(messages, self.tools)
        call = message["tool_calls"][0]
        self.history.append(message)
        return call

    def feedback(self, call, result):
        self.history.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})


def run_cap(bench, task, run_dir, *, decide=None, max_decisions=30):
    env = RealWrapper(bench)
    public = public_profiles(bench.profiles)
    agent = RUAgent(task, SYSTEM + " Station: " + json.dumps(public), env.schemas(), decide)
    with (run_dir / "trajectory.jsonl").open("x") as log:
        def record(**event):
            log.write(json.dumps(event, allow_nan=False) + "\n")
            log.flush()

        started = time.monotonic()
        decisions = 0
        try:
            obs, _ = env.reset()
            reason = "max_decisions"
            for turn in range(max_decisions):
                images = {}
                for name, rgb in obs["images"].items():
                    images[name] = f"{name}-{turn:03}.png"
                    Image.fromarray(rgb).save(run_dir / images[name])
                record(turn=turn, task=task, images=images, arms=obs["arms"],
                       rgb_received_ns=obs["rgb_received_ns"], state_received_ns=obs["state_received_ns"])
                call = agent.get_action(obs["images"], obs["arms"])
                decisions += 1
                record(turn=turn, request=call["function"])
                obs, _, done, _, info = env.step(call["function"])
                if done:
                    reason = "finish"
                    break
                agent.feedback(call, info["result"])
                record(turn=turn, feedback=info["result"])
            record(stop=reason, task_success=None, station=public)
        except Exception as exc:
            record(fault=str(exc))
            raise
        finally:
            env.close_program()

    report = {**({"decision_backend": "injected"} if decide is not None else decision_settings()), "task": task,
              "stop_reason": reason, "decisions": decisions, "agent_finished": reason == "finish",
              "elapsed_s": round(time.monotonic() - started, 3), "task_success": None}
    print(f"Exploration ended ({reason}). Generating {run_dir / 'controller.py'}", flush=True)
    try:
        artifact = compile_controller(run_dir, public, decide=decide)
        report.update(controller=artifact.name, controller_verified=False)
    except Exception as exc:
        report["controller_error"] = str(exc)
        raise
    finally:
        (run_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return reason


@dataclass
class Args:
    task: str
    backend: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    base_url: str | None = None
    output_dir: Path = Path("runs")
    out: Path | None = None  # An explicit, new run directory (legacy CLI).
    config: Path | None = None
    enable_motion: bool = False
    max_decisions: int = 30


@dataclass
class CompileArgs:
    run_dir: Path
    config: Path | None = None
    backend: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    base_url: str | None = None


def main():
    if sys.argv[1:2] == ["compile"]:
        args = tyro.cli(CompileArgs, args=sys.argv[2:])
        configure_decision(args.backend, args.model, args.reasoning_effort, args.base_url)
        profiles = json.loads(args.config.read_text())["arms"] if args.config else {}
        artifact = compile_controller(args.run_dir, public_profiles(profiles))
        report_path = args.run_dir / "result.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            report.pop("controller_error", None)
            report.update(controller=artifact.name, controller_verified=False,
                          controller_backend=decision_settings())
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(artifact)
        return
    args = tyro.cli(Args)
    configure_decision(args.backend, args.model, args.reasoning_effort, args.base_url)
    if args.config and not args.enable_motion:
        raise SystemExit("Configured device experiments require --enable-motion")
    config = json.loads(args.config.read_text()) if args.config else None
    task_name = re.sub(r"[^\w-]+", "_", args.task).strip("_")[:64] or "task"
    run_dir = args.out or args.output_dir / f"{task_name}__cap_real__0__{time.time_ns()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    bench = workstation(config, True)
    try:
        reason = run_cap(bench, args.task, run_dir, max_decisions=args.max_decisions)
        print(json.dumps({"run_dir": str(run_dir), "stop_reason": reason}))
    finally:
        bench.close()


if __name__ == "__main__":
    main()
