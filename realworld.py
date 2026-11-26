"""Run CleanRUA on a configurable workstation. Run from the repository root.

    python -m realworld check --config station.json
    python -m realworld calibrate-workspace --config station.json --arm left
    python -m realworld cap --task "Stack the blocks" --backend api --config station.json --enable-motion
    python -m realworld compile --run-dir RUN_DIR --config station.json --backend api
    python -m realworld replay --policy RUN_DIR/controller.py --config station.json --enable-motion

Select direct, cap, delegate or monitor. Set CLEANRUA_BASE_URL, CLEANRUA_MODEL_NAME and
CLEANRUA_API_KEY for an API endpoint with vision and tool calls. Delegate and Monitor
also require a configured action policy. CaP executes generated code in a Linux
bubblewrap worker with /usr/bin/python3 and NumPy. Replay executes a reviewed,
trusted main(robot) file in the workstation process without a model.

In station JSON, arms maps left/right to {driver: "module:Class", args, end_effector,
workspace_m, max_translation_m, max_rotation_rad}. Base-frame absolute XYZ/RPY
targets use metres and radians. An arm supplies state(), move_pose() and stop();
a gripper or hand may be integral or supplied by a separate tool. Cameras are named
drivers returning uint8 RGB arrays through read() and exposing close(). Make local
driver modules importable on PYTHONPATH. Without --config the CLI uses mock devices.
"""
import argparse
import ast
import base64
from functools import partial
from importlib import import_module
import inspect
import io
from itertools import count
import json
import os
from pathlib import Path
import re
import runpy
import socket
import stat
import string
import subprocess
import sys
import tempfile
import time
from typing import Annotated, get_type_hints
from uuid import uuid4
import warnings

import gymnasium as gym
from gymnasium import spaces
import httpx
import numpy as np
from PIL import Image
from pydantic import ConfigDict, Field, ValidationError, create_model
from scipy.spatial.transform import Rotation

from cleanrua import cap, delegate, direct, monitor
from cleanrua.direct import look


# Workstation observations, actions and measured receipts


class InvalidTarget(ValueError):
    """Rejected before issuing device writes."""


def vector(value, size):
    a = np.asarray(value)
    if a.shape != (size,) or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise InvalidTarget(f"Expected {size} finite numbers")
    return a.astype(float)


class PartialDict(spaces.Dict):
    """Any nonempty subset of the configured arms or target fields."""
    def contains(self, value):
        return (isinstance(value, dict) and bool(value)
                and all(k in self.spaces and self.spaces[k].contains(v) for k, v in value.items()))


class Bench(gym.Env):
    def __init__(self, arms, cameras, profiles, *, motion_enabled=False):
        if not arms or not set(arms) <= {"left", "right"} or set(arms) != set(profiles):
            raise ValueError("Configure left and/or right arm with matching profiles")
        self.arms, self.cameras, self.profiles = arms, cameras, profiles
        self.motion_enabled = motion_enabled
        action_spaces = {}
        self.hand_limits = {}
        for side, p in profiles.items():
            fields = {"pose": spaces.Box(-np.inf, np.inf, (6,), np.float64)}
            kind = p["end_effector"]
            if kind == "gripper":
                fields[kind] = spaces.Box(0., 1., (), np.float64)
            elif kind == "hand":
                lower, upper = p["hand_limits_rad"]
                lo, hi = vector(lower, len(lower)), vector(upper, len(lower))
                if not len(lo) or np.any(lo >= hi):
                    raise ValueError("Invalid hand joint bounds")
                self.hand_limits[side] = (lo, hi)
                fields[kind] = spaces.Box(lo, hi, dtype=np.float64)
            elif kind != "none":
                raise ValueError("Unknown end effector")
            for name in ("arrival_translation_m", "arrival_rotation_rad"):
                if name in p and not (np.isfinite(p[name]) and p[name] > 0):
                    raise ValueError(f"Invalid {name}")
            action_spaces[side] = PartialDict(fields)
            bounds = p.get("workspace_m")
            if bounds is not None:
                lo, hi = (vector(row, 3) for row in bounds)
                if np.any(lo >= hi):
                    raise ValueError("Invalid workspace corners")
            elif motion_enabled:
                raise ValueError("Configure workspace_m before motion")
            for name in ("max_translation_m", "max_rotation_rad"):
                if not np.isfinite(p[name]) or p[name] <= 0:
                    raise ValueError(f"Invalid {name}")
        self.action_space = PartialDict(action_spaces)
        obs = self.observe()
        self.observation_space = spaces.Dict({
            "images": spaces.Dict({k: spaces.Box(0, 255, a.shape, np.uint8) for k, a in obs["images"].items()}),
            "arms": spaces.Dict({side: spaces.Dict({
                key: spaces.Box(-np.inf, np.inf, np.shape(value), np.float64)
                for key, value in state.items()}) for side, state in obs["arms"].items()}),
            "rgb_received_ns": spaces.Dict({k: spaces.Discrete(2**63-1) for k in cameras}),
            "state_received_ns": spaces.Discrete(2**63-1)})

    def _arm_state(self, side):
        """Convert SDK arrays/scalars once, at the driver boundary."""
        raw = self.arms[side].state()
        state = {"pose": vector(raw["pose"], 6).tolist()}
        if "joints" in raw:
            joints = np.asarray(raw["joints"])
            if joints.ndim != 1:
                raise ValueError(f"Expected joint vector for {side}")
            state["joints"] = vector(joints, joints.size).tolist()
        if self.profiles[side]["end_effector"] == "gripper":
            opening = float(raw["gripper"])
            if not 0 <= opening <= 1:
                raise ValueError(f"Invalid measured opening for {side}")
            state["gripper"] = opening
        if side in self.hand_limits:
            state["hand"] = vector(raw["hand"], len(self.hand_limits[side][0])).tolist()
        return state

    def state(self, side=None):
        """With a side, always return its named fields. Preserve the original no-argument API."""
        if side is not None:
            if not isinstance(side, str) or side not in self.arms:
                raise InvalidTarget("Select a configured arm")
            return self._arm_state(side)
        return self._receipt({s: self._arm_state(s) for s in self.arms})

    def _receipt(self, states):
        return next(iter(states.values())) if len(self.arms) == 1 else states

    def observe(self):
        images, stamps = {}, {}
        for name, camera in self.cameras.items():
            rgb = camera.read()
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
                raise ValueError(f"Invalid RGB image from {name}")
            images[name], stamps[name] = rgb, time.monotonic_ns()
        states = {s: self._arm_state(s) for s in self.arms}
        return {"images": images, "arms": states,
                "rgb_received_ns": stamps, "state_received_ns": time.monotonic_ns()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observe(), {}

    def close(self):
        for camera in self.cameras.values():
            camera.close()
        for arm in self.arms.values():
            if callable(getattr(arm, "close", None)):
                arm.close()

    def _validate(self, side, target, current):
        fields = self.action_space[side].spaces
        if not isinstance(target, dict) or not target or set(target) - fields.keys():
            raise InvalidTarget(f"Invalid target fields for {side}: expected {list(fields)}")
        start = np.asarray(current["pose"])
        pose = vector(target.get("pose", start), 6)
        low, high = self.profiles[side]["workspace_m"]
        if any(np.any(x < low) or np.any(x > high) for x in (start[:3], pose[:3])):
            raise InvalidTarget("Outside workspace")
        rot = Rotation.from_euler("xyz", pose[3:])
        delta = (Rotation.from_euler("xyz", start[3:]).inv() * rot).magnitude()
        if (np.linalg.norm(pose[:3]-start[:3]) > self.profiles[side]["max_translation_m"]
                or delta > self.profiles[side]["max_rotation_rad"]):
            raise InvalidTarget("Step exceeds configured translation or rotation limit")
        validated = {"pose": pose} if "pose" in target else {}
        if "gripper" in target:
            opening = vector([target["gripper"]], 1)[0]
            if not 0 <= opening <= 1:
                raise InvalidTarget("Gripper outside [0,1]")
            validated["gripper"] = float(opening)
        if "hand" in target:
            lo, hi = self.hand_limits[side]
            hand = vector(target["hand"], len(lo))
            if np.any(hand < lo) or np.any(hand > hi):
                raise InvalidTarget("Hand target outside joint limits")
            validated["hand"] = hand
        return validated

    def _execute(self, targets, sides, readback):
        if not self.motion_enabled:
            raise RuntimeError("Motion disabled for this session")
        if not isinstance(targets, dict) or set(targets) - self.arms.keys():
            raise InvalidTarget("Expected targets keyed by configured arm names")
        selected = [s for s in sides if s in targets]
        if not selected:
            raise InvalidTarget("No target for any selected arm")
        current = {s: self._arm_state(s) for s in selected}
        commands = {s: self._validate(s, targets[s], current[s]) for s in selected}
        gripper_results = {}
        try:
            for side, cmd in commands.items():
                arm = self.arms[side]
                pose = cmd.get("pose")
                if pose is not None and not np.allclose(pose, current[side]["pose"], rtol=0, atol=1e-9):
                    arm.move_pose(pose)
                    actual = vector(arm.state()["pose"], 6)
                    error = Rotation.from_euler("xyz", pose[3:]).inv() * Rotation.from_euler("xyz", actual[3:])
                    profile = self.profiles[side]
                    translation_tolerance = profile.get("arrival_translation_m", 0.002)
                    rotation_tolerance = profile.get("arrival_rotation_rad", 0.05)
                    if np.linalg.norm(actual[:3]-pose[:3]) > translation_tolerance or error.magnitude() > rotation_tolerance:
                        raise RuntimeError("End effector outside arrival tolerance")
                if "gripper" in cmd and abs(cmd["gripper"] - current[side]["gripper"]) > .01:
                    result = arm.set_gripper(cmd["gripper"])
                    if isinstance(result, dict):
                        gripper_results[side] = result
                if "hand" in cmd and not np.array_equal(cmd["hand"], current[side]["hand"]):
                    arm.set_hand(cmd["hand"].tolist())
            return readback(), gripper_results
        except BaseException:
            self.motion_enabled = False
            for side in selected:
                try:
                    self.arms[side].stop()
                except Exception as stop_error:
                    warnings.warn(f"Stop failed on {side}: {stop_error}")
            raise

    def move(self, pose, gripper=None, *, side=None, hand=None):
        if side is None and len(self.arms) == 1:
            side = next(iter(self.arms))
        if not isinstance(side, str) or side not in self.arms:
            raise InvalidTarget("Select a configured arm")
        target = {"pose": pose}
        if gripper is not None:
            target["gripper"] = gripper
        if hand is not None:
            target["hand"] = hand
        measured, results = self._execute({side: target}, [side], self.state)
        if results:
            measured["gripper_results"] = results
        return measured

    def step(self, action):
        return self.step_groups(action, self.arms)

    def step_groups(self, action, sides):
        """Execute only the selected arm blocks; omitted devices receive no commands."""
        selected = tuple(sides)
        if not selected or len(set(selected)) != len(selected) or set(selected) - self.arms.keys():
            raise InvalidTarget("Select distinct configured arms")
        obs, results = self._execute(action, selected, self.observe)
        info = {"measured": self._receipt(obs["arms"])}
        if results:
            info["gripper_results"] = results
        return obs, 0., False, False, info


# Built-in offline devices and workstation configuration


class MockArm:
    def __init__(self):
        self.pose = [0.4, 0., 0.3, 0., 0., 0.]
        self.opening = 1.

    def state(self):
        return {"pose": self.pose.copy(), "joints": [0.] * 7, "gripper": self.opening}

    def move_pose(self, pose):
        self.pose = list(pose)

    def set_gripper(self, opening):
        self.opening = opening

    def stop(self):
        pass


class MockCamera:
    def __init__(self):
        self.count = 0

    def read(self):
        self.count += 1
        return np.full((48, 64, 3), self.count % 256, dtype=np.uint8)

    def close(self):
        pass


class MockHand:
    def __init__(self, joints):
        self.positions = [0.] * joints

    def state(self):
        return {"hand": self.positions.copy()}

    def set_hand(self, positions):
        self.positions = list(positions)

    def stop(self):
        pass


class PatternCamera:
    def __init__(self, width, height):
        self.rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self.rgb[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
        self.rgb[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]

    def read(self):
        return self.rgb.copy()

    def close(self):
        pass


def device(spec):
    module, name = spec["driver"].split(":")
    if module in ("realworld", "realworld.policy", "realworld.real_policy", "cleanrua_utils.real_policy",
                  "realworld.bench_mock", "cleanrua_utils.bench_mock", "realworld.tests.fixtures.devices",
                  "examples.plug_and_play.pattern_camera", "realworld.examples.plug_and_play.pattern_camera"):
        factory = getattr(sys.modules[__name__], name)
    else:
        factory = getattr(import_module(module), name)
    return factory(**spec.get("args", {}))


class ArmWithTool:
    def __init__(self, arm, tool, kind):
        self.arm, self.tool, self.kind = arm, tool, kind

    def state(self):
        return {**self.arm.state(), self.kind: self.tool.state()[self.kind]}

    def move_pose(self, pose):
        self.arm.move_pose(pose)

    def set_hand(self, joints):
        self.tool.set_hand(joints)

    def set_gripper(self, opening):
        return self.tool.set_gripper(opening)

    def close(self):
        if hasattr(self.tool, "close"):
            self.tool.close()

    def stop(self):
        try:
            self.arm.stop()
        finally:
            self.tool.stop()


def mock_profile():
    devices = "realworld"
    return {"arms": {"left": {"driver": devices + ":MockArm", "end_effector": "gripper",
                              "workspace_m": [[.2,-.2,.1],[.6,.2,.5]],
                              "max_translation_m": .031, "max_rotation_rad": .18}},
            "cameras": {"scene": {"driver": devices + ":MockCamera"}}}


def require_methods(driver, methods, label):
    missing = [m for m in methods if not callable(getattr(driver, m, None))]
    if missing:
        raise ValueError(f"{label} driver must implement {', '.join(missing)}")


def workstation(config, motion_enabled=False):
    arms, cameras = {}, {}
    if config is None:
        config = mock_profile()
    elif "arms" not in config:
        raise ValueError("Workstation JSON requires named arms with explicit driver and args")
    try:
        for side, p in config["arms"].items():
            arm = device(p)
            required = ("state", "move_pose", "stop") if motion_enabled else ("state",)
            require_methods(arm, required, side)
            kind = p["end_effector"]
            if "tool" in p:
                if kind not in ("gripper", "hand"):
                    raise ValueError(f"Unexpected tool for {side}: {kind}")
                tool = device(p["tool"])
                required = ("state", f"set_{kind}", "stop") if motion_enabled else ("state",)
                require_methods(tool, required, f"{side} {kind}")
                if motion_enabled and hasattr(tool, "check_motion_ready"):
                    try:
                        tool.check_motion_ready()
                    except Exception:
                        if hasattr(tool, "close"):
                            tool.close()
                        raise
                arm = ArmWithTool(arm, tool, kind)
            elif motion_enabled and kind in ("gripper", "hand"):
                require_methods(arm, (f"set_{kind}",), side)
            arms[side] = arm
        for name, spec in config["cameras"].items():
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError(f"Invalid camera name {name!r}")
            camera = device(spec)
            require_methods(camera, ("read", "close"), name)
            cameras[name] = camera
        return Bench(arms, cameras, config["arms"], motion_enabled=motion_enabled)
    except BaseException:
        for camera in cameras.values():
            camera.close()
        for arm in arms.values():
            if isinstance(arm, ArmWithTool):
                arm.close()
        raise


# Teleoperated workspace calibration


def collect_workspace_points(reader):
    points = []
    while True:
        command = input("回车采点，done 计算边界，q 取消 > ").strip().lower()
        if command == "q":
            return None
        if command == "done":
            if len(points) >= 2:
                low, high = np.min(points, axis=0), np.max(points, axis=0)
                if np.all(low < high):
                    return [low.tolist(), high.tolist()]
            print("请继续采点：X、Y、Z 每个轴都要覆盖不同的位置。")
        elif command == "":
            state = reader.state()
            try:
                xyz = vector(state["pose"], 6)[:3]
            except (KeyError, InvalidTarget) as exc:
                print(f"无效 TCP 位姿，本点未记录：{exc}")
                continue
            points.append(xyz)
            print(f"点 {len(points)} (m): {xyz.tolist()}")
        else:
            print("请输入 done、q，或直接按回车。")


def save_workspace(path, original, config):
    import fcntl

    # Serialize concurrent calibrations only during the compare-and-replace.
    with path.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.read_bytes() != original:
            raise ValueError("配置在采点期间已被修改，请重新读取后再标定。")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
            try:
                json.dump(config, tmp, ensure_ascii=False, indent=2, allow_nan=False)
                tmp.write("\n")
                tmp.flush()
                os.fchmod(tmp.fileno(), stat.S_IMODE(path.stat().st_mode))
                os.replace(tmp.name, path)
            finally:
                Path(tmp.name).unlink(missing_ok=True)


def calibrate_workspace(path, side=None):
    path = path.resolve(strict=True)
    original = path.read_bytes()
    config = json.loads(original)
    arms = config.get("arms", {})
    if side is None and len(arms) == 1:
        side = next(iter(arms))
    if side not in ("left", "right") or side not in arms:
        raise ValueError(f"请选择配置中已有的机械臂 --arm，当前可选：{', '.join(arms)}")

    profile = arms[side]
    print(f"配置：{path}\n机械臂：{side}；读取基座系 TCP XYZ，单位为米。")
    try:
        reader = device(profile.get("state_reader", profile))
        try:
            bounds = collect_workspace_points(reader)
        finally:
            close = getattr(reader, "close", None)
            if callable(close):
                close()
        if bounds is None:
            return
        print(f"arms.{side}.workspace_m\n原值：{profile.get('workspace_m')}\n新值：{bounds}")
        if input("确认写回上述配置？[y/N] ").strip().lower() not in ("y", "yes"):
            print("已取消。")
            return
    except (KeyboardInterrupt, EOFError):
        print("\n已取消。")
        return
    arms[side] = {**profile, "workspace_m": bounds}
    save_workspace(path, original, config)
    print(f"已保存 arms.{side}.workspace_m 到 {path}")


# Model backends


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


# Action policy clients for Delegate and Monitor


CONTRACT = "arms.absolute.base_m.xyz_rpy_rad.open_fraction.hand_rad.v1"


class HTTPArmPolicy:
    """POST all named RGB views, named arm measurements and instruction to a policy server."""
    def __init__(self, url):
        self.url = url

    def predict(self, obs, instruction):
        images = {}
        for name, rgb in obs["images"].items():
            stream = io.BytesIO()
            Image.fromarray(rgb).save(stream, format="PNG")
            images[name] = base64.b64encode(stream.getvalue()).decode("ascii")
        request = {"contract": CONTRACT, "instruction": instruction, "images_png": images,
                   "arms": obs["arms"],
                   "rgb_received_ns": obs["rgb_received_ns"], "state_received_ns": obs["state_received_ns"]}
        response = httpx.post(self.url, json=request, timeout=30)
        response.raise_for_status()
        result = response.json()
        if result.get("contract") != CONTRACT:
            raise ValueError("Policy response must identify the requested named-arm contract")
        return result["actions"]


class DexJoCoOpenPI:
    """Native DexJoCo Pi0.5 server. Configure matching world/flange frames and Allegro joint order."""
    def __init__(self, url, camera_mapping, joint_indices):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        if set(joint_indices) not in ({"right"}, {"right", "left"}):
            raise ValueError("DexJoCo expects right or right+left")
        self.indices = {}
        for side in ("right", "left"):
            if side in joint_indices:
                slots = joint_indices[side]
                if (len(slots) != 16 or any(type(i) != int or not 0 <= i < 16 for i in slots)
                        or len(set(slots)) != 16):
                    raise ValueError("Specify 16 driver joint indices in native Allegro order")
                self.indices[side] = np.asarray(slots)
        if not camera_mapping or {"prompt", "state"} & camera_mapping.keys():
            raise ValueError("Provide the checkpoint's mapping from image keys to camera names")
        self.cameras = camera_mapping
        self.client = WebsocketClientPolicy(url)

    def predict(self, obs, instruction):
        arms = obs["arms"]
        if arms.keys() != self.indices.keys():
            raise ValueError("Arm names do not match this DexJoCo checkpoint")
        poses, hands = [], []
        for side, ix in self.indices.items():
            arm = arms[side]
            joints = np.asarray(arm.get("hand"))
            if joints.shape != (16,):
                raise ValueError(f"{side} must provide 16 Allegro joint measurements")
            pose = arm["pose"]
            quat = Rotation.from_euler("xyz", pose[3:]).as_quat(scalar_first=True)
            poses.extend([*pose[:3], *quat])
            hands.extend(joints[ix])
        images = {key: np.asarray(Image.fromarray(obs["images"][name]).resize((224, 224), Image.Resampling.BILINEAR))
                  for key, name in self.cameras.items()}
        native = np.zeros(46)
        native[:len(poses) + len(hands)] = poses + hands
        raw = np.asarray(self.client.infer({**images, "state": native, "prompt": instruction})["actions"])
        # Some single-arm inference servers return a padded second-arm block, as in upstream.
        if (raw.ndim != 2 or not len(raw) or raw.shape[1] not in {22 * len(self.indices), 44}
                or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all()):
            raise ValueError("Invalid native DexJoCo action chunk")
        actions = [{} for _ in raw]
        for i, (side, ix) in enumerate(self.indices.items()):
            block = raw[:, 22*i:22*(i+1)]
            angles = Rotation.from_rotvec(block[:, 3:6].copy()).as_euler("xyz")
            for target, row, rpy in zip(actions, block, angles):
                hand = np.empty(16)
                hand[ix] = row[6:22]
                target[side] = {"pose": [*row[:3].tolist(), *rpy.tolist()], "hand": hand.tolist()}
        return actions


class ChunkPolicy:
    def __init__(self, client, actions_per_chunk, control_hz):
        if type(actions_per_chunk) != int or not 1 <= actions_per_chunk <= 300:
            raise ValueError("actions_per_chunk must be an integer in [1,300]")
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be positive and finite")
        self.client, self.horizon, self.period = client, actions_per_chunk, 1 / control_hz
        self.next_step = 0.

    def run(self, env, instruction, steps, active_groups=None, trace=None):
        feedback, _ = self.run_observed(env, instruction, steps, active_groups, trace)
        return feedback

    def run_observed(self, env, instruction, steps, active_groups=None, trace=None):
        """Return both the policy feedback and the observation used for its last receipt."""
        sides = list(env.arms) if active_groups is None else active_groups
        if not sides or len(set(sides)) != len(sides) or set(sides) - env.arms.keys():
            raise InvalidTarget("Select distinct configured arms")
        obs = env.observe()
        result = {"instruction": instruction, "active_groups": sides, "steps": 0, "inferences": 0}
        while result["steps"] < steps:
            raw = self.client.predict(obs, instruction)
            result["inferences"] += 1
            if not isinstance(raw, (list, tuple)) or not raw or not all(isinstance(step, dict) for step in raw):
                raise ValueError("Policy must return a nonempty sequence of named arm commands")
            count = min(self.horizon, steps - result["steps"], len(raw))
            for target in raw[:count]:
                delay = self.next_step - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                self.next_step = time.monotonic() + self.period
                if trace:
                    trace(policy_action=target, step=result["steps"] + 1, groups=sides)
                try:
                    obs, _, terminated, truncated, info = env.step_groups(target, sides)
                except InvalidTarget as exc:
                    if not env.motion_enabled:  # device failure after dispatch
                        raise
                    return {**result, "stop_reason": "target_rejected", "error": str(exc)}, env.observe()
                result["steps"] += 1
                result["measured"] = info["measured"]
                if trace:
                    trace(policy_measured=info["measured"], step=result["steps"])
                if terminated or truncated:
                    return {**result, "stop_reason": "terminated" if terminated else "truncated"}, obs
        return {**result, "stop_reason": "step_budget"}, obs


def load_policy(station, path=None):
    spec = json.loads(Path(path).read_text()) if path else (station or {}).get("policy")
    if not spec:
        raise ValueError("Supply --policy-config or the workstation's policy section")
    return ChunkPolicy(device(spec), spec["actions_per_chunk"], spec["control_hz"])


# CaP isolated program execution


def program_worker():
    import contextlib
    import io
    import json
    import resource
    import socket
    import numpy as np

    MAX_PACKET = 65536
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
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
        self.motion_fault = None
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
        deadline = time.monotonic() + 600
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
                result = message["done"]
                if self.motion_fault is not None:
                    result["motion_fault"] = self.motion_fault
                    result["motion_enabled"] = False
                    result["recovery_required"] = True
                return result
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
                    if self.motion_fault is not None:
                        raise InvalidTarget("Motion paused after hardware failure; operator recovery required")
                    try:
                        value = self.bench.move(**args)
                    except InvalidTarget:
                        raise
                    except Exception as exc:
                        if self.bench.motion_enabled:
                            raise
                        self.motion_fault = {"error": f"{type(exc).__name__}: {exc}",
                                             "requested_move": args}
                        try:
                            self.motion_fault["measured_state"] = self.bench.state()
                        except Exception as state_error:
                            self.motion_fault["state_error"] = str(state_error)
                        response = {"error": "Motion paused; operator recovery required. " +
                                    self.motion_fault["error"]}
                        self.channel.send(json.dumps(response, allow_nan=False).encode())
                        continue
                else:
                    raise InvalidTarget("Unknown robot operation")
                response = {"result": value}
            except InvalidTarget as exc:
                response = {"error": str(exc)}
            self.channel.send(json.dumps(response, allow_nan=False).encode())
        raise RuntimeError("Program exceeded the per-turn robot-call limit")


# Model tools and episode loop


def Tool(method):
    """Expose a wrapper method to the model: its signature becomes the argument schema, its docstring the description."""
    hints = get_type_hints(method, include_extras=True)
    fields = {name: (hints[name], ... if p.default is p.empty else p.default)
              for name, p in inspect.signature(method).parameters.items() if name != 'self'}
    method.arguments = create_model(method.__name__, __config__=ConfigDict(extra='forbid', allow_inf_nan=False),
                                    **fields)
    return method


class AgentEnv(gym.Wrapper):
    policy_trace = None

    def __init__(self, env):
        super().__init__(env)
        self.tools = {name: method for name, method in inspect.getmembers(type(self), inspect.isfunction)
                      if hasattr(method, "arguments")}
        self.grippers = [side for side, profile in env.profiles.items() if profile["end_effector"] == "gripper"]
        if not self.grippers:
            self.tools.pop("grip", None)
        text = gym.spaces.Text(65536, min_length=0, charset=string.printable)
        self.action_space = gym.spaces.Dict({"name": text, "arguments": text})
        self.observation_space = gym.spaces.Dict({**env.observation_space.spaces, "feedback": text})

    def schemas(self):
        tools = []
        for name, method in self.tools.items():
            schema = method.arguments.model_json_schema()
            if name == "move":
                fields = schema["properties"]
                profiles = self.env.profiles
                fields["side"] = {"type": "string", "enum": list(profiles)}
                if len(profiles) > 1:
                    schema["required"].append("side")
                for key in ("gripper", "hand"):
                    if all(p["end_effector"] != key for p in profiles.values()):
                        del fields[key]
            if name == "grip":
                schema["properties"]["side"] = {"type": "string", "enum": self.grippers}
                if len(self.grippers) > 1:
                    schema["required"].append("side")
            tools.append({"type": "function", "function": {"name": name, "description": inspect.getdoc(method),
                                                          "parameters": schema}})
        return tools

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return {**observation, "feedback": ""}, info

    def step(self, action):
        name = action["name"]
        obs = None
        try:
            method = self.tools.get(name)
            if method is None:
                raise InvalidTarget(f"Unknown tool {name}")
            arguments = method.arguments.model_validate_json(action["arguments"])
            result = method(self, **arguments.model_dump())
            if isinstance(result, tuple):  # An action policy already acquired its next observation.
                result, obs = result
            finished = name == "finish"
        except (ValidationError, InvalidTarget) as exc:
            if not self.env.motion_enabled:
                raise
            result, finished = {"error": str(exc)}, False
        if obs is None:
            obs = self.env.observe()
        terminated = finished or result.get("stop_reason") == "terminated"
        truncated = result.get("stop_reason") == "truncated"
        return {**obs, "feedback": json.dumps(result)}, 0., terminated, truncated, {"result": result}

    @Tool
    def finish(self):
        """End the episode. Task success requires independent evidence."""
        return {}


class MotionEnv(AgentEnv):
    @Tool
    def move(self, pose: Annotated[list[float], Field(min_length=6, max_length=6)],
             side: str = None, gripper: Annotated[float, Field(ge=0, le=1)] = None,
             hand: list[float] = None):
        """Move one arm to an absolute end-effector pose. Omitted tool targets hold."""
        return {"measured": self.env.move(pose, gripper, side=side, hand=hand)}


class RUAgent:
    def __init__(self, task, system, tools, decide=None):
        self.task, self.system = task, system
        self.history = []
        self.decide = decide if decide is not None else configured_decider(tools)
        if self.decide is None:
            self.decide = partial(mllm, tools=tools)

    def get_action(self, images, state):
        observation = look({"images": images, "state": state})
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": self.task},
                    *self.history[-8:], observation]
        message = self.decide(messages)
        call = message["tool_calls"][0]
        self.history.append(message)
        return call

    def feedback(self, call, result):
        self.history.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})


def public_profiles(profiles):
    return {side: {k: v for k, v in p.items() if k not in ("driver", "args", "tool")}
            for side, p in profiles.items()}


def run_episode(env, system, task, run_dir, decide, max_decisions):
    if max_decisions < 0:
        raise ValueError("max_decisions must be nonnegative; 0 means unlimited")
    public = public_profiles(env.env.profiles)
    agent = RUAgent(task, system + " Station: " + json.dumps(public), env.schemas(), decide)
    with (run_dir / "trajectory.jsonl").open("x") as log:
        def record(**event):
            log.write(json.dumps(event, allow_nan=False, default=lambda array: array.tolist()) + "\n")
            log.flush()

        started = time.monotonic()
        decisions = 0
        try:
            obs, _ = env.reset()
            reason = "max_decisions"
            for turn in (count() if max_decisions == 0 else range(max_decisions)):
                images = {}
                for name, rgb in obs["images"].items():
                    images[name] = f"{name}-{turn:03}.png"
                    Image.fromarray(rgb).save(run_dir / images[name])
                record(turn=turn, task=task, images=images, arms=obs["arms"],
                       rgb_received_ns=obs["rgb_received_ns"], state_received_ns=obs["state_received_ns"])
                call = agent.get_action(obs["images"], obs["arms"])
                decisions += 1
                record(turn=turn, request=call["function"])
                env.policy_trace = lambda **event: record(turn=turn, **event)
                obs, _, done, truncated, info = env.step(call["function"])
                record(turn=turn, feedback=info["result"])
                if done or truncated:
                    reason = info["result"].get("stop_reason", "finish")
                    break
                agent.feedback(call, info["result"])
            record(stop=reason, task_success=None, station=public)
            report = {**({"decision_backend": "injected"} if decide is not None else decision_settings()), "task": task,
                      "stop_reason": reason, "decisions": decisions, "agent_finished": reason == "finish",
                      "elapsed_s": round(time.monotonic() - started, 3), "task_success": None}
            (run_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            return report
        except Exception as exc:
            record(fault=str(exc))
            raise


# Direct


DIRECT_SYSTEM = direct.SYSTEM + (
    " For move, send an absolute base-frame XYZ/RPY pose "
    "in metres/radians. Before each move, compare the target to the selected arm's pose in "
    "the latest observation. Use that arm's Station limits: the 3D distance "
    "sqrt(dx^2 + dy^2 + dz^2) must be <= max_translation_m, the angle of the relative rotation "
    "must be <= max_rotation_rad, and the target XYZ must be inside workspace_m. "
    "Leave margin below both step limits. Split longer motions into smaller moves, reading the "
    "new measured pose before choosing each next target. Use smaller steps near objects. "
    "Gripper opening is 0..1; hand joints are in radians."
)


class DirectEnv(MotionEnv):
    @Tool
    def grip(self, opening: Annotated[float, Field(ge=0, le=1)], side: str = None):
        """Set the gripper width only (0 closed, 1 open); read measured opening and inspect the images."""
        if side is None and len(self.grippers) == 1:
            side = self.grippers[0]
        if side not in self.grippers:
            raise InvalidTarget("Select a configured gripper")
        _, _, _, _, info = self.env.step_groups({side: {"gripper": opening}}, [side])
        return info


def run_direct(bench, task, run_dir, *, decide=None, max_decisions=30):
    env = DirectEnv(bench)
    report = run_episode(env, DIRECT_SYSTEM, task, run_dir, decide, max_decisions)
    return report["stop_reason"]


# Cap


CAP_SYSTEM = cap.SYSTEM + (" Explore the task by writing Python. Develop and test functions that compute targets from live "
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

class CapEnv(AgentEnv):
    def __init__(self, env):
        super().__init__(env)
        self.program = None

    def reset(self, *, seed=None, options=None):
        self.close_program()
        return super().reset(seed=seed, options=options)

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


def run_cap(bench, task, run_dir, *, decide=None, max_decisions=30):
    env = CapEnv(bench)
    try:
        report = run_episode(env, CAP_SYSTEM, task, run_dir, decide, max_decisions)
    finally:
        env.close_program()
    reason = report["stop_reason"]
    print(f"Exploration ended ({reason}). Generating {run_dir / 'controller.py'}", flush=True)
    try:
        artifact = compile_controller(run_dir, public_profiles(bench.profiles), decide=decide)
        report.update(controller=artifact.name, controller_verified=False)
    except Exception as exc:
        report["controller_error"] = str(exc)
        raise
    finally:
        (run_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return reason


# Delegate


DELEGATE_SYSTEM = delegate.SYSTEM + " Use actor_act to delegate a bounded phase to the configured policy. Gripper opening is 0..1; hand joints are in radians."


class DelegateEnv(MotionEnv):
    def __init__(self, env, policy, task):
        super().__init__(env)
        self.policy, self.task = policy, task

    @Tool
    def actor_act(self, instruction: Annotated[str, Field(min_length=1)],
                  steps: Annotated[int, Field(ge=1, le=300)], active_groups: list[str] | None = None):
        """Give the policy a phase instruction for at most steps control commands on the selected arms."""
        return self.policy.run_observed(self.env, instruction, steps, active_groups, trace=self.policy_trace)


def run_delegate(bench, task, run_dir, *, policy, decide=None, max_decisions=30):
    env = DelegateEnv(bench, policy, task)
    report = run_episode(env, DELEGATE_SYSTEM, task, run_dir, decide, max_decisions)
    return report["stop_reason"]


# Monitor


MONITOR_SYSTEM = monitor.SYSTEM + " Use finish when the task is complete. Gripper opening is 0..1; hand joints are in radians."


class MonitorEnv(MotionEnv):
    def __init__(self, env, policy, task):
        super().__init__(env)
        self.policy, self.task = policy, task

    @Tool
    def run_policy(self, steps: Annotated[int, Field(ge=1, le=300)]):
        """Continue the original episode task for at most steps control commands."""
        return self.policy.run_observed(self.env, self.task, steps, trace=self.policy_trace)


def run_monitor(bench, task, run_dir, *, policy, decide=None, max_decisions=30):
    env = MonitorEnv(bench, policy, task)
    report = run_episode(env, MONITOR_SYSTEM, task, run_dir, decide, max_decisions)
    return report["stop_reason"]


# Command line


def resolve_algorithm(name):
    """Built-in methods follow upstream naming. External methods use trusted module:function."""
    if ":" in name:
        module, entry = name.split(":", 1)
        return getattr(import_module(module), entry)
    methods = {"direct": run_direct, "cap": run_cap, "delegate": run_delegate, "monitor": run_monitor}
    if name not in methods:
        raise ValueError(f"Unknown algorithm: {name}")
    return methods[name]


def run_method(bench, task, run_dir, *, algorithm="direct", decide=None, max_decisions=30, policy=None):
    method = resolve_algorithm(algorithm)
    options = {"max_decisions": max_decisions}
    if decide is not None:
        options["decide"] = decide
    if "policy" in inspect.signature(method).parameters:
        if policy is None:
            raise ValueError(f"{algorithm} requires an action policy")
        options["policy"] = policy
    return method(bench, task, run_dir, **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", help="check, calibrate-workspace, compile, replay, an algorithm name, or trusted module:function")
    parser.add_argument("--config", type=Path, help="Trusted workstation JSON; omit for offline fixtures")
    parser.add_argument("--arm", choices=("left", "right"), help="Arm to sample with calibrate-workspace")
    parser.add_argument("--enable-motion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out", type=Path, help="New directory for this run")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--max-decisions", type=int, default=30, help="Maximum decisions; 0 means unlimited")
    parser.add_argument("--policy-config", type=Path, help="Action policy client configuration")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--backend", help="api, local, codex or a trusted module:factory")
    parser.add_argument("--model", help="Override the configured decision model")
    parser.add_argument("--reasoning-effort", help="Provider-supported effort or default")
    parser.add_argument("--base-url", help="Compatible API or local server URL")
    parser.add_argument("--run-dir", type=Path, help="Existing CaP episode for compile")
    args = parser.parse_args(argv)
    if args.mode == "calibrate-workspace":
        if not args.config or args.enable_motion:
            parser.error("calibrate-workspace requires --config and is read-only on the hardware")
        try:
            calibrate_workspace(args.config, args.arm)
        except (OSError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")
        return
    if args.arm is not None:
        parser.error("--arm is only used by calibrate-workspace")
    if args.mode == "compile":
        if not args.run_dir or args.enable_motion:
            parser.error("compile requires --run-dir and runs offline")
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
    if args.run_dir is not None:
        parser.error("--run-dir is used with compile")
    if args.mode is None:
        try:
            args.mode = input("Method (direct / cap / delegate / monitor) [direct]: ").strip() or "direct"
            if args.mode not in ("direct", "cap", "delegate", "monitor"):
                parser.error("Choose direct, cap, delegate or monitor")
            if not args.config:
                station = input("Workstation JSON [mock]: ").strip()
                args.config = Path(station) if station else None
            args.task = args.task or input("Task: ").strip()
            backend = decision_settings()["decision_backend"]
            args.backend = args.backend or input(f"Decision backend (api / local / codex) [{backend}]: ").strip() or backend
            if args.backend in MODEL_KEYS:
                model = os.getenv(MODEL_KEYS[args.backend]) or ("CLI default" if args.backend == "codex" else "served model ID required")
                args.model = args.model or input(f"Model [{model}]: ").strip() or None
                effort = os.getenv("CLEANRUA_REASONING_EFFORT", "default")
                args.reasoning_effort = args.reasoning_effort or input(f"Reasoning effort [{effort}]: ").strip() or None
            if args.mode in ("delegate", "monitor") and not args.policy_config:
                policy_path = input("Action policy JSON [workstation policy]: ").strip()
                args.policy_config = Path(policy_path) if policy_path else None
            if args.config and not args.enable_motion:
                args.enable_motion = input(f"Enable motion for {args.config} and task {args.task!r}? [y/N]: ").strip().lower() == "y"
        except EOFError:
            parser.error("Supply a method and --task for non-interactive runs")
    if args.out is None:
        name = args.mode.replace(":", "_")
        args.out = args.output_dir / f"{name}__{time.time_ns()}"
    experiment = args.mode not in ("check", "replay")
    if experiment and (not (args.task or "").strip() or (args.config and not args.enable_motion)):
        parser.error("Model experiments require --task and live experiments also require --enable-motion")
    if args.mode == "replay" and not args.policy:
        parser.error("replay requires a trusted --policy with main(robot)")
    if args.mode == "check" and args.enable_motion:
        parser.error("check is read-only")
    if experiment:
        try:
            configure_decision(args.backend, args.model, args.reasoning_effort, args.base_url)
        except ValueError as exc:
            parser.error(str(exc))
        settings = decision_settings()
        if args.mode in ("direct", "cap", "delegate", "monitor") and settings["decision_backend"] in ("api", "local") and not settings["model"]:
            parser.error("Set the backend’s default model ID or pass --model")
        print(json.dumps({"task": args.task, "mode": args.mode, "station": str(args.config or "mock"),
                          "out": str(args.out), **settings}, ensure_ascii=False), flush=True)
    config = json.loads(args.config.read_text()) if args.config else None
    needs_policy = experiment and "policy" in inspect.signature(resolve_algorithm(args.mode)).parameters
    policy = load_policy(config, args.policy_config) if needs_policy else None
    args.out.mkdir(parents=True, exist_ok=False)
    bench = workstation(config, args.mode != "check" and (not args.config or args.enable_motion))
    try:
        if experiment:
            result = run_method(bench, args.task, args.out, algorithm=args.mode, policy=policy, max_decisions=args.max_decisions)
        elif args.mode == "replay":
            result = runpy.run_path(str(args.policy))["main"](bench)
        else:
            obs = bench.observe()
            for name, image in obs.pop("images").items():
                Image.fromarray(image).save(args.out / f"{name}.png")
            result = obs
        report = {"backend": "configured" if args.config else "mock", "mode": args.mode,
                  "result": result, "task_success": None}
        if experiment and (args.out / "result.json").is_file():
            report.update(json.loads((args.out / "result.json").read_text()))
        output = json.dumps(report, indent=2, allow_nan=False)
        (args.out / "result.json").write_text(output + "\n")
        print(output)
    finally:
        bench.close()


if __name__ == "__main__":
    main()
