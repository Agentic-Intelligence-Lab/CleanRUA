"""Run CleanRUA on a configurable workstation. Run from the repository root.

    python -m realworld check --config station.json
    python -m realworld calibrate-workspace --config station.json --arm left
    python -m realworld cap --task "Stack the blocks" --backend api --config station.json --enable-motion
    python -m realworld compile --run-dir RUN_DIR --config station.json --backend api
    python -m realworld replay --policy RUN_DIR/controller.py --config station.json --enable-motion

Install numpy, scipy, pillow, gymnasium, httpx and jsonschema.
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
driver modules importable on PYTHONPATH. A configured workstation is required for acquisition and execution.
"""
import argparse
import ast
from copy import deepcopy
import base64
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
import subprocess
import sys
import tempfile
import time
import warnings

import gymnasium as gym
from gymnasium import spaces
import httpx
import numpy as np
from PIL import Image
from jsonschema import Draft7Validator, ValidationError
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


# Workstation configuration


def device(spec):
    module, name = spec["driver"].split(":")
    if module in ("realworld.policy", "realworld.real_policy", "cleanrua_utils.real_policy") and name == "HTTPArmPolicy":
        return HTTPArmPolicy(**spec.get("args", {}))
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


def require_methods(driver, methods, label):
    missing = [m for m in methods if not callable(getattr(driver, m, None))]
    if missing:
        raise ValueError(f"{label} driver must implement {', '.join(missing)}")


def workstation(config, motion_enabled=False):
    arms, cameras = {}, {}
    if not config or "arms" not in config:
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


MODEL_KEYS = {"api": "CLEANRUA_MODEL_NAME", "local": "CLEANRUA_LOCAL_MODEL"}


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
        raise ValueError("Choose api, local or module:factory")
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
    module, sep, factory = name.partition(":")
    if not sep:
        raise ValueError("Decision backend must be api, local or a trusted module:factory")
    decide = getattr(import_module(module), factory)(tools)
    if not callable(decide):
        raise TypeError("Decision backend factory must return a callable")
    return decide


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


# The model sees the upstream method's tools, extended for the configured arms.


METHODS = {"direct": direct, "cap": cap, "delegate": delegate, "monitor": monitor}


def tool_schemas(method, profiles):
    tools = deepcopy(METHODS[method].TOOLS)
    sides = list(profiles)
    grippers = [s for s in sides if profiles[s]["end_effector"] == "gripper"]
    side_schema = {"type": "string", "enum": sides}
    if method == "direct" and grippers:
        tools.append({"type": "function", "function": {"name": "grip", "description": "Set only the gripper opening.",
                      "parameters": {"type": "object", "required": ["opening"],
                                     "properties": {"opening": {"type": "number", "minimum": 0, "maximum": 1},
                                                    "side": {"type": "string", "enum": grippers}}}}})
    for tool in tools:
        fn = tool["function"]
        schema = fn["parameters"]
        schema["additionalProperties"] = False
        fields = schema["properties"]
        if fn["name"] == "move":
            schema["required"] = ["pose"]
            fields["side"] = side_schema
            if not grippers:
                fields.pop("gripper", None)
            if any(p["end_effector"] == "hand" for p in profiles.values()):
                fields["hand"] = {"type": "array", "items": {"type": "number"}}
            if len(sides) > 1:
                schema["required"].append("side")
        elif fn["name"] == "grip" and len(grippers) > 1:
            schema["required"].append("side")
        elif fn["name"] == "vla":
            fn["name"] = "actor_act"
            fields["instruction"]["minLength"] = 1
            fields["active_groups"] = {"type": "array", "minItems": 1, "uniqueItems": True,
                                       "items": side_schema}
    return tools


def public_profiles(profiles):
    return {side: {k: v for k, v in p.items() if k not in ("driver", "args", "tool")}
            for side, p in profiles.items()}


def run_method(bench, task, run_dir, *, algorithm="direct", decide=None, max_decisions=30, policy=None):
    if algorithm not in METHODS or max_decisions < 0:
        raise ValueError("Expected a supported method and nonnegative decision budget")
    if algorithm in ("delegate", "monitor") and policy is None:
        raise ValueError("Configure the action policy for this method")
    profiles = public_profiles(bench.profiles)
    tools = tool_schemas(algorithm, profiles)
    schemas = {t["function"]["name"]: Draft7Validator(t["function"]["parameters"]) for t in tools}
    backend = decide if decide is not None else configured_decider(tools)
    system = METHODS[algorithm].SYSTEM + (
        " Use the latest measured base-frame TCP. Respect the configured workspace, total 3D "
        "translation and relative rotation limits. Use smaller steps near objects. Station: "
        + json.dumps(profiles))
    if algorithm == "cap":
        system += (" Use robot.observe() for named images and arms. Develop reusable functions based on "
                   "live observations; a separate controller will be compiled from this episode.")
    grippers = [side for side, p in profiles.items() if p["end_effector"] == "gripper"]
    history, program = [], None
    started = time.monotonic()
    decisions, reason = 0, "max_decisions"
    with (run_dir / "trajectory.jsonl").open("x") as log:
        def record(**event):
            log.write(json.dumps(event, allow_nan=False, default=lambda value: value.tolist()) + "\n")
            log.flush()

        try:
            obs = bench.observe()
            for turn in (count() if max_decisions == 0 else range(max_decisions)):
                images = {}
                for name, rgb in obs["images"].items():
                    images[name] = f"{name}-{turn:03d}.png"
                    Image.fromarray(rgb).save(run_dir / images[name])
                record(turn=turn, task=task, images=images, arms=obs["arms"],
                       rgb_received_ns=obs["rgb_received_ns"], state_received_ns=obs["state_received_ns"])
                messages = [{"role": "system", "content": system}, {"role": "user", "content": task},
                            *history[-8:], look({"images": obs["images"], "state": obs["arms"]})]
                reply = backend(messages) if backend is not None else mllm(messages, tools)
                call = reply["tool_calls"][0]
                fn = call["function"]
                name = fn["name"]
                decisions += 1
                record(turn=turn, request=fn)
                obs = None
                try:
                    if name not in schemas:
                        raise InvalidTarget(f"Unknown tool {name}")
                    args = json.loads(fn["arguments"])
                    schemas[name].validate(args)
                    if name == "finish":
                        reason = "finish"
                        break
                    if name == "move":
                        result = {"measured": bench.move(**args)}
                    elif name == "grip":
                        side = args.get("side", grippers[0])
                        obs, _, _, _, result = bench.step_groups({side: {"gripper": args["opening"]}}, [side])
                    elif name == "run_code":
                        if program is None:
                            program = ProgramSession(bench)
                        result = program.execute(args["code"])
                    else:
                        instruction = args["instruction"] if name == "actor_act" else task
                        result, obs = policy.run_observed(bench, instruction, args["steps"], args.get("active_groups"),
                                                         trace=lambda **event: record(turn=turn, **event))
                except (ValidationError, InvalidTarget, json.JSONDecodeError) as exc:
                    if not bench.motion_enabled:
                        raise
                    result = {"error": str(exc)}
                record(turn=turn, feedback=result)
                if result.get("stop_reason") in ("terminated", "truncated"):
                    reason = result["stop_reason"]
                    break
                history.extend([reply, {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}])
                if obs is None:
                    obs = bench.observe()
            record(stop=reason, task_success=None, station=profiles)
        except Exception as exc:
            record(fault=str(exc))
            raise
        finally:
            if program is not None:
                program.close()

    report = {**({"decision_backend": "injected"} if decide is not None else decision_settings()), "task": task,
              "stop_reason": reason, "decisions": decisions, "agent_finished": reason == "finish",
              "elapsed_s": round(time.monotonic() - started, 3), "task_success": None}
    try:
        if algorithm == "cap":
            report["controller"] = compile_controller(run_dir, profiles, decide=decide).name
            report["controller_verified"] = False
    except Exception as exc:
        report["controller_error"] = str(exc)
        raise
    finally:
        (run_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return reason


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


# Command line


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=[*METHODS, "check", "calibrate-workspace", "compile", "replay"])
    parser.add_argument("--config", type=Path, help="Trusted workstation JSON with arm and camera drivers")
    parser.add_argument("--arm", choices=("left", "right"), help="Arm to sample with calibrate-workspace")
    parser.add_argument("--enable-motion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out", type=Path, help="New directory for this run")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--max-decisions", type=int, default=30, help="Maximum decisions; 0 means unlimited")
    parser.add_argument("--policy-config", type=Path, help="Action policy client configuration")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--backend", help="api, local or a trusted module:factory")
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
    if args.config is None:
        parser.error("Supply --config with your workstation profile")
    args.out = args.out or args.output_dir / f"{args.mode}__{time.time_ns()}"
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
        if settings["decision_backend"] in ("api", "local") and not settings["model"]:
            parser.error("Set the backend’s default model ID or pass --model")
        print(json.dumps({"task": args.task, "mode": args.mode, "station": str(args.config or "mock"),
                          "out": str(args.out), **settings}, ensure_ascii=False), flush=True)
    config = json.loads(args.config.read_text())
    needs_policy = args.mode in ("delegate", "monitor")
    policy = load_policy(config, args.policy_config) if needs_policy else None
    args.out.mkdir(parents=True, exist_ok=False)
    bench = workstation(config, args.enable_motion)
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
        report = {"backend": "configured", "mode": args.mode,
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
