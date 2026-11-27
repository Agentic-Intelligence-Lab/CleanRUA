"""Single-arm Franka environment shared by the real-world method runners.

The station arm driver reports calibrated TCP feedback and executes blocking pose
commands. Planning, trajectory validation and hardware transport belong to it.
"""
from contextlib import ExitStack
from importlib import import_module
import math
import warnings

import numpy as np
from scipy.spatial.transform import Rotation


def _array(value, shape):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"Expected finite numeric data of shape {shape}")
    return a.astype(float)


def _plugin(spec):
    module, cls = spec["driver"].split(":")
    return getattr(import_module(module), cls)(**spec.get("args", {}))


class _USBCamera:
    def __init__(self, device):
        import cv2
        self.stream = cv2.VideoCapture(device)
        if not self.stream.isOpened():
            self.close()
            raise RuntimeError(f"Cannot open camera {device!r}")

    def read(self):
        import cv2
        ok, bgr = self.stream.read()
        if not ok:
            raise RuntimeError("Camera returned no frame")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        self.stream.release()


class FrankaEnv:
    """Base-frame TCP XYZ (m), fixed-axis XYZ RPY (rad), gripper 0 closed to 1 open."""

    def __init__(self, config, *, enable_motion=False):
        arms = config["arms"]
        if len(arms) != 1:
            raise ValueError("Configure one Franka arm")
        profile = next(iter(arms.values()))
        if profile["end_effector"] != "gripper":
            raise ValueError("These Franka runners expect a parallel gripper")
        self.motion_enabled = enable_motion
        self.last_gripper_receipt = None
        self.bounds = profile.get("workspace_m")
        if self.bounds is not None:
            self.bounds = _array(self.bounds, (2, 3))
            if np.any(self.bounds[0] >= self.bounds[1]):
                raise ValueError("workspace_m requires increasing bounds on every axis")
        self.limits = {}
        for key, default in (("max_translation_m", None), ("max_rotation_rad", None),
                             ("arrival_translation_m", .002), ("arrival_rotation_rad", .05)):
            value = profile.get(key, default)
            if value is None and not enable_motion:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Set a positive {key} in the station configuration")
            self.limits[key] = value
        if enable_motion and self.bounds is None:
            raise ValueError("Set workspace_m before enabling motion")
        if not config.get("cameras"):
            raise ValueError("Configure at least one named RGB camera")
        with ExitStack() as resources:
            def load(spec, methods):
                device = _plugin(spec)
                if callable(getattr(device, "close", None)):
                    resources.callback(device.close)
                for method in methods:
                    if not callable(getattr(device, method, None)):
                        raise ValueError(f"{spec['driver']} requires {method}()")
                return device

            self.arm = load(profile, ("state", "move_pose", "stop") if enable_motion else ("state",))
            if "tool" in profile:
                self.gripper = load(profile["tool"], ("state", "set_gripper", "stop") if enable_motion else ("state",))
            else:
                self.gripper = self.arm
                if enable_motion and not callable(getattr(self.arm, "set_gripper", None)):
                    raise ValueError("Supply an external tool or implement set_gripper()")
            for device in self._devices():
                if enable_motion and callable(getattr(device, "check_motion_ready", None)):
                    device.check_motion_ready()
            self.cameras = {}
            for name, spec in config["cameras"].items():
                camera = _plugin(spec) if isinstance(spec, dict) else _USBCamera(spec)
                resources.callback(camera.close)
                self.cameras[name] = camera
            self.resources = resources.pop_all()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.motion_enabled = False
        self.resources.close()

    def _devices(self):
        return (self.arm,) if self.gripper is self.arm else (self.arm, self.gripper)

    def state(self):
        raw = self.arm.state()
        tool_state = raw if self.gripper is self.arm else self.gripper.state()
        opening = float(_array(tool_state["gripper"], ()))
        if not 0 <= opening <= 1:
            raise ValueError("Invalid measured gripper opening")
        return {"pose": _array(raw["pose"], (6,)).tolist(),
                "joints": _array(raw["joints"], (7,)).tolist(), "gripper": opening}

    def observe(self):
        images = {}
        for name, camera in self.cameras.items():
            rgb = camera.read()
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
                raise ValueError(f"Invalid RGB frame from {name}")
            images[name] = rgb
        state = self.state()
        if self.last_gripper_receipt is not None:
            state["last_gripper_receipt"] = self.last_gripper_receipt
        return {"images": images, "state": state}

    def _allowed(self, start, goal):
        low, high = self.bounds
        rotation = Rotation.from_euler("xyz", start[3:]).inv() * Rotation.from_euler("xyz", goal[3:])
        return (np.all(start[:3] >= low) and np.all(start[:3] <= high)
                and np.all(goal[:3] >= low) and np.all(goal[:3] <= high)
                and np.linalg.norm(goal[:3] - start[:3]) <= self.limits["max_translation_m"]
                and rotation.magnitude() <= self.limits["max_rotation_rad"])

    def ik(self, pose, gripper):
        """Prepare one absolute TCP command; return an empty path for an invalid target."""
        if not self.motion_enabled:
            raise RuntimeError("Motion is disabled")
        try:
            goal = _array(pose, (6,))
            opening = float(_array(gripper, ()))
        except (ValueError, TypeError):
            return []
        if not 0 <= opening <= 1 or not self._allowed(np.array(self.state()["pose"]), goal):
            return []
        return [(goal, opening)]

    def step(self, action):
        if not self.motion_enabled:
            raise RuntimeError("Motion is disabled")
        goal, opening = action
        goal, opening = _array(goal, (6,)), float(_array(opening, ()))
        before = self.state()
        start = np.array(before["pose"])
        if not 0 <= opening <= 1 or not self._allowed(start, goal):
            raise ValueError("Command outside workstation limits at dispatch")
        try:
            if not np.allclose(start, goal, rtol=0, atol=1e-9):
                rotation = Rotation.from_euler("xyz", goal[3:])
                self.arm.move_pose(goal.tolist())
                actual = np.array(self.state()["pose"])
                if (np.linalg.norm(actual[:3] - goal[:3]) > self.limits["arrival_translation_m"]
                        or (rotation.inv() * Rotation.from_euler("xyz", actual[3:])).magnitude() > self.limits["arrival_rotation_rad"]):
                    raise RuntimeError("TCP outside arrival tolerance")
            if abs(opening - before["gripper"]) > .01:
                self.last_gripper_receipt = self.gripper.set_gripper(opening)
        except BaseException:
            self.motion_enabled = False
            for device in self._devices():
                try:
                    device.stop()
                except Exception as exc:
                    warnings.warn(f"Device stop failed: {exc}")
            raise
        return False  # Physical task success is assessed separately from motion completion.
