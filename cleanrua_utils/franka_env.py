"""One Franka arm, a parallel gripper and named RGB cameras for CleanRUA Direct.

The Direct protocol calls ik() to prepare a path and step() to execute it. With
franky's CartesianMotion, the controller performs IK during execution; ik()
only checks the Cartesian target and never sends a device command.
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
        from franky import Gripper, RelativeDynamicsFactor, Robot

        self.motion_enabled = enable_motion
        self.last_gripper_receipt = None
        self.bounds = config.get("workspace_m")
        if self.bounds is not None:
            self.bounds = _array(self.bounds, (2, 3))
            if np.any(self.bounds[0] >= self.bounds[1]):
                raise ValueError("workspace_m requires increasing bounds on every axis")
        self.limits = {}
        for key, default in (("max_translation_m", None), ("max_rotation_rad", None),
                             ("arrival_translation_m", .002), ("arrival_rotation_rad", .05)):
            value = config.get(key, default)
            if value is None and not enable_motion:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Set a positive {key} in the station configuration")
            self.limits[key] = value
        if enable_motion and self.bounds is None:
            raise ValueError("Set workspace_m before enabling motion")
        if not config.get("cameras"):
            raise ValueError("Configure at least one named RGB camera")
        self.flange_T_tcp = config.get("flange_T_tcp")
        if self.flange_T_tcp is not None:
            t = self.flange_T_tcp = _array(self.flange_T_tcp, (4, 4))
            r = t[:3, :3]
            if (not np.allclose(t[3], [0, 0, 0, 1]) or not np.allclose(r.T @ r, np.eye(3))
                    or not np.isclose(np.linalg.det(r), 1)):
                raise ValueError("flange_T_tcp must be a rigid transform in metres")

        self.robot = Robot(config["host"])
        self.robot.relative_dynamics_factor = RelativeDynamicsFactor(velocity=.02, acceleration=.02, jerk=.01)
        self.external_gripper = config.get("gripper") is not None
        with ExitStack() as resources:
            self.gripper = _plugin(config["gripper"]) if self.external_gripper else Gripper(config["host"])
            if self.external_gripper:
                if hasattr(self.gripper, "close"):
                    resources.callback(self.gripper.close)
                for method in (("state", "set_gripper", "stop") if enable_motion else ("state",)):
                    if not callable(getattr(self.gripper, method, None)):
                        raise ValueError(f"External gripper requires {method}()")
                if enable_motion and hasattr(self.gripper, "check_motion_ready"):
                    self.gripper.check_motion_ready()
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

    def _ee_T_tcp(self, sample):
        if self.flange_T_tcp is None:
            return np.eye(4)  # Use the end-effector frame configured in Desk.
        return np.linalg.solve(sample.F_T_EE.matrix, self.flange_T_tcp)

    def state(self):
        sample = self.robot.state
        tcp = sample.O_T_EE.matrix @ self._ee_T_tcp(sample)
        pose = [*tcp[:3, 3], *Rotation.from_matrix(tcp[:3, :3]).as_euler("xyz")]
        opening = (self.gripper.state()["gripper"] if self.external_gripper
                   else self.gripper.width / self.gripper.max_width)
        if not 0 <= opening <= 1:
            raise ValueError("Invalid measured gripper opening")
        return {"pose": _array(pose, (6,)).tolist(), "joints": _array(sample.q, (7,)).tolist(),
                "gripper": float(opening)}

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
        from franky import Affine, CartesianMotion

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
                tcp = Affine(goal[:3], rotation.as_quat()).matrix
                ee = tcp @ np.linalg.inv(self._ee_T_tcp(self.robot.state))
                self.robot.move(CartesianMotion(Affine(ee[:3, 3], Rotation.from_matrix(ee[:3, :3]).as_quat())))
                actual = np.array(self.state()["pose"])
                if (np.linalg.norm(actual[:3] - goal[:3]) > self.limits["arrival_translation_m"]
                        or (rotation.inv() * Rotation.from_euler("xyz", actual[3:])).magnitude() > self.limits["arrival_rotation_rad"]):
                    raise RuntimeError("TCP outside arrival tolerance")
            if abs(opening - before["gripper"]) > .01:
                if self.external_gripper:
                    self.last_gripper_receipt = self.gripper.set_gripper(opening)
                elif not self.gripper.move(opening * self.gripper.max_width, .02):
                    raise RuntimeError("Franka Hand did not reach the requested opening")
        except BaseException:
            self.motion_enabled = False
            for device in (self.robot, self.gripper):
                try:
                    device.stop()
                except Exception as exc:
                    warnings.warn(f"Stop failed: {exc}")
            raise
        return False  # Physical task success is assessed separately from motion completion.
