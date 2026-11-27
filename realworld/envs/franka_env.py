"""Named Franka TCPs in a shared station frame. Site drivers plan and execute in each arm base."""
from contextlib import ExitStack
from importlib import import_module
import math
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from cleanrua_utils.motion import TrajectoryRejected


def _array(value, shape):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"Expected finite numeric data of shape {shape}")
    return a.astype(float)


def _plugin(spec, **bindings):
    module, cls = spec["driver"].split(":")
    return getattr(import_module(module), cls)(**spec.get("args", {}), **bindings)


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


class _InvalidTarget(ValueError):
    """Invalid command before dispatch; feedback failures propagate separately."""


class _Arm:
    def __init__(self, profile, load, enable_motion, dual):
        if dual and "station_T_base" not in profile:
            raise ValueError("Dual arms each require calibrated station_T_base [xyz, rpy]")
        self.base = _array(profile.get("station_T_base", [0] * 6), (6,))
        self.rotation = Rotation.from_euler("xyz", self.base[3:])
        if profile["end_effector"] != "gripper":
            raise ValueError("These Franka runners expect a parallel gripper")
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
        self.arm = load(profile, ("state", "move_pose", "stop") if enable_motion else ("state",))
        if "tool" in profile:
            self.gripper = load(profile["tool"], ("state", "set_gripper", "stop") if enable_motion else ("state",))
        else:
            self.gripper = self.arm
            if enable_motion and not callable(getattr(self.arm, "set_gripper", None)):
                raise ValueError("Supply an external tool or implement set_gripper()")

    def transform(self, pose, *, to_base):
        if not np.any(self.base):
            return pose.copy()
        xyz, r = pose[:3], Rotation.from_euler("xyz", pose[3:])
        if to_base:
            inverse = self.rotation.inv()
            return np.r_[inverse.apply(xyz - self.base[:3]), (inverse * r).as_euler("xyz")]
        return np.r_[self.base[:3] + self.rotation.apply(xyz), (self.rotation * r).as_euler("xyz")]

    def _devices(self):
        return (self.arm,) if self.gripper is self.arm else (self.arm, self.gripper)

    def base_state(self):
        raw = self.arm.state()
        tool_state = raw if self.gripper is self.arm else self.gripper.state()
        opening = float(_array(tool_state["gripper"], ()))
        if not 0 <= opening <= 1:
            raise ValueError("Invalid measured gripper opening")
        return {"pose": _array(raw["pose"], (6,)).tolist(),
                "joints": _array(raw["joints"], (7,)).tolist(), "gripper": opening}

    def _allowed(self, start, goal):
        low, high = self.bounds
        rotation = Rotation.from_euler("xyz", start[3:]).inv() * Rotation.from_euler("xyz", goal[3:])
        return (np.all(start[:3] >= low) and np.all(start[:3] <= high)
                and np.all(goal[:3] >= low) and np.all(goal[:3] <= high)
                and np.linalg.norm(goal[:3] - start[:3]) <= self.limits["max_translation_m"]
                and rotation.magnitude() <= self.limits["max_rotation_rad"])

    def goal(self, target, current):
        if not isinstance(target, dict) or not {"arm", "pose"} <= target.keys() <= {"arm", "pose", "gripper"}:
            raise ValueError("Expected an arm, pose and optional gripper")
        goal = self.transform(_array(target["pose"], (6,)), to_base=True)
        opening = target.get("gripper")
        if "gripper" in target and not 0 <= float(_array(opening, ())) <= 1:
            raise ValueError("Invalid gripper opening")
        if not self._allowed(np.array(current["pose"]), goal):
            raise ValueError("Target outside workstation limits")
        return goal, opening

    def verify_arrival(self, goal):
        actual = np.array(self.base_state()["pose"])
        rotation = Rotation.from_euler("xyz", goal[3:]).inv() * Rotation.from_euler("xyz", actual[3:])
        if (np.linalg.norm(actual[:3] - goal[:3]) > self.limits["arrival_translation_m"]
                or rotation.magnitude() > self.limits["arrival_rotation_rad"]):
            raise RuntimeError("TCP outside arrival tolerance")


class FrankaEnv:
    def __init__(self, config, *, enable_motion=False):
        profiles = config["arms"]
        if not profiles or not set(profiles) <= {"left", "right"}:
            raise ValueError("Configure left and/or right Franka arms")
        if not config.get("cameras"):
            raise ValueError("Configure at least one named RGB camera")
        self.camera_calibration = config.get("camera_calibration", {})
        if not self.camera_calibration.keys() <= config["cameras"].keys():
            raise ValueError("Calibration must reference configured camera names")
        self.motion_enabled = enable_motion
        with ExitStack() as resources:
            def load(spec, methods, **bindings):
                device = _plugin(spec, **bindings)
                if callable(getattr(device, "close", None)):
                    resources.callback(device.close)
                for method in methods:
                    if not callable(getattr(device, method, None)):
                        raise ValueError(f"{spec['driver']} requires {method}()")
                return device

            self.arms = {name: _Arm(profile, load, enable_motion, len(profiles) == 2)
                         for name, profile in profiles.items()}
            self.coordinator = None
            if enable_motion and "coordinator" in config:
                if len(profiles) != 2:
                    raise ValueError("A coordinator requires two arms")
                self.coordinator = load(config["coordinator"], ("move_poses", "stop"),
                                        arms={name: arm.arm for name, arm in self.arms.items()})
            self.devices = [d for arm in self.arms.values() for d in arm._devices()]
            if self.coordinator is not None:
                self.devices.append(self.coordinator)
            for device in self.devices:
                if enable_motion and callable(getattr(device, "check_motion_ready", None)):
                    device.check_motion_ready()
            self.cameras = {}
            for name, spec in config["cameras"].items():
                camera = _plugin(spec) if isinstance(spec, dict) else _USBCamera(spec)
                resources.callback(camera.close)
                self.cameras[name] = camera
            self.resources = resources.pop_all()

        self.capabilities = {
            "pose_frame": "station", "workspace_frame": "arm_base",
            "max_targets": 2 if len(profiles) == 2 and "coordinator" in config else 1,
            "arms": {name: {"station_T_base": arm.base.tolist(),
                            "workspace_m": None if arm.bounds is None else arm.bounds.tolist(), **arm.limits}
                     for name, arm in self.arms.items()},
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.motion_enabled = False
        self.resources.close()

    def state(self):
        measured = {}
        for name, arm in self.arms.items():
            state = arm.base_state()
            state["pose"] = arm.transform(np.array(state["pose"]), to_base=False).tolist()
            if arm.last_gripper_receipt is not None:
                state["last_gripper_receipt"] = arm.last_gripper_receipt
            measured[name] = state
        state = {"arms": measured}
        if self.camera_calibration:
            state["camera_calibration"] = self.camera_calibration
        if len(measured) == 1:  # Older single-arm policies read state["pose"] directly.
            state.update(next(iter(measured.values())))
        return state

    def observe(self):
        images = {}
        for name, camera in self.cameras.items():
            rgb = camera.read()
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
                raise ValueError(f"Invalid RGB frame from {name}")
            if name in self.camera_calibration:
                intrinsics = self.camera_calibration[name]["intrinsics"]
                if rgb.shape[:2] != (intrinsics["height"], intrinsics["width"]):
                    raise ValueError(f"RGB dimensions from {name} differ from its calibrated mode")
            images[name] = rgb
        return {"images": images, "state": self.state()}

    def _prepare(self, targets):
        if not self.motion_enabled:
            raise RuntimeError("Motion is disabled")
        if not isinstance(targets, list) or not 1 <= len(targets) <= self.capabilities["max_targets"]:
            raise _InvalidTarget("Unsupported number of targets for this workstation")
        commands = {}
        for target in targets:
            name = target.get("arm") if isinstance(target, dict) else None
            if not isinstance(name, str) or name not in self.arms or name in commands:
                raise _InvalidTarget("Each target must name a distinct configured arm")
            arm = self.arms[name]
            current = arm.base_state()
            try:
                pose, opening = arm.goal(target, current)
            except (TypeError, ValueError) as exc:
                raise _InvalidTarget(str(exc)) from exc
            commands[name] = (pose, opening, current)
        return commands

    def ik(self, *, targets):
        """Validate before returning a step. The site driver owns trajectory planning."""
        try:
            self._prepare(targets)
        except _InvalidTarget:
            return []
        return [{"targets": targets}]

    def step(self, action):
        if not isinstance(action, dict) and len(self.arms) == 1:
            pose, opening = action  # Compatibility with existing single-arm VLA chunks.
            action = {"targets": [{"arm": next(iter(self.arms)), "pose": pose, "gripper": opening}]}
        if not isinstance(action, dict) or set(action) != {"targets"}:
            raise ValueError("Expected a named targets action")
        commands = self._prepare(action["targets"])
        try:
            rejection = self._dispatch(commands)
        except BaseException as fault:
            self.motion_enabled = False
            for device in self.devices:
                try:
                    device.stop()
                except Exception as exc:
                    warnings.warn(f"Device stop failed: {exc}")
            if isinstance(fault, TrajectoryRejected):
                raise RuntimeError("Unconfirmed execution state; motion disabled") from fault
            raise
        if rejection is not None:
            raise rejection
        return False

    def _dispatch(self, commands):
        if len(commands) == 2:
            self.coordinator.move_poses({name: pose.tolist() for name, (pose, _, _) in commands.items()})
        else:
            name, (pose, _, before) = next(iter(commands.items()))
            if not np.allclose(before["pose"], pose, rtol=0, atol=1e-9):
                try:
                    self.arms[name].arm.move_pose(pose.tolist())
                except TrajectoryRejected as exc:
                    # Only this single-arm driver can attest that no command was dispatched.
                    exc.arm = name
                    return exc
        for name, (pose, _, _) in commands.items():
            self.arms[name].verify_arrival(pose)
        for name, (_, opening, before) in commands.items():
            if opening is not None and abs(opening - before["gripper"]) > .01:
                arm = self.arms[name]
                arm.last_gripper_receipt = arm.gripper.set_gripper(opening)
