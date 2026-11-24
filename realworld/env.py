"""Named, base-frame end-effector targets for configurable workstations."""
import time
import warnings

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation


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
                    if np.linalg.norm(actual[:3]-pose[:3]) > 0.002 or error.magnitude() > 0.05:
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
