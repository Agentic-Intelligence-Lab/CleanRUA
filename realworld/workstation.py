"""Assemble existing drivers from one trusted workstation profile."""
import importlib
import re

from realworld.env import Bench


def device(spec):
    module, name = spec["driver"].split(":")
    if module in ("cleanrua_utils.real_policy", "realworld.real_policy"):
        module = "realworld.policy"
    if module in ("cleanrua_utils.bench_mock", "realworld.bench_mock", "examples.plug_and_play.pattern_camera",
                  "realworld.examples.plug_and_play.pattern_camera"):
        module = "realworld.tests.fixtures.devices"
    return getattr(importlib.import_module(module), name)(**spec.get("args", {}))


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
    devices = "realworld.tests.fixtures.devices"
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
