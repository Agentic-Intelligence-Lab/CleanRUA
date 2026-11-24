"""Real-workstation Direct agent."""
from typing import Annotated

from pydantic import Field

from cleanrua.direct import SYSTEM as UPSTREAM_SYSTEM
from cleanrua_utils.tools import Tool
from realworld.agent import MotionEnv
from realworld.episode import run_episode
from realworld.env import InvalidTarget


SYSTEM = UPSTREAM_SYSTEM + (
    " For move, send an absolute base-frame XYZ/RPY pose "
    "in metres/radians. Before each move, compare the target to the selected arm's pose in "
    "the latest observation. Use that arm's Station limits: the 3D distance "
    "sqrt(dx^2 + dy^2 + dz^2) must be <= max_translation_m, the angle of the relative rotation "
    "must be <= max_rotation_rad, and the target XYZ must be inside workspace_m. "
    "Leave margin below both step limits. Split longer motions into smaller moves, reading the "
    "new measured pose before choosing each next target. Use smaller steps near objects. "
    "Gripper opening is 0..1; hand joints are in radians."
)


class RealWrapper(MotionEnv):
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
    env = RealWrapper(bench)
    report = run_episode(env, SYSTEM, task, run_dir, decide, max_decisions)
    return report["stop_reason"]


def main():
    from realworld.__main__ import module_main
    module_main("direct")


if __name__ == "__main__":
    main()
