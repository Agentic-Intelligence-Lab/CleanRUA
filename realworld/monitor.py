"""Real-workstation Monitor agent."""
from typing import Annotated

from pydantic import Field

from cleanrua.monitor import SYSTEM as UPSTREAM_SYSTEM
from cleanrua_utils.tools import Tool
from realworld.agent import MotionEnv
from realworld.episode import run_episode


SYSTEM = UPSTREAM_SYSTEM + " Use finish when the task is complete. Gripper opening is 0..1; hand joints are in radians."


class RealWrapper(MotionEnv):
    def __init__(self, env, policy, task):
        super().__init__(env)
        self.policy, self.task = policy, task

    @Tool
    def run_policy(self, steps: Annotated[int, Field(ge=1, le=300)]):
        """Continue the original episode task for at most steps control commands."""
        return self.policy.run_observed(self.env, self.task, steps, trace=self.policy_trace)


def run_monitor(bench, task, run_dir, *, policy, decide=None, max_decisions=30):
    env = RealWrapper(bench, policy, task)
    report = run_episode(env, SYSTEM, task, run_dir, decide, max_decisions)
    return report["stop_reason"]


def main():
    from realworld.__main__ import module_main
    module_main("monitor")


if __name__ == "__main__":
    main()
