"""Real-workstation Delegate agent."""
from typing import Annotated

from pydantic import Field

from cleanrua.delegate import SYSTEM as UPSTREAM_SYSTEM
from cleanrua_utils.tools import Tool
from realworld.agent import MotionEnv
from realworld.episode import run_episode


SYSTEM = UPSTREAM_SYSTEM + " Use actor_act to delegate a bounded phase to the configured policy. Gripper opening is 0..1; hand joints are in radians."


class RealWrapper(MotionEnv):
    def __init__(self, env, policy, task):
        super().__init__(env)
        self.policy, self.task = policy, task

    @Tool
    def actor_act(self, instruction: Annotated[str, Field(min_length=1)],
                  steps: Annotated[int, Field(ge=1, le=300)], active_groups: list[str] | None = None):
        """Give the policy a phase instruction for at most steps control commands on the selected arms."""
        return self.policy.run_observed(self.env, instruction, steps, active_groups, trace=self.policy_trace)


def run_delegate(bench, task, run_dir, *, policy, decide=None, max_decisions=30):
    env = RealWrapper(bench, policy, task)
    report = run_episode(env, SYSTEM, task, run_dir, decide, max_decisions)
    return report["stop_reason"]


def main():
    from realworld.__main__ import module_main
    module_main("delegate")


if __name__ == "__main__":
    main()
