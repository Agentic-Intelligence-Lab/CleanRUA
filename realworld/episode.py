"""Run a workstation agent episode and record observable actions and receipts."""
import json
from itertools import count
import time

from PIL import Image

from realworld.agent import RUAgent
from realworld.model import decision_settings


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
