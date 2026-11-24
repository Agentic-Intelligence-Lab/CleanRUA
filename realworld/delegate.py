# Delegate on real workstations, following cleanrua/delegate_dexjoco.py (CleanRUA, 1c61b497).
# Usage: python -m realworld.delegate --task "..." --out runs/example
from dataclasses import dataclass
import base64
import inspect
import io
import json
import os
import re
from pathlib import Path
import time
import string
from typing import Annotated, get_type_hints

import gymnasium as gym
import tyro
from PIL import Image
from pydantic import ConfigDict, Field, ValidationError, create_model

from realworld.decision_cli import configured_decider
from realworld.real_decision import configure_decision, decision_settings, mllm
from realworld.real_arm_env import InvalidTarget
from realworld.workstation import workstation
from realworld.real_policy import load_policy

SYSTEM = "Use actor_act to delegate a bounded phase to the action policy. Give one short instruction per phase; inspect feedback before the next. Use move for a precise correction. End with finish when the task is complete. Base-frame XYZ/RPY in metres/radians, gripper opening 0..1, hand joints in radians."

def Tool(method):
    """Expose a wrapper method to the model: its signature becomes the argument schema, its docstring the description."""
    hints = get_type_hints(method, include_extras=True)
    fields = {name: (hints[name], ... if p.default is p.empty else p.default)
              for name, p in inspect.signature(method).parameters.items() if name != 'self'}
    method.arguments = create_model(method.__name__, __config__=ConfigDict(extra='forbid', allow_inf_nan=False),
                                    **fields)
    return method


class RealWrapper(gym.Wrapper):
    def __init__(self, env, policy, task):
        super().__init__(env)
        self.policy_trace = None
        self.policy = policy
        self.tools = {name: method for name, method in inspect.getmembers(type(self), inspect.isfunction)
                      if hasattr(method, "arguments")}
        text = gym.spaces.Text(65536, min_length=0, charset=string.printable)
        self.action_space = gym.spaces.Dict({"name": text, "arguments": text})
        self.observation_space = gym.spaces.Dict({**env.observation_space.spaces, "feedback": text})

    def schemas(self):
        tools = []
        for name, method in self.tools.items():
            schema = method.arguments.model_json_schema()
            if name == "move":
                fields = schema["properties"]
                profiles = self.env.profiles
                fields["side"] = {"type": "string", "enum": list(profiles)}
                if len(profiles) > 1:
                    schema["required"].append("side")
                for key in ("gripper", "hand"):
                    if all(p["end_effector"] != key for p in profiles.values()):
                        del fields[key]
            tools.append({"type": "function", "function": {"name": name, "description": inspect.getdoc(method),
                                                          "parameters": schema}})
        return tools

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return {**observation, "feedback": ""}, info

    def step(self, action):
        name = action["name"]
        obs = None
        finished = False
        try:
            method = self.tools.get(name)
            if method is None:
                raise InvalidTarget(f"Unknown tool {name}")
            arguments = method.arguments.model_validate_json(action["arguments"])
            if name == "actor_act":
                result, obs = method(self, **arguments.model_dump())
            else:
                result = method(self, **arguments.model_dump())
            finished = name == "finish"
        except (ValidationError, InvalidTarget) as exc:
            if not self.env.motion_enabled:
                raise
            result = {"error": str(exc)}
        if obs is None:
            obs = self.env.observe()
        terminated = finished or result.get("stop_reason") == "terminated"
        truncated = result.get("stop_reason") == "truncated"
        return {**obs, "feedback": json.dumps(result)}, 0., terminated, truncated, {"result": result}

    @Tool
    def actor_act(self, instruction: Annotated[str, Field(min_length=1)],
                  steps: Annotated[int, Field(ge=1, le=300)], active_groups: list[str] | None = None):
        """Give the policy a phase instruction for at most steps control commands on the selected arms."""
        return self.policy.run_observed(self.env, instruction, steps, active_groups, trace=self.policy_trace)

    @Tool
    def finish(self):
        """End the episode. Task success requires independent evidence."""
        return {}

    @Tool
    def move(self, pose: Annotated[list[float], Field(min_length=6, max_length=6)],
             side: str = None, gripper: Annotated[float, Field(ge=0, le=1)] = None,
             hand: list[float] = None):
        """Move one arm to an absolute end-effector pose. Omitted tool targets hold."""
        return {"measured": self.env.move(pose, gripper, side=side, hand=hand)}


def look(observation: dict) -> dict:
    """The current state and images as one user message."""
    content = [{'type': 'text', 'text': 'State: ' + json.dumps(observation['state'])}]
    for camera, image in observation['images'].items():
        png = io.BytesIO()
        Image.fromarray(image).save(png, format='PNG')
        content += [{'type': 'text', 'text': f'Camera {camera}:'}, {'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(png.getvalue()).decode()}}]
    return {'role': 'user', 'content': content}


class RUAgent:
    def __init__(self, task, system, tools, decide=None):
        self.task, self.system, self.tools = task, system, tools
        self.history = []
        self.decide = decide if decide is not None else configured_decider(tools)

    def get_action(self, images, state):
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": self.task},
                    *self.history[-8:], look({"images": images, "state": state})]
        message = self.decide(messages) if self.decide is not None else mllm(messages, self.tools)
        call = message["tool_calls"][0]
        self.history.append(message)
        return call

    def feedback(self, call, result):
        self.history.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})


def run_delegate(bench, task, run_dir, *, policy, decide=None, max_decisions=30):
    env = RealWrapper(bench, policy, task)
    public = {side: {k: v for k, v in p.items() if k not in ("driver", "args", "tool")}
              for side, p in bench.profiles.items()}
    agent = RUAgent(task, SYSTEM + " Station: " + json.dumps(public), env.schemas(), decide)
    with (run_dir / "trajectory.jsonl").open("x") as log:
        def record(**event):
            log.write(json.dumps(event, allow_nan=False, default=lambda array: array.tolist()) + "\n")
            log.flush()

        started = time.monotonic()
        decisions = 0
        try:
            obs, _ = env.reset()
            reason = "max_decisions"
            for turn in range(max_decisions):
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
                if done or truncated:
                    reason = info["result"].get("stop_reason", "finish")
                    record(turn=turn, feedback=info["result"])
                    break
                agent.feedback(call, info["result"])
                record(turn=turn, feedback=info["result"])
            record(stop=reason, task_success=None)
            report = {**({"decision_backend": "injected"} if decide is not None else decision_settings()), "task": task,
                      "stop_reason": reason, "decisions": decisions, "agent_finished": reason == "finish",
                      "elapsed_s": round(time.monotonic() - started, 3), "task_success": None}
            (run_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            return reason
        except Exception as exc:
            record(fault=str(exc))
            raise


@dataclass
class Args:
    task: str
    backend: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    base_url: str | None = None
    output_dir: Path = Path("runs")
    out: Path | None = None  # An explicit, new run directory (legacy CLI).
    config: Path | None = None
    policy_config: Path | None = None  # Otherwise use the policy section of config.
    enable_motion: bool = False
    max_decisions: int = 30


def main():
    args = tyro.cli(Args)
    configure_decision(args.backend, args.model, args.reasoning_effort, args.base_url)
    if args.config and not args.enable_motion:
        raise SystemExit("Configured device experiments require --enable-motion")
    config = json.loads(args.config.read_text()) if args.config else None
    policy = load_policy(config, args.policy_config)
    task_name = re.sub(r"[^\w-]+", "_", args.task).strip("_")[:64] or "task"
    run_dir = args.out or args.output_dir / f"{task_name}__delegate_real__0__{time.time_ns()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    bench = workstation(config, True)
    try:
        reason = run_delegate(bench, args.task, run_dir, policy=policy, max_decisions=args.max_decisions)
        print(json.dumps({"run_dir": str(run_dir), "stop_reason": reason}))
    finally:
        bench.close()


if __name__ == "__main__":
    main()
