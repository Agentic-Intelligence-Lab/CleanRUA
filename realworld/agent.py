"""Real-workstation tools and their measured feedback."""
import inspect
import json
import string
from functools import partial
from typing import Annotated

import gymnasium as gym
from pydantic import Field, ValidationError

from cleanrua_utils.agent import request_tool
from cleanrua_utils.tools import Tool
from realworld.env import InvalidTarget
from realworld.model import configured_decider, mllm


class AgentEnv(gym.Wrapper):
    policy_trace = None

    def __init__(self, env):
        super().__init__(env)
        self.tools = {name: method for name, method in inspect.getmembers(type(self), inspect.isfunction)
                      if hasattr(method, "arguments")}
        self.grippers = [side for side, profile in env.profiles.items() if profile["end_effector"] == "gripper"]
        if not self.grippers:
            self.tools.pop("grip", None)
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
            if name == "grip":
                schema["properties"]["side"] = {"type": "string", "enum": self.grippers}
                if len(self.grippers) > 1:
                    schema["required"].append("side")
            tools.append({"type": "function", "function": {"name": name, "description": inspect.getdoc(method),
                                                          "parameters": schema}})
        return tools

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return {**observation, "feedback": ""}, info

    def step(self, action):
        name = action["name"]
        obs = None
        try:
            method = self.tools.get(name)
            if method is None:
                raise InvalidTarget(f"Unknown tool {name}")
            arguments = method.arguments.model_validate_json(action["arguments"])
            result = method(self, **arguments.model_dump())
            if isinstance(result, tuple):  # An action policy already acquired its next observation.
                result, obs = result
            finished = name == "finish"
        except (ValidationError, InvalidTarget) as exc:
            if not self.env.motion_enabled:
                raise
            result, finished = {"error": str(exc)}, False
        if obs is None:
            obs = self.env.observe()
        terminated = finished or result.get("stop_reason") == "terminated"
        truncated = result.get("stop_reason") == "truncated"
        return {**obs, "feedback": json.dumps(result)}, 0., terminated, truncated, {"result": result}

    @Tool
    def finish(self):
        """End the episode. Task success requires independent evidence."""
        return {}


class MotionEnv(AgentEnv):
    @Tool
    def move(self, pose: Annotated[list[float], Field(min_length=6, max_length=6)],
             side: str = None, gripper: Annotated[float, Field(ge=0, le=1)] = None,
             hand: list[float] = None):
        """Move one arm to an absolute end-effector pose. Omitted tool targets hold."""
        return {"measured": self.env.move(pose, gripper, side=side, hand=hand)}


class RUAgent:
    def __init__(self, task, system, tools, decide=None):
        self.task, self.system = task, system
        self.history = []
        self.decide = decide if decide is not None else configured_decider(tools)
        if self.decide is None:
            self.decide = partial(mllm, tools=tools)

    def get_action(self, images, state):
        message, call = request_tool(self.decide, self.system, self.task, self.history, 4,
                                     {"images": images, "state": state})
        self.history.append(message)
        return call

    def feedback(self, call, result):
        self.history.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
