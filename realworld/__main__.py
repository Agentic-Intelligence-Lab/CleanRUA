"""Run an upstream CleanRUA method with the environment selected by the station."""
import argparse
import json
import os
import time
from functools import partial
from importlib import import_module
from itertools import islice
from pathlib import Path

from PIL import Image

from cleanrua import delegate, direct, monitor


def resolve(reference):
    module, name = reference.split(":")
    return getattr(import_module(module), name)


def load_policy(path, limit):
    spec = json.loads(path.read_text())
    policy = resolve(spec["driver"])(**spec.get("args", {}))
    predict = getattr(policy, "predict", policy)
    if not callable(predict):
        raise TypeError("Expected a callable action policy or a predict() method")

    def actions(obs, instruction):
        chunk = list(islice(predict(obs, instruction), limit))
        if not chunk:
            raise ValueError("Empty action chunk")
        return chunk
    return actions


def named_tools(tools, capabilities):
    original = next(tool["function"] for tool in tools if tool["function"]["name"] == "move")
    target = {"type": "object", "additionalProperties": False, "required": ["arm", "pose"],
              "properties": {"arm": {"type": "string", "enum": list(capabilities["arms"])},
                             **original["parameters"]["properties"]}}
    move = {"type": "function", "function": {
        "name": "move", "description": "Move named TCPs in the station frame (XYZ metres, fixed-axis XYZ radians). "
                                      "Omit gripper to retain its opening. Station limits: " + json.dumps(capabilities),
        "parameters": {"type": "object", "additionalProperties": False, "required": ["targets"],
                       "properties": {"targets": {"type": "array", "items": target, "minItems": 1,
                                                  "maxItems": capabilities["max_targets"]}}}}}
    return [move if t["function"]["name"] == "move" else t for t in tools]


def add_model_options(parser):
    parser.add_argument("--model", default=os.environ.get("CLEANRUA_MODEL_NAME"))
    parser.add_argument("--reasoning-effort", default=os.environ.get("CLEANRUA_REASONING_EFFORT"))
    parser.add_argument("--tool-choice", choices=("required", "auto"),
                        default=os.environ.get("CLEANRUA_TOOL_CHOICE", "required"))


def model_decider(args, tools):
    if args.tool_choice not in ("required", "auto"):
        raise ValueError("Tool choice must be required or auto")
    return partial(direct.mllm, tools=tools, model=args.model, reasoning_effort=args.reasoning_effort,
                   tool_choice=args.tool_choice)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("direct", "cap", "delegate", "monitor"))
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-decisions", type=int, default=50)
    parser.add_argument("--policy-config", type=Path, help="VLA driver and args for Delegate or Monitor")
    parser.add_argument("--replay", type=Path, help="Reviewed CaP controller")
    parser.add_argument("--memory-dir", type=Path, help="CaP Explore memory for one task; defaults to this run")
    add_model_options(parser)
    args = parser.parse_args(argv)
    vla = args.method in ("delegate", "monitor")
    if args.replay and (args.method != "cap" or args.check or args.task):
        parser.error("--replay requires cap --enable-motion and takes no task")
    if args.memory_dir and (args.method != "cap" or args.replay or args.check):
        parser.error("--memory-dir applies to CaP learning")
    if args.policy_config and not vla:
        parser.error("--policy-config applies to delegate and monitor")
    if args.enable_motion and not args.replay:
        if not (args.task or "").strip() or args.max_decisions <= 0:
            parser.error("Motion requires --task and a positive decision budget")
        if vla and not args.policy_config:
            parser.error("Supply --policy-config for this method")
        if not args.model:
            parser.error("Set CLEANRUA_MODEL_NAME or pass --model before connecting to hardware")
        for key in ("CLEANRUA_BASE_URL", "CLEANRUA_API_KEY"):
            if not os.environ.get(key):
                parser.error(f"Set {key} before connecting to hardware")

    config = json.loads(args.config.read_text())
    env = resolve(config.get("env", "realworld.envs.franka_env:FrankaEnv"))
    out = args.out or Path("runs") / f"{args.method}_{time.time_ns()}"
    out.mkdir(parents=True, exist_ok=False)
    with env(config, enable_motion=args.enable_motion) as robot:
        if args.check:
            obs = robot.observe()
            images = {}
            for i, (name, rgb) in enumerate(obs["images"].items()):
                images[name] = f"camera_{i}.png"
                Image.fromarray(rgb).save(out / images[name])
            result = {"state": obs["state"], "images": images, "capabilities": robot.capabilities}
        elif args.method == "cap":
            from realworld.cap import run
            result = run(robot, args.task, args, out)
        else:
            module = {"direct": direct, "delegate": delegate, "monitor": monitor}[args.method]
            decide = model_decider(args, named_tools(module.TOOLS, robot.capabilities))
            policy = [load_policy(args.policy_config, module.ACTIONS_PER_CHUNK)] if vla else []
            reason = getattr(module, args.method)(robot, *policy, args.task, max_decisions=args.max_decisions,
                                                  decide=decide)
            result = {"stop_reason": reason, "task_success": None}
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(out / "result.json")


if __name__ == "__main__":
    main()
