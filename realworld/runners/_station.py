"""Station setup for the Franka real-world runners.

Station JSON contains one entry in arms and named cameras. The arm's driver
("module:Class" and constructor args) provides state() with measured pose and
seven joints, blocking move_pose(pose) and stop(). The station owns its planner,
controller connection, model, joint order, payload, tool calibration and
trajectory validation. A cuRoboV2-Fast deployment selects its own arm adapter
through this configuration. Install the site modules or add them to PYTHONPATH.

Use end_effector="gripper". An integrated gripper supplies state()["gripper"]
and set_gripper(opening); otherwise put a separate driver spec in the arm's tool
field with state(), set_gripper() and stop(). Cameras map names to local video
devices (OpenCV) or site drivers providing read() and close().

    python -m realworld.runners.direct_realworld_franka --config station.json --check
    python -m realworld.runners.direct_realworld_franka --config station.json --enable-motion --task "Stack the blocks"

Within the arm profile, workspace_m is [[xmin,ymin,zmin],[xmax,ymax,zmax]] in the arm base frame.
Use metres for translation and radians for rotation.
Set CLEANRUA_BASE_URL, CLEANRUA_MODEL_NAME and CLEANRUA_API_KEY for a
vision/tool-capable Chat Completions endpoint. finish is a model decision.
"""
import argparse
import json
from itertools import islice
import os
from pathlib import Path
import time

from PIL import Image

from realworld.envs.franka_env import FrankaEnv


def load_policy(path, chunk_limit):
    """Site factory returns policy(observation, instruction) for the upstream VLA methods."""
    from importlib import import_module
    spec = json.loads(path.read_text())
    module, cls = spec["driver"].split(":")
    policy = getattr(import_module(module), cls)(**spec.get("args", {}))
    predict = getattr(policy, "predict", policy)
    if not callable(predict):
        raise TypeError("Expected a callable action policy or a predict() method")

    def actions(obs, instruction):
        # Each low-level step is an absolute base-frame TCP [XYZ, RPY] and opening.
        chunk = list(islice(predict(obs, instruction), chunk_limit))
        if not chunk:
            raise ValueError("Empty action chunk")
        return chunk
    return actions


def run_cli(method, run, argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-decisions", type=int, default=50)
    if method in ("delegate", "monitor"):
        parser.add_argument("--policy-config", type=Path, help="JSON with driver and args; returns a sequence of (pose, opening)")
    if method == "cap":
        parser.add_argument("--replay", type=Path, help="Run a reviewed main(robot) in a new isolated session, without a model")
    args = parser.parse_args(argv)
    replay = getattr(args, "replay", None)
    if replay and (args.check or args.task):
        parser.error("--replay requires --enable-motion and takes no task")
    if args.enable_motion and not replay:
        if not (args.task or "").strip() or args.max_decisions <= 0:
            parser.error("Motion requires --task and a positive decision budget")
        if method in ("delegate", "monitor") and not args.policy_config:
            parser.error("Supply --policy-config for this method")
        for key in ("CLEANRUA_BASE_URL", "CLEANRUA_MODEL_NAME", "CLEANRUA_API_KEY"):
            if not os.environ.get(key):
                parser.error(f"Set {key} before connecting to hardware")
    config = json.loads(args.config.read_text())
    out = args.out or Path("runs") / f"{method}_franka_{time.time_ns()}"
    out.mkdir(parents=True, exist_ok=False)
    with FrankaEnv(config, enable_motion=args.enable_motion) as robot:
        if args.check:
            obs = robot.observe()
            images = {}
            for i, (name, rgb) in enumerate(obs["images"].items()):
                images[name] = f"camera_{i}.png"
                Image.fromarray(rgb).save(out / images[name])
            result = {"state": obs["state"], "images": images}
        else:
            limits = {"workspace_m": robot.bounds.tolist(), **robot.limits}
            instruction = f"{args.task}\nStation TCP limits: {json.dumps(limits)}. Plan within each single-step limit."
            result = run(robot, instruction, args, out)
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(out / "result.json")
