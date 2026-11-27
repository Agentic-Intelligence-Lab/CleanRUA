"""Run upstream Direct on one Franka workstation.

Use a Linux real-time workstation with configured FCI access. Install NumPy,
SciPy, Pillow, httpx and the franky-control build matching the robot's firmware.
Install opencv-python-headless for local video devices.

Create a station JSON with your FCI host and named cameras, for example:
    {"host": "YOUR_FCI_HOST", "cameras": {"scene": 0}}
Integer indices or device paths use OpenCV. A camera may instead specify
{"driver": "module:Class", "args": {...}} with read() returning uint8 RGB and
close(). The default tool is Franka Hand; an external gripper uses the same
driver specification in "gripper", with state()["gripper"], set_gripper() and
stop(). Install site drivers or add their directory to PYTHONPATH.

    python -m cleanrua.direct_realworld_franka --config station.json --check
    python -m cleanrua.direct_realworld_franka --config station.json --enable-motion --task "Stack the blocks"

Before motion, supply workspace_m = [[xmin,ymin,zmin],[xmax,ymax,zmax]],
max_translation_m and max_rotation_rad from your station. The default TCP is
the Desk EE. Set flange_T_tcp to your calibrated 4x4 transform to override it.
Set CLEANRUA_BASE_URL, CLEANRUA_MODEL_NAME and CLEANRUA_API_KEY for a
vision/tool-capable Chat Completions endpoint. finish is a model decision.
"""
import argparse
import json
import os
from pathlib import Path
import time

from PIL import Image

from cleanrua.direct import direct
from cleanrua_utils.franka_env import FrankaEnv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Read the current state and save camera images")
    mode.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--out", type=Path, help="New output directory")
    parser.add_argument("--max-decisions", type=int, default=50)
    args = parser.parse_args(argv)
    if args.enable_motion:
        if not (args.task or "").strip() or args.max_decisions <= 0:
            parser.error("Motion requires --task and a positive decision budget")
        for key in ("CLEANRUA_BASE_URL", "CLEANRUA_MODEL_NAME", "CLEANRUA_API_KEY"):
            if not os.environ.get(key):
                parser.error(f"Set {key} before connecting to hardware")
    config = json.loads(args.config.read_text())
    out = args.out or Path("runs") / f"direct_franka_{time.time_ns()}"
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
            limits = {"workspace_m": config["workspace_m"], **robot.limits}
            instruction = f"{args.task}\nStation TCP limits: {json.dumps(limits)}. Plan within each single-step limit."
            result = {"stop_reason": direct(robot, instruction, max_decisions=args.max_decisions),
                      "task_success": None}
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(out / "result.json")


if __name__ == "__main__":
    main()
