"""Workstation checks, trusted policy replay, and configurable real-agent entry points."""
import argparse
from importlib import import_module
import inspect
import json
import re
import runpy
import sys
import os
import time
from pathlib import Path

from PIL import Image

from realworld.model import MODEL_KEYS, configure_decision, decision_settings
from realworld.policy import load_policy
from realworld.workstation import workstation


def resolve_algorithm(name):
    """Built-in methods follow upstream naming. External methods use trusted module:function."""
    if ":" in name:
        module, entry = name.split(":", 1)
    else:
        if not name.isidentifier():
            raise ValueError(f"Invalid algorithm name: {name}")
        module, entry = f"realworld.{name}", f"run_{name}"
    return getattr(import_module(module), entry)


def run_method(bench, task, run_dir, *, algorithm="direct", decide=None, max_decisions=30, policy=None):
    method = resolve_algorithm(algorithm)
    options = {"max_decisions": max_decisions}
    if decide is not None:
        options["decide"] = decide
    if "policy" in inspect.signature(method).parameters:
        if policy is None:
            raise ValueError(f"{algorithm} requires an action policy")
        options["policy"] = policy
    return method(bench, task, run_dir, **options)


def module_main(method):
    args = sys.argv[1:]
    if method == "direct" and args[:1] in (["check"], ["replay"], ["direct"], ["cap"]):
        return main(args)
    return main([method, *args], standalone=True)


def main(argv=None, *, standalone=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", help="check, replay, an algorithm name, or trusted module:function")
    parser.add_argument("--config", type=Path, help="Trusted workstation JSON; omit for offline fixtures")
    parser.add_argument("--enable-motion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out", type=Path, help="New directory for this run")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--max-decisions", type=int, default=30)
    parser.add_argument("--policy-config", type=Path, help="Action policy client configuration")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--backend", help="api, local, codex or a trusted module:factory")
    parser.add_argument("--model", help="Override the configured decision model")
    parser.add_argument("--reasoning-effort", help="Provider-supported effort or default")
    parser.add_argument("--base-url", help="Compatible API or local server URL")
    args = parser.parse_args(argv)
    if args.mode is None:
        try:
            args.mode = input("Method (direct / cap / delegate / monitor) [direct]: ").strip() or "direct"
            if args.mode not in ("direct", "cap", "delegate", "monitor"):
                parser.error("Choose direct, cap, delegate or monitor")
            if not args.config:
                station = input("Workstation JSON [mock]: ").strip()
                args.config = Path(station) if station else None
            args.task = args.task or input("Task: ").strip()
            backend = decision_settings()["decision_backend"]
            args.backend = args.backend or input(f"Decision backend (api / local / codex) [{backend}]: ").strip() or backend
            if args.backend in MODEL_KEYS:
                model = os.getenv(MODEL_KEYS[args.backend]) or ("CLI default" if args.backend == "codex" else "served model ID required")
                args.model = args.model or input(f"Model [{model}]: ").strip() or None
                effort = os.getenv("CLEANRUA_REASONING_EFFORT", "default")
                args.reasoning_effort = args.reasoning_effort or input(f"Reasoning effort [{effort}]: ").strip() or None
            if args.mode in ("delegate", "monitor") and not args.policy_config:
                policy_path = input("Action policy JSON [workstation policy]: ").strip()
                args.policy_config = Path(policy_path) if policy_path else None
            if args.config and not args.enable_motion:
                args.enable_motion = input(f"Enable motion for {args.config} and task {args.task!r}? [y/N]: ").strip().lower() == "y"
        except EOFError:
            parser.error("Supply a method and --task for non-interactive runs")
    if args.out is None:
        name = args.mode.replace(":", "_")
        if standalone:
            task_name = re.sub(r"[^\w-]+", "_", args.task or "").strip("_")[:64] or "task"
            name = f"{task_name}__{name}_real__0"
        args.out = args.output_dir / f"{name}__{time.time_ns()}"
    experiment = args.mode not in ("check", "replay")
    if experiment and (not (args.task or "").strip() or (args.config and not args.enable_motion)):
        parser.error("Model experiments require --task and live experiments also require --enable-motion")
    if args.mode == "replay" and not args.policy:
        parser.error("replay requires a trusted --policy with main(robot)")
    if args.mode == "check" and args.enable_motion:
        parser.error("check is read-only")
    if experiment:
        try:
            configure_decision(args.backend, args.model, args.reasoning_effort, args.base_url)
        except ValueError as exc:
            parser.error(str(exc))
        settings = decision_settings()
        if args.mode in ("direct", "cap", "delegate", "monitor") and settings["decision_backend"] in ("api", "local") and not settings["model"]:
            parser.error("Set the backend’s default model ID or pass --model")
        print(json.dumps({"task": args.task, "mode": args.mode, "station": str(args.config or "mock"),
                          "out": str(args.out), **settings}, ensure_ascii=False), flush=True)
    config = json.loads(args.config.read_text()) if args.config else None
    needs_policy = experiment and "policy" in inspect.signature(resolve_algorithm(args.mode)).parameters
    policy = load_policy(config, args.policy_config) if needs_policy else None
    args.out.mkdir(parents=True, exist_ok=False)
    bench = workstation(config, args.mode != "check" and (not args.config or args.enable_motion))
    try:
        if experiment:
            result = run_method(bench, args.task, args.out, algorithm=args.mode, policy=policy, max_decisions=args.max_decisions)
            if standalone:
                print(json.dumps({"run_dir": str(args.out), "stop_reason": result}))
                return
        elif args.mode == "replay":
            result = runpy.run_path(str(args.policy))["main"](bench)
        else:
            obs = bench.observe()
            for name, image in obs.pop("images").items():
                Image.fromarray(image).save(args.out / f"{name}.png")
            result = obs
        report = {"backend": "configured" if args.config else "mock", "mode": args.mode,
                  "result": result, "task_success": None}
        if experiment and (args.out / "result.json").is_file():
            report.update(json.loads((args.out / "result.json").read_text()))
        output = json.dumps(report, indent=2, allow_nan=False)
        (args.out / "result.json").write_text(output + "\n")
        print(output)
    finally:
        bench.close()


if __name__ == "__main__":
    main()
