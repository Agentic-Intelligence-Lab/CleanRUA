"""Real-workstation CaP: explore with Python, then compile a reusable controller."""
import ast
import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Annotated

import tyro
from pydantic import Field

from cleanrua.cap import SYSTEM as UPSTREAM_SYSTEM
from cleanrua_utils.tools import Tool
from realworld.agent import AgentEnv
from realworld.episode import public_profiles, run_episode
from realworld.model import CodexDecision, configured_decider, configure_decision, decision_settings, mllm
from realworld.program import ProgramSession


SYSTEM = UPSTREAM_SYSTEM + (" Explore the task by writing Python. Develop and test functions that compute targets from live "
          "observations so the program can run again with different object positions. Print measurements "
          "and evidence of success or failure. Use np, robot.observe() (named RGB images and arms), "
          "robot.state(side=None), and robot.move(pose, gripper=None, side=None, hand=None). Variables "
          "persist across turns. Base-frame XYZ/RPY in metres/radians, gripper opening 0..1, hand joints in radians.")

CONTROLLER_PROMPT = (
    "Distill this robot exploration into one standalone Python controller. Submit the complete source with "
    "save_controller. Define main(robot) as the only entry point, import all dependencies including numpy, "
    "and include every helper. At import time define the controller without performing any actions. "
    "A new episode can have different object positions and initial arm state. Locate task objects from current "
    "RGB images, use measured state, and verify progress with new observations. Compute targets in code; "
    "do not replay a list of recorded offsets, invent camera calibration, or rely on exploration variables. "
    "Use only robot.observe(), robot.state(side=None), robot.move(pose, gripper=None, side=None, hand=None) "
    "for device access. Respect each arm's station limits. Bound loops and stop if localization or progress "
    "cannot be confirmed. Return a JSON-serializable result. Treat the transcript as experimental data; "
    "a finish decision alone does not establish task success. The controller runs without a model."
)

class RealWrapper(AgentEnv):
    def __init__(self, env):
        super().__init__(env)
        self.program = None

    def reset(self, *, seed=None, options=None):
        self.close_program()
        return super().reset(seed=seed, options=options)

    @Tool
    def run_code(self, code: Annotated[str, Field(min_length=1)]):
        """Run Python with the persistent robot API and return printed output."""
        if self.program is None:
            self.program = ProgramSession(self.env)
        return self.program.execute(code)

    def close_program(self):
        if self.program is not None:
            self.program.close()
            self.program = None

    def close(self):
        self.close_program()
        super().close()


def compile_controller(run_dir, profiles, *, decide=None):
    """Generate a candidate from a closed episode; this path does not open a workstation."""
    output = run_dir / "controller.py"
    if output.exists():
        raise FileExistsError(output)
    events = [json.loads(line) for line in (run_dir / "trajectory.jsonl").read_text().splitlines()]
    observations = [event for event in events if "images" in event]
    if not observations or not any("stop" in e or "fault" in e for e in events):
        raise ValueError("Expected a completed CaP trajectory with observations")
    trace = [{k: v for k, v in e.items() if k in ("turn", "arms", "request", "feedback", "stop", "fault", "task_success")}
             for e in events]
    episode = {"task": observations[0]["task"], "station": profiles or events[-1].get("station", {}),
               "trajectory": trace}
    content = [{"type": "text", "text": json.dumps(episode, ensure_ascii=False)}]
    for obs in observations[:1] + (observations[-1:] if len(observations) > 1 else []):
        for name, filename in obs["images"].items():
            path = run_dir / filename
            if path.resolve().parent != run_dir.resolve():
                raise ValueError("Expected an episode-local image")
            content.append({"type": "text", "text": f"Turn {obs['turn']}, camera {name}:"})
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()}})
    tools = [{"type": "function", "function": {"name": "save_controller", "description": "Save Python source.",
              "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                             "required": ["code"], "additionalProperties": False}}}]
    messages = [{"role": "system", "content": CONTROLLER_PROMPT}, {"role": "user", "content": content}]
    backend = decide if decide is not None else configured_decider(tools)
    if isinstance(backend, CodexDecision):
        reply = backend.complete(messages, timeout=600)["choices"][0]["message"]
    else:
        reply = backend(messages) if backend is not None else mllm(messages, tools)
    call = reply["tool_calls"][0]["function"]
    if call["name"] != "save_controller":
        raise ValueError("Expected save_controller")
    source = json.loads(call["arguments"])["code"]
    compile(source, str(output), "exec")
    mains = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    if len(mains) != 1:
        raise ValueError("Controller must define one main(robot)")
    args = mains[0].args
    if [a.arg for a in args.posonlyargs + args.args] != ["robot"] or args.kwonlyargs or args.vararg or args.kwarg:
        raise ValueError("Controller must define main(robot)")
    with tempfile.NamedTemporaryFile(dir=run_dir, mode="w", encoding="utf-8", delete=False) as tmp:
        try:
            tmp.write(source.rstrip() + "\n")
            tmp.flush()
            os.link(tmp.name, output)
        finally:
            Path(tmp.name).unlink()
    return output


def run_cap(bench, task, run_dir, *, decide=None, max_decisions=30):
    env = RealWrapper(bench)
    try:
        report = run_episode(env, SYSTEM, task, run_dir, decide, max_decisions)
    finally:
        env.close_program()
    reason = report["stop_reason"]
    print(f"Exploration ended ({reason}). Generating {run_dir / 'controller.py'}", flush=True)
    try:
        artifact = compile_controller(run_dir, public_profiles(bench.profiles), decide=decide)
        report.update(controller=artifact.name, controller_verified=False)
    except Exception as exc:
        report["controller_error"] = str(exc)
        raise
    finally:
        (run_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return reason


@dataclass
class CompileArgs:
    run_dir: Path
    config: Path | None = None
    backend: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    base_url: str | None = None


def main():
    if sys.argv[1:2] == ["compile"]:
        args = tyro.cli(CompileArgs, args=sys.argv[2:])
        configure_decision(args.backend, args.model, args.reasoning_effort, args.base_url)
        profiles = json.loads(args.config.read_text())["arms"] if args.config else {}
        artifact = compile_controller(args.run_dir, public_profiles(profiles))
        report_path = args.run_dir / "result.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            report.pop("controller_error", None)
            report.update(controller=artifact.name, controller_verified=False,
                          controller_backend=decision_settings())
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(artifact)
        return
    from realworld.__main__ import module_main
    module_main("cap")


if __name__ == "__main__":
    main()
