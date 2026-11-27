"""Explore with upstream CaP tools and an isolated worker, then save a reusable controller.

    python -m realworld.runners.cap_realworld_franka --config station.json --enable-motion --task "Stack the blocks"
    python -m realworld.runners.cap_realworld_franka compile --run-dir RUN_DIR
    python -m realworld.runners.cap_realworld_franka --config station.json --enable-motion --replay RUN_DIR/controller.py

Linux bubblewrap and NumPy in /usr/bin/python3 are required for execution and replay.
Compilation reads the saved episode and calls the model without opening the station.
"""
import argparse
import ast
from contextlib import closing
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

from cleanrua import cap
from ._station import run_cli


class InvalidTarget(ValueError):
    """Malformed or infeasible program request, before motion dispatch."""


def program_worker():
    import contextlib
    import io
    import json
    import resource
    import socket
    import numpy as np

    MAX_PACKET = 65536
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    channel = socket.socket(fileno=0)

    def send(data):
        channel.send(json.dumps(data, allow_nan=False).encode())

    def request(method, **args):
        send({"method": method, "args": args})
        reply = json.loads(channel.recv(MAX_PACKET))
        if "error" in reply:
            raise ValueError(reply["error"])
        return reply["result"]

    class Robot:
        def state(self):
            return request("state")

        def observe(self):
            obs = request("observe")
            images = {}
            for name, shape in obs.pop("image_shapes").items():
                image = np.empty(shape, dtype=np.uint8)
                data = memoryview(image).cast("B")
                for offset in range(0, len(data), MAX_PACKET):
                    chunk = data[offset:offset+MAX_PACKET]
                    if channel.recv_into(chunk) != len(chunk):
                        raise RuntimeError("Incomplete RGB frame")
                images[name] = image
            obs["images"] = images
            return obs

        def move(self, pose, gripper=None):
            return request("move", pose=np.asarray(pose).tolist(), gripper=gripper)

    namespace = {"np": np, "robot": Robot()}
    while packet := channel.recv(MAX_PACKET):
        out, result = io.StringIO(), {}
        try:
            with contextlib.redirect_stdout(out):
                exec(compile(json.loads(packet)["code"], "<policy>", "exec"), namespace)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        result["output"] = out.getvalue()[-4000:]
        send({"done": result})


class ProgramSession:
    def __init__(self, bench):
        self.bench = bench
        self.channel, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        worker = inspect.getsource(program_worker) + "\nprogram_worker()\n"
        command = ["bwrap", "--die-with-parent", "--new-session", "--unshare-all",
                   "--clearenv", "--setenv", "OPENBLAS_NUM_THREADS", "1",
                   "--ro-bind", "/usr", "/usr",
                   "--ro-bind-try", "/etc/alternatives", "/etc/alternatives",
                   "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache", "--symlink", "usr/lib", "/lib",
                   "--symlink", "usr/lib64", "/lib64", "--dev", "/dev", "--tmpfs", "/tmp",
                   "/usr/bin/python3", "-IB", "-c", worker]
        try:
            self.process = subprocess.Popen(command, stdin=child, stdout=subprocess.DEVNULL)
        except BaseException:
            self.channel.close()
            raise
        finally:
            child.close()

    def close(self):
        self.channel.close()
        self.process.kill()
        self.process.wait()

    def execute(self, code):
        payload = json.dumps({"code": code}).encode()
        if len(payload) >= 65536:
            raise InvalidTarget("Program exceeds packet limit")
        deadline = time.monotonic() + 600
        self.channel.send(payload)
        for _ in range(129):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Program deadline exceeded")
            self.channel.settimeout(remaining)
            packet = self.channel.recv(65536)
            if not packet:
                raise RuntimeError("Isolated Python worker exited")
            message = json.loads(packet)
            if "done" in message:
                return message["done"]
            args = message.get("args", {})
            try:
                if message["method"] == "observe" and not args:
                    obs = self.bench.observe()
                    images = obs.pop("images")
                    header = {**obs,
                              "image_shapes": {k: v.shape for k, v in images.items()}}
                    packet = json.dumps({"result": header}, allow_nan=False).encode()
                    if len(packet) >= 65536:
                        raise ValueError("Observation metadata exceeds packet limit")
                    self.channel.send(packet)
                    for image in images.values():
                        data = memoryview(image.tobytes())
                        for offset in range(0, len(data), 65536):
                            self.channel.send(data[offset:offset+65536])
                    continue
                if message["method"] == "state" and not args:
                    value = self.bench.state()
                elif message["method"] == "move" and isinstance(args, dict) and set(args) == {"pose", "gripper"}:
                    opening = args["gripper"]
                    if opening is None:
                        opening = self.bench.state()["gripper"]
                    path = self.bench.ik(args["pose"], opening)
                    if not path:
                        raise InvalidTarget("Target rejected before dispatch")
                    for action in path:
                        self.bench.step(action)
                    value = self.bench.state()
                    if self.bench.last_gripper_receipt is not None:
                        value["last_gripper_receipt"] = self.bench.last_gripper_receipt
                else:
                    raise InvalidTarget("Unknown robot operation")
                response = {"result": value}
            except InvalidTarget as exc:
                response = {"error": str(exc)}
            self.channel.send(json.dumps(response, allow_nan=False).encode())
        raise RuntimeError("Program exceeded the per-turn robot-call limit")


def compile_controller(folder):
    output = folder / "controller.py"
    if output.exists():
        raise FileExistsError(output)
    events = [json.loads(line) for line in (folder / "trajectory.jsonl").read_text().splitlines()]
    if not events or not ("stop" in events[-1] or "fault" in events[-1]):
        raise ValueError("Expected a completed exploration log")
    observations = [e for e in events if "images" in e]
    if not observations:
        raise ValueError("No saved observations")
    messages = [{"role": "system", "content":
                 "Distill this exploration into a self-contained Python main(robot) for a fresh scene. "
                 "Include all imports and helpers. Locate objects using fresh images and state; check "
                 "progress after each stage and stop on uncertainty. Use only robot.state(), robot.observe(), "
                 "robot.move(pose, gripper=None) for hardware. Submit source with run_code; this offline "
                 "call saves the candidate without executing it. Do not call main at module scope."},
                {"role": "user", "content": json.dumps(events, ensure_ascii=False)}]
    for index in sorted({0, len(observations) - 1}):
        event = observations[index]
        images = {}
        for name, filename in event["images"].items():
            path = (folder / filename).resolve()
            if path.parent != folder.resolve():
                raise ValueError("Expected an episode-local image")
            with Image.open(path) as im:
                images[name] = np.array(im.convert("RGB"))
        messages.append(cap.look({"state": event["state"], "images": images}))
    call = cap.mllm(messages)["tool_calls"][0]["function"]
    if call["name"] != "run_code":
        raise ValueError("Expected controller source in run_code")
    source = json.loads(call["arguments"])["code"]
    compile(source, str(output), "exec")
    mains = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main"]
    if len(mains) != 1:
        raise ValueError("Expected one main(robot)")
    args = mains[0].args
    if [a.arg for a in args.posonlyargs + args.args] != ["robot"] or args.kwonlyargs or args.vararg or args.kwarg:
        raise ValueError("Expected main(robot)")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, delete=False) as tmp:
        try:
            tmp.write(source.rstrip() + "\n")
            tmp.flush()
            os.link(tmp.name, output)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    return output


def run(robot, instruction, args, out):
    if args.replay:
        with closing(ProgramSession(robot)) as program:
            result = program.execute(args.replay.read_text() + "\nprint(main(robot))")
        if "error" in result:
            raise RuntimeError(result["error"])
        return {"stop_reason": "controller_returned", "feedback": result, "task_success": None}

    history = []
    reason = "max_decisions"
    with (out / "trajectory.jsonl").open("x") as log, closing(ProgramSession(robot)) as program:
        def record(**event):
            log.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()

        try:
            for turn in range(args.max_decisions):
                obs = robot.observe()
                images = {}
                for i, (name, rgb) in enumerate(obs["images"].items()):
                    images[name] = f"camera_{i}-{turn:03d}.png"
                    Image.fromarray(rgb).save(out / images[name])
                record(turn=turn, task=instruction, images=images, state=obs["state"])
                reply = cap.mllm([{"role": "system", "content": cap.SYSTEM +
                                   " Use robot.observe() for live RGB and state. Test reusable functions "
                                   "for changed object positions. Motion errors require inspection."},
                                  {"role": "user", "content": instruction}, *history[-8:], cap.look(obs)])
                call = reply["tool_calls"][0]
                fn = call["function"]
                record(turn=turn, request=fn)
                if fn["name"] == "finish":
                    reason = "finish"
                    break
                if fn["name"] != "run_code":
                    raise ValueError(f"Unexpected tool {fn['name']}")
                result = program.execute(json.loads(fn["arguments"])["code"])
                record(turn=turn, feedback=result)
                history.extend([reply, {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}])
            record(stop=reason, task_success=None)
        except Exception as exc:
            record(fault=str(exc))
            raise
    report = {"stop_reason": reason, "task_success": None, "controller_verified": False}
    try:
        report["controller"] = compile_controller(out).name
    except Exception as exc:
        report["controller_error"] = str(exc)
        (out / "result.json").write_text(json.dumps(report, indent=2))
        raise
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["compile"]:
        parser = argparse.ArgumentParser(description="Compile a completed CaP episode without connecting to hardware.")
        parser.add_argument("--run-dir", type=Path, required=True)
        folder = parser.parse_args(argv[1:]).run_dir
        artifact = compile_controller(folder)
        report_path = folder / "result.json"
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            report.pop("controller_error", None)
            report.update(controller=artifact.name, controller_verified=False)
            report_path.write_text(json.dumps(report, indent=2))
        print(artifact)
    else:
        run_cli("cap", run, argv)


if __name__ == "__main__":
    main()
