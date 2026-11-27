"""Explore with upstream CaP tools and an isolated worker, then save a reusable controller.

    python -m realworld cap --config station.json --enable-motion --task "Stack the blocks"
    python -m realworld.cap --run-dir RUN_DIR
    python -m realworld cap --config station.json --enable-motion --replay RUN_DIR/controller.py

Linux bubblewrap and NumPy in /usr/bin/python3 are required for execution and replay.
Compilation reads the saved episode and calls the model without opening the station.
"""
import argparse
from contextlib import closing, contextmanager
import inspect
import json
import os
from pathlib import Path
import socket
import subprocess
import fcntl
import httpx
import time

import numpy as np
from PIL import Image

from cleanrua import cap, cap_explore_dexjoco as explore


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

        def move(self, pose=None, gripper=None, *, targets=None):
            if targets is not None:
                if pose is not None or gripper is not None:
                    raise ValueError("Specify targets or a single-arm pose")
                return request("move", targets=[{**t, "pose": np.asarray(t["pose"]).tolist()} for t in targets])
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
        self.steps = 0
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
                elif message["method"] == "move" and isinstance(args, dict):
                    if set(args) == {"targets"}:
                        targets = args["targets"]
                    elif set(args) == {"pose", "gripper"} and len(self.bench.capabilities["arms"]) == 1:
                        target = {"arm": next(iter(self.bench.capabilities["arms"])), "pose": args["pose"]}
                        if args["gripper"] is not None:
                            target["gripper"] = args["gripper"]
                        targets = [target]
                    else:
                        raise InvalidTarget("Use move(targets=[...]) with explicit arm names")
                    path = self.bench.ik(targets=targets)
                    if not path:
                        raise InvalidTarget("Target rejected before dispatch")
                    for action in path:
                        self.bench.step(action)
                        self.steps += 1
                    value = self.bench.state()
                else:
                    raise InvalidTarget("Unknown robot operation")
                response = {"result": value}
            except InvalidTarget as exc:
                response = {"error": str(exc)}
            self.channel.send(json.dumps(response, allow_nan=False).encode())
        raise RuntimeError("Program exceeded the per-turn robot-call limit")


PROGRAM_API = ("np, robot.state(), robot.observe() returning images and state, "
               "robot.move(targets=[dict(arm=..., pose=..., gripper=...)]) returning measured state. "
               "Pose is station XYZ metres and fixed-axis XYZ RPY radians; omit gripper to hold. "
               "Only RGB is available. No depth, reset, or automatic task-success sensor")
STATION_PROMPT = ("Explore the task with Python using run_code, or finish. Use " + PROGRAM_API +
                  ". Find objects again from live observations and check progress after each stage. "
                  "The physical scene continues from its measured state after rehearsal.")


class StationEpisode:
    """Map real measurements and isolated program execution onto the upstream Explore env."""
    success = None
    max_attempts = attempts = 1
    attempt_over = False

    def __init__(self, robot, program, instruction):
        self.robot, self.program = robot, program
        self.instruction = instruction
        self.finished = False
        self.outcomes = []

    @property
    def unwrapped(self):
        return self

    @property
    def total_steps(self):
        return self.program.steps

    def restart(self):
        # A new observation, without physically resetting the scene.
        obs = self.robot.observe()
        state = obs['state']
        calibration = state.get('camera_calibration', {})
        return {'arms': state['arms'], 'native_steps': self.total_steps,
                'cameras': {name: {'rgb': rgb, **calibration.get(name, {})} for name, rgb in obs['images'].items()}}

    def reset(self, *, seed):
        return {**self.restart(), 'feedback': ''}, {'instruction': self.instruction, 'step_limit': None}

    def step(self, action):
        before = self.total_steps
        args = json.loads(action['arguments'])
        if action['name'] == 'finish' and args == {}:
            self.finished = True
            result = {}
        elif action['name'] == 'run_code' and isinstance(args, dict) and set(args) == {'code'} and isinstance(args['code'], str):
            result = self.program.execute(args['code'])
        else:
            result = {'error': 'Expected run_code(code) or finish()'}
        feedback = {**result, 'ok': 'error' not in result, 'steps': self.total_steps - before, 'task_success': None}
        return {**self.restart(), 'feedback': json.dumps(feedback)}, 0., False, False, {}


def make_agent(options, args):
    client = httpx.Client(base_url=os.environ['CLEANRUA_BASE_URL'].rstrip('/') + '/', timeout=options.request_timeout,
                          headers={'Authorization': f'Bearer {os.environ["CLEANRUA_API_KEY"]}'})
    return explore.RUAgent(options, cap.TOOLS, model=args.model, client=client, system_prompt=STATION_PROMPT,
                           program_api=PROGRAM_API)


@contextmanager
def learning_memory(folder):
    # Keep two learners from rehearsing and rewriting the same version concurrently.
    folder.mkdir(parents=True, exist_ok=True)
    fd = os.open(folder, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def run(robot, instruction, args, out):
    with closing(ProgramSession(robot)) as session:
        env = StationEpisode(robot, session, instruction)
        if args.replay:
            obs, *_ = explore.replay_controller(env, args.replay.read_text())
            feedback = json.loads(obs['feedback'])
            if not feedback['ok']:
                raise RuntimeError(feedback['error'])
            return {'stop_reason': 'controller_returned', 'feedback': feedback, 'task_success': None, 'requests': 0}
        env.instruction += '\nStation limits: ' + json.dumps(robot.capabilities)
        options = explore.Args(task=instruction, exp_name='cap_explore_realworld', max_decisions=args.max_decisions,
                               tool_choice=args.tool_choice, reasoning_effort=args.reasoning_effort)
        memory = args.memory_dir or out
        with learning_memory(memory), closing(make_agent(options, args)) as agent:
            result = explore.run_episode(options, env, agent, out, memory, save_compile_input=True)
        result.update(task_success=result['success'], controller_verified=False)
        if result['compile_error'] or result['model_error_detail']:
            raise RuntimeError(result['compile_error'] or result['model_error_detail'])
        result['controller'] = str(memory / 'controller.py')
        return result


def main(argv=None):
    """Retry compilation using the saved episode, with no station and no code execution."""
    from realworld.__main__ import add_model_options
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    add_model_options(parser)
    args = parser.parse_args(argv)
    folder = args.run_dir
    # Offline recovery writes only to the failed run, never over an existing candidate.
    with learning_memory(folder):
        output = folder / 'controller.py'
        if output.exists():
            raise FileExistsError(output)
        path = folder / 'compile_input.json'
        if path.exists():
            record = json.loads(path.read_text())
        else:
            # Pre-Explore logs recorded the task, images and model feedback per turn.
            events = [json.loads(line) for line in (folder / 'trajectory.jsonl').read_text().splitlines()]
            if not events or not ('stop' in events[-1] or 'fault' in events[-1]):
                raise ValueError('No completed legacy trajectory or saved Explore compile input')
            record = {'sections': [json.dumps(events, ensure_ascii=False)], 'goals': [],
                      'images': next(e['images'] for e in reversed(events) if 'images' in e)}
        cameras = {}
        for name, filename in record['images'].items():
            image = (folder / filename).resolve()
            if image.parent != folder.resolve():
                raise ValueError('Expected an episode-local image')
            with Image.open(image) as im:
                cameras[name] = {'rgb': np.array(im.convert('RGB'))}
        options = explore.Args(reasoning_effort=args.reasoning_effort, tool_choice=args.tool_choice)
        with closing(make_agent(options, args)) as agent:
            agent.reset('', 0)
            notes, program = agent.compile(record['sections'], {'cameras': cameras}, record['goals'])
        (folder / 'notes.md').write_text(notes)
        output.write_text(program)
        report_path = folder / 'result.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            report.pop('controller_error', None)
            report.update(compile_error=None, controller_verified=False, controller=str(output))
            status_path = folder / 'status.json'
            if status_path.is_file():
                status = json.loads(status_path.read_text())
                status.update(version=status['version'] + 1, consecutive_successes=0)
                status_path.write_text(json.dumps(status, indent=1))
                report['status'] = status
            report_path.write_text(json.dumps(report, indent=2))
        print(output)


if __name__ == "__main__":
    main()
