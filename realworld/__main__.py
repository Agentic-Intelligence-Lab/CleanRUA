"""Real workstations for CleanRUA. Run from the repository root:

    python -m realworld direct --config station.json --check
    python -m realworld direct --config station.json --enable-motion --task TASK
    python -m realworld cap --config station.json --enable-motion --task TASK --memory-dir MEMORY
    python -m realworld cap --config station.json --enable-motion --replay MEMORY/controller.py
    python -m realworld compile --run-dir RUN_DIR
    python -m realworld delegate --config station.json --policy-config policy.json --enable-motion --task TASK
    python -m realworld monitor --config station.json --policy-config policy.json --enable-motion --task TASK

Model runs use CLEANRUA_BASE_URL, CLEANRUA_API_KEY and --model or CLEANRUA_MODEL_NAME.
Install the dependencies of cleanrua.cap_explore_dexjoco for model runs. Each VLA policy JSON
names a driver and constructor args; its predict(obs, task) returns action chunks. Use --tool-choice auto --reasoning-effort xhigh with a compatible endpoint.
Reusing CaP memory rehearses its existing controller on the current scene before exploration.
CaP programs run in Linux bubblewrap with NumPy in /usr/bin/python3. Use trusted local driver configurations.
"""
import argparse
import ast
import base64
from collections import deque
from contextlib import closing, contextmanager, nullcontext
import fcntl
from importlib import import_module
import inspect
from itertools import count, islice
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import httpx
import numpy as np
from PIL import Image
from cleanrua import cap, delegate, direct, monitor


class TrajectoryRejected(ValueError):
    """A site driver confirmed that this step sent no hardware command."""

    def __init__(self, violation, *, trajectory=None):
        super().__init__(str(violation))
        self.violation = violation
        self.trajectory = trajectory
        self.arm = None

    def record(self, out, decision, request, completed):
        result = {"status": "rejected_before_execution", "step_executed": False, "steps": completed,
                  "arm": self.arm, "violation": self.violation,
                  "instruction": "This rejected step sent no motion or gripper command. "
                                 "Replan from the fresh observation with a different target or approach."}
        path = out / f"rejected-trajectory-{decision:03d}.json"
        with path.open("x") as handle:
            json.dump({"decision": decision, "request": request,
                       "feedback": result, "trajectory": self.trajectory}, handle, allow_nan=False)
        print(f"Trajectory rejected: {json.dumps(result)} (saved to {path})", flush=True)
        return result


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

    class EndRejectedCell(BaseException):
        pass

    def request(method, **args):
        send({"method": method, "args": args})
        reply = json.loads(channel.recv(MAX_PACKET))
        if "end_cell" in reply:
            raise EndRejectedCell()
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
        except EndRejectedCell:
            pass  # The parent supplies the authoritative motion receipt.
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        result["output"] = out.getvalue()[-4000:]
        send({"done": result})


class ProgramSession:
    def __init__(self, bench, out):
        self.bench = bench
        self.out = out
        self.cell = 0
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
        self.cell += 1
        start_steps = self.steps
        rejection = None
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
                result = message["done"]
                if rejection is not None:
                    result.update(rejection, error=f"Move rejected before execution: {rejection['violation']}")
                return result
            if rejection is not None:
                # Also block RPCs from finally blocks or a program that caught the cell abort.
                self.channel.send(b'{"end_cell":true}')
                continue
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
                    try:
                        path = self.bench.ik(targets=targets)
                        if not path:
                            raise TrajectoryRejected({"reason": "Target rejected by the workstation before dispatch"})
                        for action in path:
                            self.bench.step(action)
                            self.steps += 1
                    except TrajectoryRejected as exc:
                        rejection = exc.record(self.out, self.cell, {"targets": targets}, self.steps - start_steps)
                        self.channel.send(b'{"end_cell":true}')
                        continue
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


@contextmanager
def station_agent(args, tools, prompt):
    from cleanrua import cap_explore_dexjoco as explore

    if args.tool_choice not in ('required', 'auto'):
        raise ValueError("Tool choice must be required or auto")
    options = explore.Args(task=args.task, reasoning_effort=args.reasoning_effort, tool_choice=args.tool_choice)
    # Upstream embeds simulator instructions. Supply the station API at the HTTP boundary
    # while leaving observation encoding, tool selection and accounting in RUAgent.
    compile_prompt = (f"Produce two fenced blocks: markdown notes (up to {options.notes_chars} characters), "
                      "then python defining a standalone main(robot). Generalize to new object positions using "
                      "fresh observations. Include imports and helpers and do not call main at module scope. "
                      + PROGRAM_API)

    class StationClient(httpx.Client):
        retry_thinking = None

        def post(self, url, *, json):
            system = prompt if 'tools' in json else compile_prompt
            messages = [{'role': 'system', 'content': system}, *json['messages'][1:]]
            if self.retry_thinking and len(messages) >= 2 and messages[-2]['role'] == 'assistant':
                messages[-2] = {**messages[-2], 'reasoning_content': self.retry_thinking}
            self.retry_thinking = None
            body = {**json, 'messages': messages}
            if args.method != 'cap' and 'tools' in body:
                body.pop('max_tokens', None)  # Keep the endpoint's default for Direct and the VLA methods.
            response = super().post(url, json=body)
            if response.is_success and 'tools' in body:
                data = response.json()
                message = data['choices'][0]['message']
                if not message.get('tool_calls'):
                    self.retry_thinking = message.get('reasoning_content')
                if args.method != 'cap' and data.get('usage') is None:
                    # Non-CaP methods previously accepted endpoints without token accounting.
                    return httpx.Response(response.status_code, json={**data, 'usage': {}}, request=response.request)
            return response

    class StationAgent(explore.RUAgent):
        def __init__(self, client):
            self.args, self.tools, self.client = options, tools, client
            self.model = args.model
            self.history = deque(maxlen=options.history_size if args.method == 'cap' else 4)

        def get_action(self, observation):
            action = super().get_action(observation)
            if self.message.get('reasoning_content'):
                self.pending['reasoning_content'] = self.message['reasoning_content']
            return action

    with StationClient(base_url=os.environ['CLEANRUA_BASE_URL'].rstrip('/')+'/', timeout=options.request_timeout,
                       headers={'Authorization': f'Bearer {os.environ["CLEANRUA_API_KEY"]}'}) as client:
        yield StationAgent(client)


def observe(robot, turn, feedback):
    obs = robot.observe()
    state = obs['state']
    calibration = state.get('camera_calibration', {})
    return {'arms': state['arms'], 'native_steps': turn, 'feedback': json.dumps(feedback),
            'cameras': {name: {'rgb': rgb, **calibration.get(name, {})} for name, rgb in obs['images'].items()}}


def replay(session, path):
    result = session.execute(path.read_text() + '\n\nmain(robot)\n')
    if 'error' in result:
        raise RuntimeError(result['error'])
    return {'stop_reason': 'controller_returned', 'feedback': result, 'requests': 0, 'task_success': None}


def execute_action(name, arguments, *, robot, policy, task, session, out, decision):
    if name == 'run_code' and session is not None:
        if set(arguments) != {'code'} or not isinstance(arguments['code'], str):
            return {'ok': False, 'error': 'Expected run_code(code)'}
        before = session.steps
        result = session.execute(arguments['code'])
        return {**result, 'ok': 'error' not in result, 'steps': session.steps - before}
    if name == 'move' and session is None:
        if set(arguments) != {'targets'}:
            return {'ok': False, 'error': 'Expected named targets'}
        path = robot.ik(targets=arguments['targets'])
        if not path:
            return {'ok': False, 'steps': 0, 'error': 'Targets rejected before dispatch'}
        for completed, action in enumerate(path):
            try:
                ended = robot.step(action)
            except TrajectoryRejected as rejection:
                return rejection.record(out, decision, arguments, completed)
            if ended:
                return {'steps': completed + 1, 'episode_ended': True}
        return {'steps': len(path)}
    if name in ('vla', 'run_policy') and policy is not None:
        steps = arguments.get('steps')
        instruction = arguments.get('instruction') if name == 'vla' else task
        if type(steps) is not int or not 1 <= steps <= 300 or not isinstance(instruction, str):
            return {'ok': False, 'error': 'Expected a VLA instruction and 1..300 steps'}
        chunk = iter(())
        for i in range(steps):
            action = next(chunk, None)
            if action is None:
                chunk = iter(policy(robot.observe(), instruction))
                action = next(chunk)
            if robot.step(action):
                return {'steps': i + 1, 'episode_ended': True}
        return {'steps': steps}
    return {'ok': False, 'error': 'Unknown station tool'}


def write_artifact(path, text):
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as tmp:
        try:
            tmp.write(text)
            tmp.flush()
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def compile_candidate(agent, record, folder, memory):
    cameras = {}
    for name, filename in record['images'].items():
        path = (folder / filename).resolve()
        if path.parent != folder.resolve():
            raise ValueError('Expected an episode-local image')
        with Image.open(path) as im:
            cameras[name] = {'rgb': np.array(im.convert('RGB'))}
    notes, program = agent.compile(record['sections'], {'cameras': cameras}, record['goals'])
    mains = [n.args for n in ast.parse(program).body if isinstance(n, ast.FunctionDef) and n.name == 'main']
    if len(mains) != 1:
        raise ValueError('Expected one main(robot)')
    params = mains[0]
    if [a.arg for a in params.posonlyargs + params.args] != ['robot'] or params.kwonlyargs or params.vararg or params.kwarg:
        raise ValueError('Expected main(robot)')
    write_artifact(memory/'notes.md', notes)
    write_artifact(memory/'controller.py', program)
    write_artifact(memory/'status.json', json.dumps({'version': record['version'] + 1,
                   'verified_version': None, 'consecutive_successes': 0}, indent=2))


def run_episode(robot, args, out, session):
    module = {'direct': direct, 'delegate': delegate, 'monitor': monitor, 'cap': cap}[args.method]
    tools = module.TOOLS if session else named_tools(module.TOOLS, robot.capabilities)
    names = {t['function']['name'] for t in tools}
    prompt = STATION_PROMPT if session else module.SYSTEM
    task = args.task + '\nStation capabilities: ' + json.dumps(robot.capabilities)
    prompt += ' steps_remaining counts remaining model decisions; zero denotes unlimited.'
    policy = load_policy(args.policy_config, module.ACTIONS_PER_CHUNK) if args.policy_config else None
    memory = args.memory_dir or out
    with (learning_memory(memory) if session else nullcontext()), station_agent(args, tools, prompt) as agent, \
            (out/'trajectory.jsonl').open('x') as log:
        def record(event, **fields):
            log.write(json.dumps({'event': event, **fields}, ensure_ascii=False, allow_nan=False) + '\n')
            log.flush()

        goals, prior, transcript = [], [], []
        version = 0
        if session:
            for path in sorted((memory/'goal').glob('*.png')):
                goals.extend([{'type': 'text', 'text': f'Goal image {path.stem}:'}, {'type': 'image_url', 'image_url': {
                    'url': 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()}}])
            if (memory/'notes.md').exists():
                prior.append('Earlier notes:\n' + (memory/'notes.md').read_text())
            if (memory/'controller.py').exists():
                program = (memory/'controller.py').read_text()
                rehearsal = session.execute(program + '\n\nmain(robot)\n')
                prior.append(f'Previous program:\n{program}\nRehearsal: {json.dumps(rehearsal)}')
                record('rehearsal', feedback=rehearsal, task_success=None)
            if (memory/'status.json').exists():
                version = json.loads((memory/'status.json').read_text())['version']
        instruction = '\n'.join([task, *prior])
        agent.reset([{'type':'text', 'text':instruction}, *goals] if session else instruction, args.max_decisions)
        decision = 0
        reason = 'max_decisions'
        start = time.monotonic()
        obs = observe(robot, 0, {})
        record('reset', task=args.task, model=args.model)
        try:
            for decision in (range(1, args.max_decisions + 1) if args.max_decisions else count(1)):
                if session:
                    images = out/'images'
                    images.mkdir(exist_ok=True)
                    for i, cam in enumerate(obs['cameras'].values()):
                        Image.fromarray(cam['rgb']).save(images/f'{decision:03d}_{i}.png')
                call = agent.get_action(obs)
                name = call['name']
                record('action', decision=decision, action=call)
                if session and agent.total_tokens >= agent.args.max_total_tokens:
                    reason = 'token_budget'
                    break
                if session and time.monotonic() - start >= agent.args.episode_timeout:
                    reason = 'timeout'
                    break
                try:
                    arguments = json.loads(call['arguments'])
                except (TypeError, ValueError):
                    arguments = None
                if name not in names or not isinstance(arguments, dict):
                    feedback = {'ok': False, 'error': 'Invalid tool name or arguments'}
                elif name == 'finish' and not arguments:
                    reason = 'finish'
                    break
                else:
                    feedback = execute_action(name, arguments, robot=robot, policy=policy, task=args.task,
                                              session=session, out=out, decision=decision)
                record('step', decision=decision, feedback=feedback)
                if session:
                    transcript.append(f'{name}({call["arguments"]})\nResult: {json.dumps(feedback)}')
                if feedback.get('episode_ended'):
                    reason = 'episode_ended'
                    break
                obs = observe(robot, decision if args.max_decisions else 0, feedback)
            if session:
                # Keep the exact final images and compilation inputs for offline recovery.
                images = {}
                for i, (name, cam) in enumerate(obs['cameras'].items()):
                    images[name] = f'final_{i}.png'
                    Image.fromarray(cam['rgb']).save(out/images[name])
                materials = {'sections': [instruction, '\n\n'.join(transcript)[-30000:],
                                          f'Stop: {reason}. Task success unknown.'],
                             'images': images, 'goals': goals, 'version': version}
                write_artifact(out/'compile_input.json', json.dumps(materials))
                compile_candidate(agent, materials, out, memory)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            record('fault', error=error)
            failed_compile = (out/'compile_input.json').is_file()
            report = {'stop_reason': reason if failed_compile else 'error', 'decisions': decision,
                      'task_success': None, **agent.usage}
            report['compile_error' if failed_compile else 'error'] = error
            write_artifact(out/'result.json', json.dumps(report, indent=2))
            if failed_compile:
                raise RuntimeError(f'Controller compilation failed: {exc}') from exc
            raise
        result = {'stop_reason': reason, 'decisions': decision, 'task_success': None, **agent.usage}
        if session:
            result.update(controller=str(memory/'controller.py'), controller_verified=False)
        return result


def offline_compile(argv):
    parser = argparse.ArgumentParser(description='Compile a saved CaP exploration without opening hardware.')
    parser.add_argument('--run-dir', type=Path, required=True)
    add_model_options(parser)
    args = parser.parse_args(argv)
    args.method, args.task = 'cap', ''
    if not args.model:
        parser.error('Set CLEANRUA_MODEL_NAME or pass --model')
    out = args.run_dir
    with learning_memory(out):
        if (out/'controller.py').exists():
            raise FileExistsError(out/'controller.py')
        p = out/'compile_input.json'
        if p.exists():
            materials = json.loads(p.read_text())
            materials.setdefault('version', 0)
        else:
            events = [json.loads(l) for l in (out/'trajectory.jsonl').read_text().splitlines()]
            if not events or not ('stop' in events[-1] or 'fault' in events[-1]):
                raise ValueError('Expected a completed legacy episode or compile_input.json')
            materials = {'sections':[json.dumps(events, ensure_ascii=False)], 'goals':[], 'version':0,
                         'images':next(e['images'] for e in reversed(events) if 'images' in e)}
        with station_agent(args, cap.TOOLS, STATION_PROMPT) as agent:
            agent.reset('', 0)
            compile_candidate(agent, materials, out, out)
        p = out/'result.json'
        result = json.loads(p.read_text()) if p.exists() else {'task_success': None}
        result.update(controller=str(out/'controller.py'), controller_verified=False, compile_error=None)
        result.pop('controller_error', None)
        write_artifact(p, json.dumps(result, indent=2))
    print(out/'controller.py')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['compile']:
        return offline_compile(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method", choices=("direct", "cap", "delegate", "monitor"))
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-decisions", type=int, default=os.environ.get("CLEANRUA_MAX_DECISIONS", "50"),
                        help="Decision budget; 0 means unlimited. Defaults to CLEANRUA_MAX_DECISIONS or 50.")
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
        if not (args.task or "").strip() or args.max_decisions < 0:
            parser.error("Motion requires --task and a nonnegative decision budget (0 means unlimited)")
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
        else:
            with (closing(ProgramSession(robot, out)) if args.method == "cap" else nullcontext()) as session:
                result = replay(session, args.replay) if args.replay else run_episode(robot, args, out, session)
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(out / "result.json")


if __name__ == "__main__":
    # Drivers import TrajectoryRejected by its package name. Use that same module for the CLI.
    from realworld.__main__ import main as run_cli
    run_cli()
