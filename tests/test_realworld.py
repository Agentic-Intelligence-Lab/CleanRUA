"""Hardware-free tests of station-frame dispatch and the four launcher paths."""
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np

from realworld.__main__ import main
from realworld.envs.franka_env import FrankaEnv


class Arm:
    def __init__(self):
        self.pose = [0.0] * 6
        self.opening = 1
        self.moves = []
        self.stopped = False

    def state(self):
        return {"pose": self.pose, "joints": [0] * 7, "gripper": self.opening}

    def move_pose(self, pose):
        self.moves.append(pose)
        self.pose = pose

    def set_gripper(self, opening):
        self.opening = opening

    def stop(self):
        self.stopped = True


class Camera:
    def read(self):
        return np.zeros((4, 4, 3), np.uint8)

    def close(self):
        pass


class Coordinator:
    def __init__(self, arms):
        self.arms = arms
        self.calls = []

    def move_poses(self, poses):
        self.calls.append(poses)
        for name, pose in poses.items():
            self.arms[name].move_pose(pose)

    def stop(self):
        pass


class Policy:
    instructions = []

    def predict(self, obs, instruction):
        self.instructions.append(instruction)
        return [{"targets": [{"arm": "right", "pose": obs["state"]["arms"]["right"]["pose"]}]}]


def spec(cls):
    return {"driver": f"{__name__}:{cls.__name__}"}


def station():
    return {"arms": {arm: {**spec(Arm), "end_effector": "gripper", "workspace_m": [[-1]*3, [1]*3],
                           "max_translation_m": .1, "max_rotation_rad": .2, "station_T_base": base}
                     for arm, base in [("left", [0]*6), ("right", [1, 0, 0, 0, 0, np.pi/2])]},
            "cameras": {"front": spec(Camera)}}


def reply(name, **args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def response(message, status=200):
    return httpx.Response(status, json={"choices": [{"message": message}]},
                          request=httpx.Request("POST", "http://unused/chat/completions"))


class StationTest(unittest.TestCase):
    def test_frames_selection_and_preflight(self):
        cfg = station()
        with FrankaEnv(cfg, enable_motion=True) as env:
            poses = env.state()["arms"]
            np.testing.assert_allclose(poses["right"]["pose"], [1, 0, 0, 0, 0, np.pi/2])
            target = {"arm": "right", "pose": [1, .04, 0, 0, 0, np.pi/2]}
            action = {"targets": [target]}
            for targets in ([target, {"arm": "left", "pose": [0]*6}], [{**target, "arm": "other"}],
                            [{**target, "pose": [0]*6}], [{**target, "gripper": 2}]):
                self.assertEqual(env.ik(targets=targets), [])
            self.assertEqual(env.arms["right"].arm.moves, [])
            env.step(action)
            np.testing.assert_allclose(env.arms["right"].arm.pose, [.04, 0, 0, 0, 0, 0], atol=1e-12)
            self.assertEqual(env.arms["left"].arm.moves, [])
            self.assertEqual(env.arms["right"].arm.opening, 1)
            np.testing.assert_allclose(env.state()["arms"]["right"]["pose"], target["pose"])
        del cfg["arms"]["right"]["station_T_base"]
        with self.assertRaisesRegex(ValueError, "station_T_base"):
            FrankaEnv(cfg)

    def test_dispatch_rechecks_measured_start(self):
        with FrankaEnv(station(), enable_motion=True) as env:
            arm = env.arms["left"].arm
            target = {"arm": "left", "pose": [.02, 0, 0, 0, 0, 0]}
            action, = env.ik(targets=[target])
            arm.pose[0] = .5
            with self.assertRaisesRegex(ValueError, "limits"):
                env.step(action)
            arm.pose[0] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                env.ik(targets=[target])
            self.assertEqual(arm.moves, [])

    def test_coordinator_validates_all(self):
        cfg = {**station(), "coordinator": spec(Coordinator)}
        with FrankaEnv(cfg, enable_motion=True) as env:
            targets = [{"arm": n, "pose": state["pose"]} for n, state in env.state()["arms"].items()]
            for bad in ([targets[0], targets[0]], [targets[0], {**targets[1], "pose": [2]*6}]):
                with self.assertRaises(ValueError):
                    env.step({"targets": bad})
            self.assertEqual(env.coordinator.calls, [])
            env.step({"targets": targets})
            self.assertEqual(len(env.coordinator.calls), 1)
            with patch.object(env.coordinator, "move_poses", side_effect=RuntimeError("driver fault")):
                with self.assertRaisesRegex(RuntimeError, "driver fault"):
                    env.step({"targets": targets})
            self.assertTrue(all(arm.arm.stopped for arm in env.arms.values()))
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                env.step({"targets": targets})

    def test_single_arm_and_read_only(self):
        cfg = station()
        del cfg["arms"]["right"]
        del cfg["arms"]["left"]["station_T_base"]
        with FrankaEnv(cfg) as env:
            self.assertIn("front", env.observe()["images"])
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                env.step(([0]*6, .2))
        with FrankaEnv(cfg, enable_motion=True) as env:
            env.step(([0]*6, .2))
            self.assertEqual(env.state()["gripper"], .2)

    def launch(self, method, folder, *extra):
        cfg = folder / "station.json"
        cfg.write_text(json.dumps(station()))
        out = folder / method
        with redirect_stdout(io.StringIO()):
            main([method, "--config", str(cfg), "--out", str(out), *extra])
        return json.loads((out / "result.json").read_text())

    def test_check_loads_selected_env(self):
        class OtherEnv:
            capabilities = {"arms": {"right": {}}, "max_targets": 1}

            def __init__(self, config, *, enable_motion):
                assert not enable_motion

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                pass

            def observe(self):
                return {"state": {"arms": {"right": {"pose": [0]*6}}}, "images": {"wrist": Camera().read()}}

        with tempfile.TemporaryDirectory() as temp, patch.object(sys.modules[__name__], "OtherEnv", OtherEnv, create=True):
            folder = Path(temp)
            cfg = folder / "station.json"
            cfg.write_text(json.dumps({"env": f"{__name__}:OtherEnv"}))
            with redirect_stdout(io.StringIO()):
                main(["direct", "--config", str(cfg), "--out", str(folder / "check"), "--check"])
            result = json.loads((folder / "check/result.json").read_text())
            self.assertEqual(result["capabilities"], OtherEnv.capabilities)
            self.assertTrue((folder / "check" / result["images"]["wrist"]).is_file())

    @patch.dict(os.environ, CLEANRUA_BASE_URL="http://unused", CLEANRUA_MODEL_NAME="test", CLEANRUA_API_KEY="test")
    def test_direct_delegate_monitor(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            policy = folder / "policy.json"
            policy.write_text(json.dumps(spec(Policy)))
            target = {"arm": "left", "pose": [.04, 0, 0, 0, 0, 0]}
            for method in ("direct", "delegate", "monitor"):
                calls = [reply("move", targets=[target]), reply("finish")]
                if method == "delegate":
                    calls.insert(0, reply("vla", instruction="grasp", steps=1))
                if method == "monitor":
                    calls.insert(0, reply("run_policy", steps=1))
                text_only = {"role": "assistant", "content": "Inspecting."}
                with patch("httpx.post", side_effect=[response(text_only), *map(response, calls)]) as model:
                    options = ["--policy-config", str(policy)] if method != "direct" else []
                    report = self.launch(method, folder, "--enable-motion", "--task", "stack", *options,
                                         "--model", "qwen4-max", "--reasoning-effort", "xhigh", "--tool-choice", "auto")
                self.assertEqual(report["stop_reason"], "finish")
                bodies = [call.kwargs["json"] for call in model.call_args_list]
                self.assertTrue(all((b["model"], b["reasoning_effort"], b["tool_choice"], b["parallel_tool_calls"])
                                    == ("qwen4-max", "xhigh", "auto", False) for b in bodies))
                self.assertEqual(bodies[1]["messages"][-2:], [text_only,
                                 {"role": "user", "content": "Reply with exactly one tool call."}])
                tools = bodies[-1]["tools"]
                move = next(t["function"] for t in tools if t["function"]["name"] == "move")
                schema = move["parameters"]["properties"]["targets"]
                self.assertEqual(schema["maxItems"], 1)
                self.assertEqual(schema["items"]["properties"]["arm"]["enum"], ["left", "right"])
                feedback = bodies[-1]["messages"][-1]["content"][0]["text"]
                self.assertEqual(json.loads(feedback.removeprefix("State: "))["arms"]["left"]["pose"], target["pose"])
            self.assertEqual(Policy.instructions[-2:], ["grasp", "stack"])

    @patch.dict(os.environ, CLEANRUA_BASE_URL="http://unused", CLEANRUA_MODEL_NAME="custom-model", CLEANRUA_API_KEY="test")
    def test_text_only_retry_limit_and_http_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with patch("httpx.post", return_value=response({"content": "Wait"})) as model:
                with self.assertRaisesRegex(RuntimeError, "without a tool call"):
                    self.launch("direct", folder, "--enable-motion", "--task", "inspect")
            self.assertEqual(model.call_count, 3)
            body = model.call_args.kwargs["json"]
            self.assertEqual(body["model"], "custom-model")
            self.assertNotIn("reasoning_effort", body)
        with tempfile.TemporaryDirectory() as temp, patch("httpx.post", return_value=response({}, 403)) as model:
            with self.assertRaises(httpx.HTTPStatusError):
                self.launch("direct", Path(temp), "--enable-motion", "--task", "inspect")
            model.assert_called_once()

    @unittest.skipUnless(sys.platform == "linux" and shutil.which("bwrap"), "needs Linux bubblewrap")
    @patch.dict(os.environ, CLEANRUA_BASE_URL="http://unused", CLEANRUA_MODEL_NAME="test", CLEANRUA_API_KEY="test")
    def test_cap_explore_compile_replay(self):
        cell = ('s = robot.observe()["state"]["arms"]["right"]\n'
                'print(robot.move(targets=[dict(arm="right", pose=s["pose"], gripper=.3)])["arms"]["right"]["gripper"])')
        source = "def main(robot):\n    " + cell.replace("\n", "\n    ")
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            memory = folder / "cap"
            compiled = {"role": "assistant", "content": f"```markdown\nUse live poses.\n```\n```python\n{source}\n```"}
            replies = [{"role": "assistant", "content": "Inspecting."}, reply("run_code", code=cell),
                       reply("finish"), compiled]

            def fake_response(message):
                return httpx.Response(200, json={"choices": [{"message": message}], "usage": {"total_tokens": 7}},
                                      request=httpx.Request("POST", "http://unused/chat/completions"))

            with patch("httpx.Client.post", side_effect=map(fake_response, replies)) as model:
                report = self.launch("cap", folder, "--enable-motion", "--task", "inspect", "--model", "qwen4-max",
                                     "--reasoning-effort", "xhigh", "--tool-choice", "auto")
            self.assertEqual((model.call_count, report["requests"], report["compile_requests"]), (4, 3, 1))
            for call in model.call_args_list:
                body = call.kwargs["json"]
                self.assertEqual((body["model"], body["reasoning_effort"]), ("qwen4-max", "xhigh"))
            self.assertEqual(model.call_args_list[0].kwargs["json"]["tool_choice"], "auto")
            self.assertNotIn("tools", model.call_args.kwargs["json"])  # Upstream text compiler.
            self.assertIsNone(report["task_success"])
            self.assertFalse(report["controller_verified"])
            self.assertEqual((memory / "notes.md").read_text(), "Use live poses.")
            self.assertEqual((memory / "controller.py").read_text().strip(), source)
            self.assertFalse((memory / "verified.py").exists())
            events = [json.loads(l) for l in (memory / "trajectory.jsonl").read_text().splitlines()]
            self.assertEqual(next(e["feedback"]["output"] for e in events if e["event"] == "step"), "0.3\n")
            with redirect_stdout(io.StringIO()), patch.dict(os.environ, {}, clear=True), patch("httpx.Client.post") as http:
                main(["cap", "--config", str(folder / "station.json"), "--out", str(folder / "replay"),
                      "--enable-motion", "--replay", str(memory / "controller.py")])
                http.assert_not_called()
            replay = json.loads((folder / "replay/result.json").read_text())
            self.assertEqual((replay["feedback"]["output"], replay["requests"]), ("0.3\n", 0))
            with redirect_stdout(io.StringIO()), patch("httpx.Client.post", side_effect=map(fake_response, [
                    reply("finish"), {"content": "invalid compiler response"}])) as model:
                with self.assertRaisesRegex(RuntimeError, "python block"):
                    main(["cap", "--config", str(folder / "station.json"), "--out", str(folder / "next"),
                          "--memory-dir", str(memory), "--enable-motion", "--task", "inspect"])
            prompt = model.call_args_list[0].kwargs["json"]["messages"][1]["content"][0]["text"]
            self.assertIn("Use live poses.", prompt)
            self.assertIn(source, prompt)
            self.assertIn("0.3", prompt)  # Rehearsal output, prior to any model request.
            self.assertEqual((memory / "controller.py").read_text().strip(), source)
            self.assertEqual(json.loads((memory / "status.json").read_text())["version"], 1)
            from realworld.cap import main as retry_compile
            with redirect_stdout(io.StringIO()), patch("httpx.Client.post", return_value=fake_response(compiled)) as model:
                retry_compile(["--run-dir", str(folder / "next"), "--model", "qwen4-max", "--reasoning-effort", "xhigh"])
            self.assertEqual(model.call_count, 1)
            self.assertEqual((folder / "next/controller.py").read_text().strip(), source)
