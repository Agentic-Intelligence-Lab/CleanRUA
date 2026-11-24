import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx

from cleanrua import cap, direct
from realworld.__main__ import main as workstation_cli, run_method
from realworld.cap import compile_controller, main as cap_cli, run_cap
from realworld.workstation import workstation


def message(name, arguments):
    return {"role": "assistant", "tool_calls": [{"type": "function", "id": "call",
            "function": {"name": name, "arguments": json.dumps(arguments)}}]}


class MethodParityTest(unittest.TestCase):
    def test_direct_prototype_and_station(self):
        bench = workstation(None, True)
        self.addCleanup(bench.close)
        target = [0.41, 0., 0.3, 0., 0., 0.]
        replies = [message("move", {"pose": target, "gripper": 1.}), message("finish", {})]
        sim = Mock()
        sim.observe.side_effect = lambda: {"images": {}, "state": {}}
        sim.ik.return_value = ["waypoint"]
        sim.step.return_value = False
        model = Mock(side_effect=replies)
        with patch.object(direct, "mllm", model):
            self.assertEqual(direct.direct(sim, "translate"), "finish")
        sim.step.assert_called_once_with("waypoint")
        with tempfile.TemporaryDirectory() as folder:
            model = Mock(side_effect=replies)
            self.assertEqual(run_method(bench, "translate", Path(folder), decide=model), "finish")
            self.assertEqual(bench.state()["pose"], target)
            self.assertIn("measured", model.call_args.args[0][-2]["content"])

    @unittest.skipUnless(shutil.which("bwrap"), "isolated program executor requires Linux bubblewrap")
    def test_cap_program_across_turns(self):
        bench = workstation(None, True)
        self.addCleanup(bench.close)
        code = "target = robot.state()['pose'].copy(); target[0] += 0.005"
        controller = ("def main(robot):\n"
                      "    pose = robot.state()['pose'].copy()\n"
                      "    pose[0] += 0.005\n"
                      "    return robot.move(pose, gripper=1.0)\n")
        model = Mock(side_effect=[message("run_code", {"code": code}),
                                  message("run_code", {"code": "print(robot.move(target, 1.0))"}),
                                  message("finish", {}), message("save_controller", {"code": controller})])
        with tempfile.TemporaryDirectory() as folder:
            reason = run_method(bench, "translate", Path(folder), algorithm="cap", decide=model)
            self.assertEqual(reason, "finish")
            self.assertAlmostEqual(bench.state()["pose"][0], .405)
            self.assertIn("0.405", model.call_args_list[-2].args[0][-2]["content"])
            request = model.call_args.args[0][-1]["content"]
            self.assertIn(code, request[0]["text"])
            self.assertTrue(any(part["type"] == "image_url" for part in request))
            path = Path(folder) / "controller.py"
            self.assertEqual(path.read_text(), controller)
            self.assertFalse(json.loads((path.parent / "result.json").read_text())["controller_verified"])
            next_bench = workstation(None, True)
            next_bench.arms["left"].pose[0] = .45
            with patch("realworld.__main__.workstation", return_value=next_bench):
                workstation_cli(["replay", "--policy", str(path), "--out", str(path.parent / "replay")])
            self.assertAlmostEqual(next_bench.state()["pose"][0], .455)
            self.assertEqual(model.call_count, 4)

    def test_api_only_compiles_and_retries_without_devices(self):
        config = {"CLEANRUA_DECISION_BACKEND": "api", "CLEANRUA_API_KEY": "test"}
        source = "def main(robot):\n    return robot.state()\n"
        for responses_api in (True, False):
            config["CLEANRUA_BASE_URL"] = "https://api.openai.com/v1" if responses_api else "https://example.test/v1"
            config["CLEANRUA_MODEL_NAME"] = "gpt-6.2-astra" if responses_api else "fixture-vlm"
            sent = []
            def endpoint(request):
                body = json.loads(request.content)
                self.assertEqual(body["model"], config["CLEANRUA_MODEL_NAME"])
                self.assertEqual(request.headers["Authorization"], "Bearer test")
                sent.append(request.url.path)
                tools = body["tools"] if responses_api else [t["function"] for t in body["tools"]]
                if len(sent) == 1:
                    self.assertIn("finish", [t["name"] for t in tools])
                    name, args = "finish", {}
                else:
                    self.assertEqual([t["name"] for t in tools], ["save_controller"])
                    if not responses_api and len(sent) == 2:
                        return httpx.Response(503)
                    name, args = "save_controller", {"code": source}
                if responses_api:
                    payload = {"status": "completed", "output": [{"type": "function_call", "call_id": "c",
                               "name": name, "arguments": json.dumps(args)}]}
                else:
                    payload = {"choices": [{"message": message(name, args)}]}
                return httpx.Response(200, json=payload)
            with self.subTest(responses_api=responses_api), tempfile.TemporaryDirectory() as folder, \
                    patch.dict(os.environ, config), httpx.Client(transport=httpx.MockTransport(endpoint)) as client, \
                    patch("realworld.real_decision.httpx.post", client.post):
                root = Path(folder)
                bench = workstation(None, False)
                self.addCleanup(bench.close)
                if responses_api:
                    self.assertEqual(run_cap(bench, "read the current state", root), "finish")
                    self.assertEqual(sent, ["/v1/responses"] * 2)
                else:
                    with self.assertRaises(httpx.HTTPStatusError):
                        run_cap(bench, "read the current state", root)
                    self.assertIn("controller_error", json.loads((root / "result.json").read_text()))
                    with patch.object(sys, "argv", ["cap", "compile", "--run-dir", folder]), \
                            patch("realworld.cap.workstation", side_effect=AssertionError("devices opened")):
                        cap_cli()
                    self.assertEqual(sent, ["/v1/chat/completions"] * 3)
                self.assertEqual((root / "controller.py").read_text(), source)
                report = json.loads((root / "result.json").read_text())
                self.assertFalse(report["controller_verified"])
                self.assertNotIn("controller_error", report)

    def test_offline_controller_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [{"turn": 0, "task": "observe", "images": {}, "arms": {}},
                    {"turn": 0, "request": {"name": "run_code", "arguments": "{}"}},
                    {"stop": "finish", "task_success": None}]
            (root / "trajectory.jsonl").write_text("\n".join(map(json.dumps, rows)))
            for invalid in ("def main(robot)", "def main(): pass"):
                with self.assertRaises((SyntaxError, ValueError)):
                    compile_controller(root, {}, decide=Mock(return_value=message("save_controller", {"code": invalid})))
                self.assertFalse((root / "controller.py").exists())
            src = "def main(robot):\n    return robot.state()\n"
            model = Mock(return_value=message("save_controller", {"code": src}))
            self.assertEqual(compile_controller(root, {}, decide=model).read_text(), src)
            with self.assertRaises(FileExistsError):
                compile_controller(root, {}, decide=model)
            model.assert_called_once()

    def test_cap_simulation_default(self):
        sim = Mock()
        sim.observe.return_value = {"images": {}, "state": {"pose": [0.] * 6}}
        sim.ik.return_value = [0]
        sim.step.return_value = False
        model = Mock(side_effect=[message("run_code", {"code": "p = robot.state()['pose']"}),
                                  message("run_code", {"code": "robot.move(p, 1.0)"}),
                                  message("finish", {})])
        with patch.object(cap, "mllm", model):
            self.assertEqual(cap.cap(sim, "test"), "finish")
        sim.step.assert_called_once_with(0)
