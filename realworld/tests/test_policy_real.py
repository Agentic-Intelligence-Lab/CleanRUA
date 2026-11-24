"""CPU checks for both real-station policy wrappers and their shared policy clients."""
import base64
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from realworld.delegate import RealWrapper as Delegate
from realworld.monitor import RealWrapper as Monitor
from realworld.__main__ import run_method
from realworld.real_policy import CONTRACT, ChunkPolicy, DexJoCoOpenPI, HTTPArmPolicy
from realworld.workstation import legacy_profile, workstation


class PolicyTest(unittest.TestCase):
    def setUp(self):
        cfg = legacy_profile()
        cfg["arms"]["right"] = copy.deepcopy(cfg["arms"]["left"])
        cfg["cameras"]["wrist"] = cfg["cameras"]["scene"]
        self.bench = workstation(cfg, True)
        self.addCleanup(self.bench.close)
        self.client = Mock()
        self.policy = ChunkPolicy(self.client, 1, 1000)
        self.seen = []

        def predict(obs, instruction):
            self.seen.append((instruction, obs["arms"]["left"]["pose"][0], tuple(obs["images"])))
            target = {s: {"pose": np.array(a["pose"])} for s, a in obs["arms"].items()}
            for arm in target.values():
                arm["pose"][0] += .005
            return [target, {"left": {"pose":[100.]*6}}]  # discard the unused prediction
        self.client.predict.side_effect = predict

    def test_methods_keep_distinct_instruction_and_replan(self):
        for name, tool, instruction in [("delegate","actor_act","reach"), ("monitor","run_policy","place")]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                start = self.bench.state("right")["pose"][0]
                args = {"steps": 2}
                if name == "delegate":
                    args.update(instruction=instruction, active_groups=["left"])
                def message(name, args):
                    return {"role": "assistant", "tool_calls": [{"id": name, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}
                agent = Mock(side_effect=[message(tool,args), message("finish",{})])
                stop = run_method(self.bench, "place", Path(directory), algorithm=name, policy=self.policy, decide=agent)
                self.assertEqual(stop, "finish")
                self.assertEqual([x[0] for x in self.seen[-2:]], [instruction]*2)
                self.assertAlmostEqual(self.seen[-1][1] - self.seen[-2][1], .005)
                self.assertEqual(self.seen[-1][2], ("scene","wrist"))
                self.assertAlmostEqual(self.bench.state("right")["pose"][0] - start, 0 if name == "delegate" else .01)
                rows = list(map(json.loads,(Path(directory)/"trajectory.jsonl").read_text().splitlines()))
                self.assertEqual(sum("policy_measured" in r for r in rows), 2)

    def test_monitor_rejects_rewritten_prompt_and_malformed_chunks(self):
        env = Monitor(self.bench, self.policy, "original")
        def call(args):
            return env.step({"name": "run_policy", "arguments": json.dumps(args)})[-1]["result"]
        self.assertIn("error", call({"steps": 1, "instruction":"replace"}))
        self.client.predict.assert_not_called()
        self.client.predict.side_effect = None
        with patch.object(self.bench, "step_groups") as step:
            for chunk in ([], [[0]*79], [[float("nan")]*80]):
                self.client.predict.return_value = chunk
                with self.assertRaises(ValueError):
                    call({"steps": 1})
            step.assert_not_called()

    def test_partial_rejection_no_retry(self):
        env = Delegate(self.bench, ChunkPolicy(self.client, 3, 1000), "place")
        pose = self.bench.state("left")["pose"]
        pose[0] += .005
        first = {"left": {"pose": pose}}
        self.client.predict.side_effect = lambda *_: [first, {"left":{"gripper":2}}, first]
        result = env.step({"name": "actor_act", "arguments": '{"instruction":"reach","steps":3}'})[-1]["result"]
        self.assertEqual((result["steps"],result["stop_reason"]), (1,"target_rejected"))
        self.client.predict.assert_called_once()
        self.assertAlmostEqual(self.bench.state("left")["pose"][0], .405)

    def test_http_adapter_sends_all_views_and_rejects_wrong_contract(self):
        obs = self.bench.observe()
        with patch("realworld.real_policy.httpx.post") as post:
            post.return_value.json.return_value = {"contract":"delta", "actions":[]}
            with self.assertRaises(ValueError):
                HTTPArmPolicy("http://example.invalid/infer").predict(obs,"phase")
            request = post.call_args.kwargs["json"]
            self.assertEqual(request["contract"], CONTRACT)
            for name, data in request["images_png"].items():
                np.testing.assert_array_equal(Image.open(io.BytesIO(base64.b64decode(data))), obs["images"][name])
            self.assertEqual(set(request["images_png"]),set(obs["images"]))
            self.assertEqual(request["arms"],obs["arms"])
            self.assertNotIn("mask", request)

    def test_native_openpi_single_and_dual(self):
        for sides in [("right",),("right","left")]:
            with self.subTest(sides=sides):
                slots = list(reversed(range(16)))
                with patch.dict(sys.modules, {"openpi_client.websocket_client_policy": Mock()}):
                    adapter = DexJoCoOpenPI("unused", {"ego":"scene"}, {s: slots for s in sides})
                obs = self.bench.observe()
                obs["arms"] = {side: obs["arms"][side] for side in sides}
                raw = np.zeros((2,44))
                for i, side in enumerate(sides):
                    obs["arms"][side].pop("gripper")
                    obs["arms"][side]["hand"] = (np.arange(16)[::-1] + i).tolist()
                    raw[:,22*i:22*i+3] = [i+.3,0,.2]
                    raw[:,22*i+5] = np.pi/2
                    raw[:,22*i+6:22*i+22] = np.arange(16)/20 + i
                raw.setflags(write=False)  # openpi_client decodes an immutable msgpack buffer
                adapter.client.infer.return_value = {"actions":raw}
                out = adapter.predict(obs, "reach")
                req = adapter.client.infer.call_args.args[0]
                self.assertEqual(req["state"].shape,(46,))
                self.assertEqual(req["ego"].shape,(224,224,3))
                np.testing.assert_allclose(req["state"][3:7],[1,0,0,0])
                for i, side in enumerate(sides):
                    np.testing.assert_allclose(req["state"][7*len(sides)+16*i:7*len(sides)+16*(i+1)], np.arange(16)+i)
                    np.testing.assert_allclose(out[0][side]["pose"],[i+.3,0,.2,0,0,np.pi/2], atol=1e-9)
                    np.testing.assert_allclose(out[0][side]["hand"],np.arange(16)[::-1]/20 + i)
                if len(sides)==1:
                    np.testing.assert_array_equal(req["state"][23:],0)
