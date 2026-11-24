"""Offline plug-in boundary tests: use the same fixture on both station layouts."""
import json
from pathlib import Path
import runpy
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from realworld.cap import ProgramSession
from realworld.__main__ import run_method
from realworld.delegate import RealWrapper as Delegate
from realworld.monitor import RealWrapper as Monitor
from realworld.real_policy import ChunkPolicy
from realworld.workstation import workstation


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class RealPluginTest(unittest.TestCase):
    def bench(self, dual):
        config = json.loads((FIXTURES / "dual_mock.json").read_text()) if dual else None
        bench = workstation(config, motion_enabled=True)
        self.addCleanup(bench.close)
        return bench

    def test_saved_policy_on_both_layouts(self):
        main = runpy.run_path(str(FIXTURES / "nudge.py"))["main"]
        for dual in (False, True):
            with self.subTest(dual=dual):
                bench = self.bench(dual)
                result = main(bench)
                self.assertEqual(result["views"], ["front", "right_wrist"] if dual else ["scene"])
                self.assertEqual(set(result["measured_dx_m"]), set(bench.arms))
                for displacement in result["measured_dx_m"].values():
                    self.assertAlmostEqual(displacement, .005)

    def test_external_algorithm_loaded_by_name(self):
        bench = self.bench(True)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            reason = run_method(bench, "verify X", out, algorithm="realworld.tests.fixtures.nudge:run")
            self.assertEqual(reason, "finish")
            measured = json.loads((out / "measurement.json").read_text())["measured_dx_m"]
            for side in ("left", "right"):
                self.assertAlmostEqual(measured[side], .005)

    def test_sdk_arrays_normalized_at_device_boundary(self):
        bench = self.bench(False)
        measured = {"pose": np.zeros(6, dtype=np.float32), "joints": np.zeros(7), "gripper": np.float32(.5)}
        with patch.object(bench.arms["left"], "state", return_value=measured):
            obs = bench.observe()
            self.assertEqual(json.loads(json.dumps(obs["arms"]))["left"], bench.state(side="left"))
            self.assertEqual(obs["arms"]["left"]["gripper"], .5)
            self.assertTrue(bench.observation_space.contains(obs))

    def test_fault_stops_only_selected_side(self):
        bench = self.bench(True)
        left, right = bench.arms["left"], bench.arms["right"]
        with patch.object(left, "move_pose", side_effect=TimeoutError), patch.object(left, "stop") as stop_left, \
                patch.object(right, "stop") as stop_right:
            goal = bench.state(side="left")["pose"]
            goal[0] += .005
            with self.assertRaises(TimeoutError):
                bench.move(goal, side="left")
            stop_left.assert_called_once()
            stop_right.assert_not_called()
            self.assertFalse(bench.motion_enabled)

    def test_policy_wrappers_reuse_last_control_frame(self):
        for wrapper, name, args in [(Delegate, "actor_act", {"instruction":"hold"}), (Monitor, "run_policy", {})]:
            bench = self.bench(False)
            policy = Mock()
            policy.predict.side_effect = lambda obs, instruction: [{"left": {"pose": obs["arms"]["left"]["pose"]}}]
            agent = wrapper(bench, ChunkPolicy(policy, 1, 10), "hold")
            cam = bench.cameras["scene"]
            with patch.object(cam, "read", wraps=cam.read) as read:
                obs, _, _, _, info = agent.step({"name": name, "arguments": json.dumps({**args, "steps": 1})})
                self.assertEqual(read.call_count, 2)  # before inference, after execution
                self.assertEqual(info["result"]["measured"], obs["arms"]["left"])
                self.assertEqual(int(obs["images"]["scene"][0,0,0]), cam.count)

    @unittest.skipUnless(shutil.which("bwrap"), "Linux CaP worker")
    def test_isolated_policy_uses_same_observations_and_moves_as_replay(self):
        source = (FIXTURES / "nudge.py").read_text()
        for dual in (False, True):
            bench = self.bench(dual)
            session = ProgramSession(bench)
            self.addCleanup(session.close)
            result = session.execute(source + "\nimport json\nprint(json.dumps(main(robot)))")
            self.assertNotIn("error", result, result)
            replay = runpy.run_path(str(FIXTURES / "nudge.py"))["main"](self.bench(dual))
            self.assertEqual(json.loads(result["output"]), replay)
            if dual:
                result = session.execute("im = robot.observe()['images']['right_wrist']; "
                                         "print(im.shape, im.dtype, im[-1,-1].tolist())")
                self.assertEqual(result["output"].strip(), "(240, 320, 3) uint8 [255, 255, 0]")
