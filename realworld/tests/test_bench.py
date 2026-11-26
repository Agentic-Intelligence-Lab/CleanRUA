import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from realworld.direct import run_direct
from realworld.tests.fixtures.devices import MockArm, MockCamera
from realworld.env import Bench
from realworld.program import ProgramSession
from realworld.workstation import mock_profile


class FreshBenchTest(unittest.TestCase):
    def setUp(self):
        self.arm = MockArm()
        self.profiles = mock_profile()["arms"]
        self.bench = Bench({"left": self.arm}, {"scene": MockCamera()}, self.profiles, motion_enabled=True)

    def test_decision_to_measured_state(self):
        def call(name, args):
            return {"role": "assistant", "tool_calls": [{"id": name, "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}}]}
        pose = [0.41, 0, 0.3, 0, 0, 0]
        agent = Mock(side_effect=[call("move", {"pose": pose, "gripper": 0.5}), call("finish", {})])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(run_direct(self.bench, "move once", root, decide=agent), "finish")
            events = [json.loads(line) for line in (root / "trajectory.jsonl").read_text().splitlines()]
            self.assertEqual(events[2]["feedback"]["measured"]["pose"], pose)
            self.assertTrue((root / events[3]["images"]["scene"]).is_file())
            self.assertIsNone(events[-1]["task_success"])
            self.assertEqual(self.arm.opening, 0.5)

    def test_read_only_and_invalid_target(self):
        self.arm.move_pose = Mock()
        with self.assertRaises(ValueError):
            self.bench.move([2, 0, 0.3, 0, 0, 0], 1.)
        bench = Bench({"left": self.arm}, {"scene": MockCamera()}, self.profiles)
        self.assertTrue(bench.observe()["images"])
        with self.assertRaises(RuntimeError):
            bench.move(self.arm.pose, 0.)
        self.arm.move_pose.assert_not_called()

    def test_fault_blocks_followup_and_gripper(self):
        self.arm.move_pose = Mock(side_effect=TimeoutError("unknown result"))
        self.arm.set_gripper, self.arm.stop = Mock(), Mock()
        with self.assertRaises(TimeoutError):
            self.bench.move([0.41, 0., 0.3, 0., 0., 0.], 0.)
        with self.assertRaises(RuntimeError):
            self.bench.move([0.41, 0., 0.3, 0., 0., 0.], 0.)
        self.arm.stop.assert_called_once()
        self.arm.set_gripper.assert_not_called()

    def test_arrival_tolerance_is_separate_from_step_limit(self):
        self.profiles["left"].update(arrival_translation_m=.006, arrival_rotation_rad=.09)
        self.arm.move_pose = Mock()  # Measured pose stays at the start.
        self.bench.move([.405, 0., .3, 0., 0., .08])
        with self.assertRaisesRegex(ValueError, "Step exceeds"):
            self.bench.move([.45, 0., .3, 0., 0., 0.])
        with self.assertRaisesRegex(RuntimeError, "arrival tolerance"):
            self.bench.move([.405, 0., .3, 0., 0., .10])

    def test_decision_budget(self):
        finish = {"role": "assistant", "tool_calls": [{"id": "f", "type": "function",
                  "function": {"name": "finish", "arguments": "{}"}}]}
        action = {"role": "assistant", "tool_calls": [{"id": "g", "type": "function",
                  "function": {"name": "grip", "arguments": '{"opening": 1}'}}]}
        for limit, reason, count in ((0, "finish", 2), (1, "max_decisions", 1)):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as folder:
                model = Mock(side_effect=[action, finish])
                self.assertEqual(run_direct(self.bench, "test", Path(folder), decide=model, max_decisions=limit), reason)
                self.assertEqual(model.call_count, count)
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            run_direct(self.bench, "test", Path(folder), decide=Mock(), max_decisions=-1)

    @unittest.skipUnless(shutil.which("bwrap"), "isolated CaP worker requires Linux bubblewrap")
    def test_program_reports_fault_and_blocks_second_move(self):
        self.arm.move_pose = Mock(side_effect=TimeoutError("unknown result"))
        program = ProgramSession(self.bench)
        self.addCleanup(program.close)
        code = "robot.move([0.41, 0., 0.3, 0., 0., 0.])"
        first = program.execute(code)
        self.assertTrue(first["recovery_required"])
        self.assertFalse(first["motion_enabled"])
        self.assertIn("unknown result", first["motion_fault"]["error"])
        self.assertEqual(first["motion_fault"]["measured_state"]["pose"], self.arm.pose)
        second = program.execute(code)
        self.assertIn("operator recovery required", second["error"])
        self.assertEqual(first["motion_fault"], second["motion_fault"])
        self.arm.move_pose.assert_called_once()
