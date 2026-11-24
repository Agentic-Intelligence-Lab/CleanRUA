import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from realworld.direct_franka import run_direct
from realworld.bench_mock import MockArm, MockCamera
from realworld.real_arm_env import Bench
from realworld.workstation import legacy_profile


class FreshBenchTest(unittest.TestCase):
    def setUp(self):
        self.arm = MockArm()
        self.profiles = legacy_profile()["arms"]
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
