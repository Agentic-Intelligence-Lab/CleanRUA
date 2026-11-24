import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from realworld.direct import RealWrapper
from realworld.drivers.robotiq_ros2 import Robotiq2F85
from realworld.workstation import legacy_profile, workstation


class GripToolTest(unittest.TestCase):
    def test_gripper_only_action(self):
        env = workstation(None, True)
        self.addCleanup(env.close)
        arm = env.arms["left"]
        arm.move_pose = Mock()
        wrapper = RealWrapper(env)
        self.assertIn("grip", [s["function"]["name"] for s in wrapper.schemas()])
        for opening in (.8, 1.):
            _, _, _, _, info = wrapper.step({"name": "grip", "arguments": json.dumps({"opening":opening})})
            self.assertEqual(info["result"]["measured"]["gripper"], opening)
        arm.move_pose.assert_not_called()

    def test_arm_only_has_no_grip_tool(self):
        config = legacy_profile()
        config["arms"]["left"]["end_effector"] = "none"
        env = workstation(config)
        self.addCleanup(env.close)
        self.assertNotIn("grip", RealWrapper(env).tools)

    def test_recovered_receipt_reaches_direct(self):
        for name in ("grip", "move"):
            env = workstation(None, True)
            self.addCleanup(env.close)
            arm = env.arms["left"]
            report = {"controller_status": "aborted_stalled", "target_reached": False,
                      "requested_opening": 0., "measured_opening": .25}
            def close_to_object(opening):
                arm.opening = .25
                return report
            arm.set_gripper = close_to_object
            args = {"opening": 0.} if name == "grip" else {"pose": arm.pose, "gripper": 0.}
            obs, _, _, _, info = RealWrapper(env).step({"name": name, "arguments": json.dumps(args)})
            self.assertEqual(json.loads(obs["feedback"]), info["result"])
            measured = info["result"]["measured"]
            self.assertEqual(measured["gripper"], .25)
            self.assertEqual(info["result"].get("gripper_results", measured.get("gripper_results")),
                             {"left": report})
            self.assertTrue(env.motion_enabled)


class RobotiqReceiptTest(unittest.TestCase):
    def setUp(self):
        self.tool = Robotiq2F85.__new__(Robotiq2F85)
        self.tool.closed_rad, self.tool.max_effort = .7929, 20.
        self.tool.node, self.tool.ros, self.tool.client = Mock(), Mock(), Mock()
        self.tool.goal_type = SimpleNamespace(Goal=lambda: SimpleNamespace(command=SimpleNamespace()))
        self.tool.check_motion_ready = Mock()
        self.tool.state = Mock(return_value={"gripper": 1.})
        self.tool._goal_tolerance = Mock(return_value=.1)
        self.tool._settled_opening = Mock(return_value=1. - .7580 / .7929)
        handle = self.tool.client.send_goal_async.return_value.result.return_value
        self.receipt = handle.get_result_async.return_value.result.return_value
        self.receipt.status, self.receipt.result.stalled = 6, True
        module = SimpleNamespace(GoalStatus=SimpleNamespace(STATUS_ABORTED=6, STATUS_SUCCEEDED=4))
        mock_ros = patch.dict(sys.modules, {"action_msgs.msg": module})
        mock_ros.start()
        self.addCleanup(mock_ros.stop)

    def test_stalled_close_reports_actual_result(self):
        self.assertTrue(self.tool.set_gripper(0.)["target_reached"])
        self.tool._settled_opening.return_value = .4
        result = self.tool.set_gripper(0.)
        self.assertEqual(result["measured_opening"], .4)
        self.assertFalse(result["target_reached"])

    def test_stalled_open_and_other_aborts(self):
        self.tool.state.return_value = {"gripper": 0.}
        self.tool._settled_opening.return_value = 1.
        self.assertTrue(self.tool.set_gripper(1.)["target_reached"])
        self.tool._settled_opening.return_value = .3
        with self.assertRaisesRegex(RuntimeError, "opening stopped short"):
            self.tool.set_gripper(1.)
        self.receipt.result.stalled = False
        with self.assertRaisesRegex(RuntimeError, "did not confirm"):
            self.tool.set_gripper(1.)

    def test_settling_requires_fresh_stationary_samples(self):
        del self.tool._settled_opening
        self.tool.node.get_clock.return_value.now.return_value.nanoseconds = 0
        for mode in ("moving_then_still", "always_moving", "no_feedback", "sparse_feedback"):
            now = 0
            self.tool.stamp = 0
            def spin(*args, **kwargs):
                nonlocal now
                now += 100_000_000
                if mode == "no_feedback" or (mode == "sparse_feedback" and now % 500_000_000):
                    return
                self.tool.stamp = now
                self.tool.opening = max(0., 1. - now / 2e9) if mode == "moving_then_still" else .3
                if mode == "always_moving":
                    self.tool.opening = (now % 2_000_000_000) / 2e9
            self.tool.ros.spin_once.side_effect = spin
            with self.subTest(mode=mode), patch("realworld.drivers.robotiq_ros2.time.monotonic",
                                                side_effect=lambda: now / 1e9):
                if mode == "moving_then_still":
                    self.assertEqual(self.tool._settled_opening(), 0.)
                    self.assertGreaterEqual(now, 2_750_000_000)
                else:
                    with self.assertRaisesRegex(RuntimeError, "No stable fresh"):
                        self.tool._settled_opening()
