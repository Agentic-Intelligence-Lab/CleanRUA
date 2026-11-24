import json
import unittest
from unittest.mock import Mock

from realworld.direct import RealWrapper
from realworld.workstation import mock_profile, workstation


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
        config = mock_profile()
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
