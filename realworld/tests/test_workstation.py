import copy
import unittest
from unittest.mock import Mock

import numpy as np

from realworld.workstation import mock_profile, workstation


class WorkstationTest(unittest.TestCase):
    def setUp(self):
        config = mock_profile()
        hand = copy.deepcopy(config["arms"]["left"])
        hand.update(end_effector="hand",
                    hand_limits_rad=[[-1]*3, [1]*3],
                    tool={"driver": "realworld.tests.fixtures.devices:MockHand", "args": {"joints": 3}})
        config["arms"]["right"] = hand
        config["cameras"] = dict.fromkeys(("front", "left_wrist", "right_wrist", "head"), config["cameras"]["scene"])
        self.env = workstation(config, True)
        self.addCleanup(self.env.close)

    def test_mixed_tools_and_four_views(self):
        obs, _ = self.env.reset()
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertEqual(len(obs["images"]), 4)
        self.assertEqual(set(obs["arms"]), {"left", "right"})
        self.assertNotIn("mask", obs)
        self.assertNotIn("state", obs)
        pose = obs["arms"]["right"]["pose"].copy()
        pose[5] += .1
        target = {"left": {"gripper": 0.}, "right": {"pose": pose, "hand": [.1,.2,.3]}}
        self.assertTrue(self.env.action_space.contains(target))
        after, *_ = self.env.step(target)
        np.testing.assert_allclose(after["arms"]["right"]["pose"], pose)
        self.assertEqual(after["arms"]["left"]["gripper"], 0.)
        self.assertEqual(self.env.arms["right"].tool.positions, [.1,.2,.3])

    def test_tool_only_command_leaves_poses_and_other_arm_untouched(self):
        left = self.env.arms["left"]
        left.move_pose = Mock()
        self.env.arms["right"].move_pose = Mock()
        self.env.step({"left": {"gripper": .25}})
        self.assertEqual(left.opening, .25)
        left.move_pose.assert_not_called()
        self.env.arms["right"].move_pose.assert_not_called()

    def test_all_arms_validated_before_dispatch(self):
        left = self.env.arms["left"]
        left.move_pose = Mock()
        pose = self.env.state("left")["pose"]
        pose[0] += .01
        target = {"left": {"pose": pose}, "right": {"hand": [2., 0., 0.]}}
        with self.assertRaises(ValueError):
            self.env.step(target)
        left.move_pose.assert_not_called()

    def test_named_move_does_not_address_other_arm(self):
        left = self.env.arms["left"]
        left.move_pose = Mock()
        pose = self.env.state()["right"]["pose"]
        self.env.move(pose, side="right", hand=[.3, .1, .2])
        left.move_pose.assert_not_called()
        with self.assertRaises(ValueError):
            self.env.move(pose, side="right", gripper=0.)
