import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from realworld.drivers.franky import FrankaArm


def transform(xyz, angles):
    mat = np.eye(4)
    mat[:3, 3] = xyz
    mat[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    return mat


def affine(xyz, quat):
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_quat(quat).as_matrix()
    mat[:3, 3] = xyz
    return SimpleNamespace(matrix=mat)


class FrankaTCPTest(unittest.TestCase):
    def test_observed_tcp_and_commanded_ee(self):
        base_flange = transform([.4, .1, .5], [2.8, -.2, .5])
        flange_tcp = transform([0, 0, .18], [0, 0, 0])
        goal = transform([.45, .05, .35], [2.6, .1, -.2])
        for flange_ee in (np.eye(4), flange_tcp, transform([.01, .02, .09], [.1, .2, .3])):
            robot = Mock()
            robot.state = SimpleNamespace(q=np.zeros(7), F_T_EE=SimpleNamespace(matrix=flange_ee),
                                          O_T_EE=SimpleNamespace(matrix=base_flange @ flange_ee))
            fake = SimpleNamespace(Robot=lambda host: robot, Gripper=None, RelativeDynamicsFactor=SimpleNamespace,
                                   Affine=affine, CartesianMotion=lambda t: t)
            with patch.dict(sys.modules, franky=fake):
                for calibration in (flange_tcp, None):
                    arm = FrankaArm("mock", with_gripper=False, flange_T_tcp=calibration)
                    self.assertEqual(vars(robot.relative_dynamics_factor),
                                     {"velocity": 0.02, "acceleration": 0.02, "jerk": 0.01})
                    pose = arm.state()["pose"]
                    np.testing.assert_allclose(transform(pose[:3], pose[3:]), base_flange @ (
                        flange_tcp if calibration is not None else flange_ee), atol=1e-10)
                    arm.move_pose([.45, .05, .35, 2.6, .1, -.2])
                    expected = goal @ np.linalg.inv(flange_tcp) @ flange_ee if calibration is not None else goal
                    np.testing.assert_allclose(robot.move.call_args.args[0].matrix, expected, atol=1e-10)
