"""CPU tests for the RoboDojo cap_tools wrapper: geometry from calibrated RGB cameras and the reachability check."""
import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from cleanrua.experimental import cap_tools_robodojo as cap_tools
from test_direct_robodojo import FakeRoboDojo

K = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])


def looking_at(position, target) -> np.ndarray:
    """World-from-optical pose of a camera at `position` looking at `target`: z forward, x right, y down."""
    z = np.asarray(target, float) - position
    z /= np.linalg.norm(z)
    x = np.cross(z, [0., 0., 1.])
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = np.column_stack([x, np.cross(z, x), z]), position
    return pose


class TwoCameraRoboDojo(FakeRoboDojo):
    """Two calibrated 640 x 480 cameras looking at the table from the front and from the side."""

    def _obs(self):
        observation = super()._obs()
        observation['cameras'] = {name: {'rgb': np.zeros((480, 640, 3), np.uint8), 'intrinsic_K': K,
                                         'T_world_from_optical': looking_at(np.array(position), [0., 0., .75])}
                                  for name, position in (('head', (0., -.8, 1.4)), ('side', (.9, .1, 1.2)))}
        return observation


class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.env = cap_tools.RoboDojoWrapper(TwoCameraRoboDojo())
        self.env.reset()

    def run_code(self, code):
        obs, *_ = self.env.step({'name': 'run_code', 'arguments': json.dumps({'code': code})})
        return json.loads(obs['feedback'])

    def test_pixels_of_one_point_triangulate_back_to_it(self):
        feedback = self.run_code(
            "p = np.array([.12, .05, .78])\n"
            "pixels = {name: robot.project(name, p) for name in ('head', 'side')}\n"
            "found = robot.triangulate(pixels)\n"
            "print(np.round(found['point'], 4).tolist(), found['error'] < 1e-6)\n"
            "wrong = robot.triangulate({'head': pixels['head'], 'side': pixels['side'] + [40, 0]})\n"
            "print(wrong['error'] > .01)")
        self.assertTrue(feedback['ok'], feedback)
        self.assertEqual(feedback['output'], '[0.12, 0.05, 0.78] True\nTrue\n')
        self.assertEqual(feedback['steps'], 0)

    def test_a_pixel_ray_meets_the_table_where_the_point_is(self):
        feedback = self.run_code(
            "u, v = robot.project('head', [-.2, .1, .745])\n"
            "print(np.round(robot.point_at_height('head', u, v, .745), 4).tolist())\n"
            "origin, direction = robot.ray('head', 320, 240)\n"
            "print(round(float(np.linalg.norm(direction)), 6))\n"
            "robot.point_at_height('head', 320, 0, 5.)")
        self.assertEqual(feedback['output'], '[-0.2, 0.1, 0.745]\n1.0\n')
        self.assertIn('does not meet the plane', feedback['error'])

    def test_reachable_checks_poses_without_moving(self):
        solve_ik = self.env.unwrapped.solve_ik
        self.env.unwrapped.solve_ik = lambda arm, position, quaternion: (  # Nothing beyond x = 1 m is reachable.
            None if position[0] > 1 else solve_ik(arm, position, quaternion))
        feedback = self.run_code(
            "print(robot.reachable([{'arm': 'left', 'pose': [-.3, 0., .9, 0., 0., 0.]},"
            " {'arm': 'right', 'pose': [2., 0., .9, 0., 0., 0.]}]))")
        self.assertTrue(feedback['ok'], feedback)
        self.assertEqual((feedback['output'], feedback['steps']), ('[True, False]\n', 0))

    def test_move_turns_a_grasp_half_a_turn_to_keep_the_wrist_off_its_limit(self):
        solve_ik = self.env.unwrapped.solve_ik

        def near_limit(arm, position, quaternion):  # The wrist roll nears pi as the fingers close along -world y.
            joints = solve_ik(arm, position, quaternion).copy()
            joints[5] = 1.55 * (1 - Rotation.from_quat(quaternion, scalar_first=True).apply([0., 1., 0.])[1])
            return joints

        self.env.unwrapped.solve_ik = near_limit
        feedback = self.run_code(
            "print(robot.move([{'arm': 'left', 'pose': [-.3, 0., .85, np.pi, 0., 0.]}], max_steps=4)['half_turned'])\n"
            "print(robot.move([{'arm': 'left', 'pose': [-.3, 0., .85, 0., 0., 0.]}], max_steps=4)['half_turned'])")
        self.assertTrue(feedback['ok'], feedback)
        self.assertEqual(feedback['output'], "['left']\n[]\n")


if __name__ == '__main__':
    unittest.main()
