"""Single-arm bench interface. Positions use base-frame metres, angles fixed XYZ radians."""
import numpy as np
from scipy.spatial.transform import Rotation


class FrankaArm:
    def __init__(self, host, with_gripper=True, flange_T_tcp=None):
        from franky import Gripper, RelativeDynamicsFactor, Robot

        self.flange_T_tcp = None
        if flange_T_tcp is not None:
            transform = np.asarray(flange_T_tcp, dtype=float)
            if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                    or not np.allclose(transform[3], [0, 0, 0, 1])
                    or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3))
                    or not np.isclose(np.linalg.det(transform[:3, :3]), 1)):
                raise ValueError("flange_T_tcp must be a rigid 4x4 transform in metres")
            self.flange_T_tcp = transform
        self.robot = Robot(host)
        self.hand = Gripper(host) if with_gripper else None
        self.robot.relative_dynamics_factor = RelativeDynamicsFactor(velocity=0.02, acceleration=0.02, jerk=0.01)

    def _ee_T_tcp(self, sample):
        # F_T_EE already includes the end-effector transform configured in Desk.
        return np.linalg.solve(sample.F_T_EE.matrix, self.flange_T_tcp)

    def state(self):
        sample = self.robot.state
        pose = sample.O_T_EE.matrix
        if self.flange_T_tcp is not None:
            pose = pose @ self._ee_T_tcp(sample)
        angles = Rotation.from_matrix(pose[:3, :3]).as_euler("xyz")
        state = {"pose": [*pose[:3, 3], *angles], "joints": np.asarray(sample.q).tolist()}
        if self.hand is not None:
            state["gripper"] = self.hand.width / self.hand.max_width
        return state

    def move_pose(self, pose):
        from franky import Affine, CartesianMotion

        target = Affine(pose[:3], Rotation.from_euler("xyz", pose[3:]).as_quat())
        if self.flange_T_tcp is not None:
            # The requested pose is base_T_tcp. CartesianMotion expects base_T_EE.
            ee = target.matrix @ np.linalg.inv(self._ee_T_tcp(self.robot.state))
            target = Affine(ee[:3, 3], Rotation.from_matrix(ee[:3, :3]).as_quat())
        self.robot.move(CartesianMotion(target))

    def set_gripper(self, opening):
        if not self.hand.move(opening * self.hand.max_width, 0.02):
            raise RuntimeError("Gripper did not reach the requested width")

    def stop(self):
        try:
            self.robot.stop()
        finally:
            if self.hand is not None:
                self.hand.stop()
