"""Deterministic interface fixtures. They model neither dynamics nor task success."""
import numpy as np


class MockArm:
    def __init__(self):
        self.pose = [0.4, 0., 0.3, 0., 0., 0.]
        self.opening = 1.

    def state(self):
        return {"pose": self.pose.copy(), "joints": [0.] * 7, "gripper": self.opening}

    def move_pose(self, pose):
        self.pose = list(pose)

    def set_gripper(self, opening):
        self.opening = opening

    def stop(self):
        pass


class MockCamera:
    def __init__(self):
        self.count = 0

    def read(self):
        self.count += 1
        return np.full((48, 64, 3), self.count % 256, dtype=np.uint8)

    def close(self):
        pass


class MockHand:
    def __init__(self, joints):
        self.positions = [0.] * joints

    def state(self):
        return {"hand": self.positions.copy()}

    def set_hand(self, positions):
        self.positions = list(positions)

    def stop(self):
        pass


class PatternCamera:
    def __init__(self, width, height):
        self.rgb = np.zeros((height, width, 3), dtype=np.uint8)
        self.rgb[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
        self.rgb[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]

    def read(self):
        return self.rgb.copy()

    def close(self):
        pass
