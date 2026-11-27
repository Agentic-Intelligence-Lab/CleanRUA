"""Driver feedback for a trajectory definitively rejected before hardware dispatch."""


class TrajectoryRejected(Exception):
    def __init__(self, violation, *, trajectory=None):
        super().__init__(str(violation))
        self.violation = violation
        self.trajectory = trajectory
        self.arm = None
