"""Robotiq 2F-85 with a ROS 2 GripperCommand action and measured knuckle position.

The action controller must accept knuckle radians. Hardware speed/force limits must
be commissioned on the controller. This client never loads or activates controllers.
"""
import math
import time


class Robotiq2F85:
    def __init__(self, action, joint_states, max_effort, joint="robotiq_85_left_knuckle_joint",
                 closed_rad=0.7929):
        import rclpy
        from control_msgs.action import GripperCommand
        from rcl_interfaces.srv import GetParameters
        from rclpy.action import ActionClient
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState

        if not (math.isfinite(closed_rad) and closed_rad > 0 and math.isfinite(max_effort) and max_effort > 0):
            raise ValueError("Expected positive calibrated joint range and effort limit")
        self.ros, self.goal_type = rclpy, GripperCommand
        self.joint, self.closed_rad, self.max_effort = joint, closed_rad, max_effort
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("cleanrua_robotiq_client")
        self.client = ActionClient(self.node, GripperCommand, action)
        controller = self.node.resolve_topic_name(action).rsplit("/", 1)[0]
        self.params = self.node.create_client(GetParameters, controller + "/get_parameters")
        self.sub = self.node.create_subscription(JointState, joint_states, self._sample, qos_profile_sensor_data)
        self.opening, self.received, self.goal = None, 0., None
        self.stamp = 0

    def _sample(self, msg):
        if self.joint in msg.name:
            angle = float(msg.position[msg.name.index(self.joint)])
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            age = (self.node.get_clock().now().nanoseconds - stamp) / 1e9
            if math.isfinite(angle) and -.05 <= age <= .5:
                self.opening = max(0., min(1., 1. - angle / self.closed_rad))
                self.received = time.monotonic()
                self.stamp = stamp

    def state(self):
        began = time.monotonic()
        while self.received < began:
            if time.monotonic() - began > 2.:
                raise RuntimeError("No fresh measured Robotiq joint state")
            self.ros.spin_once(self.node, timeout_sec=.1)
        return {"gripper": self.opening}

    def check_motion_ready(self):
        if not self.client.wait_for_server(timeout_sec=1.):
            raise RuntimeError("Robotiq GripperCommand action is unavailable; check controller ownership")

    def _goal_tolerance(self):
        from rcl_interfaces.msg import ParameterType
        from rcl_interfaces.srv import GetParameters

        future = self.params.call_async(GetParameters.Request(names=["goal_tolerance"]))
        self.ros.spin_until_future_complete(self.node, future, timeout_sec=1.)
        if not future.done():
            future.cancel()
            raise RuntimeError("No Robotiq goal_tolerance response")
        values = future.result().values
        if len(values) != 1 or values[0].type != ParameterType.PARAMETER_DOUBLE:
            raise RuntimeError("Missing Robotiq goal_tolerance")
        tolerance = values[0].double_value
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise RuntimeError("Invalid Robotiq goal_tolerance")
        return tolerance

    def _settled_opening(self):
        # Source timestamps exclude pre-abort, duplicated and out-of-order feedback.
        previous = self.node.get_clock().now().nanoseconds
        start = None
        deadline = time.monotonic() + 5.
        while time.monotonic() < deadline:
            self.ros.spin_once(self.node, timeout_sec=.05)
            stamp = self.stamp
            if stamp <= previous:
                continue
            angle = (1. - self.opening) * self.closed_rad
            if start is None or stamp - previous > 250_000_000:
                start, low, high = stamp, angle, angle
            low, high = min(low, angle), max(high, angle)
            # Allow approximately one hardware encoder count across the window.
            if high - low > .004:
                start, low, high = stamp, angle, angle
            previous = stamp
            if stamp - start >= 750_000_000:
                return self.opening
        raise RuntimeError("No stable fresh Robotiq opening after stalled action")

    def set_gripper(self, opening):
        from action_msgs.msg import GoalStatus

        if not 0. <= opening <= 1.:
            raise ValueError("Opening must be in [0,1]")
        self.check_motion_ready()
        closing = opening < self.state()["gripper"]
        goal = self.goal_type.Goal()
        goal.command.position = (1. - opening) * self.closed_rad
        goal.command.max_effort = self.max_effort
        # Retain the pending goal request so stop() can also cancel a late acceptance.
        self.goal = self.client.send_goal_async(goal)
        self.ros.spin_until_future_complete(self.node, self.goal, timeout_sec=2.)
        if not self.goal.done():
            raise RuntimeError("Robotiq goal acknowledgement timed out")
        handle = self.goal.result()
        if not handle.accepted:
            self.goal = None
            raise RuntimeError("Robotiq rejected the goal")
        result = handle.get_result_async()
        self.ros.spin_until_future_complete(self.node, result, timeout_sec=15.)
        if not result.done():
            raise RuntimeError("Robotiq completion timed out")
        receipt = result.result()
        self.goal = None
        if receipt.status == GoalStatus.STATUS_ABORTED and receipt.result.stalled:
            tolerance = self._goal_tolerance()
            measured = self._settled_opening()
            reached = abs(measured - opening) * self.closed_rad < tolerance
            if not reached and not closing:
                raise RuntimeError(f"Robotiq opening stopped short: requested {opening}, measured {measured}")
            return {"controller_status": "aborted_stalled", "target_reached": reached,
                    "requested_opening": opening, "measured_opening": measured}
        settled = receipt.result.reached_goal or (closing and receipt.result.stalled)
        if receipt.status != GoalStatus.STATUS_SUCCEEDED or not settled:
            raise RuntimeError("Robotiq did not confirm target width; check obstruction and measured opening")

    def stop(self):
        if self.goal is None:
            return
        self.ros.spin_until_future_complete(self.node, self.goal, timeout_sec=2.)
        if not self.goal.done():
            raise RuntimeError("Robotiq delivery uncertain; cancellation unconfirmed")
        handle = self.goal.result()
        if handle.accepted:
            cancel = handle.cancel_goal_async()
            self.ros.spin_until_future_complete(self.node, cancel, timeout_sec=2.)
            if not cancel.done() or not cancel.result().goals_canceling:
                raise RuntimeError("Robotiq cancellation unconfirmed")
        self.goal = None

    def close(self):
        try:
            self.stop()
        finally:
            self.client.destroy()
            self.node.destroy_node()
