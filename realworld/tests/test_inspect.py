"""Read-only inventory tests, requiring no ROS installation or device connection."""
import contextlib
from concurrent.futures import Future
import io
import json
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from realworld.inspect import main, message_summary, read_services, robot_description


ROBOT = """<robot name="fixture">
<link name="base"/><link name="arm"/><link name="tool"/><link name="finger"/>
<joint name="j1" type="revolute"><parent link="base"/><child link="arm"/>
<limit lower="-1" upper="2" effort="10" velocity="0.5"/></joint>
<joint name="tcp" type="fixed"><parent link="arm"/><child link="tool"/>
<origin xyz="0 0 0.12"/></joint>
<joint name="finger_slave" type="prismatic"><parent link="tool"/><child link="finger"/>
<mimic joint="finger_master" multiplier="-1"/></joint>
</robot>"""


class InspectionTest(unittest.TestCase):
    def test_model_retains_mimic_limits_and_tool_transform(self):
        info = robot_description(ROBOT)
        self.assertEqual(info["independent_joint_names"], ["j1"])
        self.assertEqual(info["root_candidates"], ["base"])
        self.assertEqual(info["tip_candidates"], ["finger"])
        self.assertEqual(info["joints"][0]["limits"]["velocity"], .5)
        self.assertEqual(info["joints"][1]["origin"]["xyz_m"], [0., 0., .12])
        self.assertEqual(info["joints"][2]["mimic"]["joint"], "finger_master")
        self.assertNotIn("tcp", info)  # A named fixed joint is a candidate, not a confirmed installed TCP.

    def test_unexpanded_xacro_rejected(self):
        with self.assertRaises(ValueError):
            robot_description('<robot xmlns:xacro="http://www.ros.org/wiki/xacro"><xacro:arm/></robot>')

    def test_camera_metadata_does_not_copy_pixels(self):
        stamp = NS(sec=10, nanosec=12)
        message = NS(header=NS(stamp=stamp, frame_id="optical"), width=640, height=480,
                     encoding="rgb8", data=b"pixels")
        summary = message_summary("sensor_msgs/msg/Image", message)
        self.assertNotIn("data", summary)
        self.assertEqual(summary["stamp_ns"], 10000000012)
        message.k, message.d, message.r, message.p = [0.]*9, [], [0.]*9, [0.]*12
        message.distortion_model = "plumb_bob"
        self.assertFalse(message_summary("sensor_msgs/msg/CameraInfo", message)["calibrated"])

    def test_query_only_description_and_controller_list(self):
        class Request:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
        service = NS(Request=Request)
        calls = []

        class Node:
            def get_service_names_and_types(self):
                return [("/robot/get_parameters", ["rcl_interfaces/srv/GetParameters"]),
                        ("/manager/list_controllers", ["controller_manager_msgs/srv/ListControllers"]),
                        ("/manager/switch_controller", ["controller_manager_msgs/srv/SwitchController"]),
                        ("/robot/set_parameters", ["rcl_interfaces/srv/SetParameters"])]

            def create_client(self, kind, name):
                def call(request):
                    calls.append((name, request.__dict__))
                    future = Future()
                    response = (NS(values=[NS(type=4, string_value=ROBOT)]) if "get_parameters" in name
                                else {"controller": [{"name": "arm", "state": "active"}]})
                    future.set_result(response)
                    return future
                return NS(call_async=call)

        modules = {"rcl_interfaces.srv": NS(GetParameters=service),
                   "controller_manager_msgs.srv": NS(ListControllers=service),
                   "rosidl_runtime_py.convert": NS(message_to_ordereddict=lambda x: x)}
        report = {"descriptions": {}, "controllers": {}, "errors": []}
        seen = set()
        with patch.dict(sys.modules, modules):
            read_services(Node(), report, seen)
            read_services(Node(), report, seen)
        self.assertEqual(calls, [("/robot/get_parameters", {"names": ["robot_description"]}),
                                 ("/manager/list_controllers", {})])
        self.assertEqual(report["descriptions"]["/robot/get_parameters"]["name"], "fixture")
        self.assertEqual(report["errors"], [])

    def test_cli_stdout_report_from_local_file(self):
        from pathlib import Path
        with patch.object(Path, "read_text", return_value=ROBOT), contextlib.redirect_stdout(io.StringIO()) as output:
            main(["--urdf", "fixture.urdf"])
        report = json.loads(output.getvalue())
        self.assertEqual(report["urdf"]["name"], "fixture")
        self.assertTrue(report["confirmation_required"])
