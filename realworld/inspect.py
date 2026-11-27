"""Read robot descriptions and ROS 2 observations into a workstation inventory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from uuid import uuid4
import xml.etree.ElementTree as ET


def numbers(value):
    result = [float(x) for x in value.split()]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("Robot description contains non-finite numbers")
    return result


def robot_description(xml):
    root = ET.fromstring(xml)
    if root.tag != "robot" or any("xacro" in element.tag for element in root.iter()):
        raise ValueError("Supply expanded URDF with a robot root")
    links = [link.attrib["name"] for link in root.findall("link")]
    joints = []
    for joint in root.findall("joint"):
        item = dict(joint.attrib)
        for field in ("parent", "child", "mimic"):
            element = joint.find(field)
            if element is not None:
                item[field] = dict(element.attrib)
        origin = joint.find("origin")
        item["origin"] = {
            "xyz_m": numbers(origin.get("xyz", "0 0 0")) if origin is not None else [0., 0., 0.],
            "rpy_rad": numbers(origin.get("rpy", "0 0 0")) if origin is not None else [0., 0., 0.]}
        axis = joint.find("axis")
        if item["type"] in ("revolute", "continuous", "prismatic"):
            item["axis"] = numbers(axis.get("xyz", "1 0 0")) if axis is not None else [1., 0., 0.]
        limits = joint.find("limit")
        if limits is not None:
            item["limits"] = {key: numbers(value)[0] for key, value in limits.attrib.items()}
        joints.append(item)
    parents = {j["parent"]["link"] for j in joints if "parent" in j}
    children = {j["child"]["link"] for j in joints if "child" in j}
    return {
        "name": root.get("name"), "sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "links": links, "joints": joints,
        "independent_joint_names": [j["name"] for j in joints if j["type"] != "fixed" and "mimic" not in j],
        "root_candidates": sorted(set(links) - children),
        "tip_candidates": sorted(set(links) - parents),
        "collision_links": [link.get("name") for link in root.findall("link") if link.find("collision") is not None],
    }


def message_summary(kind, message):
    """Copy measurement metadata only; image pixels stay in the ROS transport."""
    if kind == "tf2_msgs/msg/TFMessage":
        return {"transforms": [{
            "parent": t.header.frame_id, "child": t.child_frame_id,
            "stamp_ns": t.header.stamp.sec * 10**9 + t.header.stamp.nanosec,
            "translation_m": [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z],
            "quaternion_xyzw": [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
        } for t in message.transforms]}
    summary = {"frame_id": message.header.frame_id,
               "stamp_ns": message.header.stamp.sec * 10**9 + message.header.stamp.nanosec}
    if kind == "sensor_msgs/msg/JointState":
        summary.update(names=list(message.name), position=list(message.position),
                       velocity=list(message.velocity), effort=list(message.effort))
    elif kind == "sensor_msgs/msg/CameraInfo":
        summary.update(width=message.width, height=message.height, distortion_model=message.distortion_model,
                       K=list(message.k), D=list(message.d), R=list(message.r), P=list(message.p),
                       calibrated=bool(message.k[0]))
    else:
        summary.update(width=message.width, height=message.height, encoding=message.encoding)
    return summary


def read_services(node, report, requested):
    """Call only these two query services; never request controller state changes."""
    from rcl_interfaces.srv import GetParameters
    from rosidl_runtime_py.convert import message_to_ordereddict

    pending = []
    for name, types in node.get_service_names_and_types():
        if name in requested:
            continue
        if "rcl_interfaces/srv/GetParameters" in types:
            service, request = GetParameters, GetParameters.Request(names=["robot_description"])
            field = "descriptions"
        elif "controller_manager_msgs/srv/ListControllers" in types:
            try:
                from controller_manager_msgs.srv import ListControllers
            except ImportError:
                requested.add(name)
                report["errors"].append({"source": name, "error": "Install controller_manager_msgs to read controller metadata"})
                continue
            service, request, field = ListControllers, ListControllers.Request(), "controllers"
        else:
            continue
        client = node.create_client(service, name)
        requested.add(name)
        future = client.call_async(request)

        def complete(done, source=name, destination=field):
            try:
                response = done.result()
                if destination == "descriptions":
                    if response.values and response.values[0].type == 4 and response.values[0].string_value:
                        report[destination][source] = robot_description(response.values[0].string_value)
                else:
                    report[destination][source] = message_to_ordereddict(response)
            except Exception as exc:
                report["errors"].append({"source": source, "error": str(exc)})
        future.add_done_callback(complete)
        pending.append((name, future))
    return pending


def scan_ros(duration):
    import rclpy
    from rclpy.context import Context
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rosidl_runtime_py.utilities import get_message

    context = Context()
    rclpy.init(args=[], context=context)
    node = rclpy.create_node("cleanrua_inspect_" + uuid4().hex[:8], context=context,
                            enable_rosout=False, start_parameter_services=False)
    report = {"nodes": [], "topics": {}, "samples": {}, "transforms": {},
              "controllers": {}, "descriptions": {}, "errors": []}
    supported = {"sensor_msgs/msg/JointState", "sensor_msgs/msg/Image",
                 "sensor_msgs/msg/CameraInfo", "tf2_msgs/msg/TFMessage"}
    subscribed, requested, pending = set(), set(), []
    began = time.monotonic()
    discover_at = began
    try:
        while time.monotonic() - began < duration:
            if time.monotonic() >= discover_at:
                report["nodes"] = sorted({namespace.rstrip("/") + "/" + name
                                          for name, namespace in node.get_node_names_and_namespaces()
                                          if name != node.get_name()})
                for topic, types in node.get_topic_names_and_types():
                    report["topics"][topic] = types
                    for kind in (supported | ({"std_msgs/msg/String"} if topic.endswith("/robot_description") else set())).intersection(types):
                        if (topic, kind) in subscribed:
                            continue
                        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
                        publishers = node.get_publishers_info_by_topic(topic)
                        if publishers and all(p.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL for p in publishers):
                            qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

                        def receive(message, source=topic, msg_type=kind):
                            if msg_type == "std_msgs/msg/String":
                                try:
                                    report["descriptions"][source] = robot_description(message.data)
                                except ValueError as exc:
                                    report["errors"].append({"source": source, "error": str(exc)})
                                return
                            record = message_summary(msg_type, message)
                            record["received_utc"] = datetime.now(timezone.utc).isoformat()
                            report["samples"][source] = record
                            if msg_type == "tf2_msgs/msg/TFMessage":
                                for transform in record["transforms"]:
                                    key = source + ":" + transform["parent"] + "->" + transform["child"]
                                    report["transforms"][key] = transform
                        node.create_subscription(get_message(kind), topic, receive, qos)
                        subscribed.add((topic, kind))
                pending.extend(read_services(node, report, requested))
                discover_at = time.monotonic() + .5
            rclpy.spin_once(node, timeout_sec=min(.1, max(0., began + duration - time.monotonic())))
        for name, future in pending:
            if not future.done():
                future.cancel()
                report["errors"].append({"source": name, "error": "Query did not complete within scan window"})
    finally:
        node.destroy_node()
        rclpy.shutdown(context=context)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, help="Expanded local URDF")
    parser.add_argument("--ros", action="store_true", help="Read the current ROS_DOMAIN_ID graph")
    parser.add_argument("--seconds", type=float, default=5., help="ROS discovery and sample window")
    parser.add_argument("--out", type=Path, help="New inventory JSON; default prints to stdout")
    args = parser.parse_args(argv)
    if not args.urdf and not args.ros:
        parser.error("Select --urdf and/or --ros")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    report = {"captured_utc": datetime.now(timezone.utc).isoformat(), "confirmation_required": [
        "Which arm chain and TCP the task controls",
        "Which active controller should execute trajectories",
        "Whether published camera/TF calibration matches this installation",
        "World obstacles and the permitted physical workspace",
        "Joint units, zero offsets and end-effector command semantics",
    ]}
    if args.urdf:
        report["urdf"] = {"path": str(args.urdf.resolve()), **robot_description(args.urdf.read_text())}
    if args.ros:
        try:
            report["ros"] = scan_ros(args.seconds)
        except ImportError as exc:
            parser.error(f"ROS inspection needs the sourced ROS 2 Python environment: {exc}")
    output = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.out:
        with args.out.open("x") as stream:
            stream.write(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
