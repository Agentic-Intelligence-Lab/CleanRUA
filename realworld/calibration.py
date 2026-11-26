"""Sample a teleoperated TCP and update one arm's workspace in a station profile."""
import fcntl
import json
import os
from pathlib import Path
import stat
import tempfile

import numpy as np

from realworld.env import InvalidTarget, vector
from realworld.workstation import device


def _collect(reader):
    points = []
    while True:
        command = input("回车采点，done 计算边界，q 取消 > ").strip().lower()
        if command == "q":
            return None
        if command == "done":
            if len(points) >= 2:
                low, high = np.min(points, axis=0), np.max(points, axis=0)
                if np.all(low < high):
                    return [low.tolist(), high.tolist()]
            print("请继续采点：X、Y、Z 每个轴都要覆盖不同的位置。")
        elif command == "":
            state = reader.state()
            try:
                xyz = vector(state["pose"], 6)[:3]
            except (KeyError, InvalidTarget) as exc:
                print(f"无效 TCP 位姿，本点未记录：{exc}")
                continue
            points.append(xyz)
            print(f"点 {len(points)} (m): {xyz.tolist()}")
        else:
            print("请输入 done、q，或直接按回车。")


def _save(path, original, config):
    # Serialize concurrent calibrations only during the compare-and-replace.
    with path.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.read_bytes() != original:
            raise ValueError("配置在采点期间已被修改，请重新读取后再标定。")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
            try:
                json.dump(config, tmp, ensure_ascii=False, indent=2, allow_nan=False)
                tmp.write("\n")
                tmp.flush()
                os.fchmod(tmp.fileno(), stat.S_IMODE(path.stat().st_mode))
                os.replace(tmp.name, path)
            finally:
                Path(tmp.name).unlink(missing_ok=True)


def calibrate_workspace(path, side=None):
    path = path.resolve(strict=True)
    original = path.read_bytes()
    config = json.loads(original)
    arms = config.get("arms", {})
    if side is None and len(arms) == 1:
        side = next(iter(arms))
    if side not in ("left", "right") or side not in arms:
        raise ValueError(f"请选择配置中已有的机械臂 --arm，当前可选：{', '.join(arms)}")

    profile = arms[side]
    print(f"配置：{path}\n机械臂：{side}；读取基座系 TCP XYZ，单位为米。")
    try:
        reader = device(profile.get("state_reader", profile))
        try:
            bounds = _collect(reader)
        finally:
            close = getattr(reader, "close", None)
            if callable(close):
                close()
        if bounds is None:
            return
        print(f"arms.{side}.workspace_m\n原值：{profile.get('workspace_m')}\n新值：{bounds}")
        if input("确认写回上述配置？[y/N] ").strip().lower() not in ("y", "yes"):
            print("已取消。")
            return
    except (KeyboardInterrupt, EOFError):
        print("\n已取消。")
        return
    arms[side] = {**profile, "workspace_m": bounds}
    _save(path, original, config)
    print(f"已保存 arms.{side}.workspace_m 到 {path}")
