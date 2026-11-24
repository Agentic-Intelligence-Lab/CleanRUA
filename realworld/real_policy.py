"""Policy clients for configurable workstations. Adapters return time-ordered commands keyed by arm name."""
import base64
import io
import json
from pathlib import Path
import time

import httpx
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from realworld.real_arm_env import InvalidTarget
from realworld.workstation import device


CONTRACT = "arms.absolute.base_m.xyz_rpy_rad.open_fraction.hand_rad.v1"


class HTTPArmPolicy:
    """POST all named RGB views, named arm measurements and instruction to a policy server."""
    def __init__(self, url):
        self.url = url

    def predict(self, obs, instruction):
        images = {}
        for name, rgb in obs["images"].items():
            stream = io.BytesIO()
            Image.fromarray(rgb).save(stream, format="PNG")
            images[name] = base64.b64encode(stream.getvalue()).decode("ascii")
        request = {"contract": CONTRACT, "instruction": instruction, "images_png": images,
                   "arms": obs["arms"],
                   "rgb_received_ns": obs["rgb_received_ns"], "state_received_ns": obs["state_received_ns"]}
        response = httpx.post(self.url, json=request, timeout=30)
        response.raise_for_status()
        result = response.json()
        if result.get("contract") != CONTRACT:
            raise ValueError("Policy response must identify the requested named-arm contract")
        return result["actions"]


class DexJoCoOpenPI:
    """Native DexJoCo Pi0.5 server. Configure matching world/flange frames and Allegro joint order."""
    def __init__(self, url, camera_mapping, joint_indices):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        if set(joint_indices) not in ({"right"}, {"right", "left"}):
            raise ValueError("DexJoCo expects right or right+left")
        self.indices = {}
        for side in ("right", "left"):
            if side in joint_indices:
                slots = joint_indices[side]
                if (len(slots) != 16 or any(type(i) != int or not 0 <= i < 16 for i in slots)
                        or len(set(slots)) != 16):
                    raise ValueError("Specify 16 driver joint indices in native Allegro order")
                self.indices[side] = np.asarray(slots)
        if not camera_mapping or {"prompt", "state"} & camera_mapping.keys():
            raise ValueError("Provide the checkpoint's mapping from image keys to camera names")
        self.cameras = camera_mapping
        self.client = WebsocketClientPolicy(url)

    def predict(self, obs, instruction):
        arms = obs["arms"]
        if arms.keys() != self.indices.keys():
            raise ValueError("Arm names do not match this DexJoCo checkpoint")
        poses, hands = [], []
        for side, ix in self.indices.items():
            arm = arms[side]
            joints = np.asarray(arm.get("hand"))
            if joints.shape != (16,):
                raise ValueError(f"{side} must provide 16 Allegro joint measurements")
            pose = arm["pose"]
            quat = Rotation.from_euler("xyz", pose[3:]).as_quat(scalar_first=True)
            poses.extend([*pose[:3], *quat])
            hands.extend(joints[ix])
        images = {key: np.asarray(Image.fromarray(obs["images"][name]).resize((224, 224), Image.Resampling.BILINEAR))
                  for key, name in self.cameras.items()}
        native = np.zeros(46)
        native[:len(poses) + len(hands)] = poses + hands
        raw = np.asarray(self.client.infer({**images, "state": native, "prompt": instruction})["actions"])
        # Some single-arm inference servers return a padded second-arm block, as in upstream.
        if (raw.ndim != 2 or not len(raw) or raw.shape[1] not in {22 * len(self.indices), 44}
                or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all()):
            raise ValueError("Invalid native DexJoCo action chunk")
        actions = [{} for _ in raw]
        for i, (side, ix) in enumerate(self.indices.items()):
            block = raw[:, 22*i:22*(i+1)]
            angles = Rotation.from_rotvec(block[:, 3:6].copy()).as_euler("xyz")
            for target, row, rpy in zip(actions, block, angles):
                hand = np.empty(16)
                hand[ix] = row[6:22]
                target[side] = {"pose": [*row[:3].tolist(), *rpy.tolist()], "hand": hand.tolist()}
        return actions


class ChunkPolicy:
    def __init__(self, client, actions_per_chunk, control_hz):
        if type(actions_per_chunk) != int or not 1 <= actions_per_chunk <= 300:
            raise ValueError("actions_per_chunk must be an integer in [1,300]")
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be positive and finite")
        self.client, self.horizon, self.period = client, actions_per_chunk, 1 / control_hz
        self.next_step = 0.

    def run(self, env, instruction, steps, active_groups=None, trace=None):
        feedback, _ = self.run_observed(env, instruction, steps, active_groups, trace)
        return feedback

    def run_observed(self, env, instruction, steps, active_groups=None, trace=None):
        """Return both the policy feedback and the observation used for its last receipt."""
        sides = list(env.arms) if active_groups is None else active_groups
        if not sides or len(set(sides)) != len(sides) or set(sides) - env.arms.keys():
            raise InvalidTarget("Select distinct configured arms")
        obs = env.observe()
        result = {"instruction": instruction, "active_groups": sides, "steps": 0, "inferences": 0}
        while result["steps"] < steps:
            raw = self.client.predict(obs, instruction)
            result["inferences"] += 1
            if not isinstance(raw, (list, tuple)) or not raw or not all(isinstance(step, dict) for step in raw):
                raise ValueError("Policy must return a nonempty sequence of named arm commands")
            count = min(self.horizon, steps - result["steps"], len(raw))
            for target in raw[:count]:
                delay = self.next_step - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                self.next_step = time.monotonic() + self.period
                if trace:
                    trace(policy_action=target, step=result["steps"] + 1, groups=sides)
                try:
                    obs, _, terminated, truncated, info = env.step_groups(target, sides)
                except InvalidTarget as exc:
                    if not env.motion_enabled:  # device failure after dispatch
                        raise
                    return {**result, "stop_reason": "target_rejected", "error": str(exc)}, env.observe()
                result["steps"] += 1
                result["measured"] = info["measured"]
                if trace:
                    trace(policy_measured=info["measured"], step=result["steps"])
                if terminated or truncated:
                    return {**result, "stop_reason": "terminated" if terminated else "truncated"}, obs
        return {**result, "stop_reason": "step_budget"}, obs


def load_policy(station, path=None):
    spec = json.loads(Path(path).read_text()) if path else (station or {}).get("policy")
    if not spec:
        raise ValueError("Supply --policy-config or the workstation's policy section")
    return ChunkPolicy(device(spec), spec["actions_per_chunk"], spec["control_hz"])
