"""Small CPU-only fixtures shared by the native baseline's unittest tests."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import direct_robodojo as direct
from cleanrua_utils.robodojo_wrappers import RoboDojoEnv
from direct_robodojo import Observation

ROOT = Path(__file__).resolve().parents[1]


def native_root(parent: Path) -> Path:
    """Only parse_args' filesystem contract; never a usable simulator."""
    root = parent / 'native'
    (root / 'env_cfg').mkdir(parents=True)
    (root / 'env_cfg' / 'arx_x5.yml').write_text('config: {}\n', encoding='utf-8')
    (root / 'Assets').mkdir()
    return root


def fake_observation() -> Observation:
    arms = {}
    for name, x in [('left', -.3), ('right', .3)]:
        arms[name] = {'control_pose': {'position': [x, 0., .9], 'quaternion': [1., 0., 0., 0.]},
                      'joints': [.1, -.2], 'finger_joints': [.25, -.25],
                      'gripper': .25, 'gripper_command': .8}
    return Observation('snapshot', {'arms': arms, 'state_version': 0, 'native_steps': 0, 'ended': False}, {})


def model_reply(identity='call-ID', *, tool='capture_scene', tokens=10) -> dict:
    return {'usage': {'total_tokens': tokens}, 'choices': [{'message': {
        'role': 'assistant', 'content': 'Look', 'tool_calls': [{
            'id': identity, 'type': 'function',
            'function': {'name': tool, 'arguments': '{"summary":"Look"}'},
        }],
    }}]}


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cleanrua-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = direct.Args(native_root=native_root(self.root), capture_video=False,
                                model='offline-model', base_url='https://model.invalid/v1',
                                api_key_env='CLEANRUA_TEST_API_KEY', output_dir=self.root / 'runs')
        self.events = []
        self.enterContext(patch.dict(os.environ, {'CLEANRUA_TEST_API_KEY': 'offline-key'}))
        self.enterContext(patch('socket.socket.connect', side_effect=AssertionError('No real network in CPU tests')))
        self.enterContext(patch('socket.create_connection', side_effect=AssertionError('No real network in CPU tests')))
        self.enterContext(patch('httpx.HTTPTransport.handle_request', side_effect=AssertionError('Use fake HTTP transport')))
        self.enterContext(patch.object(RoboDojoEnv, '_launch', side_effect=AssertionError('No native startup in CPU tests')))

    def record(self, event, **payload):
        self.events.append(copy.deepcopy({'event': event, **payload}))