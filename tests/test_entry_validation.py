"""Entry-owned CLI, schemas and smoke validation; no native runtime is loaded."""
import copy
from dataclasses import asdict, is_dataclass
import os
import sys
import unittest
from unittest.mock import Mock, patch

import jsonschema

import direct_robodojo as m
from offline_helpers import OfflineTestCase


class ConfigTests(OfflineTestCase):
    def parse(self, *extra, smoke=True):
        argv = ['direct_robodojo.py', '--native-root', str(self.args.native_root),
                '--output-dir', str(self.args.output_dir), '--api-key-env', self.args.api_key_env]
        if smoke:
            argv.append('--smoke-test')
        with patch.object(sys, 'argv', [*argv, *extra]), patch('sys.stderr', new=Mock()):
            return m.parse_args()

    def test_minimal_native_root_smoke_needs_no_credentials(self):
        with patch.dict(os.environ, {self.args.api_key_env: ''}):
            args = self.parse()
        self.assertTrue(args.smoke_test)
        self.assertEqual(args.native_root, self.args.native_root.resolve())
        self.assertEqual(args.output_dir, self.args.output_dir.resolve())
        self.assertNotIn('env_url', asdict(args))
        self.assertNotIn('env_timeout', asdict(args))

    def test_model_defaults_preserved_and_explicit_overrides_used(self):
        args = self.parse(smoke=False)
        self.assertEqual(args.model, 'micu-gpt-6-astra')
        self.assertEqual(args.base_url, 'https://blackwall.qicore.tech/v1')
        self.assertEqual(args.provider, 'openai-compatible')
        args = self.parse('--model', 'chosen-model', '--base-url', 'https://model.invalid/v1',
                          '--provider', 'aliyun', '--no-capture-video', smoke=False)
        self.assertEqual((args.model, args.base_url, args.provider),
                         ('chosen-model', 'https://model.invalid/v1', 'aliyun'))
        self.assertFalse(args.capture_video)

    def test_native_root_requires_checkout_file_and_assets_directory(self):
        for missing in ('env_cfg/arx_x5.yml', 'Assets'):
            path = self.args.native_root / missing
            backup = path.with_name(path.name + '.saved')
            with self.subTest(missing=missing):
                path.rename(backup)
                try:
                    with self.assertRaises(SystemExit) as raised:
                        self.parse()
                    self.assertEqual(raised.exception.code, 2)
                finally:
                    backup.rename(path)
        for invalid in (self.root / 'absent', self.args.native_root / 'env_cfg/arx_x5.yml'):
            with self.subTest(root=invalid), self.assertRaises(SystemExit):
                self.parse('--native-root', str(invalid))

    def test_optional_layout_must_be_an_existing_file(self):
        layout = self.root / 'layout.json'
        layout.write_text('{}', encoding='utf-8')
        self.assertEqual(self.parse('--layout', str(layout)).layout, layout.resolve())
        for invalid in (self.root / 'absent.json', self.root):
            with self.subTest(layout=invalid), self.assertRaises(SystemExit):
                self.parse('--layout', str(invalid))

    def test_all_budgets_are_positive_and_finite(self):
        for name in ('max-decisions', 'history-size', 'max-output-tokens', 'max-total-tokens',
                     'episode-timeout', 'request-timeout'):
            for value in ('0', '-1', 'nan', 'inf', '-inf'):
                with self.subTest(name=name, value=value), self.assertRaises(SystemExit):
                    self.parse(f'--{name}={value}')

    def test_seed_task_provider_and_smoke_decision_constraints(self):
        for field, value in [('seed', '1'), ('seed', '-1'), ('seed', 'False'), ('seed', '0.0'),
                             ('task', 'other'), ('provider', 'responses'), ('max-decisions', '2')]:
            with self.subTest(field=field, value=value), self.assertRaises(SystemExit):
                self.parse('--' + field, value)
        self.assertEqual(self.parse('--max-decisions', '3').max_decisions, 3)

    def test_model_mode_requires_credentials_and_nonempty_model_endpoint(self):
        with patch.dict(os.environ, {self.args.api_key_env: ''}), self.assertRaises(SystemExit):
            self.parse(smoke=False)
        for flag, value in [('--model', ''), ('--model', '   '), ('--base-url', '')]:
            with self.subTest(flag=flag, value=value), self.assertRaises(SystemExit):
                self.parse(flag, value, smoke=False)
        for provider in ('openai-compatible', 'aliyun'):
            self.assertEqual(self.parse('--provider', provider, smoke=False).provider, provider)

    def test_url_errors_are_cli_errors_before_native_startup(self):
        for url in ('http://remote.invalid', 'https://model.invalid:0', 'https://model.invalid/?key=secret'):
            with self.subTest(url=url), self.assertRaises(SystemExit):
                self.parse('--base-url', url)

class EntryValidationTests(unittest.TestCase):
    def test_action_and_observation_dataclasses_are_defined_locally(self):
        for cls in (m.Action, m.Observation):
            self.assertTrue(is_dataclass(cls))
            self.assertEqual(cls.__module__, 'direct_robodojo')
        first, second = m.Action('hold', {}), m.Action('hold', {})
        self.assertTrue(first.call_id)
        self.assertNotEqual(first.call_id, second.call_id)
        self.assertEqual(first.summary, '')
        obs = m.Observation('one', {}, {})
        other = m.Observation('two', {}, {})
        obs.feedback['call_ok'] = False
        self.assertEqual(other.feedback, {})
        self.assertEqual(asdict(other), {'snapshot_id': 'two', 'state': {}, 'cameras': {}, 'feedback': {}})

    def test_url_credentials_query_fragment_and_remote_plaintext_rejected(self):
        for url in ('https://user:secret@example.invalid/v1', 'https://example.invalid/v1?key=secret',
                    'https://example.invalid/#fragment', 'https://example.invalid/?', 'https://example.invalid/#',
                    'http://remote.invalid/v1', 'file:///tmp/api', '', 'https://model.invalid:0',
                    'https://model.invalid:65536', 'https://model.invalid:bad'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                m.validate_url(url)
        for url in ('http://127.0.0.1:8000/', 'http://localhost:8000/', 'http://[::1]:8000/',
                    'https://model.invalid/v1/'):
            self.assertEqual(m.validate_url(url), url.rstrip('/'))

    def test_schemas_define_four_tools_and_do_not_mutate_arguments(self):
        schemas = {tool['function']['name']: tool['function']['parameters'] for tool in m.tool_schemas(['arm'])}
        self.assertEqual(list(schemas), ['capture_scene', 'move', 'hold', 'finish'])
        for schema in schemas.values():
            jsonschema.Draft202012Validator.check_schema(schema)
        arguments = {'summary': 'Hold', 'steps': 2}
        before = copy.deepcopy(arguments)
        jsonschema.validate(arguments, schemas['hold'])
        self.assertEqual(arguments, before)
        for tool in ('capture_scene', 'finish'):
            jsonschema.validate({'summary': 'Observed'}, schemas[tool])
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate({'summary': 'Observed', 'steps': 1}, schemas[tool])

    def test_native_move_normalized_gripper_schema(self):
        schema = m.tool_schemas(['left', 'right'])[1]['function']['parameters']
        arguments = {'summary': 'Measured world link6 pose', 'arm': 'left', 'pose': [0., 0., .9, 0., 0., 0.]}
        jsonschema.validate(arguments, schema)
        for gripper in (0., .5, 1.):
            jsonschema.validate({**arguments, 'gripper': gripper}, schema)
        invalid = [{'gripper': -.01}, {'gripper': 1.01}, {'gripper': True}, {'opening_m': .04},
                   {'arm': 'unsupported'}, {'pose': [0, 0, .9]}, {'pose': [0] * 7}, {'pose': ['0'] * 6},
                   {'steps': 25}, {'steps': True}, {'steps': 0}, {'summary': ''},
                   {'summary': 'x' * 1001}, {'extra': 1}]
        for update in invalid:
            with self.subTest(update=update), self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate({**arguments, **update}, schema)
        hold = m.tool_schemas(['left'])[2]['function']['parameters']
        for steps in (1, 24):
            jsonschema.validate({'summary': 'Hold', 'steps': steps}, hold)

    def test_smoke_checks_motion_not_just_successful_calls(self):
        m.verify_smoke('capture_scene', {'call_ok': True}, 0, 0)
        m.verify_smoke('hold', {'call_ok': True, 'steps_executed': 2,
                              'stop_reason': 'requested_step_budget'}, 0, 2)
        m.verify_smoke('move', {'call_ok': True, 'targets_reached': True}, 2, 3)
        invalid = [('capture_scene', {'call_ok': False}, 0, 0),
                   ('hold', {'call_ok': True, 'steps_executed': 1, 'stop_reason': 'requested_step_budget'}, 0, 1),
                   ('hold', {'call_ok': True, 'steps_executed': 2, 'stop_reason': 'wall_time_limit'}, 0, 2),
                   ('move', {'call_ok': True, 'targets_reached': True}, 0, 0),
                   ('move', {'call_ok': True, 'targets_reached': False}, 0, 1),
                   ('unknown', {'call_ok': True}, 0, 1)]
        for tool, feedback, before, after in invalid:
            with self.subTest(tool=tool, feedback=feedback), self.assertRaises(RuntimeError):
                m.verify_smoke(tool, feedback, before, after)

    def test_capture_smoke_rejects_unexpected_physics_advance(self):
        with self.assertRaises(RuntimeError):
            m.verify_smoke('capture_scene', {'call_ok': True}, 0, 1)


if __name__ == '__main__':
    unittest.main()