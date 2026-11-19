"""CPU tests for benchmark summaries and documentation exports."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from cleanrua_utils import benchmark


class SummaryTest(unittest.TestCase):
    def test_groups_by_task_and_algorithm(self):
        with tempfile.TemporaryDirectory() as directory:
            for i, (success, tokens) in enumerate([(True, 10_000), (False, 30_000)]):
                run = Path(directory) / f'stack_bowls__direct_robodojo__0__{i}'
                run.mkdir()
                (run / 'result.json').write_text(json.dumps({
                    'stop_reason': 'native_terminated' if success else 'native_truncated', 'success': success,
                    'score': 100. if success else 15., 'decisions': 4, 'total_tokens': tokens, 'model_s': 10.,
                    'elapsed_s': 60.}))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                benchmark.summary(Namespace(output_dir=Path(directory), price=[]))
        self.assertEqual(output.getvalue().splitlines()[0],
                         '| task | algorithm | n | excluded | success | success 95% CI | score | decisions | tokens '
                         '| tokens/success | cost | cost/success | model_s | elapsed_s | stops |')
        row = output.getvalue().splitlines()[-1]
        self.assertEqual([cell.strip() for cell in row.strip('|').split('|')],
                         ['stack_bowls', 'direct_robodojo', '2', '0', '1/2', '9–91%', '57.5', '4.0', '20k', '40k',
                          '-', '-', '10', '60',
                          'native_terminated, native_truncated'])


    def test_excludes_model_errors_and_incomplete_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed = root / 'stack_bowls__direct_robodojo__0__1'
            failed.mkdir()
            (failed / 'result.json').write_text(json.dumps({'stop_reason': 'model_error: HTTP 429'}))
            (root / 'stack_bowls__direct_robodojo__0__2').mkdir()
            (root / 'stack_bowls__direct_robodojo__0__3').touch()  # A file is not an episode directory.
            for name in ('logs', 'aborted', 'too__many__parts__in__name'):
                ignored = root / name
                ignored.mkdir()
                (ignored / 'result.json').write_text('{}')
                (ignored / 'stack_blocks__code_robodojo__0__4').mkdir()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                benchmark.summary(Namespace(output_dir=root, price=[]))
            lines = output.getvalue().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual([cell.strip() for cell in lines[-1].strip('|').split('|')],
                             ['stack_bowls', 'direct_robodojo', '0', '2', *(['-'] * 10),
                              'incomplete, model_error: HTTP 429'])

    def test_wilson_intervals_use_only_valid_episodes(self):
        for k, n, interval in ((3, 8, '14–69%'), (0, 3, '0–56%'), (3, 3, '44–100%'), (3, 6, '19–81%')):
            with self.subTest(k=k, n=n), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for i in range(n + 1):
                    run = root / f'stack_bowls__direct_robodojo__0__{i}'
                    run.mkdir()
                    (run / 'result.json').write_text(json.dumps({
                        'stop_reason': 'finish' if i < n else 'model_error: ReadTimeout',
                        'success': i < k or i == n, 'total_tokens': 1000 if i < n else 1_000_000,
                        'score': 10 if i < n else 100}))
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    benchmark.summary(Namespace(output_dir=root, price=[]))
                cells = [cell.strip() for cell in output.getvalue().splitlines()[-1].strip('|').split('|')]
                self.assertEqual(cells[2:7], [str(n), '1', f'{k}/{n}', interval, '10.0'])
                self.assertEqual(cells[8], '1k')
                self.assertEqual(cells[9], f'{n / k:.0f}k' if k else '-')

    def test_cli_prices_use_cached_tokens_and_exclude_failed_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, model in enumerate(('model-a', 'model-b', 'unpriced-error')):
                run = root / f'stack_bowls__direct_robodojo__0__{i}'
                run.mkdir()
                (run / 'result.json').write_text(json.dumps({
                    'model': model, 'stop_reason': 'finish' if i < 2 else 'model_error: HTTP 503',
                    'success': i == 0, 'prompt_tokens': 100_000, 'cached_tokens': 20_000,
                    'completion_tokens': 10_000, 'total_tokens': 110_000}))
            for prices, expected in (([], ['-', '-']),
                                     (['--price', 'model-a=1.25,0.125,10'], ['-', '-']),
                                     (['--price', 'model-a=1.25,0.125,10', '--price', 'model-b=2,0.5,4'],
                                      ['$0.21', '$0.41'])):
                with self.subTest(prices=prices):
                    output = subprocess.run([sys.executable, '-m', 'cleanrua_utils.benchmark', 'summary',
                                             directory, *prices], capture_output=True, text=True, check=True).stdout
                    cells = [cell.strip() for cell in output.splitlines()[-1].strip('|').split('|')]
                    self.assertEqual(cells[2:5], ['2', '1', '1/2'])
                    self.assertEqual(cells[10:12], expected)
            first = root / 'stack_bowls__direct_robodojo__0__0' / 'result.json'
            result = json.loads(first.read_text())
            result['success'] = False
            first.write_text(json.dumps(result))
            output = subprocess.run([sys.executable, '-m', 'cleanrua_utils.benchmark', 'summary', directory,
                                     '--price', 'model-a=1.25,0.125,10', '--price', 'model-b=2,0.5,4'],
                                    capture_output=True, text=True, check=True).stdout
            cells = [cell.strip() for cell in output.splitlines()[-1].strip('|').split('|')]
            self.assertEqual(cells[9:12], ['-', '$0.21', '-'])


class ExportTest(unittest.TestCase):
    def test_cli_export_preserves_key_and_episode_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task, algorithm, success, stop in (('task_b', 'monitor', True, 'finish'),
                                                    ('task_a', 'direct', False, 'model_error: HTTP 429')):
                run = root / f'{task}__{algorithm}_robodojo__0__1'
                run.mkdir()
                (run / 'result.json').write_text(json.dumps({
                    'success': success, 'score': 100. if success else 0., 'decisions': 3, 'total_tokens': 42,
                    'native_steps': 5, 'step_limit': 20, 'elapsed_s': 1.5, 'model_s': .5,
                    'stop_reason': stop, 'agent_finished': success}))
            output = root / 'export.json'
            subprocess.run([sys.executable, '-m', 'cleanrua_utils.benchmark', 'export', directory, '-o', str(output),
                            '--tasks', 'task_b', 'task_a', '--algorithms', 'monitor', 'direct',
                            '--not-run', 'direct:task_b', 'monitor:task_a',
                            '--meta', 'round=测试', 'seed=-2', 'repeats=1', 'ratio=1.5'],
                           capture_output=True, text=True, check=True)
            text = output.read_text()
            data = json.loads(text)
        self.assertEqual(list(data),
                 ['round', 'seed', 'repeats', 'ratio', 'tasks', 'algorithms', 'episodes', 'not_run'])
        self.assertEqual([data[key] for key in ('round', 'seed', 'repeats', 'ratio')], ['测试', -2, 1, '1.5'])
        self.assertEqual(data['tasks'], ['task_b', 'task_a'])
        self.assertEqual(data['algorithms'], ['monitor', 'direct'])
        self.assertEqual([r['task'] for r in data['episodes']], ['task_a', 'task_b'])
        self.assertEqual([r['algorithm'] for r in data['episodes']], ['direct', 'monitor'])
        self.assertEqual([r['invalid'] for r in data['episodes']], [True, False])
        self.assertEqual([r['finish'] for r in data['episodes']], [False, True])
        for episode in data['episodes']:
            self.assertEqual(list(episode), ['algorithm', 'task', 'success', 'score', 'decisions', 'tokens',
                                            'native_steps', 'step_limit', 'elapsed_s', 'model_s', 'stop_reason',
                                            'finish', 'invalid'])
            self.assertEqual(episode['tokens'], 42)
        self.assertEqual(data['not_run'], [{'algorithm': 'direct', 'task': 'task_b'},
                                          {'algorithm': 'monitor', 'task': 'task_a'}])
        self.assertEqual([list(item) for item in data['not_run']], [['algorithm', 'task']] * 2)
        self.assertEqual(text, json.dumps(data, indent=1))  # Default ASCII escaping and no trailing newline.


if __name__ == '__main__':
    unittest.main()
