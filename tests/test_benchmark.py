"""CPU test for the benchmark summary."""
import contextlib
import io
import json
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
                benchmark.summary(Namespace(output_dir=Path(directory)))
        row = output.getvalue().splitlines()[-1]
        self.assertEqual([cell.strip() for cell in row.strip('|').split('|')],
                         ['stack_bowls', 'direct_robodojo', '2', '1/2', '57.5', '4.0', '20k', '40k', '10', '60',
                          'native_terminated, native_truncated'])


if __name__ == '__main__':
    unittest.main()
