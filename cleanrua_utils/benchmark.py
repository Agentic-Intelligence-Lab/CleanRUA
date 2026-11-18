"""Run CleanRUA agents on RoboDojo tasks, one process per episode, and summarize their result.json files.

    python -m cleanrua_utils.benchmark run --algorithms direct code --tasks stack_bowls stack_blocks --repeats 3 \
        --native-root /path/to/RoboDojo
    python -m cleanrua_utils.benchmark summary runs/benchmark

Each episode runs `python -m cleanrua.<algorithm>_robodojo` with the given task and writes to
`<output-dir>/<task>__<algorithm>_robodojo__<seed>__<time>/`. An episode that exceeds `--episode-timeout` or
reports GPU out-of-memory is killed and recorded as such, so one stuck simulator does not stop the batch.
Delegate and Monitor need a Pi0.5 server; see examples/installation.md.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ALGORITHMS = ('direct', 'code', 'delegate', 'monitor')


def run(args) -> None:
    logs = args.output_dir / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    episodes = [(task, algorithm, repeat) for repeat in range(args.repeats) for task in args.tasks
                for algorithm in args.algorithms]
    for number, (task, algorithm, repeat) in enumerate(episodes, 1):
        log = logs / f'{task}__{algorithm}__{repeat}__{int(time.time())}.log'
        command = [sys.executable, '-m', f'cleanrua.{algorithm}_robodojo', '--native-root', str(args.native_root),
                   '--task', task, '--output-dir', str(args.output_dir), *args.extra]
        print(f'[{number}/{len(episodes)}] {algorithm} {task} repeat {repeat}', flush=True)
        start, outcome = time.time(), None
        with log.open('w') as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(10)
                if time.time() - start > args.episode_timeout:
                    outcome = 'killed: episode timeout'
                elif b'ERROR_OUT_OF_DEVICE_MEMORY' in log.read_bytes():
                    outcome = 'killed: GPU out of memory'
                if outcome:
                    process.kill()
                    process.wait()
        new = [path for path in args.output_dir.glob(f'{task}__{algorithm}_robodojo__*/result.json')
               if path.stat().st_mtime >= start]
        result = json.loads(new[0].read_text()) if new else {}
        print(f'    {outcome or f"exit {process.returncode}"} after {time.time() - start:.0f} s:',
              {key: result.get(key) for key in ('stop_reason', 'success', 'score', 'decisions', 'total_tokens')},
              flush=True)


def load(directory: Path) -> list[dict]:
    results = []
    for path in sorted(directory.glob('*/result.json')):
        result = json.loads(path.read_text())
        result.setdefault('algorithm', path.parent.name.split('__')[1])
        result.setdefault('task', path.parent.name.split('__')[0])
        results.append(result)
    return results


def summary(args) -> None:
    results = load(args.output_dir)
    groups = {}
    for result in results:
        groups.setdefault((result['task'], result['algorithm']), []).append(result)
    header = ('task', 'algorithm', 'n', 'success', 'score', 'decisions', 'tokens', 'tokens/success', 'model_s',
              'elapsed_s', 'stops')
    rows = []
    for (task, algorithm), group in sorted(groups.items()):
        successes = sum(r['success'] for r in group)
        mean = lambda key: statistics.mean(r.get(key, 0) for r in group)
        rows.append((task, algorithm, len(group), f'{successes}/{len(group)}', f'{mean("score"):.1f}',
                     f'{mean("decisions"):.1f}', f'{mean("total_tokens") / 1e3:.0f}k',
                     f'{sum(r["total_tokens"] for r in group) / successes / 1e3:.0f}k' if successes else '-',
                     f'{mean("model_s"):.0f}', f'{mean("elapsed_s"):.0f}',
                     ', '.join(sorted({r['stop_reason'] for r in group}))))
    print('| ' + ' | '.join(header) + ' |\n|' + '---|' * len(header))
    for row in rows:
        print('| ' + ' | '.join(map(str, row)) + ' |')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    runner = commands.add_parser('run', help='run episodes')
    runner.add_argument('--algorithms', nargs='+', choices=ALGORITHMS, required=True)
    runner.add_argument('--tasks', nargs='+', required=True)
    runner.add_argument('--repeats', type=int, default=1)
    runner.add_argument('--native-root', type=Path, required=True)
    runner.add_argument('--output-dir', type=Path, default=Path('runs/benchmark'))
    runner.add_argument('--episode-timeout', type=float, default=4200., help='seconds before an episode is killed')
    runner.add_argument('extra', nargs=argparse.REMAINDER, help='after --, arguments passed to every agent')
    reporter = commands.add_parser('summary', help='summarize result.json files as a Markdown table')
    reporter.add_argument('output_dir', type=Path, nargs='?', default=Path('runs/benchmark'))
    args = parser.parse_args()
    if args.command == 'run':
        args.extra = [a for a in args.extra if a != '--']
        run(args)
    else:
        summary(args)
