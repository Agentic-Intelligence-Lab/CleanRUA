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
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ALGORITHMS = ('direct', 'code', 'delegate', 'monitor', 'code_delegate', 'code_jev', 'direct_jev')


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
    prices = {}
    for price in args.price:
        model, rates = price.split('=', 1)
        input_price, cached_price, output_price = map(float, rates.split(','))
        prices[model] = input_price, cached_price, output_price
    groups = {}
    for directory in sorted(args.output_dir.iterdir()):
        parts = directory.name.split('__')
        if not directory.is_dir() or len(parts) != 4:
            continue
        path = directory / 'result.json'
        result = json.loads(path.read_text()) if path.is_file() else {'stop_reason': 'incomplete'}
        result.setdefault('task', parts[0])
        result.setdefault('algorithm', parts[1])
        groups.setdefault((result['task'], result['algorithm']), []).append((result, path.is_file()))
    header = ('task', 'algorithm', 'n', 'excluded', 'success', 'success 95% CI', 'score', 'decisions', 'tokens',
              'tokens/success', 'cost', 'cost/success', 'model_s', 'elapsed_s', 'stops')
    rows = []
    for (task, algorithm), group in sorted(groups.items()):
        stops = ', '.join(sorted({r['stop_reason'] for r, complete in group}))
        valid = [r for r, complete in group if complete and not r['stop_reason'].startswith('model_error')]
        n = len(valid)
        excluded = len(group) - n
        if not n:
            rows.append((task, algorithm, n, excluded, *(['-'] * 10), stops))
            continue
        successes = sum(r['success'] for r in valid)
        p, z = successes / n, 1.96
        d = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / d
        half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
        interval = f'{max(0., center - half) * 100:.0f}–{min(1., center + half) * 100:.0f}%'
        cost = None
        if all(r.get('model') in prices for r in valid):
            cost = 0.
            for result in valid:
                input_price, cached_price, output_price = prices[result['model']]
                cached = result.get('cached_tokens', 0)
                cost += ((result.get('prompt_tokens', 0) - cached) * input_price + cached * cached_price
                         + result.get('completion_tokens', 0) * output_price) / 1e6
        mean = lambda key: statistics.mean(r.get(key, 0) for r in valid)
        rows.append((task, algorithm, n, excluded, f'{successes}/{n}', interval, f'{mean("score"):.1f}',
                     f'{mean("decisions"):.1f}', f'{mean("total_tokens") / 1e3:.0f}k',
                     f'{sum(r["total_tokens"] for r in valid) / successes / 1e3:.0f}k' if successes else '-',
                     f'${cost / n:.2f}' if cost is not None else '-',
                     f'${cost / successes:.2f}' if cost is not None and successes else '-',
                     f'{mean("model_s"):.0f}', f'{mean("elapsed_s"):.0f}', stops))
    print('| ' + ' | '.join(header) + ' |\n|' + '---|' * len(header))
    for row in rows:
        print('| ' + ' | '.join(map(str, row)) + ' |')


def export(args) -> None:
    data = {}
    for item in args.meta:
        key, value = item.split('=', 1)
        data[key] = int(value) if re.fullmatch(r'-?\d+', value) else value
    data['tasks'] = args.tasks
    data['algorithms'] = args.algorithms
    data['episodes'] = [
        {'algorithm': r['algorithm'].removesuffix('_robodojo'), 'task': r['task'], 'success': r['success'],
         'score': r['score'], 'decisions': r['decisions'], 'tokens': r['total_tokens'],
         'native_steps': r['native_steps'], 'step_limit': r['step_limit'], 'elapsed_s': r['elapsed_s'],
         'model_s': r['model_s'], 'stop_reason': r['stop_reason'], 'finish': r['agent_finished'],
         'invalid': r['stop_reason'].startswith('model_error')}
        for r in load(args.output_dir)]
    data['not_run'] = []
    for item in args.not_run:
        algorithm, task = item.split(':', 1)
        data['not_run'].append({'algorithm': algorithm, 'task': task})
    with args.output.open('w') as file:
        json.dump(data, file, indent=1)


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
    reporter.add_argument('--price', action='append', default=[], metavar='MODEL=IN,CACHED,OUT',
                          help='USD per million input, cached input and output tokens; repeat for each model')
    exporter = commands.add_parser('export', help='export result.json files for documentation charts')
    exporter.add_argument('output_dir', type=Path)
    exporter.add_argument('-o', '--output', type=Path, required=True)
    exporter.add_argument('--tasks', nargs='+', required=True)
    exporter.add_argument('--algorithms', nargs='+', choices=ALGORITHMS, required=True)
    exporter.add_argument('--not-run', nargs='+', default=[], metavar='ALGORITHM:TASK')
    exporter.add_argument('--meta', nargs='+', default=[], metavar='KEY=VALUE')
    args = parser.parse_args()
    if args.command == 'run':
        args.extra = [a for a in args.extra if a != '--']
        run(args)
    elif args.command == 'summary':
        summary(args)
    else:
        export(args)
