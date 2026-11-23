"""Run CleanRUA agents on RoboDojo or DexJoCo tasks, one process per episode, and summarize their result.json files.

    python -m cleanrua_utils.benchmark run --algorithms direct cap --tasks stack_bowls stack_blocks --repeats 3 \
        --native-root /path/to/RoboDojo
    python -m cleanrua_utils.benchmark run --benchmark dexjoco --algorithms delegate monitor --tasks pick_bucket \
        --seeds 0-9 --native-root /path/to/dexjoco
    python -m cleanrua_utils.benchmark summary runs/benchmark

Each episode runs `python -m cleanrua.<algorithm>_<benchmark>`, or `cleanrua.experimental.` for an experimental
variant, with the given task and writes to
`<output-dir>/<task>__<algorithm>_<benchmark>__<seed>__<time>/`. RoboDojo has one evaluation scene, seed 0; DexJoCo
runs every seed in `--seeds`, such as `0-9` or `10,12`. Episodes run seed by seed, `--repeats` times each. An episode
that exceeds `--episode-timeout` or reports GPU out-of-memory is killed and recorded as such, so one stuck simulator
does not stop the batch. Delegate and Monitor need a Pi0.5 server; see examples/installation_<benchmark>.md.

Agents that learn across episodes keep their state in a memory directory. `--memory DIR --learn` passes
`--memory-dir DIR --learn` to them and appends each episode's seed, outcome and cost to `DIR/learning.jsonl`.
`--memory DIR` alone evaluates frozen memory: DIR is copied once to `<output-dir>/memory`, its file hashes go to
`<output-dir>/memory_manifest.json`, agents get `--memory-dir <output-dir>/memory --no-learn`, a task's test seeds may
not include seeds it learned from, and the run stops if the copy changes.
"""
import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ALGORITHMS = ('direct', 'cap', 'delegate', 'monitor', 'cap_delegate', 'cap_explore')
EXPERIMENTAL = ('cap_jev', 'direct_jev', 'monitor_check', 'delegate_memory')  # In cleanrua/experimental/.


def run(args) -> None:
    seeds = sorted({seed for part in args.seeds.split(',')
                    for seed in range(int(part.split('-')[0]), int(part.split('-')[-1]) + 1)})
    if args.benchmark == 'robodojo' and seeds != [0]:
        raise ValueError('RoboDojo has one evaluation scene, seed 0')
    logs = args.output_dir / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    memory_args, snapshot = [], None
    if args.memory and args.learn:
        args.memory.mkdir(parents=True, exist_ok=True)
        memory_args = ['--memory-dir', str(args.memory), '--learn']
    elif args.memory:  # Frozen evaluation on a copy, so learning elsewhere cannot change it mid-run.
        learning = args.memory / 'learning.jsonl'
        learned = {(e['task'], e['seed']) for e in map(json.loads, learning.read_text().splitlines())} \
            if learning.exists() else set()
        if leaked := sorted(learned & {(task, seed) for task in args.tasks for seed in seeds}):
            raise ValueError(f'These test seeds were used for learning: {leaked}')
        snapshot = args.output_dir / 'memory'
        digest = lambda root: {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sorted(root.rglob('*')) if path.is_file()}
        if not snapshot.exists():
            shutil.copytree(args.memory, snapshot)
        frozen = digest(snapshot)
        if frozen != digest(args.memory):
            raise ValueError(f'{args.memory} differs from the snapshot in {snapshot}; use a new --output-dir')
        (args.output_dir / 'memory_manifest.json').write_text(json.dumps(
            {'source': str(args.memory), 'files': frozen}, indent=1))
        memory_args = ['--memory-dir', str(snapshot), '--no-learn']
    episodes = [(task, algorithm, seed, repeat) for seed in seeds for repeat in range(args.repeats)
                for task in args.tasks for algorithm in args.algorithms]
    for number, (task, algorithm, seed, repeat) in enumerate(episodes, 1):
        log = logs / f'{task}__{algorithm}__{seed}__{int(time.time())}.log'
        package = 'cleanrua.experimental' if algorithm in EXPERIMENTAL else 'cleanrua'
        command = [sys.executable, '-m', f'{package}.{algorithm}_{args.benchmark}',
                   '--native-root', str(args.native_root), '--task', task, '--output-dir', str(args.output_dir),
                   *memory_args, *args.extra]
        if args.benchmark == 'dexjoco':
            command += ['--seed', str(seed)]
        print(f'[{number}/{len(episodes)}] {algorithm} {task} seed {seed} repeat {repeat}', flush=True)
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
        new = [path for path in args.output_dir.glob(f'{task}__{algorithm}_{args.benchmark}__*/result.json')
               if path.stat().st_mtime >= start]
        result = json.loads(new[0].read_text()) if new else {}
        print(f'    {outcome or f"exit {process.returncode}"} after {time.time() - start:.0f} s:',
              {key: result.get(key) for key in ('stop_reason', 'success', 'score', 'decisions', 'total_tokens')},
              flush=True)
        if memory_args and args.learn:  # Every learning episode counts, including failed and killed ones.
            entry = {'task': task, 'algorithm': algorithm, 'seed': seed,
                     'run': new[0].parent.name if new else None,
                     'stop_reason': result.get('stop_reason', outcome or 'incomplete'),
                     'success': result.get('success', False), 'model': result.get('model'),
                     'elapsed_s': round(time.time() - start, 1),
                     **{key: result.get(key, 0) for key in ('native_steps', 'total_tokens', 'prompt_tokens',
                                                            'cached_tokens', 'completion_tokens')}}
            with (args.memory / 'learning.jsonl').open('a') as file:
                file.write(json.dumps(entry) + '\n')
        if snapshot and digest(snapshot) != frozen:
            raise ValueError(f'The frozen memory in {snapshot} changed during the episode in {log}')


def summary(args) -> None:
    prices = {}
    for item in args.price:
        model, rates = item.split('=', 1)
        prices[model] = tuple(map(float, rates.split(',')))

    def price(results):  # USD, or None when a model has no price.
        if not all(r.get('model') in prices for r in results):
            return None
        return sum(((r.get('prompt_tokens', 0) - r.get('cached_tokens', 0)) * prices[r['model']][0]
                    + r.get('cached_tokens', 0) * prices[r['model']][1]
                    + r.get('completion_tokens', 0) * prices[r['model']][2]) / 1e6 for r in results)
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
    header = ('task', 'algorithm', 'n', 'excluded', 'success', 'success 95% CI', 'score', 'decisions', 'steps',
              'tokens', 'tokens/success', 'cost', 'cost/success', 'model_s', 'elapsed_s', 'stops')
    rows = []
    for (task, algorithm), group in sorted(groups.items()):
        stops = ', '.join(sorted({r['stop_reason'] for r, complete in group}))
        valid = [r for r, complete in group if complete and not r['stop_reason'].startswith('model_error')]
        n = len(valid)
        excluded = len(group) - n
        if not n:
            rows.append((task, algorithm, n, excluded, *(['-'] * 11), stops))
            continue
        successes = sum(r['success'] for r in valid)
        p, z = successes / n, 1.96
        d = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / d
        half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
        interval = f'{max(0., center - half) * 100:.0f}–{min(1., center + half) * 100:.0f}%'
        cost = price(valid)
        mean = lambda key: statistics.mean(r.get(key, 0) for r in valid)
        rows.append((task, algorithm, n, excluded, f'{successes}/{n}', interval, f'{mean("score"):.1f}',
                     f'{mean("decisions"):.1f}', f'{mean("native_steps"):.0f}', f'{mean("total_tokens") / 1e3:.0f}k',
                     f'{sum(r.get("total_tokens", 0) for r in valid) / successes / 1e3:.0f}k' if successes else '-',
                     f'${cost / n:.2f}' if cost is not None else '-',
                     f'${cost / successes:.2f}' if cost is not None and successes else '-',
                     f'{mean("model_s"):.0f}', f'{mean("elapsed_s"):.0f}', stops))
    print('| ' + ' | '.join(header) + ' |\n|' + '---|' * len(header))
    for row in rows:
        print('| ' + ' | '.join(map(str, row)) + ' |')
    learning = args.output_dir / 'memory' / 'learning.jsonl'
    if learning.exists():  # The cost of building the frozen memory these episodes used.
        entries = {}
        for entry in map(json.loads, learning.read_text().splitlines()):
            entries.setdefault((entry['task'], entry['algorithm']), []).append(entry)
        print('\nLearning behind the frozen memory:\n\n| task | algorithm | episodes | seeds | success | steps | '
              'tokens | cost | elapsed_s |\n|---|---|---|---|---|---|---|---|---|')
        for (task, algorithm), group in sorted(entries.items()):
            cost, seeds = price(group), ','.join(str(seed) for seed in sorted({e['seed'] for e in group}))
            print(f'| {task} | {algorithm} | {len(group)} | {seeds} | {sum(e["success"] for e in group)}/{len(group)} '
                  f'| {sum(e["native_steps"] for e in group)} | {sum(e["total_tokens"] for e in group) / 1e3:.0f}k '
                  f'| {f"${cost:.2f}" if cost is not None else "-"} | {sum(e["elapsed_s"] for e in group):.0f} |')


def export(args) -> None:
    data = {}
    for item in args.meta:
        key, value = item.split('=', 1)
        data[key] = int(value) if re.fullmatch(r'-?\d+', value) else value
    data['tasks'] = args.tasks
    data['algorithms'] = args.algorithms
    data['episodes'] = []
    for path in sorted(args.output_dir.glob('*/result.json')):  # Run folders are <task>__<algorithm>__<seed>__<time>.
        task, algorithm = path.parent.name.split('__')[:2]
        r = {'task': task, 'algorithm': algorithm, **json.loads(path.read_text())}
        data['episodes'].append({  # An episode whose first model request failed has no usage fields.
            # Runs before 2026-11-21 named CaP agents code_*.
            'algorithm': re.sub('^code', 'cap', re.sub('_(robodojo|dexjoco)$', '', r['algorithm'])),
            'task': r['task'], 'success': r['success'],
            'score': r['score'], 'decisions': r['decisions'], 'tokens': r.get('total_tokens', 0),
            'native_steps': r['native_steps'], 'step_limit': r['step_limit'], 'elapsed_s': r['elapsed_s'],
            'model_s': r.get('model_s', 0.), 'stop_reason': r['stop_reason'], 'finish': r['agent_finished'],
            'invalid': r['stop_reason'].startswith('model_error')})
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
    runner.add_argument('--benchmark', choices=('robodojo', 'dexjoco'), default='robodojo')
    runner.add_argument('--algorithms', nargs='+', choices=ALGORITHMS + EXPERIMENTAL, required=True)
    runner.add_argument('--tasks', nargs='+', required=True)
    runner.add_argument('--seeds', default='0', help='seeds such as 0-9 or 10,12; RoboDojo has only seed 0')
    runner.add_argument('--repeats', type=int, default=1, help='episodes per task, algorithm and seed')
    runner.add_argument('--memory', type=Path, help='memory directory of agents that learn across episodes')
    runner.add_argument('--learn', action='store_true', help='let agents update --memory instead of freezing it')
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
    exporter.add_argument('--algorithms', nargs='+', choices=ALGORITHMS + EXPERIMENTAL, required=True)
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
