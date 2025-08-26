# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

import neptune
from neptune import Run
import os
import time
import signal

results_dir = 'results'
ids_file = os.path.join(results_dir, 'synced_ids')
if os.path.exists(ids_file):
    with open(ids_file, 'r') as f:
        synced_ids = [int(id) for id in f.readlines() if id.strip()]
else:
    synced_ids = []

current_runs = [d for d in os.listdir(results_dir) if d.startswith('run_') and int(d.split('_')[1]) not in synced_ids]

if not current_runs:
    print('No new runs to sync')
    exit()
print(f'Syncing runs: {current_runs}')

with open(ids_file, 'a') as f:
    f.write('\n'.join(run.split('_')[1] for run in current_runs))
    f.write('\n')

neptune_runs: dict[str, Run] = {}
for run in current_runs:
    run_path = os.path.join(results_dir, run)
    with open(os.path.join(run_path, 'tags'), 'r') as f:
        tags = f.read().splitlines()
    neptune_run = neptune.init_run(
        project=os.getenv('NEPTUNE_PROJECT'),
        tags=tags,
    )
    neptune_runs[run] = neptune_run

runs_counters: dict[str, dict[str, int]] = {run: {} for run in current_runs}


def kill_handler(signum, frame):
    print('Killing...')
    for runs in neptune_runs.values():
        runs.stop()
    exit()


signal.signal(signal.SIGTERM, kill_handler)

while True:
    print('Syncing...')
    for run in current_runs:
        neptune_run = neptune_runs[run]
        run_path = os.path.join(results_dir, run, 'logs')
        files: list[str] = []
        for root, _, filenames in os.walk(run_path):
            relative_root = root.replace(run_path, '')
            relative_root = relative_root[1:] if relative_root.startswith('/') else relative_root
            for filename in filenames:
                files.append(os.path.join(relative_root, filename) if relative_root else filename)
        for file in files:
            if file not in runs_counters[run]:
                runs_counters[run][file] = 0
            read_idx = runs_counters[run][file]
            with open(os.path.join(run_path, file), 'r') as f:
                values = [float(line) for line in f.readlines()]
            neptune_run[file].extend(values[read_idx:])
            runs_counters[run][file] = len(values)
    time.sleep(60)
