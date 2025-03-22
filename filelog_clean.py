import os


results_dir = 'results'
with open(os.path.join(results_dir, 'synced_ids'), 'r') as f:
    synced_ids = [int(id) for id in f.readlines() if id.strip()]

current_runs = [d for d in os.listdir(results_dir) if d.startswith('run_') and int(d.split('_')[1]) in synced_ids]

if not current_runs:
    print('No leftover runs to clean')
    exit()

print(f'Cleaning runs: {current_runs}')
for run in current_runs:
    run_path = os.path.join(results_dir, run)
    for root, _, filenames in os.walk(run_path, topdown=False):
        for filename in filenames:
            os.remove(os.path.join(root, filename))
        os.rmdir(root)

with open(os.path.join(results_dir, 'synced_ids'), 'w') as f:
    pass
