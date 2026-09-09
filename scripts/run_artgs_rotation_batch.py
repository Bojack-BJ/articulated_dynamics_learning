#!/usr/bin/env python3
"""Run an isolated shard of the aligned ArtGS rotation benchmark."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    base = args.root / 'outputs/paper_experiments/external_rotation_v1'
    base.mkdir(parents=True, exist_ok=True)
    with (args.root / 'outputs/external_baseline_suite_v1/manifest.csv').open() as stream:
        objects = [row['object_id'] for row in csv.DictReader(stream)]
    for obj in objects[args.shard::2]:
        work = base / ('artgs_' + obj.removeprefix('partnet_'))
        work.mkdir(parents=True, exist_ok=True)
        status = work / 'status.json'
        complete = status.exists() and json.loads(status.read_text()).get('status') == 'complete'
        if not complete:
            command = [sys.executable, str(args.root / 'scripts/run_artgs_rotation_comparison.py'),
                       '--root', str(args.root), '--repo', str(args.repo), '--work', str(work),
                       '--object', obj, '--gpu', args.gpu]
            with (work / 'driver.log').open('a') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print(obj, 'failed', result.returncode, flush=True)
                continue
        command = [sys.executable, str(args.root / 'scripts/evaluate_artgs_rotation_pair.py'),
                   '--root', str(args.root), '--work', str(work)]
        with (work / 'evaluation.log').open('a') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        print(obj, 'evaluation returncode', result.returncode, flush=True)


if __name__ == '__main__':
    main()
