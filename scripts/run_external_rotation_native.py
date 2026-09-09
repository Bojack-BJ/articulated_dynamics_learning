#!/usr/bin/env python3
"""Rerun native external methods on globally rotated observations."""
import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def rotate_tree(source, target, q):
    from plyfile import PlyData
    target.mkdir(parents=True, exist_ok=True)
    t = np.eye(4)
    t[:3, :3] = q
    for path in source.iterdir():
        dest = target / path.name
        if path.name in {'init', 'final'} or dest.exists():
            continue
        if path.is_dir():
            rotate_tree(path, dest, q)
        elif path.suffix == '.ply':
            ply = PlyData.read(path)
            v = ply['vertex'].data
            for keys in [('x', 'y', 'z'), ('nx', 'ny', 'nz')]:
                if all(k in v.dtype.names for k in keys):
                    xyz = np.column_stack([v[k] for k in keys]) @ q.T
                    for i, key in enumerate(keys):
                        v[key] = xyz[:, i]
            ply.write(dest)
        elif path.name == 'data.npz':
            with np.load(path) as z:
                data = {k: z[k] for k in z.files}
            data['poses'] = (t @ data['poses']).astype(data['poses'].dtype)
            data['extrinsics'] = (data['extrinsics'] @ t.T).astype(data['extrinsics'].dtype)
            np.savez_compressed(dest, **data)
        elif path.name in {'filtered.npz', 'filtered_vis.npz'}:
            with np.load(path) as z:
                data = {k: z[k] for k in z.files}
            if 'coords' in data:
                if data['coords'].shape[-1] != 3:
                    raise ValueError('Expected world-space XYZ tracks')
                data['coords'] = (data['coords'] @ q.T).astype(data['coords'].dtype)
            np.savez_compressed(dest, **data)
        elif path.name.startswith('transforms') and path.suffix == '.json':
            data = json.loads(path.read_text())
            for frame in data['frames']:
                frame['transform_matrix'] = (t @ np.asarray(frame['transform_matrix'])).tolist()
            dest.write_text(json.dumps(data))
        else:
            dest.symlink_to(path.resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--method', choices=['aim', 'reart', 'videoartgs', 'aim_then_videoartgs'], required=True)
    p.add_argument('--gpu', required=True)
    args = p.parse_args()
    if args.method == 'aim_then_videoartgs':
        for method in ['aim', 'videoartgs']:
            subprocess.run([sys.executable, __file__, '--root', str(args.root),
                            '--method', method, '--gpu', args.gpu], check=True)
        return
    root = args.root
    q = Rotation.random(random_state=20260909).as_matrix()
    base = root / 'outputs/paper_experiments/external_rotation_v1' / args.method
    with (root / 'outputs/external_baseline_suite_v1/manifest.csv').open() as f:
        objects = [r['object_id'] for r in csv.DictReader(f)]
    for obj in objects:
        out = base / obj
        out.mkdir(parents=True, exist_ok=True)
        status_path = out / 'status.json'
        if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'native_complete':
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(root / 'src'))
        status = dict(object=obj, method=args.method, rotation=q.tolist(), protocol='global coordinate rotation; RGB-D unchanged', status='preparing')
        status_path.write_text(json.dumps(status))
        try:
            if args.method == 'reart':
                source = root / 'outputs/external_baselines_v1/reart_sequences_dense20' / obj
                target = out / 'input' / obj
                rotate_tree(source, target, q)
                cwd = root / 'ReArt'
                commands = [[sys.executable, '-m', 'rgbd_urdf_mvp.perception.reart_run_wrapper',
                             '--reart-root', str(cwd), '--seq-path', str(target), '--save-root', str(out / 'native'),
                             '--stage', 'both', '--base-n-iter', '2000', '--kinematic-n-iter', '200',
                             '--num-points', '512', '--num-parts', '10', '--use-flow-loss', '--use-assign-loss', '--use-nproc']]
            elif args.method == 'aim':
                source = root / 'outputs/external_baselines_v1/aim_style_aligned_dataset' / obj
                target = out / 'input'
                rotate_tree(source, target, q)
                original = root / 'outputs/external_baselines_v1/aim_style_aligned_runs' / obj
                commands = []
                for name in ['train.log', 'segmentation.log']:
                    cmd = shlex.split((original / name).read_text().splitlines()[0].removeprefix('$ '))
                    cmd[cmd.index('--source_path') + 1] = str(target)
                    cmd[cmd.index('--model_path') + 1] = str(out / 'native')
                    commands.append(cmd)
                cwd = Path(commands[0][1]).parent
                env['PYTHONPATH'] = ':'.join(map(str, [cwd, cwd / 'submodules/diff-gaussian-rasterization1/build/lib.linux-x86_64-cpython-310', cwd / 'submodules/simple-knn/build/lib.linux-x86_64-cpython-310', cwd / 'lib/pointops/build/lib.linux-x86_64-cpython-310']))
            else:
                old = root / 'outputs/external_baseline_suite_v1/videoartgs_native_v2'
                new = out / 'input'
                source = old / 'videoartgs/realscan' / obj
                rotate_tree(source, new / 'videoartgs/realscan' / obj, q)
                plan = json.loads((root / 'outputs/external_baseline_suite_v1/per_object' / obj / 'videoartgs/run_plan.json').read_text())
                commands = [[s.replace(str(old), str(new)) for s in cmd] for cmd in plan['commands']]
                cwd = root.parent / 'third_party/VideoArtGS'
                env.update(dict(plan.get('environment', [])))
            for index, command in enumerate(commands):
                status.update(status='running', stage=index, command=command)
                status_path.write_text(json.dumps(status))
                with (out / f'stage_{index}.log').open('w') as log:
                    subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            status.update(status='native_complete', evaluation='pending')
        except Exception as error:
            status.update(status='failed', error=str(error))
        status_path.write_text(json.dumps(status))
        print(obj, status['status'], flush=True)


if __name__ == '__main__':
    main()
