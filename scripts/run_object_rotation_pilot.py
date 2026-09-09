#!/usr/bin/env python3
"""Run fresh object-mask tracks and a frozen checkpoint on rendered rotations."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--slot', required=True)
    p.add_argument('--relation', required=True)
    p.add_argument('--gpu', default='6')
    args = p.parse_args()
    base = args.root / 'outputs/paper_experiments/object_rotation_visual_v1'
    env = dict(os.environ, PYTHONPATH=str(args.root / 'src'), CUDA_VISIBLE_DEVICES=args.gpu)
    cli = [sys.executable, '-m', 'rgbd_urdf_mvp']
    for path in sorted(base.glob('*/yaw_*/episode.json')):
        for mode, name in [('triview', 'episode.json'), ('single', 'episode_single.json')]:
            payload = json.loads((path.parent / name).read_text())
            # Part labels are evaluation-only; never use them for query allocation.
            payload['metadata'].pop('part_segmentation', None)
            episode = path.parent / f'episode_tracking_{mode}.json'
            episode.write_text(json.dumps(payload))
            out = path.parent / mode
            out.mkdir(exist_ok=True)
            tracks, features = out / 'tracks.json', out / 'features.npz'
            stages = [
                ('tracking', [*cli, 'track-part-pixels', str(episode), '--output-json', str(tracks),
                 '--device', 'cuda', '--cotracker-repo', str(args.root / 'co-tracker'),
                 '--cotracker-checkpoint', str(args.root / 'co-tracker/ckpt/scaled_offline.pth'),
                 '--frame-stride', '2', '--seed-stride-px', '8', '--max-tracks-per-part-view', '512',
                 '--export-cotracker-features', '--cotracker-features-output', str(features)]),
                ('slots', [*cli, 'infer-motion-part-slots', str(tracks), str(features), args.slot,
                 '--output-json', str(out / 'slots.json'), '--device', 'cuda']),
                ('relation', [sys.executable, '-c',
                 'import sys; from rgbd_urdf_mvp.kinematics.pairwise_relation_head import SlotRelationInferencer, SlotRelationInferenceConfig; '
                 'SlotRelationInferencer(SlotRelationInferenceConfig(tracks_path=sys.argv[1], features_npz=sys.argv[2], '
                 'slot_model_path=sys.argv[3], relation_model_path=sys.argv[4], output_json=sys.argv[5], '
                 'device="cuda", quality_weighted_trajectories=True)).infer()',
                 str(tracks), str(features), args.slot, args.relation, str(out / 'joints.json')]),
                ('viewer', [*cli, 'visualize-object-mask-flow-html', str(out / 'slots.json'),
                 '--joint-inference', str(out / 'joints.json'), '--output-html', str(out / 'viewer.html'),
                 '--axis-remap', 'x,y,z']),
            ]
            for stage, command in stages:
                artifact = {'tracking': features, 'slots': out / 'slots.json',
                            'relation': out / 'joints.json', 'viewer': out / 'viewer.html'}[stage]
                if artifact.exists() and (stage != 'tracking' or tracks.exists()):
                    continue
                status = dict(stage=stage, status='running', command=command)
                (out / 'status.json').write_text(json.dumps(status))
                with (out / f'{stage}.log').open('w') as log:
                    stage_env = dict(env)
                    if stage == 'relation':
                        stage_env['PYTHONPATH'] = str(args.root.parent / 'neural_head_so3_pilot_workspace/src')
                    result = subprocess.run(command, env=stage_env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    status.update(status='failed', returncode=result.returncode)
                    (out / 'status.json').write_text(json.dumps(status))
                    raise SystemExit(result.returncode)
            (out / 'status.json').write_text(json.dumps(dict(status='complete', slot=args.slot, relation=args.relation)))
            print(path.parent, mode, 'complete', flush=True)


if __name__ == '__main__':
    main()
