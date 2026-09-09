#!/usr/bin/env python3
"""Replay joint states with a rotated object and unchanged cameras."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--objects', nargs='+', default=['partnet_10849', 'partnet_46556'])
    args = p.parse_args()
    sys.path.insert(0, str(args.root / 'src'))
    import mujoco
    from rgbd_urdf_mvp.sim.mujoco_recorder import MuJoCoEpisodeRecorder, MuJoCoRecordConfig

    for obj in args.objects:
        source = args.source / obj / 'episode.json'
        original = json.loads(source.read_text())
        meta = original['metadata']
        for angle in (0, 45, 90):
            out = args.output / obj / f'yaw_{angle}'
            out.mkdir(parents=True, exist_ok=True)
            model = mujoco.MjModel.from_xml_path(meta['model_path'])
            root = meta['target_root_body_id']
            if model.body_parentid[root] != 0 or model.body_jntnum[root] != 0:
                raise ValueError('Pilot requires a fixed world-child object root')
            q = Rotation.from_euler('z', angle, degrees=True)
            center = np.asarray(meta['lookat'])
            model.body_pos[root] = center + q.apply(model.body_pos[root] - center)
            old = model.body_quat[root].copy()
            xyzw = (q * Rotation.from_quat(old[[1, 2, 3, 0]])).as_quat()
            model.body_quat[root] = xyzw[[3, 0, 1, 2]]
            mujoco.mj_saveLastXML(str(out / 'model.xml'), model)
            data = mujoco.MjData(model)
            first = original['frames'][0]
            w, h = Image.open(source.parent / first['depth_paths_by_view'][0]).size
            model.vis.global_.fovy = meta['camera_fovy_deg']
            renderer = mujoco.Renderer(model, height=h, width=w)
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = meta['lookat']
            camera.distance = meta['camera_distance']
            camera.elevation = meta['camera_elevation_deg']
            recorder = MuJoCoEpisodeRecorder(MuJoCoRecordConfig(
                model_path=meta['model_path'], output_dir=out,
                object_instance_id=obj, category=original['category']))
            option = recorder._scene_option_hiding_geom_ids(
                mujoco, model, set(meta.get('hidden_clear_geom_ids', [])))
            payload = copy.deepcopy(original)
            payload['metadata']['model_path'] = str(out / 'model.xml')
            payload['metadata']['object_rotation_test'] = dict(
                yaw_degrees=angle, center=center.tolist(), rotation=q.as_matrix().tolist(),
                source=str(source), protocol='fixed cameras; replayed qpos; rerendered RGB-D')
            coverage = []
            for i, frame in enumerate(payload['frames']):
                for name, value in frame['action_log']['joint_positions'].items():
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if jid < 0:
                        raise ValueError(name)
                    data.qpos[model.jnt_qposadr[jid]] = value
                mujoco.mj_forward(model, data)
                for key in ('rgb', 'depth', 'mask', 'part_mask'):
                    frame[key + '_paths_by_view'] = []
                counts = []
                for v, az in enumerate(meta['camera_azimuths_deg']):
                    camera.azimuth = az
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    rgb = renderer.render().copy()
                    renderer.enable_depth_rendering()
                    depth = renderer.render().copy()
                    renderer.disable_depth_rendering()
                    renderer.enable_segmentation_rendering()
                    seg = renderer.render().copy()
                    renderer.disable_segmentation_rendering()
                    part = np.zeros((h, w), dtype=np.uint16)
                    for item in meta['part_segmentation']['parts']:
                        hit = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 0], item['geom_ids'])
                        part[hit] = item['part_id']
                    counts.append({str(pid): int((part == pid).sum()) for pid in np.unique(part) if pid})
                    arrays = dict(rgb=rgb, depth=np.clip(np.nan_to_num(depth) * 1000, 0, 65535).astype(np.uint16),
                                  mask=(part > 0).astype(np.uint16) * 65535, part_mask=part)
                    for key, array in arrays.items():
                        rel = f'assets/view_{v}/frame_{i:04d}_{key}.png'
                        path = out / rel
                        path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(array).save(path)
                        frame[key + '_paths_by_view'].append(rel)
                for key in ('rgb', 'depth', 'mask', 'part_mask'):
                    frame[key + '_path'] = frame[key + '_paths_by_view'][1]
                coverage.append(counts)
            renderer.close()
            (out / 'episode.json').write_text(json.dumps(payload))
            (out / 'visibility.json').write_text(json.dumps(coverage))
            single = copy.deepcopy(payload)
            single['metadata']['camera_mode'] = 'orbit'
            single['metadata']['camera_azimuths_deg'] = [meta['camera_azimuths_deg'][1]]
            single['metadata']['selected_source_view'] = 1
            for frame in single['frames']:
                for key in ('rgb', 'depth', 'mask', 'part_mask'):
                    frame[key + '_paths_by_view'] = [frame[key + '_path']]
                frame['camera_poses_by_view'] = [frame['camera_pose']]
            (out / 'episode_single.json').write_text(json.dumps(single))
            print(obj, angle, 'rendered', len(coverage), flush=True)


if __name__ == '__main__':
    main()
