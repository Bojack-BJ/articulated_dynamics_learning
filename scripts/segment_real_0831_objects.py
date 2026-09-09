"""Generate object-only SAM2 masks and a calibrated tracking episode."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument('scene', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--sam-root', type=Path, required=True)
    p.add_argument('--prepare', type=Path, required=True)
    p.add_argument('--views', default='0,1,2')
    args = p.parse_args()
    sys.path.insert(0, str(args.sam_root))
    from sam2.build_sam import build_sam2_video_predictor
    model = build_sam2_video_predictor(
        'configs/sam2.1/sam2.1_hiera_l.yaml',
        str(args.sam_root/'checkpoints/sam2.1_hiera_large.pt'), device='cuda')
    # Boxes are in the inspected 848x480 source images; include the swept door.
    boxes = ([[340,110,640,360],[375,285,530,470],[235,145,430,340]]
             if 'microwave' in args.scene.name else
             [[275,55,840,405],[330,255,550,480],[135,70,495,370]])
    count = json.loads((args.scene/'metadata.json').read_text())['frame_num']
    anchor = count//2
    args.output.mkdir(parents=True, exist_ok=True)
    stats = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for view, box in enumerate(boxes):
            if view not in [int(v) for v in args.views.split(',')]:
                continue
            frames = args.output/'sam_video'/str(view)
            frames.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                target = frames/f'{i:06d}.jpg'
                if not target.exists():
                    Image.open(args.scene/'color'/str(view)/f'{i}.png').convert('RGB').save(target, quality=95)
            state = model.init_state(str(frames), offload_video_to_cpu=True, offload_state_to_cpu=True)
            negatives = ([[500,320]], [[430,450]], []) if 'microwave' in args.scene.name else ([], [], [])
            points = negatives[view]
            kwargs = {'points':np.asarray(points,np.float32), 'labels':np.zeros(len(points),np.int32)} if points else {}
            model.add_new_points_or_box(state, frame_idx=anchor, obj_id=1, box=np.asarray(box,np.float32), **kwargs)
            if view == 2 and args.scene.name in ('ofen_hand_open', 'ofen_freefall_2'):
                # Separate mask identity preserves the paper-covered interior of the door.
                model.add_new_points_or_box(state, frame_idx=anchor, obj_id=2,
                    box=np.asarray([140,210,395,320],np.float32),
                    points=np.asarray([[260,263],[200,259],[322,264]],np.float32),
                    labels=np.ones(3,np.int32))
            masks = args.scene/'mask'/str(view)/'1'
            masks.mkdir(parents=True, exist_ok=True)
            written = set()
            for reverse in (False, True):
                for i, ids, logits in model.propagate_in_video(state, start_frame_idx=anchor, reverse=reverse):
                    mask = (logits[:,0]>0).any(dim=0).cpu().numpy()
                    Image.fromarray((mask*255).astype(np.uint8)).save(masks/f'{i}.png')
                    written.add(i)
                    if i in (0,anchor,count-1):
                        rgb = np.asarray(Image.open(args.scene/'color'/str(view)/f'{i}.png').convert('RGB')).copy()
                        rgb[mask] = (rgb[mask]*0.55+np.array([20,220,140])*0.45).astype(np.uint8)
                        Image.fromarray(rgb).save(args.output/f'qa_view{view}_frame{i}.jpg')
            if len(written)!=count:
                raise RuntimeError(f'Incomplete propagation: {len(written)}/{count}')
            stats.append({'view':view,'frames':len(written),'box':box,'anchor':anchor})
            del state
            torch.cuda.empty_cache()
    (args.output/'object_mask_provenance.json').write_text(json.dumps({
        'source':'SAM2 prompted object masks, not GT parts','views':stats},indent=2))
    subprocess.run([sys.executable,str(args.prepare),str(args.scene),str(args.output),
        '--views','0,1,2','--object-mask-id','1','--hand-mask-ids','',
        '--object-erosion-px','2','--frame-stride','1',
        '--category','microwave' if 'microwave' in args.scene.name else 'oven'],check=True)


if __name__ == '__main__':
    main()
