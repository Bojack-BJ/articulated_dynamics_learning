from pathlib import Path
from PIL import Image, ImageDraw

scenes = sorted(Path('outputs').glob('real_0831_*_track2art_v1'))
canvas = Image.new('RGB',(1272,len(scenes)*270),'white')
draw = ImageDraw.Draw(canvas)
for row, scene in enumerate(scenes):
    for view in range(3):
        files = sorted(scene.glob(f'qa_view{view}_frame*.jpg'),key=lambda p:int(p.stem.split('frame')[-1]))
        if not files:
            continue
        frame = Image.open(files[len(files)//2])
        frame.thumbnail((424,240))
        canvas.paste(frame,(view*424,row*270+30))
        draw.text((view*424+4,row*270+5),scene.name.removeprefix('real_0831_').removesuffix('_track2art_v1')+f' v{view}',fill='black')
canvas.save('/private/tmp/real0831_mask_preview.jpg')
