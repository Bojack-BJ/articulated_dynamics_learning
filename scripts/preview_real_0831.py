from pathlib import Path
from PIL import Image, ImageDraw

root = Path('data/track2art_real_new_20260831/extracted')
scenes = sorted(root.iterdir())
canvas = Image.new('RGB', (1272, len(scenes)*270), 'white')
draw = ImageDraw.Draw(canvas)
for row, scene in enumerate(scenes):
    files = sorted((scene/'color/0').glob('*.png'), key=lambda p: int(p.stem))
    anchor = files[len(files)//2].stem
    for view in range(3):
        frame = Image.open(scene/'color'/str(view)/f'{anchor}.png').convert('RGB')
        frame.thumbnail((424,240))
        canvas.paste(frame,(view*424,row*270+30))
        draw.text((view*424+4,row*270+5),f'{scene.name} v{view} f{anchor}',fill='black')
canvas.save('/private/tmp/real0831_preview.jpg')
