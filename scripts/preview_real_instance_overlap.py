from pathlib import Path
from PIL import Image, ImageDraw

roots = [Path('data/track2art_real_20260814/scenes'),
         Path('data/track2art_real_new_20260830/extracted'),
         Path('data/track2art_real_new_20260831/extracted')]
scenes = [p for root in roots for p in sorted(root.iterdir())
          if (p / 'color/0').is_dir()]
canvas = Image.new('RGB', (1272, len(scenes) * 270), 'white')
draw = ImageDraw.Draw(canvas)
for row, scene in enumerate(scenes):
    frames = sorted((scene / 'color/0').glob('*.png'), key=lambda p: int(p.stem))
    anchor = frames[len(frames) // 2].name
    for view in range(3):
        path = scene / 'color' / str(view) / anchor
        if not path.exists():
            continue
        with Image.open(path) as source:
            frame = source.convert('RGB')
        frame.thumbnail((424, 240))
        canvas.paste(frame, (view * 424, row * 270 + 30))
        draw.text((view * 424 + 4, row * 270 + 5),
                  f'{scene.name} v{view} {anchor}', fill='black')
for start in range(0, len(scenes), 4):
    canvas.crop((0, start * 270, 1272, min(start + 4, len(scenes)) * 270)).save(
        f'/private/tmp/instance_overlap_{start}.jpg')
