"""Create isolated annotation episodes without modifying source proposals."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('destination', type=Path)
args = p.parse_args()
if args.destination.exists():
    raise SystemExit('Destination exists; refusing to replace annotation state')
episode = json.loads(args.source.read_text())
for frame in episode['frames']:
    for key, value in list(frame.items()):
        if key.endswith('_path') and isinstance(value, str) and value:
            frame[key] = str((args.source.parent / value).resolve())
        elif key.endswith('_paths_by_view') and isinstance(value, list):
            frame[key] = [str((args.source.parent / v).resolve()) if v else '' for v in value]
episode.setdefault('metadata', {})['part_segmentation'] = {
    'provider': 'unreviewed-object-proposal-not-part-gt',
    'parts': [
        {'part_id': 1, 'name': 'base', 'role': 'base', 'color': '#62d26f'},
        {'part_id': 2, 'name': 'moving', 'role': 'moving', 'color': '#a56de2'}
    ]
}
episode['metadata']['manual_review'] = {
    'source_episode': str(args.source.resolve()), 'status': 'pending',
    'warning': 'Initial labels are object proposals, not accepted part GT.'
}
args.destination.parent.mkdir(parents=True, exist_ok=True)
args.destination.write_text(json.dumps(episode, indent=2) + '\n')
