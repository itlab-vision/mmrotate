# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert evaluation JSON report to Markdown table.')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the evaluation results JSON file')
    parser.add_argument(
        '--work-dir',
        type=str,
        default='work_dirs',
        help=
        'Directory where the output Markdown file will be saved (default: work_dirs)'
    )
    parser.add_argument(
        '--sort-map',
        choices=['asc', 'desc', 'none'],
        default='desc',
        help='Sort results by mAP (asc: ascending, desc: descending)')
    return parser.parse_args()


def generate_markdown(data, sort_map=None):
    results = data.get('results', {})

    rows = []
    for family, models in results.items():
        for model in models:
            name = model.get('name', '')

            raw_scale = model.get('scale', '').lower()
            scale = '-' if raw_scale == 'ss' else raw_scale.upper()

            raw_rotation = model.get('rotation', '').lower()
            rotation = '-' if raw_rotation == 'none' else raw_rotation.upper()

            angle = model.get('angle', '')
            map_val = model.get('mAP')
            fps_val = model.get('FPS')
            config_path = model.get('config', '')
            weights_url = model.get('weights_url', '')

            config_link = f'[config](../../{config_path})' if config_path else 'N/A'
            download_link = f'[model]({weights_url})' if weights_url else 'N/A'

            rows.append({
                'family':
                family,
                'name':
                name,
                'scale':
                scale,
                'rotation':
                rotation,
                'angle':
                angle,
                'map':
                map_val if map_val is not None else -1.0,
                'map_str':
                f'{map_val:.2f}' if map_val is not None else 'N/A',
                'fps_str':
                f'{fps_val:.1f}' if fps_val is not None else 'N/A',
                'config_link':
                config_link,
                'download_link':
                download_link
            })

    if sort_map == 'asc':
        rows.sort(key=lambda x: x['map'])
    elif sort_map == 'desc':
        rows.sort(key=lambda x: x['map'], reverse=True)

    lines = []
    lines.append(
        '| Family | Model Name | mAP (%) | FPS | Scale | Rotation | Angle | Config | Download |'
    )
    lines.append('|---|---|---|---|---|---|---|---|---|')

    for r in rows:
        lines.append(
            f"| {r['family']} | `{r['name']}` | {r['map_str']} | {r['fps_str']} | "
            f"{r['scale']} | {r['rotation']} | {r['angle']} | {r['config_link']} | {r['download_link']} |"
        )

    return '\n'.join(lines)


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        sys.exit(1)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    output_filename = input_path.stem + '.md'
    output_path = work_dir / output_filename

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    md_content = generate_markdown(data, sort_map=args.sort_map)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)


if __name__ == '__main__':
    main()
