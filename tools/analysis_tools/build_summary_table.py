# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import json
import sys
from pathlib import Path

import yaml


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
        help='Output directory for the Markdown file (default: work_dirs)')
    parser.add_argument(
        '--sort-map',
        choices=['asc', 'desc', 'none'],
        default='desc',
        help='Sort results by mAP (asc: ascending, desc: descending)')
    parser.add_argument(
        '--include-ref-map',
        action='store_true',
        help='Extract and include Ref mAP from metafile.yml files')
    return parser.parse_args()


class ReportConverter:
    """Handles the conversion of evaluation JSON reports to Markdown."""

    def __init__(self, args):
        self.input_path = Path(args.input)
        self.work_dir = Path(args.work_dir)
        self.sort_map = args.sort_map
        self.include_ref_map = args.include_ref_map

    def _get_ref_map_dict(self):
        """Scans configs/ for metafile.yml and extracts reference mAP."""
        ref_map_dict = {}
        if not self.include_ref_map:
            return ref_map_dict

        metafiles = glob.glob('configs/**/metafile.yml', recursive=True)

        for mf_path in metafiles:
            with open(mf_path, 'r', encoding='utf-8') as f:
                try:
                    data = yaml.safe_load(f)
                except yaml.YAMLError:
                    continue

            if not data or 'Models' not in data:
                continue

            for model in data['Models']:
                name = model.get('Name', '')
                if not name:
                    continue

                ref_map = '-'
                results = model.get('Results')

                # Prioritize Paper mAP, fallback to mAP
                if isinstance(results, list) and len(results) > 0:
                    metrics = results[0].get('Metrics', {})
                    if 'Paper mAP' in metrics:
                        ref_map = f"{metrics['Paper mAP']}\\*"
                    elif 'mAP' in metrics:
                        ref_map = str(metrics['mAP'])

                ref_map_dict[name] = ref_map

        return ref_map_dict

    def _generate_markdown(self, data, ref_map_dict):
        """Formats the parsed JSON data into a Markdown table."""
        results = data.get('results', {})
        rows = []

        for family, models in results.items():
            for model in models:
                name = model.get('name', '')

                raw_scale = model.get('scale', '').lower()
                scale = '-' if raw_scale == 'ss' else raw_scale.upper()

                raw_rot = model.get('rotation', '').lower()
                rotation = '-' if raw_rot == 'none' else raw_rot.upper()

                angle = model.get('angle', '')
                map_val = model.get('mAP')
                fps_val = model.get('FPS')
                cfg_path = model.get('config', '')
                w_url = model.get('weights_url', '')

                cfg_link = f'[config](../../{cfg_path})' if cfg_path else '-'
                d_link = (f'[model]({w_url})'
                          if w_url and w_url.startswith('http') else '-')

                ref_map_str = ref_map_dict.get(name, '-')
                map_num = map_val if map_val is not None else -1.0
                map_str = f'{map_val:.2f}' if map_val is not None else 'N/A'
                fps_str = f'{fps_val:.1f}' if fps_val is not None else 'N/A'

                rows.append({
                    'family': family,
                    'name': name,
                    'scale': scale,
                    'rotation': rotation,
                    'angle': angle,
                    'map': map_num,
                    'map_str': map_str,
                    'ref_map_str': ref_map_str,
                    'fps_str': fps_str,
                    'config_link': cfg_link,
                    'download_link': d_link
                })

        if self.sort_map == 'asc':
            rows.sort(key=lambda x: x['map'])
        elif self.sort_map == 'desc':
            rows.sort(key=lambda x: x['map'], reverse=True)

        lines = []

        # Build headers based on the include_ref_map flag
        if self.include_ref_map:
            lines.append(
                '| Family | Model Name | mAP (%) | Ref mAP (%) | FPS | '
                'Scale | Rotation | Angle | Config | Download |')
            lines.append('|---|---|---|---|---|---|---|---|---|---|')
        else:
            lines.append('| Family | Model Name | mAP (%) | FPS | '
                         'Scale | Rotation | Angle | Config | Download |')
            lines.append('|---|---|---|---|---|---|---|---|---|')

        # Build table rows
        for r in rows:
            if self.include_ref_map:
                lines.append(
                    f"| {r['family']} | `{r['name']}` | {r['map_str']} | "
                    f"{r['ref_map_str']} | {r['fps_str']} | {r['scale']} | "
                    f"{r['rotation']} | {r['angle']} | {r['config_link']} | "
                    f"{r['download_link']} |")
            else:
                lines.append(
                    f"| {r['family']} | `{r['name']}` | {r['map_str']} | "
                    f"{r['fps_str']} | {r['scale']} | {r['rotation']} | "
                    f"{r['angle']} | {r['config_link']} | "
                    f"{r['download_link']} |")

        return '\n'.join(lines)

    def process(self):
        """Executes the full parsing, generating, and saving pipeline."""
        if not self.input_path.exists():
            print(f'Error: Input file {self.input_path} not found.')
            sys.exit(1)

        self.work_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.work_dir / f'{self.input_path.stem}.md'

        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ref_dict = self._get_ref_map_dict()
        md_content = self._generate_markdown(data, ref_dict)

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f'Success: Markdown table saved to {out_path}')


def main():
    args = parse_args()
    converter = ReportConverter(args)
    converter.process()


if __name__ == '__main__':
    main()
