# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile

logger = logging.getLogger('splitter')


def init_logger():
    """Initializes global logger for clean console output."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(stream_handler)


def parse_args():
    parser = argparse.ArgumentParser(
        description='DOTA dataset splitting wrapper.')
    parser.add_argument(
        '--dota-version',
        nargs='+',
        choices=['1.0', '1.5', '2.0'],
        default=['1.0'],
        help='DOTA dataset version(s) (default: 1.0)')
    parser.add_argument(
        '--data-split',
        nargs='+',
        choices=['train', 'val', 'test', 'trainval'],
        default=['val'],
        help='Dataset split(s) to process (default: val)')
    parser.add_argument(
        '--scale',
        nargs='+',
        choices=['ss', 'ms', 'ss-cfa', 'ms-cfa'],
        default=['ss', 'ms'],
        help='Scale mode(s): single-scale (ss) or multi-scale (ms) '
        '(default: ss ms)')
    parser.add_argument(
        '--nproc',
        type=int,
        default=10,
        help='Number of processes for img_split.py (default: 10)')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing target directories (default: False)')

    return parser.parse_args()


def generate_split_config(version, split, scale, nproc):
    """Generates a dictionary representing the JSON config for img_split.py."""
    v_str = version.replace('.', '_')

    config = {
        'nproc': nproc,
        'sizes': [1024],
        'img_rate_thr': 0.6,
        'iof_thr': 0.7,
        'no_padding': False,
        'padding_value': [104, 116, 124],
        'save_dir': f'data/split_{scale}_dota_{v_str}/{split}/',
        'save_ext': '.png'
    }

    if split == 'trainval':
        config['img_dirs'] = [
            f'data/DOTA_{v_str}/train/images/',
            f'data/DOTA_{v_str}/val/images/'
        ]
        config['ann_dirs'] = [
            f'data/DOTA_{v_str}/train/labelTxt/',
            f'data/DOTA_{v_str}/val/labelTxt/'
        ]
    else:
        config['img_dirs'] = [f'data/DOTA_{v_str}/{split}/images/']

        if split != 'test':
            config['ann_dirs'] = [f'data/DOTA_{v_str}/{split}/labelTxt/']

    if scale == 'ss':
        config['gaps'] = [200]
        config['rates'] = [1.0]
    elif scale == 'ms':
        config['gaps'] = [500]
        config['rates'] = [0.5, 1.0, 1.5]
    elif scale == 'ss-cfa':
        config['gaps'] = [200]
        config['rates'] = [1.0]
    elif scale == 'ms-cfa':
        config['gaps'] = [500]
        config['rates'] = [0.75, 1.0, 1.25]

    return config


def check_prerequisites(config, split):
    for img_dir in config.get('img_dirs', []):
        if not os.path.exists(img_dir):
            return False, f'Missing source images: {img_dir}'

    if split != 'test':
        for ann_dir in config.get('ann_dirs', []):
            if not os.path.exists(ann_dir):
                return False, f'Missing annotations: {ann_dir}'

    return True, ''


def run_split_command(config, version, split, scale, overwrite):
    is_valid, reason = check_prerequisites(config, split)
    if not is_valid:
        logger.info(
            f'[-] Skipping {version} | {split} | {scale.upper()} -> {reason}')
        return

    save_dir = config['save_dir']
    if os.path.exists(save_dir):
        if overwrite:
            shutil.rmtree(save_dir)
        else:
            logger.info(
                f'[-] Skipping {version} | {split} | {scale.upper()} -> '
                f"Directory '{save_dir}' exists (use --overwrite).")
            return

    logger.info(
        f'\n[+] Starting {version} | {split} | {scale.upper()} -> {save_dir}')

    with tempfile.NamedTemporaryFile(
            mode='w+', suffix='.json', encoding='utf-8') as tmp_file:
        json.dump(config, tmp_file, indent=2)
        tmp_file.flush()

        cmd = (f'python tools/data/dota/split/img_split.py '
               f'--base-json {tmp_file.name}')

        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True)

        for line in process.stdout:
            clean_line = line.strip()
            if clean_line:
                logger.info(clean_line)

        process.wait()

        if process.returncode != 0:
            logger.error(
                f'[!] Process failed with exit code {process.returncode}')
        else:
            logger.info('[OK] Split completed successfully.')


def main():
    init_logger()
    args = parse_args()

    logger.info('=' * 60)
    logger.info('DOTA Splitter Workflow Initialization')
    logger.info('=' * 60)

    for version in args.dota_version:
        for split in args.data_split:
            for scale in args.scale:
                config = generate_split_config(version, split, scale,
                                               args.nproc)
                run_split_command(config, version, split, scale,
                                  args.overwrite)

    logger.info('\n' + '=' * 60)
    logger.info('All splitting tasks completed.')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
