# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

import torch
import yaml

logger = logging.getLogger('evaluator')

DATASET_TYPE = {
    '1.0': 'DOTADataset',
    '1.5': 'DOTAv15Dataset',
    '2.0': 'DOTAv2Dataset',
}

EVALUATION_SCRIPT = {
    '1.0': '3rdparty/DOTA_devkit/dota_evaluation_task1.py',
    '1.5': '3rdparty/DOTA_devkit/dota-v1.5_evaluation_task1.py',
    # '2.0': '3rdparty/DOTA_devkit/dota-v2.0_evaluation_task1.py',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMRotate model evaluation and metric benchmarking script.'
    )
    parser.add_argument(
        '--dota-version',
        choices=['1.0', '1.5', '2.0'],
        default='1.0',
        help='DOTA dataset version')
    parser.add_argument(
        '--data-split',
        choices=['val', 'test'],
        default='val',
        help='Dataset split to evaluate on')
    parser.add_argument(
        '--map-samples-per-gpu',
        type=int,
        default=2,
        help='Batch size per GPU for mAP evaluation')
    parser.add_argument(
        '--benchmark-samples-per-gpu',
        type=int,
        default=1,
        help='Batch size per GPU for benchmark (FPS calculation)')
    parser.add_argument(
        '--benchmark-workers-per-gpu',
        type=int,
        default=0,
        help='Dataloader workers for benchmark evaluation')
    parser.add_argument(
        '--map-workers-per-gpu',
        type=int,
        default=2,
        help='Dataloader workers for mAP evaluation')
    parser.add_argument(
        '--tasks',
        choices=['map', 'benchmark', 'map+benchmark'],
        default='map+benchmark',
        help='Evaluation tasks to run for each model')
    parser.add_argument(
        '--work-dir',
        default='work_dirs',
        help='Directory where reports and execution logs will be saved')
    parser.add_argument(
        '--models',
        nargs='+',
        default=[],
        help='Specific model names to evaluate. Runs all models if empty.')
    parser.add_argument(
        '--metafiles',
        nargs='+',
        default=[],
        help='Specific metafile paths to process. Scans configs/ if empty.')

    return parser.parse_args()


def get_submission_dir(work_dir, name, dota_version, data_split):
    """Generates the directory path for saving formatted prediction results."""
    dota_v_str = dota_version.replace('.', '_')
    res_name = f"{name}_dota{dota_v_str}_{data_split}"
    return os.path.join(
        work_dir, 'formatted_results', res_name).replace('\\', '/')


def get_dataset_paths(version, split):
    """Returns paths to the original dataset imageset text file and annotation
    directory for a given DOTA version and split."""
    v_str = version.replace('.', '_')
    imagesetfile = f'data/DOTA_{v_str}/{split}_set.txt'
    annopath = f'data/DOTA_{v_str}/{split}/labelTxt/{{:s}}.txt'
    return imagesetfile, annopath


def get_model_dota_version(name):
    """Determines the DOTA dataset version from the model name based on name
    keywords."""
    if not name:
        return None

    name_lower = name.lower()

    if '_dota15' in name_lower:
        return '1.5'
    elif '_dota2' in name_lower:
        return '2.0'
    elif '_dota' in name_lower:
        return '1.0'

    return None


def generate_out_prefix(args):
    """Generates a structured output file path prefix based on CLI arguments
    and current timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    v_str = args.dota_version.replace('.', '_')
    base_name = (f'models_stats_dota{v_str}_{args.data_split}_'
                 f'{args.tasks}_{timestamp}')
    return os.path.join(args.work_dir, base_name)


def init_logger(log_filepath):
    """Initializes global logger handlers for synchronized console and file
    output."""
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def check_directories(version, split):
    """Validates the existence of required DOTA dataset directories for both
    Single-Scale (ss) and Multi-Scale (ms) splits.

    Terminates execution if any required data directory is missing.
    """
    v_str = version.replace('.', '_')
    dirs = {
        'ss': {
            'img': f'data/split_ss_dota_{v_str}/{split}/images',
            'ann': f'data/split_ss_dota_{v_str}/{split}/annfiles'
        },
        'ms': {
            'img': f'data/split_ms_dota_{v_str}/{split}/images',
            'ann': f'data/split_ms_dota_{v_str}/{split}/annfiles'
        }
    }

    missing = []
    for scale in dirs:
        for p_type in dirs[scale]:
            if not os.path.exists(dirs[scale][p_type]):
                missing.append(dirs[scale][p_type])

    if missing:
        logger.error(
            '\nError: The following required data directories are missing:')
        for m in missing:
            logger.error(f'  - {m}')
        logger.error(
            '\nPlease verify your dataset paths and ensure DOTA data is split '
            'correctly.')
        sys.exit(1)

    logger.info('All required data directories exist.')
    return dirs


def check_metafiles(metafiles_args):
    """Validates the existence of explicitly provided metafiles or scans the
    configs directory if none are provided.

    Terminates execution if explicitly provided files are missing or if no
    files are found overall.
    """
    if metafiles_args:
        metafiles = metafiles_args
        missing_mfs = [mf for mf in metafiles if not os.path.exists(mf)]
        if missing_mfs:
            logger.error(
                '\nError: The following specified metafiles are missing:')
            for m in missing_mfs:
                logger.error(f'  - {m}')
            sys.exit(1)
        logger.info(f'Using {len(metafiles)} manually specified metafile(s).')
    else:
        metafiles = glob.glob('configs/**/metafile.yml', recursive=True)
        logger.info(f'Found {len(metafiles)} metafiles in configs directory.')

    if not metafiles:
        logger.error('\nError: No metafiles found or specified to process.')
        sys.exit(1)

    return metafiles


def check_checkpoints(metafiles, target_models, dota_version):
    """Scans all metafile configurations and verifies local availability of
    required model weights.

    Terminates execution if any required checkpoint files are missing.
    """
    missing_checkpoints = []
    found_models_count = 0

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

            if target_models and name not in target_models:
                continue

            config = model.get('Config', '')
            weights_url = model.get('Weights', '')
            meta = model.get('Metadata', {})
            training_data = meta.get('Training Data', '').lower()

            if 'dota' not in training_data and 'dota' not in config.lower():
                continue

            if get_model_dota_version(name) != dota_version:
                continue

            found_models_count += 1
            checkpoint_path = os.path.join('checkpoints',
                                           os.path.basename(weights_url))
            if not os.path.exists(checkpoint_path):
                missing_checkpoints.append((name, checkpoint_path))

    if found_models_count == 0:
        logger.error(
            'Error: No matching models found for evaluation based on the '
            'provided arguments.')
        sys.exit(1)

    if missing_checkpoints:
        logger.error(
            '\nError: The following required model checkpoints are missing:')
        for name, path in missing_checkpoints:
            logger.error(f'  - Model: {name}')
            logger.error(f'    Path:  {path}')
        logger.error(
            "\nPlease download missing checkpoints into the 'checkpoints/' "
            'folder before running.')
        sys.exit(1)

    logger.info('All required model checkpoints exist. Proceeding...')


def run_command(cmd, env=None):
    """Executes a shell command, streams line-by-line output to the logger, and
    returns status.

    Returns: (success_bool, output_string, error_message)
    """
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1)

        output = []
        for line in process.stdout:
            clean_line = line.rstrip('\r\n')
            if clean_line:
                logger.info(clean_line)
            output.append(line)

        returncode = process.wait()
        out_str = ''.join(output)

        if returncode != 0:
            return False, out_str, f'Process exited with code {returncode}'

        return True, out_str, ''

    except Exception as e:
        logger.error(f'\nExecution error: {e}')
        return False, '', str(e)


def extract_collection_name(data, mf_path):
    """Extracts collection name from metafile YAML or falls back to parent
    directory name."""
    if 'Collections' in data and isinstance(
            data['Collections'], list) and len(data['Collections']) > 0:
        coll_name = data['Collections'][0].get('Name')
        if coll_name:
            return coll_name
    return os.path.basename(os.path.dirname(mf_path))


def prepare_model_paths(model_entry, data_dirs, dota_version):
    """Extracts model metadata and constructs corresponding dataset and
    checkpoint paths."""
    name = model_entry.get('Name', '')
    config = model_entry.get('Config', '')
    weights_url = model_entry.get('Weights', '')
    meta = model_entry.get('Metadata', {})
    training_data = meta.get('Training Data', '').lower()

    if 'dota' not in training_data and 'dota' not in config.lower():
        return None

    if get_model_dota_version(name) != dota_version:
        return None

    scale = 'ms' if '_ms_' in name else 'ss'
    rotation = 'rr' if '_rr_' in name else 'none'
    img_prefix = data_dirs[scale]['img']
    ann_file = data_dirs[scale]['ann']
    checkpoint_path = os.path.join('checkpoints',
                                   os.path.basename(weights_url))
    angle = name.rsplit('_', 1)[-1]

    return {
        'name': name,
        'config': config,
        'weights_url': weights_url,
        'checkpoint_path': checkpoint_path,
        'scale': scale,
        'rotation': rotation,
        'angle': angle,
        'img_prefix': img_prefix,
        'ann_file': ann_file
    }


def evaluate_mAP(paths, args):
    """Executes mAP evaluation for a given model and parses results."""
    logger.info('\n[+] Running mAP evaluation...')

    submission_dir = get_submission_dir(
        args.work_dir, paths['name'], args.dota_version, args.data_split)
    os.removedirs(submission_dir) if os.path.exists(submission_dir) else None

    try:
        # Run tools/test.py with --format-only
        cmd_format = (
            f"python -W ignore ./tools/test.py {paths['config']} "
            f"{paths['checkpoint_path']} --format-only "
            f'--eval-options submission_dir={submission_dir} '
            f'--cfg-options data.test_dataloader.workers_per_gpu='
            f'{args.map_workers_per_gpu} '
            f'data.test_dataloader.samples_per_gpu={args.map_samples_per_gpu} '
            f"data.test.ann_file={paths['ann_file']} "
            f"data.test.img_prefix={paths['img_prefix']} "
            f'data.test.type={DATASET_TYPE[args.dota_version]}')
        success_format, out_format, err_format = run_command(cmd_format)

        if not success_format:
            logger.info(f'\n[-] Test formatting failed: {err_format}')
            return None, err_format

        imagesetfile, annopath = get_dataset_paths(args.dota_version, args.data_split)
        if not os.path.exists(imagesetfile):
            err_msg = (
                f'Imageset file not found at {imagesetfile}. '
                'Please ensure the original dataset imageset file exists.')
            logger.info(f'\n[-] {err_msg}')
            return None, err_msg

        # Run DOTA devkit evaluation script
        if args.dota_version not in EVALUATION_SCRIPT:
            err_msg = (f'Evaluation script for DOTA version '
                       f'{args.dota_version} is not configured.')
            logger.info(f'\n[-] {err_msg}')
            return None, err_msg

        eval_script = EVALUATION_SCRIPT[args.dota_version]

        detpath = os.path.join(submission_dir,
                               'Task1_{:s}.txt').replace('\\', '/')

        cmd_eval = (f'python {eval_script} '
                    f'--detpath "{detpath}" '
                    f'--annopath "{annopath}" '
                    f'--imagesetfile "{imagesetfile}"')
        success_eval, out_eval, err_eval = run_command(cmd_eval)

        if not success_eval:
            logger.info(f'\n[-] DOTA devkit evaluation failed: {err_eval}')
            return None, err_eval

        # Extract mAP metric from output
        map_match = re.search(r'^map:\s*([0-9.]+)', out_eval, re.MULTILINE)
        if map_match:
            raw_map = float(map_match.group(1))
            if raw_map <= 1.0:
                raw_map *= 100
            val = round(raw_map, 2)
            logger.info(f'\n[OK] Extracted mAP: {val}')
            return val, None
        else:
            logger.info('\n[-] Failed to extract mAP metric from output.')
            return None, 'Regex match failed. Output might be malformed.'
    finally:
        if os.path.exists(submission_dir):
            shutil.rmtree(submission_dir, ignore_errors=True)


def evaluate_benchmark(paths, args):
    """Executes benchmark evaluation (FPS calculation) and parses output."""
    logger.info('\n[+] Running Benchmark...')
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore'
    cmd_bench = (
        f'python -m torch.distributed.launch --nproc_per_node=1 '
        f'--master_port=29500 tools/analysis_tools/benchmark.py '
        f"{paths['config']} {paths['checkpoint_path']} --launcher pytorch "
        f'--log-interval 5 --cfg-options '
        f'data.test_dataloader.workers_per_gpu='
        f'{args.benchmark_workers_per_gpu} '
        f'data.test_dataloader.samples_per_gpu='
        f'{args.benchmark_samples_per_gpu} '
        f"data.test.ann_file={paths['ann_file']} "
        f"data.test.img_prefix={paths['img_prefix']} "
        f'data.test.type={DATASET_TYPE[args.dota_version]}')
    success, out_bench, err_msg = run_command(cmd_bench, env=env)

    if success:
        fps_match = re.search(r'Overall fps:\s*([0-9.]+)', out_bench)
        if fps_match:
            val = float(fps_match.group(1))
            logger.info(f'\n[OK] Extracted FPS: {val}')
            return val, None
        else:
            logger.info('\n[-] Failed to extract FPS metric from output.')
            return None, 'Regex match failed. Output might be malformed.'
    else:
        logger.info(f'\n[-] Benchmark failed: {err_msg}')
        return None, err_msg


def process_model(model_entry, data_dirs, collection_name, args, errors_db):
    """Coordinates overall evaluation workflow for an individual model."""
    name = model_entry.get('Name', '')
    if args.models and name not in args.models:
        return None

    paths = prepare_model_paths(
        model_entry, data_dirs, dota_version=args.dota_version)
    if not paths:
        return None

    logger.info(f"\n{'='*80}")
    logger.info(f'Processing model: {name} (Collection: {collection_name})')
    logger.info(f"{'='*80}")

    model_info = {
        'name': name,
        'config': paths['config'],
        'weights_url': paths['weights_url'],
        'scale': paths['scale'],
        'rotation': paths['rotation'],
        'angle': paths['angle']
    }

    # Evaluate mAP
    if args.tasks in ['map', 'map+benchmark']:
        mAP, err = evaluate_mAP(paths, args)
        model_info['mAP'] = mAP
        if err:
            errors_db.setdefault(name, {})['mAP'] = err

    # Evaluate FPS
    if args.tasks in ['benchmark', 'map+benchmark']:
        fps, err = evaluate_benchmark(paths, args)
        model_info['FPS'] = fps
        if err:
            errors_db.setdefault(name, {})['Benchmark'] = err

    return model_info


def save_report(results_db, errors_db, gpu_name, args, out_prefix):
    """Saves structured evaluation metrics and hardware configuration to
    JSON."""
    out_filepath = f'{out_prefix}.json'

    with open(out_filepath, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'run_config': vars(args),
                'hardware': {
                    'gpu_name': gpu_name
                },
                'results': results_db,
                'errors': errors_db
            },
            f,
            indent=4,
            ensure_ascii=False)

    logger.info(f"\n{'='*80}")
    logger.info('Evaluation finished.')
    logger.info(f'JSON Report: {out_filepath}')
    logger.info(f'Text Log:    {out_prefix}.log')
    logger.info(f"{'='*80}\n")


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    out_prefix = generate_out_prefix(args)
    init_logger(f'{out_prefix}.log')

    gpu_name = torch.cuda.get_device_name(
        0) if torch.cuda.is_available() else 'No GPU detected'

    data_dirs = check_directories(args.dota_version, args.data_split)
    metafiles = check_metafiles(args.metafiles)
    check_checkpoints(metafiles, args.models, args.dota_version)

    results_db = {}
    errors_db = {}

    for mf_path in metafiles:
        with open(mf_path, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError:
                continue

        if not data or 'Models' not in data:
            continue

        collection_name = extract_collection_name(data, mf_path)
        group_results = []

        for model_entry in data['Models']:
            model_info = process_model(model_entry, data_dirs, collection_name,
                                       args, errors_db)
            if model_info:
                group_results.append(model_info)

        if group_results:
            results_db[collection_name] = group_results

    save_report(results_db, errors_db, gpu_name, args, out_prefix)


if __name__ == '__main__':
    main()
