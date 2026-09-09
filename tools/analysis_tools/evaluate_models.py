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
from mmcv import Config

logger = logging.getLogger('evaluator')

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
        nargs='+',
        choices=['map', 'benchmark'],
        default=['map', 'benchmark'],
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


class EnvironmentValidator:
    """A unified validator for metafiles, model configs, directories, and
    checkpoints."""

    def __init__(self, args):
        self.args = args
        self.metafiles = []
        self.configs = set()

    def validate_all(self):
        """Runs the entire validation pipeline."""
        self._check_metafiles()
        self._get_configs_to_check()
        self._check_directories()
        self._check_checkpoints()
        return self.metafiles

    def _check_metafiles(self):
        """Validates the existence of explicitly provided metafiles or scans
        configs/."""
        if self.args.metafiles:
            self.metafiles = self.args.metafiles
            missing_mfs = [
                mf for mf in self.metafiles if not os.path.exists(mf)
            ]
            if missing_mfs:
                logger.error(
                    '\nError: The following specified metafiles are missing:')
                for m in missing_mfs:
                    logger.error(f'  - {m}')
                sys.exit(1)
            logger.info(
                f'Using {len(self.metafiles)} manually specified metafile(s).')
        else:
            self.metafiles = glob.glob(
                'configs/**/metafile.yml', recursive=True)
            logger.info(
                f'Found {len(self.metafiles)} metafiles in configs directory.')

        if not self.metafiles:
            logger.error(
                '\nError: No metafiles found or specified to process.')
            sys.exit(1)

    def _get_configs_to_check(self):
        """Parses metafiles and extracts valid model config paths to
        evaluate."""
        for mf_path in self.metafiles:
            with open(mf_path, 'r', encoding='utf-8') as f:
                try:
                    data = yaml.safe_load(f)
                except yaml.YAMLError:
                    continue

            if not data or 'Models' not in data:
                continue

            for model in data['Models']:
                name = model.get('Name', '')
                if self.args.models and name not in self.args.models:
                    continue

                config_path = model.get('Config', '')
                meta = model.get('Metadata', {})
                training_data = meta.get('Training Data', '').lower()

                if ('dota' not in training_data
                        and 'dota' not in config_path.lower()):
                    continue
                if get_model_dota_version(name) != self.args.dota_version:
                    continue
                if config_path:
                    self.configs.add(config_path)

    def _extract_dataset_paths(self, cfg):
        """Extracts all required dataset paths from a parsed MMCV config."""
        paths = set()
        for split in ['train', 'val', 'test']:
            split_info = cfg.data.get(split)
            if not split_info:
                continue

            if isinstance(split_info, list):
                datasets = split_info
            elif split_info.get('type') == 'ConcatDataset':
                datasets = split_info.get('datasets', [])
            elif split_info.get('type') == 'MultiImageMixDataset':
                datasets = [split_info.get('dataset', {})]
            else:
                datasets = [split_info]

            for ds in datasets:
                if not isinstance(ds, dict):
                    continue

                for key in ['img_prefix', 'ann_file']:
                    val = ds.get(key, '')
                    if not val:
                        continue

                    paths_to_check = val if isinstance(val, list) else [val]
                    for p in paths_to_check:
                        if split == 'test':
                            val_p = p.replace('test/',
                                              f'{self.args.data_split}/')
                            if (key == 'ann_file' and 'images' in val_p
                                    and self.args.data_split != 'test'):
                                val_p = val_p.replace('images', 'annfiles')
                            paths.add(val_p)
                        else:
                            paths.add(p)
        return paths

    def _check_directories(self):
        """Verifies local availability of required dataset directories for all
        targets."""
        missing_dirs = set()

        for config_path in self.configs:
            try:
                cfg = Config.fromfile(config_path)
                paths = self._extract_dataset_paths(cfg)
                missing_dirs.update(p for p in paths if not os.path.exists(p))
            except Exception as e:
                logger.error(f'Failed to load config {config_path}'
                             f' during directory check: {e}')
                continue

        if missing_dirs:
            logger.error('\nError: The following required data'
                         ' directories or files are missing:')
            for m in sorted(missing_dirs):
                logger.error(f'  - {m}')
            logger.error('\nPlease verify your dataset paths'
                         ' and ensure data is split correctly.')
            sys.exit(1)

        if self.configs:
            logger.info('All required data directories exist.')

    def _check_checkpoints(self):
        """Scans all metafile configurations and verifies local availability of
        required model weights."""
        missing_checkpoints = []
        found_models_count = 0

        for mf_path in self.metafiles:
            with open(mf_path, 'r', encoding='utf-8') as f:
                try:
                    data = yaml.safe_load(f)
                except yaml.YAMLError:
                    continue

            if not data or 'Models' not in data:
                continue

            for model in data['Models']:
                name = model.get('Name', '')

                if self.args.models and name not in self.args.models:
                    continue

                config = model.get('Config', '')
                weights_url = model.get('Weights', '')
                meta = model.get('Metadata', {})
                training_data = meta.get('Training Data', '').lower()

                if 'dota' not in training_data and 'dota' not in config.lower(
                ):
                    continue

                if get_model_dota_version(name) != self.args.dota_version:
                    continue

                found_models_count += 1
                checkpoint_path = os.path.join('checkpoints',
                                               os.path.basename(weights_url))
                if not os.path.exists(checkpoint_path):
                    missing_checkpoints.append((name, checkpoint_path))

        if found_models_count == 0:
            logger.error('Error: No matching models found for evaluation'
                         ' based on the provided arguments.')
            sys.exit(1)

        if missing_checkpoints:
            logger.error('\nError: The following required model'
                         ' checkpoints are missing:')
            for name, path in missing_checkpoints:
                logger.error(f'  - Model: {name}')
                logger.error(f'    Path:  {path}')
            logger.error('\nPlease download missing checkpoints into'
                         " the 'checkpoints/' folder before running.")
            sys.exit(1)

        logger.info('All required model checkpoints exist. Proceeding...')


class ModelEvaluator:
    """Handles the evaluation workflow (mAP, FPS) for models."""

    def __init__(self, args, metafiles):
        self.args = args
        self.metafiles = metafiles
        self.results_db = {}
        self.errors_db = {}

        if self.args.dota_version == '1.0':
            from mmrotate.datasets.dota import DOTADataset
            self.classes = DOTADataset.CLASSES
        elif self.args.dota_version == '1.5':
            from mmrotate.datasets.dotav15 import DOTAv15Dataset
            self.classes = DOTAv15Dataset.CLASSES
        elif self.args.dota_version == '2.0':
            from mmrotate.datasets.dotav2 import DOTAv2Dataset
            self.classes = DOTAv2Dataset.CLASSES
        else:
            self.classes = []

    @staticmethod
    def _run_command(cmd, env=None):
        """Executes a shell command, streams line-by-line output to the logger,
        and returns status.

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

    def _get_submission_dir(self, name):
        """Generates the directory path for saving formatted prediction
        results."""
        dota_v_str = self.args.dota_version.replace('.', '_')
        res_name = f'{name}_dota{dota_v_str}_{self.args.data_split}'
        return os.path.join(self.args.work_dir, 'formatted_results',
                            res_name).replace('\\', '/')

    def _get_dataset_paths(self):
        """Returns paths to the original dataset imageset text file and
        annotation directory for a given DOTA version and split."""
        v_str = self.args.dota_version.replace('.', '_')
        imagesetfile = f'data/DOTA_{v_str}/{self.args.data_split}_set.txt'
        annopath = (f'data/DOTA_{v_str}/'
                    f'{self.args.data_split}/labelTxt/{{:s}}.txt')
        return imagesetfile, annopath

    @staticmethod
    def _extract_collection_name(data, mf_path):
        """Extracts collection name from metafile YAML or falls back to parent
        directory name."""
        if 'Collections' in data and isinstance(
                data['Collections'], list) and len(data['Collections']) > 0:
            coll_name = data['Collections'][0].get('Name')
            if coll_name:
                return coll_name
        return os.path.basename(os.path.dirname(mf_path))

    def _prepare_model_paths(self, model_entry):
        """Extracts model metadata and constructs corresponding dataset and
        checkpoint paths."""
        name = model_entry.get('Name', '')
        config_path = model_entry.get('Config', '')
        weights_url = model_entry.get('Weights', '')
        meta = model_entry.get('Metadata', {})
        training_data = meta.get('Training Data', '').lower()

        if 'dota' not in training_data and 'dota' not in config_path.lower():
            return None

        if get_model_dota_version(name) != self.args.dota_version:
            return None

        try:
            cfg = Config.fromfile(config_path)
            test_dict = cfg.data.get('test', {})
            img_prefix = test_dict.get('img_prefix', '')
            ann_file = test_dict.get('ann_file', '')

            # Replace 'test' with the requested data_split (e.g. 'val')
            if self.args.data_split != 'test':
                img_prefix = img_prefix.replace('test/',
                                                f'{self.args.data_split}/')
                ann_file = ann_file.replace('test/',
                                            f'{self.args.data_split}/')

                # In DOTA configs, 'test' ann_file often points to 'images/'.
                # When switching to 'val' or another split,
                # we need actual annotations.
                if 'images' in ann_file:
                    ann_file = ann_file.replace('images', 'annfiles')

        except Exception as e:
            logger.error(f'Failed to load config {config_path}: {e}')
            return None

        scale = 'ms' if '_ms_' in name else 'ss'
        rotation = 'rr' if '_rr_' in name else 'none'
        checkpoint_path = os.path.join('checkpoints',
                                       os.path.basename(weights_url))
        angle = name.rsplit('_', 1)[-1]

        return {
            'name': name,
            'config': config_path,
            'weights_url': weights_url,
            'checkpoint_path': checkpoint_path,
            'scale': scale,
            'rotation': rotation,
            'angle': angle,
            'img_prefix': img_prefix,
            'ann_file': ann_file
        }

    def _evaluate_map(self, paths):
        """Executes mAP evaluation for a given model and parses results."""
        logger.info('\n[+] Running mAP evaluation...')

        submission_dir = self._get_submission_dir(paths['name'])
        os.removedirs(submission_dir) if os.path.exists(
            submission_dir) else None

        try:
            # Run tools/test.py with --format-only
            cmd_format = (
                f"python -W ignore ./tools/test.py {paths['config']} "
                f"{paths['checkpoint_path']} --format-only "
                f'--eval-options submission_dir={submission_dir} '
                f'--cfg-options data.test_dataloader.workers_per_gpu='
                f'{self.args.map_workers_per_gpu} '
                f'data.test_dataloader.samples_per_gpu='
                f'{self.args.map_samples_per_gpu} '
                f"data.test.ann_file={paths['ann_file']} "
                f"data.test.img_prefix={paths['img_prefix']}")
            success_format, out_format, err_format = self._run_command(
                cmd_format)

            if not success_format:
                logger.info(f'\n[-] Test formatting failed: {err_format}')
                return None, None, err_format

            imagesetfile, annopath = self._get_dataset_paths()
            if not os.path.exists(imagesetfile):
                err_msg = (
                    f'Imageset file not found at {imagesetfile}. '
                    'Please ensure the original dataset imageset file exists.')
                logger.info(f'\n[-] {err_msg}')
                return None, None, err_msg

            # Run DOTA devkit evaluation script
            if self.args.dota_version not in EVALUATION_SCRIPT:
                err_msg = (f'Evaluation script for DOTA version '
                           f'{self.args.dota_version} is not configured.')
                logger.info(f'\n[-] {err_msg}')
                return None, None, err_msg

            eval_script = EVALUATION_SCRIPT[self.args.dota_version]
            detpath = os.path.join(submission_dir,
                                   'Task1_{:s}.txt').replace('\\', '/')
            cmd_eval = (f'python {eval_script} '
                        f'--detpath "{detpath}" '
                        f'--annopath "{annopath}" '
                        f'--imagesetfile "{imagesetfile}"')

            success_eval, out_eval, err_eval = self._run_command(cmd_eval)
            if not success_eval:
                logger.info(f'\n[-] DOTA devkit evaluation failed: {err_eval}')
                return None, None, err_eval

            # Extract mAP metric from output
            map_match = re.search(r'^map:\s*([0-9.]+)', out_eval, re.MULTILINE)
            if map_match:
                raw_map = float(map_match.group(1))
                if raw_map <= 1.0:
                    raw_map *= 100
                val = round(raw_map, 2)
                logger.info(f'\n[OK] Extracted mAP: {val}')

                classaps_dict = {}
                classaps_match = re.search(r'^classaps:\s*\[(.*?)\]', out_eval,
                                           re.MULTILINE | re.DOTALL)
                if classaps_match:
                    aps = [float(x) for x in classaps_match.group(1).split()]

                    for cls_name, ap in zip(self.classes, aps):
                        classaps_dict[cls_name] = round(ap, 2)

                return val, classaps_dict, None
            else:
                logger.info('\n[-] Failed to extract mAP metric from output.')
                return (None, None,
                        'Regex match failed. Output might be malformed.')
        finally:
            if os.path.exists(submission_dir):
                shutil.rmtree(submission_dir, ignore_errors=True)

    def _evaluate_benchmark(self, paths):
        """Executes benchmark evaluation (FPS calculation) and parses
        output."""
        logger.info('\n[+] Running Benchmark...')
        env = os.environ.copy()
        env['PYTHONWARNINGS'] = 'ignore'
        cmd_bench = (
            f'python -m torch.distributed.launch --nproc_per_node=1 '
            f'--master_port=29500 tools/analysis_tools/benchmark.py '
            f"{paths['config']} {paths['checkpoint_path']} --launcher pytorch "
            f'--log-interval 5 --cfg-options '
            f'data.test_dataloader.workers_per_gpu='
            f'{self.args.benchmark_workers_per_gpu} '
            f'data.test_dataloader.samples_per_gpu='
            f'{self.args.benchmark_samples_per_gpu} '
            f"data.test.ann_file={paths['ann_file']} "
            f"data.test.img_prefix={paths['img_prefix']}")
        success, out_bench, err_msg = self._run_command(cmd_bench, env=env)

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

    def process(self, model_entry, collection_name):
        """Coordinates overall evaluation workflow for an individual model."""
        name = model_entry.get('Name', '')
        if self.args.models and name not in self.args.models:
            return None

        paths = self._prepare_model_paths(model_entry)
        if not paths:
            return None

        logger.info(f"\n{'='*80}")
        logger.info(
            f'Processing model: {name} (Collection: {collection_name})')
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
        if 'map' in self.args.tasks:
            mAP, classaps_dict, err = self._evaluate_map(paths)
            model_info['mAP'] = mAP

            for cls_name in self.classes:
                model_info[cls_name] = classaps_dict.get(
                    cls_name) if classaps_dict else None

            if err:
                self.errors_db.setdefault(name, {})['mAP'] = err

        # Evaluate FPS
        if 'benchmark' in self.args.tasks:
            fps, err = self._evaluate_benchmark(paths)
            model_info['FPS'] = fps
            if err:
                self.errors_db.setdefault(name, {})['Benchmark'] = err

        return model_info

    def evaluate_all(self):
        """Iterates over all metafiles and processes their models."""
        for mf_path in self.metafiles:
            with open(mf_path, 'r', encoding='utf-8') as f:
                try:
                    data = yaml.safe_load(f)
                except yaml.YAMLError:
                    continue

            if not data or 'Models' not in data:
                continue

            collection_name = self._extract_collection_name(data, mf_path)
            group_results = []

            for model_entry in data['Models']:
                model_info = self.process(model_entry, collection_name)
                if model_info:
                    group_results.append(model_info)

            if group_results:
                self.results_db[collection_name] = group_results

        return self.results_db, self.errors_db


def generate_out_prefix(args):
    """Generates a structured output file path prefix based on CLI arguments
    and current timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    v_str = args.dota_version.replace('.', '_')
    tasks_str = '+'.join(args.tasks)
    base_name = (f'models_stats_dota{v_str}_{args.data_split}_'
                 f'{tasks_str}_{timestamp}')
    return os.path.join(args.work_dir, base_name)


def save_report(results_db, errors_db, args, out_prefix):
    """Saves structured evaluation metrics and hardware configuration to
    JSON."""
    out_filepath = f'{out_prefix}.json'

    gpu_name = torch.cuda.get_device_name(
        0) if torch.cuda.is_available() else 'No GPU detected'

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

    validator = EnvironmentValidator(args)
    metafiles = validator.validate_all()

    evaluator = ModelEvaluator(args, metafiles)
    results_db, errors_db = evaluator.evaluate_all()

    save_report(results_db, errors_db, args, out_prefix)


if __name__ == '__main__':
    main()
