import os
import sys
import yaml
import glob
import json
import re
import argparse
import subprocess
import torch
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='mmrotate evaluate models metrics')
    parser.add_argument(
        '--dota-version', 
        choices=['1.0', '1.5', '2.0'], 
        default='1.0',
        help='DOTA dataset version')
    parser.add_argument(
        '--data-split', 
        choices=['val', 'test'], 
        default='val',
        help='Dataset split to use')
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
        help='Dataloader workers for benchmark')
    parser.add_argument(
        '--map-workers-per-gpu', 
        type=int, 
        default=2,
        help='Dataloader workers for mAP evaluation')
    parser.add_argument(
        '--tasks', 
        choices=['map', 'benchmark', 'map+benchmark'], 
        default='map+benchmark',
        help='Tasks to run for each model')
    parser.add_argument(
        '--work-dir', 
        default='work_dirs',
        help='Directory to save the final JSON report')
    parser.add_argument(
        '--models', 
        nargs='+', 
        default=[],
        help='Specific model names to evaluate. If empty, runs all models.')
    
    return parser.parse_args()


def check_directories(version, split):
    """
    Validates the existence of required DOTA dataset directories for both 
    Single-Scale (ss) and Multi-Scale (ms) splits.
    Terminates execution if any required data paths are missing.
    """
    v_str = version.replace('.', '_')
    dirs = {
        'ss': {
            'img': f"data/split_ss_dota_{v_str}/{split}/images",
            'ann': f"data/split_ss_dota_{v_str}/{split}/annfiles",
            'ann_hbb': f"data/split_ss_dota_{v_str}/{split}/annfiles_hbb"
        },
        'ms': {
            'img': f"data/split_ms_dota_{v_str}/{split}/images",
            'ann': f"data/split_ms_dota_{v_str}/{split}/annfiles",
            'ann_hbb': f"data/split_ms_dota_{v_str}/{split}/annfiles_hbb"
        }
    }
    
    missing = []
    for scale in dirs:
        for p_type in dirs[scale]:
            if not os.path.exists(dirs[scale][p_type]):
                missing.append(dirs[scale][p_type])
                
    if missing:
        print("\nError: The following required data directories are missing:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease check your dataset paths and ensure DOTA data is split correctly.")
        sys.exit(1)
        
    print("All required data directories exist.")
    return dirs


def check_checkpoints(metafiles, target_models):
    """
    Scans all metafile configs and verifies that required model checkpoints exist locally.
    Terminates execution if any required checkpoints are missing.
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

            found_models_count += 1
            checkpoint_path = os.path.join('checkpoints', os.path.basename(weights_url))
            if not os.path.exists(checkpoint_path):
                missing_checkpoints.append((name, checkpoint_path))

    if found_models_count == 0:
        print("Error: No matching models found to evaluate based on arguments.")
        sys.exit(1)

    if missing_checkpoints:
        print("\nError: The following required model checkpoints are missing:")
        for name, path in missing_checkpoints:
            print(f"  - Model: {name}")
            print(f"    Path:  {path}")
        print("\nPlease download the missing checkpoints into 'checkpoints/' directory before running.")
        sys.exit(1)

    print("All required checkpoints for models exist. Proceeding...")


def run_command(cmd, env=None):
    """
    Runs a shell command, streams the output, and captures the exit code.
    Returns: (success_bool, output_string, error_message)
    """
    try:
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            env=env
        )
        
        output = []
        while True:
            char_bytes = process.stdout.read(1)
            if not char_bytes:
                break
                
            char = char_bytes.decode('utf-8', errors='replace')
            sys.stdout.write(char)
            sys.stdout.flush()
            output.append(char)
            
        returncode = process.wait()
        out_str = ''.join(output)
        
        if returncode != 0:
            return False, out_str, f"Process exited with code {returncode}"
            
        return True, out_str, ""
        
    except Exception as e:
        print(f"\nExecution error: {e}")
        return False, "", str(e)


def extract_collection_name(data, mf_path):
    """Extracts collection name from metafile YAML or falls back to parent folder name."""
    if 'Collections' in data and isinstance(data['Collections'], list) and len(data['Collections']) > 0:
        coll_name = data['Collections'][0].get('Name')
        if coll_name:
            return coll_name
    return os.path.basename(os.path.dirname(mf_path))


def prepare_model_paths(model_entry, data_dirs):
    """Extracts model metadata and constructs dataset/checkpoint paths."""
    name = model_entry.get('Name', '')
    config = model_entry.get('Config', '')
    weights_url = model_entry.get('Weights', '')
    meta = model_entry.get('Metadata', {})
    training_data = meta.get('Training Data', '').lower()

    if 'dota' not in training_data and 'dota' not in config.lower():
        return None

    scale = 'ms' if '_ms_' in name else 'ss'
    img_prefix = data_dirs[scale]['img']
    ann_file = data_dirs[scale]['ann_hbb'] if '_hbb_' in name else data_dirs[scale]['ann']
    checkpoint_path = os.path.join('checkpoints', os.path.basename(weights_url))

    return {
        'name': name,
        'config': config,
        'weights_url': weights_url,
        'checkpoint_path': checkpoint_path,
        'scale': scale,
        'img_prefix': img_prefix,
        'ann_file': ann_file
    }


def evaluate_mAP(paths, args):
    """Executes mAP evaluation for a given model and parses the output."""
    print("\n[+] Running mAP evaluation...")
    cmd_map = (
        f"python -W ignore ./tools/test.py {paths['config']} {paths['checkpoint_path']} "
        f"--eval mAP "
        f"--cfg-options data.test_dataloader.workers_per_gpu={args.map_workers_per_gpu} "
        f"data.test_dataloader.samples_per_gpu={args.map_samples_per_gpu} "
        f"data.test.ann_file={paths['ann_file']} "
        f"data.test.img_prefix={paths['img_prefix']}"
    )
    success, out_map, err_msg = run_command(cmd_map)

    if success:
        map_match = re.search(r"'mAP':\s*([0-9.]+)", out_map)
        if map_match:
            raw_map = float(map_match.group(1))
            if raw_map <= 1.0:
                raw_map *= 100
            val = round(raw_map, 2)
            print(f"\n[OK] Extracted mAP: {val}")
            return val, None
        else:
            print("\n[-] Failed to extract mAP from output.")
            return None, "Regex match failed. Output might be malformed."
    else:
        print(f"\n[-] mAP evaluation failed: {err_msg}")
        return None, err_msg


def evaluate_benchmark(paths, args):
    """Executes benchmark (FPS calculation) for a given model and parses the output."""
    print("\n[+] Running Benchmark...")
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    cmd_bench = (
        f"python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 "
        f"tools/analysis_tools/benchmark.py {paths['config']} {paths['checkpoint_path']} "
        f"--launcher pytorch --log-interval 5 "
        f"--cfg-options data.test_dataloader.workers_per_gpu={args.benchmark_workers_per_gpu} "
        f"data.test_dataloader.samples_per_gpu={args.benchmark_samples_per_gpu} "
        f"data.test.ann_file={paths['ann_file']} "
        f"data.test.img_prefix={paths['img_prefix']}"
    )
    success, out_bench, err_msg = run_command(cmd_bench, env=env)

    if success:
        fps_match = re.search(r"Overall fps:\s*([0-9.]+)", out_bench)
        if fps_match:
            val = float(fps_match.group(1))
            print(f"\n[OK] Extracted FPS: {val}")
            return val, None
        else:
            print("\n[-] Failed to extract FPS.")
            return None, "Regex match failed. Output might be malformed."
    else:
        print(f"\n[-] Benchmark failed: {err_msg}")
        return None, err_msg


def process_model(model_entry, data_dirs, collection_name, args, errors_db):
    """Handles evaluation pipeline for a single model entry."""
    name = model_entry.get('Name', '')
    if args.models and name not in args.models:
        return None

    paths = prepare_model_paths(model_entry, data_dirs)
    if not paths:
        return None

    print(f"\n{'='*80}")
    print(f"Processing model: {name} (Collection: {collection_name})")
    print(f"{'='*80}")

    model_info = {
        'name': name,
        'config': paths['config'],
        'weights_url': paths['weights_url'],
        'scale': paths['scale']
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


def save_report(results_db, errors_db, gpu_name, args):
    """Saves final evaluation results and metadata to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    v_str = args.dota_version.replace('.', '_')
    out_filename = f"models_stats_dota{v_str}_{args.data_split}_{args.tasks}_{timestamp}.json"
    out_filepath = os.path.join(args.work_dir, out_filename)

    with open(out_filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'run_config': vars(args),
            'hardware': {
                'gpu_name': gpu_name
            },
            'results': results_db,
            'errors': errors_db
        }, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Evaluation finished. Results saved to: {out_filepath}")
    print(f"{'='*80}\n")


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected"
    data_dirs = check_directories(args.dota_version, args.data_split)
    metafiles = glob.glob('configs/**/metafile.yml', recursive=True)
    
    check_checkpoints(metafiles, args.models)
                     
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
            model_info = process_model(model_entry, data_dirs, collection_name, args, errors_db)
            if model_info:
                group_results.append(model_info)

        if group_results:
            results_db[collection_name] = group_results

    save_report(results_db, errors_db, gpu_name, args)


if __name__ == '__main__':
    main()