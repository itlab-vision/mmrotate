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
        '--samples-per-gpu', 
        type=int, 
        default=2,
        help='Batch size per GPU')
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
        print("Error: The following required data directories are missing:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
        
    print("All required data directories exist. Proceeding...")
    return dirs

def run_command(cmd, env=None):
    """
    Runs a shell command, streams the output to the console in real-time,
    and returns the full output as a string for parsing.
    Reads in binary mode to prevent Python from automatically converting
    carriage returns (\\r) into newlines (\\n).
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
            
            if not char_bytes and process.poll() is not None:
                break
                
            if char_bytes:
                char = char_bytes.decode('utf-8', errors='replace')
                sys.stdout.write(char)
                sys.stdout.flush()
                output.append(char)
                
        return ''.join(output)
    except Exception as e:
        print(f"\nExecution error: {e}")
        return str(e)

def main():
    args = parse_args()
    
    os.makedirs(args.work_dir, exist_ok=True)
    
    gpu_name = "CPU or No GPU detected"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    
    data_dirs = check_directories(args.dota_version, args.data_split)
    
    metafiles = glob.glob('configs/**/metafile.yml', recursive=True)
                     
    results_db = {}
    
    for mf_path in metafiles:
        with open(mf_path, 'r', encoding='utf-8') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError:
                continue
                
        if not data or 'Models' not in data:
            continue
            
        group_results = []
        
        for model in data['Models']:
            name = model.get('Name', '')

            if args.models and name not in args.models:
                continue

            config = model.get('Config', '')
            weights_url = model.get('Weights', '')
            
            meta = model.get('Metadata', {})
            training_data = meta.get('Training Data', '').lower()
            
            if 'dota' not in training_data and 'dota' not in config.lower():
                continue
                
            checkpoint_path = os.path.join('checkpoints', os.path.basename(weights_url))
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint {checkpoint_path} not found. Skipping {name}.")
                continue
                
            scale = 'ms' if '_ms_' in name else 'ss'
            img_prefix = data_dirs[scale]['img']
            
            if '_hbb_' in name:
                ann_file = data_dirs[scale]['ann_hbb']
            else:
                ann_file = data_dirs[scale]['ann']
            
            model_info = {
                'name': name,
                'config': config,
                'weights_url': weights_url,
                'scale': scale
            }
            
            print(f"\n{'='*80}")
            print(f"Processing model: {name}")
            print(f"{'='*80}")
            
            # Evaluate mAP
            if args.tasks in ['map', 'map+benchmark']:
                print("\n[+] Running mAP evaluation...")
                cmd_map = (
                    f"python -W ignore ./tools/test.py {config} {checkpoint_path} "
                    f"--eval mAP "
                    f"--cfg-options data.test_dataloader.workers_per_gpu={args.map_workers_per_gpu} "
                    f"data.test_dataloader.samples_per_gpu={args.samples_per_gpu} "
                    f"data.test.ann_file={ann_file} "
                    f"data.test.img_prefix={img_prefix}"
                )
                out_map = run_command(cmd_map)
                map_match = re.search(r"'mAP':\s*([0-9.]+)", out_map)
                model_info['mAP'] = float(map_match.group(1)) if map_match else None
                if not map_match:
                    print("\n[-] Failed to extract mAP.")
                else:
                    print(f"\n[OK] Extracted mAP: {model_info['mAP']}")

            # Evaluate FPS (Benchmark)
            if args.tasks in ['benchmark', 'map+benchmark']:
                print("\n[+] Running Benchmark...")
                env = os.environ.copy()
                env["PYTHONWARNINGS"] = "ignore"
                cmd_bench = (
                    f"python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 "
                    f"tools/analysis_tools/benchmark.py {config} {checkpoint_path} "
                    f"--launcher pytorch --log-interval 5 "
                    f"--cfg-options data.test_dataloader.workers_per_gpu={args.benchmark_workers_per_gpu} "
                    f"data.test_dataloader.samples_per_gpu={args.samples_per_gpu} "
                    f"data.test.ann_file={ann_file} "
                    f"data.test.img_prefix={img_prefix}"
                )
                out_bench = run_command(cmd_bench, env=env)
                fps_match = re.search(r"Overall fps:\s*([0-9.]+)", out_bench)
                model_info['FPS'] = float(fps_match.group(1)) if fps_match else None
                if not fps_match:
                    print("\n[-] Failed to extract FPS.")
                else:
                    print(f"\n[OK] Extracted FPS: {model_info['FPS']}")
                    
            group_results.append(model_info)
            
        if group_results:
            results_db[mf_path] = group_results

    # Save results to JSON
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
            'results': results_db
        }, f, indent=4, ensure_ascii=False)
        
    print(f"\n{'='*80}")
    print(f"Evaluation finished. Results saved to: {out_filepath}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()