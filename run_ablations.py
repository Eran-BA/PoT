#!/usr/bin/env python3
"""
Ablation study runner for Pointer-over-Heads Transformer.
Executes multiple configurations and logs results to CSV.
"""
import subprocess
import csv
import os
import sys
from datetime import datetime

# Base configuration
BASE_CONFIG = {
    "data_source": "hf",
    "epochs": 3,
    "batch_size": 16,
    "lr": 5e-5,
    "weight_decay": 0.01,
}

# Ablation configurations
ABLATIONS = {
    "iterations": [
        {"max_inner_iters": 1, "name": "iter_1"},
        {"max_inner_iters": 2, "name": "iter_2"},
        {"max_inner_iters": 3, "name": "iter_3"},
    ],
    "routing": [
        {"routing_topk": 0, "name": "soft_mixture"},
        {"routing_topk": 1, "name": "top1"},
        {"routing_topk": 2, "name": "top2"},
    ],
    "halting": [
        {"halting_mode": "fixed", "max_inner_iters": 2, "name": "fixed"},
        {"halting_mode": "entropy", "max_inner_iters": 3, "name": "entropy"},
        {"halting_mode": "halting", "max_inner_iters": 3, "name": "halting"},
    ],
    "combination": [
        {"combination": "mask_concat", "name": "mask_concat"},
        {"combination": "mixture", "name": "mixture"},
    ],
    "learning_rate": [
        {"lr": 3e-5, "name": "lr_3e5"},
        {"lr": 5e-5, "name": "lr_5e5"},
        {"lr": 8e-5, "name": "lr_8e5"},
    ],
}

# Multi-seed runs for best configs
SEEDS = [42, 123, 456]

def build_command(config):
    """Build command line from config dict."""
    cmd = ["python", "ab_ud_pointer_vs_baseline.py"]
    for key, value in config.items():
        if key != "name":
            cmd.extend([f"--{key}", str(value)])
    return cmd

def parse_output(output):
    """Extract metrics from script output."""
    lines = output.strip().split('\n')
    results = []
    
    for line in lines:
        if line.startswith("[Epoch"):
            # Parse epoch results
            parts = line.split()
            epoch = int(parts[1].strip(']'))
            model_type = parts[2]  # BASE or PoH
            
            # Extract metrics (simple parsing)
            metrics = {"epoch": epoch, "model": model_type}
            
            # Parse loss and UAS
            for i, part in enumerate(parts):
                if part == "loss":
                    metrics["train_loss"] = float(parts[i+1])
                elif part == "UAS":
                    metrics["train_uas"] = float(parts[i+1].strip('()s'))
                elif part == "dev" and parts[i+1] == "UAS":
                    metrics["dev_uas"] = float(parts[i+2].strip('()s'))
                elif part == "iters":
                    metrics["mean_iters"] = float(parts[i+1])
            
            results.append(metrics)
    
    return results

def run_ablation(ablation_name, configs, output_file, quick=False):
    """Run a set of ablation experiments."""
    print(f"\n{'='*80}")
    print(f"Running ablation: {ablation_name}")
    print(f"{'='*80}\n")
    
    # Adjust epochs for quick mode
    epochs = 2 if quick else BASE_CONFIG.get("epochs", 3)
    
    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = ['ablation', 'config_name', 'seed', 'epoch', 'model', 
                      'train_loss', 'train_uas', 'dev_uas', 'mean_iters']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if file is empty
        if csvfile.tell() == 0:
            writer.writeheader()
        
        for config_dict in configs:
            config_name = config_dict.pop("name")
            
            # Merge with base config
            full_config = {**BASE_CONFIG, **config_dict, "epochs": epochs}
            
            # Run with single seed for ablations
            full_config["seed"] = 42
            
            print(f"Running {ablation_name}/{config_name}...")
            print(f"  Config: {full_config}")
            
            cmd = build_command(full_config)
            
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=3600  # 1 hour timeout
                )
                
                if result.returncode != 0:
                    print(f"  ✗ Failed: {result.stderr}")
                    continue
                
                # Parse output
                metrics_list = parse_output(result.stdout)
                
                # Write to CSV
                for metrics in metrics_list:
                    row = {
                        'ablation': ablation_name,
                        'config_name': config_name,
                        'seed': full_config["seed"],
                        **metrics
                    }
                    writer.writerow(row)
                
                # Print summary (last epoch)
                if metrics_list:
                    last_metrics = [m for m in metrics_list if m['epoch'] == epochs]
                    for m in last_metrics:
                        print(f"  ✓ {m['model']}: dev UAS {m.get('dev_uas', 'N/A'):.4f}")
                
            except subprocess.TimeoutExpired:
                print(f"  ✗ Timeout")
            except Exception as e:
                print(f"  ✗ Error: {e}")
            
            # Restore name for next iteration
            config_dict["name"] = config_name

def run_multiseed(config, seeds, output_file):
    """Run best config with multiple seeds."""
    print(f"\n{'='*80}")
    print(f"Running multi-seed evaluation")
    print(f"{'='*80}\n")
    
    full_config = {**BASE_CONFIG, **config}
    
    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = ['ablation', 'config_name', 'seed', 'epoch', 'model', 
                      'train_loss', 'train_uas', 'dev_uas', 'mean_iters']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        for seed in seeds:
            full_config["seed"] = seed
            
            print(f"Running seed {seed}...")
            cmd = build_command(full_config)
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                
                if result.returncode == 0:
                    metrics_list = parse_output(result.stdout)
                    for metrics in metrics_list:
                        row = {
                            'ablation': 'multiseed',
                            'config_name': 'best',
                            'seed': seed,
                            **metrics
                        }
                        writer.writerow(row)
                    print(f"  ✓ Completed seed {seed}")
                else:
                    print(f"  ✗ Failed seed {seed}")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--quick", action="store_true", help="Quick mode (2 epochs)")
    parser.add_argument("--ablation", type=str, choices=list(ABLATIONS.keys()) + ["all"], 
                        default="all", help="Which ablation to run")
    parser.add_argument("--multiseed", action="store_true", help="Run multi-seed evaluation")
    parser.add_argument("--output", type=str, default="ablation_results.csv", 
                        help="Output CSV file")
    args = parser.parse_args()
    
    # Create output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"ablation_results_{timestamp}.csv"
    
    print(f"\n{'='*80}")
    print(f"Pointer-over-Heads Ablation Study")
    print(f"Output: {output_file}")
    print(f"Quick mode: {args.quick}")
    print(f"{'='*80}")
    
    # Run selected ablations
    if args.ablation == "all":
        for ablation_name, configs in ABLATIONS.items():
            run_ablation(ablation_name, configs, output_file, quick=args.quick)
    else:
        run_ablation(args.ablation, ABLATIONS[args.ablation], output_file, quick=args.quick)
    
    # Run multi-seed if requested
    if args.multiseed:
        best_config = {
            "max_inner_iters": 3,
            "routing_topk": 2,
            "halting_mode": "entropy",
            "combination": "mask_concat"
        }
        run_multiseed(best_config, SEEDS, output_file)
    
    print(f"\n{'='*80}")
    print(f"✓ Ablation study complete!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()

