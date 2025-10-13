#!/usr/bin/env python3
"""
Unified experiment runner for PoH.

Single entry point for all experiments: parse, ablations, multiseed

Usage:
    python scripts/run.py parse --config experiments/configs/parsing/ud_en.yaml
    python scripts/run.py multiseed --task sorting --config experiments/configs/sorting/len20.yaml --seeds 5
    python scripts/run.py ablations --iterations 1,2,4,8,12,16
    python scripts/run.py parse --ignore_punct --language en

Author: Eran Ben Artzy
Year: 2025
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_command(cmd: list, description: str):
    """Run shell command and handle errors."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with code {result.returncode}")
        return False
    print(f"\n✅ Success: {description}")
    return True


def run_parse(args):
    """Run dependency parsing experiment."""
    print("\n🎯 MODE: Dependency Parsing")
    
    # Build command
    cmd = [
        'python', 'scripts/train.py',
        '--task', 'dependency',
        '--config', args.config,
    ]
    
    if args.device:
        cmd.extend(['--device', args.device])
    
    if args.output_dir:
        cmd.extend(['--output_dir', args.output_dir])
    
    # Add UD-specific flags
    if args.ignore_punct:
        cmd.append('--ignore_punct')
    
    if args.language:
        cmd.extend(['--language', args.language])
    
    if args.evaluation_script:
        cmd.extend(['--evaluation_script', args.evaluation_script])
    
    return run_command(cmd, "Dependency Parsing")


def run_multiseed(args):
    """Run multi-seed experiments."""
    print(f"\n🎯 MODE: Multi-Seed ({args.seeds} seeds)")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) / f"multiseed_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for seed in range(args.seeds):
        print(f"\n{'='*80}")
        print(f"Seed {seed + 1}/{args.seeds}")
        print(f"{'='*80}")
        
        seed_output = output_base / f"seed_{seed}"
        seed_output.mkdir(exist_ok=True)
        
        cmd = [
            'python', 'scripts/train.py',
            '--task', args.task,
            '--config', args.config,
            '--seed', str(seed),
            '--output_dir', str(seed_output),
        ]
        
        if args.device:
            cmd.extend(['--device', args.device])
        
        success = run_command(cmd, f"Seed {seed}")
        results.append({'seed': seed, 'success': success})
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n{'='*80}")
    print(f"MULTI-SEED SUMMARY")
    print(f"{'='*80}")
    print(f"Successful: {successful}/{args.seeds}")
    print(f"Results saved to: {output_base}/")
    
    # Save summary
    summary_path = output_base / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'total_seeds': args.seeds,
            'successful': successful,
            'timestamp': timestamp,
            'results': results
        }, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    return successful == args.seeds


def run_ablations(args):
    """Run ablation studies."""
    print(f"\n🎯 MODE: Ablations")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) / f"ablations_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Parse iterations
    iterations_list = [int(x.strip()) for x in args.iterations.split(',')]
    
    print(f"Testing iterations: {iterations_list}")
    
    results = []
    
    for iters in iterations_list:
        print(f"\n{'='*80}")
        print(f"Ablation: {iters} iterations")
        print(f"{'='*80}")
        
        iter_output = output_base / f"iters_{iters}"
        iter_output.mkdir(exist_ok=True)
        
        cmd = [
            'python', 'scripts/train.py',
            '--task', args.task,
            '--config', args.config,
            '--iterations', str(iters),
            '--output_dir', str(iter_output),
        ]
        
        if args.device:
            cmd.extend(['--device', args.device])
        
        if args.seeds:
            cmd.extend(['--seeds', str(args.seeds)])
        
        success = run_command(cmd, f"{iters} iterations")
        results.append({'iterations': iters, 'success': success})
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n{'='*80}")
    print(f"ABLATION SUMMARY")
    print(f"{'='*80}")
    print(f"Successful: {successful}/{len(iterations_list)}")
    print(f"Results saved to: {output_base}/")
    
    # Save summary
    summary_path = output_base / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'iterations_tested': iterations_list,
            'successful': successful,
            'timestamp': timestamp,
            'results': results
        }, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    return successful == len(iterations_list)


def main():
    parser = argparse.ArgumentParser(
        description='Unified experiment runner for PoH',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dependency parsing
  python scripts/run.py parse --config experiments/configs/parsing/ud_en.yaml

  # Multi-seed experiment
  python scripts/run.py multiseed --task sorting --config experiments/configs/sorting/len20.yaml --seeds 5

  # Ablation study
  python scripts/run.py ablations --task sorting --config experiments/configs/sorting/len12.yaml --iterations 1,2,4,8,12

  # Parsing with UD settings
  python scripts/run.py parse --config experiments/configs/parsing/ud_en.yaml --ignore_punct --language en
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Experiment mode')
    
    # Parse mode
    parse_parser = subparsers.add_parser('parse', help='Run dependency parsing experiment')
    parse_parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    parse_parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parse_parser.add_argument('--output_dir', type=str, default='experiments/results', help='Output directory')
    parse_parser.add_argument('--ignore_punct', action='store_true', help='Ignore punctuation in evaluation')
    parse_parser.add_argument('--language', type=str, help='Language code (e.g., en, cs, fi)')
    parse_parser.add_argument('--evaluation_script', type=str, help='Path to official eval script')
    
    # Multiseed mode
    multiseed_parser = subparsers.add_parser('multiseed', help='Run multi-seed experiment')
    multiseed_parser.add_argument('--task', type=str, required=True, help='Task name')
    multiseed_parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    multiseed_parser.add_argument('--seeds', type=int, default=3, help='Number of seeds')
    multiseed_parser.add_argument('--device', type=str, default='cuda', help='Device')
    multiseed_parser.add_argument('--output_dir', type=str, default='experiments/results', help='Output directory')
    
    # Ablations mode
    ablations_parser = subparsers.add_parser('ablations', help='Run ablation study')
    ablations_parser.add_argument('--task', type=str, required=True, help='Task name')
    ablations_parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    ablations_parser.add_argument('--iterations', type=str, default='1,2,4,8', help='Comma-separated iteration counts')
    ablations_parser.add_argument('--seeds', type=int, default=1, help='Seeds per iteration')
    ablations_parser.add_argument('--device', type=str, default='cuda', help='Device')
    ablations_parser.add_argument('--output_dir', type=str, default='experiments/results', help='Output directory')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    # Run appropriate mode
    if args.mode == 'parse':
        success = run_parse(args)
    elif args.mode == 'multiseed':
        success = run_multiseed(args)
    elif args.mode == 'ablations':
        success = run_ablations(args)
    else:
        print(f"Unknown mode: {args.mode}")
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

