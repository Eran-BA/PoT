#!/usr/bin/env python3
"""
Run maze scaling benchmark for LARGE mazes only (20×20 and up).
Skips small mazes (8-16) which are too easy.

Usage:
    python experiments/run_large_mazes_only.py --wall-prob 0.65
"""

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Run large maze benchmark (20×20+)')
    parser.add_argument('--wall-prob', type=float, default=0.65, help='Wall probability (0.6-0.7 recommended)')
    parser.add_argument('--maze-sizes', nargs='+', type=int, default=[20, 24, 30], help='Large maze sizes')
    parser.add_argument('--train', type=int, default=500, help='Training samples per size')
    parser.add_argument('--test', type=int, default=100, help='Test samples per size')
    parser.add_argument('--R', type=int, default=4, help='PoH refinement steps')
    parser.add_argument('--T', type=int, default=4, help='HRM outer loop period')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs per size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true', help='Quick mode: 20×20 only, small dataset')
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.maze_sizes = [20]
        args.train = 100
        args.test = 20
        args.epochs = 25
        print("🚀 QUICK TEST MODE (20×20 only)")
    
    # Validate maze sizes
    if any(s < 20 for s in args.maze_sizes):
        print("⚠️  Warning: You specified maze sizes < 20. These are typically too easy.")
        print("    This script is optimized for large mazes (20×20+).")
        response = input("    Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Determine difficulty name
    if args.wall_prob <= 0.55:
        difficulty = "medium"
    elif args.wall_prob <= 0.7:
        difficulty = "hard"
    else:
        difficulty = "veryhard"
    
    output_name = f"maze_large_wall{int(args.wall_prob*100):02d}_{difficulty}"
    
    print(f"\n{'='*80}")
    print(f"LARGE MAZE BENCHMARK - {difficulty.upper()} MODE")
    print(f"{'='*80}")
    print(f"Wall probability: {args.wall_prob} ({difficulty})")
    print(f"Maze sizes: {args.maze_sizes} (large mazes only)")
    print(f"Training samples: {args.train} per size")
    print(f"Test samples: {args.test} per size")
    print(f"Epochs: {args.epochs}")
    print(f"Config: R={args.R}, T={args.T}, heads={args.heads}")
    print(f"Output: experiments/results/{output_name}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        'python3', 'experiments/maze_scaling_benchmark.py',
        '--maze-sizes', *map(str, args.maze_sizes),
        '--train', str(args.train),
        '--test', str(args.test),
        '--R', str(args.R),
        '--T', str(args.T),
        '--heads', str(args.heads),
        '--epochs', str(args.epochs),
        '--seed', str(args.seed),
        '--output', f'experiments/results/{output_name}',
        '--wall-prob', str(args.wall_prob)
    ]
    
    print(f"Running command: {' '.join(cmd)}\n")
    
    # Run the benchmark
    try:
        subprocess.run(cmd, check=True)
        print(f"\n{'='*80}")
        print(f"✅ Benchmark complete!")
        print(f"Results saved to: experiments/results/{output_name}.json")
        print(f"Plot saved to: experiments/results/{output_name}.png")
        print(f"{'='*80}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Benchmark failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️  Benchmark interrupted by user")
        sys.exit(130)

if __name__ == '__main__':
    main()

