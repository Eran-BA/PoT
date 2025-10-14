#!/usr/bin/env python3
"""
Run maze scaling benchmark with configurable wall probability.
Usage:
    python experiments/run_harder_mazes.py --wall-prob 0.6 --maze-sizes 8 12 16 20 24 30
"""

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Run maze benchmark with custom wall probability')
    parser.add_argument('--wall-prob', type=float, default=0.6, help='Wall probability (0.3=easy, 0.45=medium, 0.6=hard, 0.7=very hard)')
    parser.add_argument('--maze-sizes', nargs='+', type=int, default=[8, 12, 16, 20, 24, 30], help='Maze sizes to test')
    parser.add_argument('--train', type=int, default=1000, help='Training samples per size')
    parser.add_argument('--test', type=int, default=200, help='Test samples per size')
    parser.add_argument('--R', type=int, default=4, help='PoH refinement steps')
    parser.add_argument('--T', type=int, default=4, help='HRM outer loop period')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs per size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (small dataset, 16x16 only)')
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.maze_sizes = [16]
        args.train = 100
        args.test = 20
        args.epochs = 20
        print("🚀 QUICK TEST MODE")
    
    # Determine difficulty name
    if args.wall_prob <= 0.35:
        difficulty = "easy"
    elif args.wall_prob <= 0.5:
        difficulty = "medium"
    elif args.wall_prob <= 0.65:
        difficulty = "hard"
    else:
        difficulty = "veryhard"
    
    output_name = f"maze_scaling_wall{int(args.wall_prob*100):02d}_{difficulty}"
    
    print(f"\n{'='*80}")
    print(f"MAZE SCALING BENCHMARK - {difficulty.upper()} MODE")
    print(f"{'='*80}")
    print(f"Wall probability: {args.wall_prob} ({difficulty})")
    print(f"Maze sizes: {args.maze_sizes}")
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
        '--wall-prob', str(args.wall_prob)  # Pass to benchmark script
    ]
    
    print(f"Running command: {' '.join(cmd)}\n")
    
    # Check if --wall-prob is supported, if not, modify file
    # First check if the argument exists
    result = subprocess.run(
        ['python3', 'experiments/maze_scaling_benchmark.py', '--help'],
        capture_output=True,
        text=True
    )
    
    if '--wall-prob' not in result.stdout:
        print("⚠️  Benchmark script doesn't have --wall-prob argument yet.")
        print("📝 Modifying generate_dataset calls in the script...")
        
        # Read the file
        with open('experiments/maze_scaling_benchmark.py', 'r') as f:
            content = f.read()
        
        # Replace wall_prob values
        content = content.replace('wall_prob=0.3', f'wall_prob={args.wall_prob}')
        content = content.replace('wall_prob=0.45', f'wall_prob={args.wall_prob}')
        
        # Write back
        with open('experiments/maze_scaling_benchmark.py', 'w') as f:
            f.write(content)
        
        print(f"✅ Updated wall_prob to {args.wall_prob} in maze_scaling_benchmark.py\n")
        
        # Remove --wall-prob from command since script doesn't support it
        cmd = [c for c in cmd if c not in ['--wall-prob', str(args.wall_prob)]]
    
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

