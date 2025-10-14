#!/bin/bash
# Run both NLI benchmarks

cd /Users/rnbnrzy/Desktop/PoT
export PYTHONPATH=$PWD

echo "========================================"
echo "Running NLI Benchmarks"
echo "========================================"
echo ""

# Run synthetic benchmark (10K steps, ~30-60 min)
echo "1. Starting synthetic NLI benchmark (fair_ab_nli.py)..."
echo "   This will take approximately 30-60 minutes"
echo "   Results will be saved to: experiments/results/nli/ab_results.csv"
echo ""

python3 experiments/fair_ab_nli.py > logs/fair_ab_nli.log 2>&1 &
SYNTHETIC_PID=$!
echo "   Process ID: $SYNTHETIC_PID"
echo "   Monitor: tail -f logs/fair_ab_nli.log"
echo ""

echo "✅ Synthetic benchmark started in background"
echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo ""
echo "To run real NLI benchmark (requires 'datasets' library):"
echo "  1. Install: pip install datasets"
echo "  2. Quick test (5K samples, 2K steps, ~30 min):"
echo "     python3 experiments/real_nli_benchmark.py --max_train_samples 5000 --max_steps 2000"
echo ""
echo "  3. Full benchmark (20K steps, ~2-3 hours):"
echo "     python3 experiments/real_nli_benchmark.py --dataset snli --max_steps 20000"
echo ""
echo "Monitor synthetic benchmark:"
echo "  tail -f logs/fair_ab_nli.log"
echo ""
echo "Check if still running:"
echo "  ps aux | grep fair_ab_nli"
echo ""

