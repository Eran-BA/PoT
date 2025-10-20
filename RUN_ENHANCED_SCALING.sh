#!/bin/bash
# Run parameter scaling benchmark with all enhancements
# Tests Large and XL models with MLM-U inspired techniques

cd /Users/rnbnrzy/Desktop/PoT

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

echo "🚀 Running Enhanced Parameter Scaling Benchmark"
echo "================================================"
echo "Features enabled:"
echo "  ✓ Label smoothing (0.1)"
echo "  ✓ Cosine LR warmup (2000 steps)"
echo "  ✓ Multi-horizon supervision (3-step)"
echo "  ✓ Validity-aware loss"
echo "  ✓ Routing entropy regularization (5e-4, annealed)"
echo "  ✓ CNN maze encoder"
echo "  ✓ Depth-first parameter parity"
echo "================================================"
echo ""

# Run with all enhancements
python -u experiments/parameter_scaling_benchmark.py \
  --maze-size 16 \
  --train 1000 \
  --test 100 \
  --epochs 50 \
  --R 4 \
  --T 4 \
  --seed 42 \
  --lr 1e-3 \
  --label-smoothing 0.1 \
  --warmup-steps 2000 \
  --multi-horizon 3 \
  --validity-mask \
  --route-ent-weight 5e-4 \
  --ent-anneal \
  --output experiments/results/parameter_scaling_enhanced \
  2>&1 | tee experiments/results/parameter_scaling_enhanced.log

echo ""
echo "✅ Benchmark complete!"
echo "📊 Results saved to: experiments/results/parameter_scaling_enhanced/"
echo "📝 Log saved to: experiments/results/parameter_scaling_enhanced.log"

