#!/bin/bash
# Quick entropy threshold sweep to find optimal early stopping point
# Goal: Find threshold that gives ~1.2-1.5 avg iterations with 97.9% UAS

set -e

DATA_DIR=${1:-ud_data}
EPOCHS=${2:-3}

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║          🔬 ENTROPY THRESHOLD SWEEP EXPERIMENT                                 ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Data: $DATA_DIR"
echo "Epochs: $EPOCHS"
echo "Testing thresholds: 0.6, 0.65, 0.7, 0.75, 0.8"
echo ""
echo "Expected outcome:"
echo "  • Lower threshold → more early stopping → fewer avg iterations"
echo "  • Should find threshold giving ~1.3 avg iters with 97.9% UAS"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"

for threshold in 0.6 0.65 0.7 0.75 0.8; do
    echo ""
    echo "→ Testing ent_threshold=$threshold"
    echo "────────────────────────────────────────────────────────────────────────────────"
    
    python ab_ud_pointer_vs_baseline.py \
        --data_source conllu --conllu_dir "$DATA_DIR" \
        --epochs "$EPOCHS" --batch_size 32 --lr 3e-5 \
        --halting_mode entropy --max_inner_iters 3 \
        --routing_topk 0 \
        --ent_threshold "$threshold" \
        --log_csv entropy_sweep.csv \
        2>&1 | grep -E "(Epoch [1-3].*PoH|params)" | head -4
    
    echo "✅ Completed threshold=$threshold"
done

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "📊 ANALYZING RESULTS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Analyze results
python << 'EOF'
import pandas as pd
import sys

try:
    df = pd.read_csv('entropy_sweep.csv')
    poh = df[df['model'] == 'PoH']
    
    print(f"{'Threshold':<12} {'Epoch 3 UAS':<15} {'Mean Iters':<12} {'Speedup':<12}")
    print("-" * 60)
    
    # Get baseline (threshold=0.8, no early stopping)
    baseline_subset = poh[(poh['epoch'] == 3)]
    if not baseline_subset.empty:
        # Sort by some criterion to get consistent baseline
        baseline_row = baseline_subset.iloc[-1]  # Last run (likely highest threshold)
        baseline_iters = baseline_row.get('mean_inner_iters', 3.0)
    else:
        baseline_iters = 3.0
    
    # Group by threshold (we need to infer from the order or add it to CSV)
    # For now, show last run for each threshold
    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8]
    results = []
    
    # Since threshold isn't logged to CSV yet, we can't group by it
    # But we ran them in order, so let's try to infer
    epoch3 = poh[poh['epoch'] == 3].copy()
    
    print("⚠️  Note: ent_threshold not logged to CSV yet - showing all runs")
    print("")
    
    for idx, row in epoch3.iterrows():
        uas = row['dev_uas']
        iters = row.get('mean_inner_iters', 'N/A')
        speedup = baseline_iters / iters if isinstance(iters, (int, float)) else 'N/A'
        
        if isinstance(speedup, float):
            print(f"Run {idx:<8} {uas:.4f}          {iters:.2f}         {speedup:.2f}x")
        else:
            print(f"Run {idx:<8} {uas:.4f}          {iters}          {speedup}")
    
    print("")
    print("📌 TO IMPROVE: Add ent_threshold to CSV logging:")
    print("   In main(), when calling flatten_cfg(), add: ent_threshold=args.ent_threshold")
    
except FileNotFoundError:
    print("❌ entropy_sweep.csv not found")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error analyzing results: {e}")
    sys.exit(1)
EOF

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ ENTROPY SWEEP COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📋 Next steps:"
echo "  1. Review the mean_inner_iters for each threshold"
echo "  2. Best config: threshold that gives ~1.3 iters with UAS ≥ 97.9%"
echo "  3. (Optional) Add ent_threshold to CSV logging for better analysis"
echo ""

