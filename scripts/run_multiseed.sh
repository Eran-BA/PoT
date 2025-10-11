#!/bin/bash
# Multi-seed experiment runner for reproducible results
# Usage: ./run_multiseed.sh [additional args]

SEEDS=(42 123 456)
DEFAULT_ARGS="--data_source hf --epochs 5 --batch_size 16 --lr 3e-5"

# Merge with user-provided args
ALL_ARGS="$DEFAULT_ARGS $@"

echo "================================================================================================="
echo "Running Multi-Seed Experiments"
echo "================================================================================================="
echo "Seeds: ${SEEDS[@]}"
echo "Args: $ALL_ARGS"
echo "================================================================================================="
echo ""

# Create timestamp for this batch
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="training_log_multiseed_${TIMESTAMP}.csv"

for SEED in "${SEEDS[@]}"; do
    echo "-----------------------------------------------------------------------"
    echo "Running with seed: $SEED"
    echo "-----------------------------------------------------------------------"
    python ab_ud_pointer_vs_baseline.py \
        $ALL_ARGS \
        --seed $SEED \
        --log_csv "$LOG_FILE"
    
    if [ $? -ne 0 ]; then
        echo "✗ Error: Experiment failed for seed $SEED"
        exit 1
    fi
    echo ""
done

echo "================================================================================================="
echo "✓ All seeds completed!"
echo "Results saved to: $LOG_FILE"
echo "================================================================================================="
echo ""

# Compute statistics if pandas is available
python3 << 'PYEOF'
import sys
try:
    import pandas as pd
    import glob
    
    # Find the latest log file
    log_files = glob.glob("training_log_multiseed_*.csv")
    if not log_files:
        print("No log files found.")
        sys.exit(1)
    
    latest_log = max(log_files, key=lambda x: x.split('_')[-1])
    print(f"Analyzing: {latest_log}\n")
    
    df = pd.read_csv(latest_log)
    
    # Get final epoch for each seed/model
    final_df = df.groupby(['seed', 'model']).last().reset_index()
    
    print("="*80)
    print("SUMMARY STATISTICS (Final Epoch)")
    print("="*80)
    
    for model in final_df['model'].unique():
        model_df = final_df[final_df['model'] == model]
        print(f"\n{model}:")
        print(f"  Dev UAS: {model_df['dev_uas'].mean():.4f} ± {model_df['dev_uas'].std():.4f}")
        if 'dev_las' in model_df.columns:
            print(f"  Dev LAS: {model_df['dev_las'].mean():.4f} ± {model_df['dev_las'].std():.4f}")
        if model == 'PoH' and 'mean_inner_iters' in model_df.columns:
            iters = pd.to_numeric(model_df['mean_inner_iters'], errors='coerce')
            if not iters.isna().all():
                print(f"  Mean Inner Iters: {iters.mean():.2f} ± {iters.std():.2f}")
    
    print("\n" + "="*80)
    
except ImportError:
    print("\n(Install pandas to see summary statistics)")
except Exception as e:
    print(f"\nError computing statistics: {e}")
PYEOF

echo ""
echo "To visualize results, run:"
echo "  python plot_results.py $LOG_FILE"

