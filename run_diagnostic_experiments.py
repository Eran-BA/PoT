#!/usr/bin/env python3
"""
Quick runner for diagnostic experiments to unlock multi-iteration benefits.

Usage:
    python run_diagnostic_experiments.py --experiment {entropy|distance|all}
"""
import argparse
import subprocess
import pandas as pd
import sys

def run_entropy_halting_ablation(data_dir="ud_data"):
    """
    Experiment 1: Sweep entropy thresholds to find optimal avg iterations.
    Goal: ~1.2-1.5 avg iters with same UAS
    """
    print("\n" + "="*80)
    print("🔬 EXPERIMENT 1: Entropy Halting Ablation")
    print("="*80)
    print("Goal: Find threshold giving mean_iters ~1.3 with 97.9% UAS\n")
    
    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8]
    
    for threshold in thresholds:
        print(f"\n→ Testing ent_threshold={threshold}")
        print("-" * 80)
        
        cmd = [
            "python", "ab_ud_pointer_vs_baseline.py",
            "--data_source", "conllu",
            "--conllu_dir", data_dir,
            "--epochs", "3",
            "--batch_size", "32",
            "--lr", "3e-5",
            "--halting_mode", "entropy",
            "--max_inner_iters", "3",
            "--routing_topk", "0",
            "--log_csv", "entropy_ablation.csv"
        ]
        
        # Note: Can't pass ent_threshold via CLI yet - needs patch
        print(f"⚠️  Note: ent_threshold not yet exposed as CLI arg")
        print(f"    Current default: 0.8 (hardcoded in PoHParser.__init__)")
        print(f"    Would run: {' '.join(cmd)}")
        print(f"    With: --ent_threshold {threshold}")
    
    print("\n" + "="*80)
    print("📊 TO ENABLE: Add this to argparse in ab_ud_pointer_vs_baseline.py:")
    print("="*80)
    print('    ap.add_argument("--ent_threshold", type=float, default=0.8)')
    print('    # Then pass to PoHParser: ent_threshold=args.ent_threshold')
    print()
    

def run_distance_bucket_analysis(data_dir="ud_data"):
    """
    Experiment 2: Analyze UAS by dependency distance.
    Goal: Prove iterations don't help long-range dependencies
    """
    print("\n" + "="*80)
    print("🔬 EXPERIMENT 2: Distance-Bucket Analysis")
    print("="*80)
    print("Goal: Show whether iterations help long-range (>6 tokens) dependencies\n")
    
    configs = [
        ("1 iter", 1),
        ("3 iters", 3),
        ("7 iters", 7)
    ]
    
    for name, iters in configs:
        print(f"\n→ Testing {name}")
        print("-" * 80)
        
        cmd = [
            "python", "ab_ud_pointer_vs_baseline.py",
            "--data_source", "conllu",
            "--conllu_dir", data_dir,
            "--epochs", "3",
            "--batch_size", "32",
            "--lr", "3e-5",
            "--halting_mode", "fixed",
            "--max_inner_iters", str(iters),
            "--routing_topk", "0",
            "--log_csv", f"distance_analysis_{iters}iter.csv"
        ]
        
        print(f"    Command: {' '.join(cmd)}")
        print(f"    ⚠️  Distance logging not yet integrated - needs patch")
    
    print("\n" + "="*80)
    print("📊 TO ENABLE: Add distance-bucket logging to epoch() function:")
    print("="*80)
    print("""
    # In epoch() after computing UAS:
    if log_distance_buckets:
        from utils.diagnostics import log_distance_buckets
        log_distance_buckets(all_heads_gold, all_heads_pred, prefix="  ")
    """)
    print()


def analyze_entropy_results():
    """
    Analyze entropy ablation results.
    """
    try:
        df = pd.read_csv('entropy_ablation.csv')
        poh = df[df['model'] == 'PoH']
        
        print("\n" + "="*80)
        print("📊 ENTROPY ABLATION RESULTS")
        print("="*80)
        
        # Group by threshold (would need to be logged)
        print(f"\n{'Threshold':<12} {'Epoch 3 UAS':<15} {'Mean Iters':<12} {'Efficiency':<12}")
        print("-" * 60)
        
        for threshold in [0.6, 0.65, 0.7, 0.75, 0.8]:
            # This won't work until we log threshold
            # subset = poh[(poh['ent_threshold'] == threshold) & (poh['epoch'] == 3)]
            print(f"{threshold:<12} (Need logging)")
        
        print("\n✅ Best config: threshold giving mean_iters ~1.3 with UAS ≥ 97.9%")
        
    except FileNotFoundError:
        print("⚠️  entropy_ablation.csv not found - run experiment first")


def analyze_distance_results():
    """
    Analyze distance-bucket results.
    """
    print("\n" + "="*80)
    print("📊 DISTANCE-BUCKET ANALYSIS")
    print("="*80)
    
    print("\nExpected pattern if iterations help long-range:")
    print("-" * 60)
    print(f"{'Distance':<12} {'1 iter UAS':<15} {'3 iter UAS':<15} {'7 iter UAS':<15}")
    print("-" * 60)
    print(f"{'1-2':<12} 98.5%           98.5%           98.5%  (no change)")
    print(f"{'3-5':<12} 97.8%           97.9%           97.9%  (minimal)")
    print(f"{'6-10':<12} 96.5%           97.0%           97.2%  (some gain)")
    print(f"{'>10':<12} 94.0%           95.0%           95.5%  (clear gain)")
    print()
    print("If all rows show <0.1% diff → iterations don't add multi-hop reasoning")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run diagnostic experiments")
    parser.add_argument("--experiment", choices=["entropy", "distance", "all"], 
                       default="all", help="Which experiment to run")
    parser.add_argument("--data_dir", default="ud_data", 
                       help="Path to UD CoNLL-U files")
    parser.add_argument("--analyze_only", action="store_true",
                       help="Only analyze existing results, don't run experiments")
    args = parser.parse_args()
    
    if args.analyze_only:
        if args.experiment in ["entropy", "all"]:
            analyze_entropy_results()
        if args.experiment in ["distance", "all"]:
            analyze_distance_results()
        return
    
    print("\n╔" + "="*78 + "╗")
    print("║  🔬 DIAGNOSTIC EXPERIMENTS: Unlocking Multi-Iteration Benefits" + " "*14 + "║")
    print("╚" + "="*78 + "╝\n")
    
    print("⚠️  NOTE: Some features require code patches (see output below)")
    print("    These experiments will show you what to add!\n")
    
    if args.experiment in ["entropy", "all"]:
        run_entropy_halting_ablation(args.data_dir)
    
    if args.experiment in ["distance", "all"]:
        run_distance_bucket_analysis(args.data_dir)
    
    print("\n" + "="*80)
    print("📋 SUMMARY: What Needs to Be Added")
    print("="*80)
    print("""
1. ✅ utils/diagnostics.py - DONE! (distance buckets, deep supervision helpers)

2. ⚠️  CLI args in ab_ud_pointer_vs_baseline.py:
   • --ent_threshold (float, default=0.8)
   • --deep_supervision (flag)
   • --aux_loss_weights (str, e.g., "0.3,0.5,1.0")
   • --log_distance_buckets (flag)

3. ⚠️  Pass ent_threshold to PoHParser:
   In main(), change:
     poh = PoHParser(..., halting_mode=args.halting_mode, max_inner_iters=...)
   To:
     poh = PoHParser(..., ent_threshold=args.ent_threshold)

4. ⚠️  Add distance logging in epoch():
   After computing UAS/LAS, add:
     if log_distance_buckets:
         from utils.diagnostics import log_distance_buckets
         log_distance_buckets(all_heads_gold, all_heads_pred)

5. ⚠️  (Optional) Deep supervision in PoHParser.forward():
   Store intermediate logits and use compute_deep_supervision_loss()

═══════════════════════════════════════════════════════════════════════════════

🚀 QUICK WIN: Once ent_threshold is exposed, run:

   for threshold in 0.6 0.65 0.7 0.75 0.8; do
       python ab_ud_pointer_vs_baseline.py \\
           --data_source conllu --conllu_dir ud_data \\
           --epochs 3 --batch_size 32 --lr 3e-5 \\
           --halting_mode entropy --max_inner_iters 3 \\
           --ent_threshold $threshold \\
           --log_csv entropy_sweep.csv
   done

Expected: Lower threshold → more early stopping → faster with same UAS
Best: threshold giving mean_iters ~1.3 with 97.9% UAS

═══════════════════════════════════════════════════════════════════════════════
    """)


if __name__ == "__main__":
    main()

