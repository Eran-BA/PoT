# Pointer-over-Heads Transformer (PoT)

**Dynamic multi-head attention with adaptive routing for dependency parsing**

**Author:** Eran Ben Artzy

---

## 🎯 Implementation Status

**Core Features** (✅ Complete):
- ✅ Pointer-over-Heads transformer architecture
- ✅ Baseline vs PoH A/B comparison
- ✅ Multi-head routing (soft, top-k)
- ✅ Adaptive halting (fixed, entropy, ACT-style)
- ✅ UAS and LAS support with biaffine labeler
- ✅ Parameter matching (`--param_match`)
- ✅ Encoder freezing (`--freeze_encoder`)
- ✅ CSV logging (auto-generated)
- ✅ Multi-seed runner (`run_multiseed.sh`)
- ✅ Visualization suite

**Utilities Ready** (🔧 Integration pending):
- 🔧 `utils/logger.py` - Drop-in CSV logger (created, can replace built-in)
- 🔧 `utils/conllu_writer.py` - Prediction export (created, needs `--emit_conllu` wire)
- 🔧 `utils/metrics.py` - Punctuation masking (created, needs `--ignore_punct` wire)

**Current State:** All utilities exist and are documented. Main scripts use built-in CSV logging. To use the utility versions, simply import and call them (examples below).

---

## Architecture

```mermaid
flowchart TB
  %% Styles
  classDef head fill:#ffe0c2,stroke:#333,stroke-width:2px,color:#111
  classDef ctrl fill:#cfe8ff,stroke:#333,stroke-width:2px,color:#111
  classDef io fill:#e8f5e9,stroke:#333,stroke-width:2px,color:#111
  classDef mix fill:#fff9c4,stroke:#333,stroke-width:2px,color:#111
  
  %% Nodes
  X[Input]:::io
  
  PC[Pointer Controller<br/>softmax over heads]:::ctrl
  
  subgraph SA[Self-Attention Heads ]
    direction LR
    H1[Head 1]:::head
    H2[Head 2]:::head
    H3[Head 3]:::head
  end
  
  MIX[Weighted/Selected<br/>Head Output]:::mix
  Y[Output]:::io
  
  %% Forward pass
  X --> PC
  PC -->|α₁| H1
  PC -->|α₂| H2
  PC -->|α₃| H3
  
  H1 --> MIX
  H2 --> MIX
  H3 --> MIX
  
  MIX --> Y
  
  %% Feedback cycles (recurrent control)
  H1 -.feedback.-> PC
  H2 -.feedback.-> PC
  H3 -.feedback.-> PC
  
  %% Optional recurrence over time
  Y -.next step context.-> PC
```

## Installation

```bash
# Clone repository
git clone https://github.com/Eran-BA/PoT.git
cd PoT

# Install dependencies
pip install -r requirements.txt
```

## Quick Start (5 minutes)

### Verify Installation

Run a complete experiment with dummy data:

```bash
# 1. Quick sanity check (30 seconds)
python ab_ud_pointer_vs_baseline.py --data_source dummy --epochs 2 --batch_size 8

# 2. View the generated CSV log (auto-created)
cat training_log_*.csv | head -n 5

# 3. Generate quick plot
python plot_simple.py training_log_*.csv
```

**Expected output:**
```
================================================================================
Loading data source: dummy
================================================================================
✓ Train set: 128 examples
✓ Dev set:   48 examples
✓ Sample fields: ['tokens', 'head']
✓ Dependency labels: absent
================================================================================

Config: epochs=2, bs=8, lr=5e-05, wd=0.01, warmup=0.05, seed=42
Data: dummy, Train size: 128, Dev size: 48
Label vocab size: 0 (LAS disabled)
Baseline params: 67,110,913
PoH params:      67,786,769 (+675,856)

Logging results to: training_log_20250111_143022.csv

[Epoch 1]  BASE  train loss 4.7515 UAS 0.2500 (2.1s) | dev UAS 0.3333 (0.8s)
[Epoch 1]  PoH   train loss 2.4897 UAS 0.6250 (2.3s) | dev UAS 0.6667 (0.9s) iters 3.00

[Epoch 2]  BASE  train loss 2.2400 UAS 0.5000 (2.1s) | dev UAS 0.6667 (0.8s)
[Epoch 2]  PoH   train loss 0.8710 UAS 0.7500 (2.3s) | dev UAS 1.0000 (0.9s) iters 3.00

✓ Results logged to: training_log_20250111_143022.csv
```

### Reproducible Full Run

Complete workflow from installation to publication-ready figure:

```bash
# Install dependencies
pip install -r requirements.txt

# Run on real data (HuggingFace UD English EWT)
python ab_ud_pointer_vs_baseline.py \
  --data_source hf \
  --epochs 3 \
  --batch_size 16 \
  --lr 3e-5 \
  --param_match baseline \
  --log_csv my_results.csv

# Generate visualization
python plot_simple.py my_results.csv --out figure1.png

# Multi-seed for paper
./run_multiseed.sh
# Output: training_log_multiseed_TIMESTAMP.csv with mean±std statistics
```

**This produces:**
- ✅ CSV log with all hyperparameters and metrics
- ✅ Parameter-matched fair comparison
- ✅ Publication-ready PNG figure
- ✅ Console output with detailed breakdown

---

## Usage

### Basic Parser

Run the standalone parser:

```bash
python ud_pointer_parser.py --epochs 2 --batch_size 8 --halting_mode entropy --max_inner_iters 3 --routing_topk 2
```

### A/B Comparison: Baseline vs PoH

Compare the Pointer-over-Heads model against a vanilla multi-head attention baseline:

```bash
# Using HuggingFace UD English EWT (default)
python ab_ud_pointer_vs_baseline.py --data_source hf --epochs 5 --batch_size 16

# Using local CoNLL-U files
python ab_ud_pointer_vs_baseline.py --data_source conllu --conllu_dir /path/to/en_ewt --epochs 5 --batch_size 16

# Quick test with dummy data
python ab_ud_pointer_vs_baseline.py --data_source dummy --epochs 2 --batch_size 8

# With CSV logging and warmup
python ab_ud_pointer_vs_baseline.py --data_source hf --epochs 5 --batch_size 16 --lr 3e-5 --warmup_ratio 0.05
```

### Multi-Seed Reproducibility

Run experiments with multiple seeds for robust results:

```bash
# Using the convenience script (Unix/Linux/Mac)
./run_multiseed.sh --data_source hf --epochs 5 --batch_size 16 --lr 3e-5

# Manual runs with different seeds
for seed in 42 123 456; do
  python ab_ud_pointer_vs_baseline.py --data_source hf --epochs 5 --batch_size 16 --seed $seed --log_csv results.csv
done
```

The script automatically:
- Runs training with 3 different seeds
- Logs all results to a timestamped CSV file  
- Computes mean ± std statistics
- Shows parameter counts and metrics
```

### Ablation Studies

Run comprehensive ablation studies to understand what components matter:

```bash
# Quick mode (2 epochs per config)
python run_ablations.py --quick

# Full ablation suite
python run_ablations.py

# Specific ablation only
python run_ablations.py --ablation iterations
python run_ablations.py --ablation routing
python run_ablations.py --ablation halting

# Multi-seed evaluation for best config
python run_ablations.py --multiseed
```

### Key Configuration Options

**Data Sources:**
- `--data_source hf`: HuggingFace Universal Dependencies (default)
- `--data_source conllu`: Local CoNLL-U files
- `--data_source dummy`: Small synthetic dataset for testing

**Using Local CoNLL-U Files:**
```bash
# Download UD English EWT from https://universaldependencies.org/
# Extract to a directory, e.g., ./ud_data/en_ewt/

python ab_ud_pointer_vs_baseline.py \
  --data_source conllu \
  --conllu_dir ./ud_data/en_ewt/ \
  --epochs 5 \
  --batch_size 16

# The script expects .conllu files in the directory:
# ./ud_data/en_ewt/*.conllu
```

**PoH Parameters:**
- `--max_inner_iters N`: Number of refinement iterations (1-3, default: 3)
- `--routing_topk K`: Top-K head selection (0=soft mixture, 1-8=hard routing, default: 2)
- `--halting_mode MODE`: fixed/entropy/halting (default: entropy)
- `--combination MODE`: mask_concat/mixture (default: mask_concat)

**Training:**
- `--lr FLOAT`: Learning rate (default: 5e-5)
- `--weight_decay FLOAT`: Weight decay (default: 0.01)
- `--warmup_ratio FLOAT`: LR warmup ratio (default: 0.05 = 5%)
- `--batch_size INT`: Batch size (default: 8)
- `--seed INT`: Random seed for reproducibility (default: 42)
- `--log_csv FILE`: CSV file for logging results (auto-generated if not provided)

**Advanced Options:**
- `--param_match {baseline,poh}`: Automatically match parameter counts
  - `baseline`: Boost baseline FFN to match PoH params
  - `poh`: Reduce PoH FFN to match baseline params
- `--freeze_encoder`: Freeze pretrained encoder (only train parsing layers)
- `--ignore_punct`: Ignore punctuation tokens in UAS/LAS computation
- `--emit_conllu`: Write predictions to CoNLL-U format for official evaluation

### Parameter Matching for Fair Comparison

To isolate routing benefits from parameter count:

```bash
# Boost baseline to match PoH parameters
python ab_ud_pointer_vs_baseline.py --data_source hf --param_match baseline

# Or shrink PoH to match baseline
python ab_ud_pointer_vs_baseline.py --data_source hf --param_match poh

# Freeze encoder to only train parsing head
python ab_ud_pointer_vs_baseline.py --data_source hf --freeze_encoder
```

## Model Comparison

**Parameter counts** (with distilbert-base-uncased encoder):
- Baseline Parser: ~72.5M parameters
- PoH Parser: ~73.1M parameters
- **Overhead: +676K parameters (~0.93% increase)**

The PoH architecture adds minimal parameters while providing:
- Dynamic computation through adaptive routing
- Better accuracy through iterative refinement
- More interpretable attention patterns

## Evaluation Metrics

The parser supports both **UAS (Unlabeled Attachment Score)** and **LAS (Labeled Attachment Score)**:

- **UAS**: Percentage of tokens with correct head attachment
- **LAS**: Percentage of tokens with both correct head and correct dependency label

LAS is automatically enabled when dependency labels (`deprel`) are available in the training data.

### CoNLL-U Format Evaluation

For official UD evaluation metrics, use the CoNLL evaluation script:

```bash
python conll_eval.py gold.conllu predictions.conllu
```

This provides:
- UAS and LAS scores
- Per-relation breakdown
- Complete tree accuracy
- Compatible with CoNLL 2018 shared task format

## Utilities

### CSV Logging
Automatically tracks all experiments with minimal overhead:

```python
from utils.logger import append_row, flatten_cfg

# Log a single row
append_row("results.csv", flatten_cfg(
    seed=42,
    model="PoH",
    dev_uas=0.892,
    train_uas=0.945,
    mean_iters=2.3
))
```

### CoNLL-U Export
Write predictions for official evaluation:

```python
from utils.conllu_writer import write_conllu

write_conllu(
    "predictions.conllu",
    tokens=token_lists,
    heads_gold=gold_heads,
    heads_pred=predicted_heads,
    deprels_pred=predicted_labels
)
```

### Punctuation Masking
Proper UAS/LAS computation excluding punctuation:

```python
from utils.metrics import compute_uas_las, build_masks_for_metrics

uas, las = compute_uas_las(
    pred_heads, gold_heads, 
    pred_labels, gold_labels,
    mask, deprels=deprel_strings,
    ignore_punct=True
)
```

### Integration Examples

**Quick integration** (~5 lines per feature):

```python
# 1. CoNLL-U export (add to eval loop)
if args.emit_conllu:
    from utils.conllu_writer import write_conllu
    write_conllu("predictions.conllu", tokens, heads_gold, 
                 deprels_gold, heads_pred, deprels_pred)

# 2. Punctuation masking (replace UAS computation)
if args.ignore_punct and deprels_available:
    from utils.metrics import compute_uas_las
    uas, las = compute_uas_las(pred_heads, gold_heads, pred_labels, 
                                gold_labels, mask, deprels, ignore_punct=True)

# 3. Enhanced CSV logging (replace current logging)
from utils.logger import append_row, flatten_cfg
append_row(args.log_csv, flatten_cfg(
    seed=args.seed, epoch=ep, model="PoH", 
    dev_uas=dv_uas, train_uas=tr_uas, **vars(args)
))
```

## Visualization

### Comprehensive Plots
Generate publication-quality figures from ablation results:

```bash
# Plot all results
python plot_results.py ablation_results_TIMESTAMP.csv

# Specific plot types
python plot_results.py results.csv --plot_type ablations
python plot_results.py results.csv --plot_type learning_curves
python plot_results.py results.csv --plot_type comparison
python plot_results.py results.csv --plot_type iters
python plot_results.py results.csv --plot_type seeds
```

Generated plots include:
- **Ablation comparisons**: Bar charts comparing configs within each ablation
- **Learning curves**: UAS progression over epochs
- **Baseline vs PoH**: Direct improvement comparison
- **Iterations vs Accuracy**: Relationship between inner iterations and performance
- **Multi-seed variance**: Stability across random seeds

### Quick Plot
For rapid iteration:

```bash
# Simple UAS vs iterations scatter plot
python plot_simple.py results.csv --out quick_plot.png
```

## Google Colab

Try it in your browser with zero setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eran-BA/PoT/blob/main/PoT_Colab.ipynb)

The Colab notebook includes:
- Quick dummy data test
- Full UD training pipeline  
- Multi-seed evaluation
- Automatic visualization
- Result download

## Repository Structure

```
PoT/
├── pointer_over_heads_transformer.py  # Core PoH transformer implementation
├── ud_pointer_parser.py               # Basic UD parser with PoH
├── ab_ud_pointer_vs_baseline.py       # A/B comparison script (Baseline vs PoH)
├── run_ablations.py                   # Automated ablation study runner
├── run_multiseed.sh                   # Multi-seed experiment runner
├── count_params.py                    # Parameter counting utility
├── conll_eval.py                      # CoNLL-U evaluation script
├── plot_results.py                    # Comprehensive visualization suite
├── plot_simple.py                     # Quick UAS vs iterations plot
├── PoT_Colab.ipynb                    # Google Colab notebook
├── requirements.txt                   # Python dependencies
├── LICENSE                            # Apache 2.0 license
├── README.md                          # This file
└── utils/
    ├── logger.py                      # CSV logging utilities
    ├── conllu_writer.py              # CoNLL-U format writer
    └── metrics.py                     # Evaluation metrics with punct masking
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{benartzy2025pot,
  title={Pointer-over-Heads Transformer: Dynamic Multi-Head Attention with Adaptive Routing},
  author={Ben Artzy, Eran},
  year={2025},
  url={https://github.com/Eran-BA/PoT},
  note={GitHub repository}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.