# PoT Usage Guide

**Comprehensive guide for using the Pointer-over-Heads Transformer**

Author: Eran Ben Artzy  
License: Apache 2.0

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU)

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/PoT.git
cd PoT
```

### Step 2: Install Dependencies

**Option A: pip install (recommended)**
```bash
pip install -e .
```

**Option B: Manual installation**
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python scripts/train_simple.py --data_source dummy --epochs 1
```

If you see training output without errors, installation is successful! ✓

---

## Quick Start

### 1. Train with Dummy Data (2 minutes)

Perfect for testing and development:

```bash
python scripts/train_simple.py \
  --data_source dummy \
  --epochs 2 \
  --batch_size 8
```

### 2. Download Real Data (1 minute)

```bash
mkdir -p data
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu -O data/en_ewt-ud-train.conllu
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu -O data/en_ewt-ud-dev.conllu
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu -O data/en_ewt-ud-test.conllu
```

### 3. Train on Real Data (10 minutes)

```bash
python scripts/train_simple.py \
  --data_source conllu \
  --conllu_dir data/ \
  --epochs 5 \
  --batch_size 32 \
  --lr 3e-5
```

---

## Training

### Basic Training

**Single PoH Parser**:
```bash
python scripts/train_simple.py \
  --data_source conllu \
  --conllu_dir data/ \
  --epochs 10 \
  --batch_size 32
```

**A/B Comparison (Baseline vs PoH)**:
```bash
python scripts/train.py \
  --data_source conllu \
  --conllu_dir data/ \
  --epochs 10 \
  --batch_size 32
```

### Configuration Options

#### Model Architecture

```bash
python scripts/train_simple.py \
  --model_name distilbert-base-uncased \  # Encoder
  --d_model 768 \                          # Hidden dimension
  --n_heads 8 \                            # Number of heads
  --d_ff 2048                              # FFN dimension
```

#### PoH-Specific

```bash
python scripts/train_simple.py \
  --halting_mode fixed \        # fixed | entropy | halting
  --max_inner_iters 1 \         # Number of iterations (1 is optimal!)
  --routing_topk 0 \            # 0=soft, >0=hard top-k
  --combination mask_concat \   # mask_concat | mixture
  --ent_threshold 0.8           # Entropy threshold (if halting_mode=entropy)
```

#### Training Hyperparameters

```bash
python scripts/train_simple.py \
  --epochs 10 \
  --batch_size 32 \
  --lr 3e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --seed 42
```

### Optimal Configuration (Recommended)

Based on empirical experiments on UD English EWT:

```bash
python scripts/train_simple.py \
  --data_source conllu \
  --conllu_dir data/ \
  --epochs 5 \
  --batch_size 32 \
  --lr 3e-5 \
  --max_inner_iters 1 \      # 1 iteration is optimal!
  --routing_topk 0 \          # Soft routing performs best
  --halting_mode fixed        # Simple and effective
```

**Expected Results**: UAS ~88%, LAS ~85% on UD English EWT dev set

---

## Evaluation

### Generate Predictions

```bash
python scripts/train_simple.py \
  --data_source conllu \
  --conllu_dir data/ \
  --epochs 5 \
  --emit_conllu \
  --log_csv results.csv
```

This will:
- Save predictions to `predictions_epochN.conllu` files
- Log metrics to `results.csv`

### Official CoNLL-U Evaluation

```bash
python src/evaluation/conll_eval.py \
  data/en_ewt-ud-dev.conllu \
  predictions_epoch5.conllu
```

### Ignore Punctuation in Metrics

```bash
python scripts/train.py \
  --data_source conllu \
  --conllu_dir data/ \
  --ignore_punct  # Exclude punctuation from metrics
```

---

## Advanced Features

### 1. Deep Supervision

Apply loss at each iteration with ramped weights:

```bash
python scripts/train.py \
  --data_source conllu \
  --conllu_dir data/ \
  --max_inner_iters 3 \
  --deep_supervision \
  --ramp_strength 1.0  # 0=flat, 1=full ramp [0.3→1.0]
```

**When to use**: When you want to encourage progressive refinement across iterations.

### 2. ACT-Style Halting

Learned halting with ponder cost:

```bash
python scripts/train.py \
  --data_source conllu \
  --conllu_dir data/ \
  --halting_mode halting \
  --max_inner_iters 5 \
  --act_halting \
  --ponder_coef 1e-3  # Penalty for computation
```

**When to use**: When you want the model to learn when to stop iterating.

### 3. Combined Mode (Recommended for Multiple Iterations)

ACT + deep supervision:

```bash
python scripts/train.py \
  --data_source conllu \
  --conllu_dir data/ \
  --halting_mode halting \
  --max_inner_iters 3 \
  --deep_supervision \
  --act_halting \
  --ponder_coef 1e-3 \
  --ramp_strength 1.0
```

### 4. Gradient Modes

**Full BPTT** (default, recommended):
```bash
python scripts/train.py --grad_mode full
```
Gradients flow through all iterations.

**Last-iterate** (HRM-style, memory efficient):
```bash
python scripts/train.py --grad_mode last
```
Gradients only through final iteration. See `docs/GRADIENT_MODES_THEORY.md` for details.

### 5. TRM-Style Training

Outer supervision steps with inner refinement:

```bash
python scripts/train.py \
  --data_source conllu \
  --conllu_dir data/ \
  --trm_mode \
  --trm_supervision_steps 2 \
  --trm_inner_updates 1 \
  --trm_ramp_strength 1.0
```

### 6. Multi-Seed Robustness

Run multiple seeds for statistical significance:

```bash
bash scripts/run_multiseed.sh data/ 5 32 3e-5 3 "42 1337 2023" results.csv
```

Or manually:
```bash
for seed in 42 1337 2023; do
  python scripts/train.py \
    --data_source conllu \
    --conllu_dir data/ \
    --seed $seed \
    --log_csv results.csv
done
```

### 7. Ablation Studies

Run automated ablations:

```bash
python scripts/run_ablations.py \
  --data_source conllu \
  --conllu_dir data/ \
  --output_csv ablations.csv
```

Or specific ablations:

**Test iterations**:
```bash
for iters in 1 2 3 4 5; do
  python scripts/train.py \
    --max_inner_iters $iters \
    --log_csv iterations_ablation.csv
done
```

**Test routing**:
```bash
for topk in 0 1 2 4; do
  python scripts/train.py \
    --routing_topk $topk \
    --log_csv routing_ablation.csv
done
```

**Test halting modes**:
```bash
for mode in fixed entropy halting; do
  python scripts/train.py \
    --halting_mode $mode \
    --log_csv halting_ablation.csv
done
```

---

## Programmatic Usage

### Basic Example

```python
import torch
from transformers import AutoTokenizer
from src.models import PoHParser
from src.data.loaders import create_dummy_dataset
from src.training.trainer import Trainer

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Create model
parser = PoHParser(
    enc_name="distilbert-base-uncased",
    d_model=768,
    n_heads=8,
    d_ff=2048,
    max_inner_iters=1,
    n_labels=50,
    use_labels=True
).to(device)

# Load data
train_data = create_dummy_dataset(n_samples=100)
dev_data = create_dummy_dataset(n_samples=50)

# Create trainer
trainer = Trainer(parser, tokenizer, device)

# Train
for epoch in range(5):
    train_metrics = trainer.train_epoch(train_data, batch_size=32)
    dev_metrics = trainer.eval_epoch(dev_data, batch_size=32)
    print(f"Epoch {epoch+1}: UAS={dev_metrics['uas']:.3f}")
```

### Advanced Example with Custom Loop

```python
from src.models import PoHParser
from src.data.collate import collate_batch
from torch.optim import AdamW

# Model
parser = PoHParser(...).to(device)

# Optimizer with differentiated LR (automatic in Trainer)
encoder_params = list(parser.encoder.parameters())
controller_params = list(parser.block.controller.parameters())
other_params = [p for n, p in parser.named_parameters() 
                if 'encoder' not in n and 'controller' not in n]

optimizer = AdamW([
    {'params': encoder_params, 'lr': 3e-5},
    {'params': controller_params, 'lr': 3e-5 * 20},  # 20x for controller
    {'params': other_params, 'lr': 3e-5 * 2}
], weight_decay=0.01)

# Training loop
for batch in dataloader:
    subw, word_ids, heads, labels = collate_batch(batch, tokenizer, device)
    
    loss, metrics = parser(subw, word_ids, heads, labels)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parser.parameters(), 1.0)
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}, UAS: {metrics['uas']:.3f}")
```

---

## Troubleshooting

### Issue: Out of Memory

**Solutions**:
1. Reduce batch size: `--batch_size 16`
2. Use last-iterate gradients: `--grad_mode last`
3. Reduce max iterations: `--max_inner_iters 1`
4. Use gradient accumulation (custom implementation needed)

### Issue: Poor Performance

**Checks**:
1. ✓ Using optimal config? (`--max_inner_iters 1 --routing_topk 0`)
2. ✓ Enough epochs? (Try 5-10)
3. ✓ Learning rate okay? (3e-5 is good starting point)
4. ✓ Real data or dummy? (Dummy won't achieve high scores)

**Try**:
- Increase epochs: `--epochs 10`
- Tune learning rate: `--lr 5e-5` or `--lr 1e-5`
- Check data quality: Verify CoNLL-U files are correct format

### Issue: HuggingFace Dataset Loading Fails

**Solution**: Use manual download instead:
```bash
# Download UD files manually
mkdir -p data
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu -O data/en_ewt-ud-train.conllu
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu -O data/en_ewt-ud-dev.conllu

# Use CoNLL-U source
python scripts/train_simple.py --data_source conllu --conllu_dir data/
```

### Issue: Import Errors

**Solution**: Install in development mode:
```bash
pip install -e .
```

Or add to Python path:
```bash
export PYTHONPATH=/path/to/PoT:$PYTHONPATH
```

### Issue: Slow Training

**Speedups**:
1. Use GPU: Check `torch.cuda.is_available()`
2. Reduce iterations: `--max_inner_iters 1`
3. Use hard routing: `--routing_topk 2` (less accurate but faster)
4. Increase batch size (if memory allows): `--batch_size 64`

### Issue: Routing Weights Not Diverse

If routing always uses the same heads:

**Fixes**:
1. Increase routing temperature (modify `routing_tau` in code)
2. Use entropy-based halting: `--halting_mode entropy`
3. Add routing entropy regularization (custom implementation)

---

## Best Practices

### For Research

1. **Multiple Seeds**: Always run 3+ seeds for statistical significance
2. **Log Everything**: Use `--log_csv` to track all metrics
3. **CoNLL-U Export**: Enable `--emit_conllu` for official evaluation
4. **Ignore Punct**: Use `--ignore_punct` for fair comparison
5. **Ablations**: Test each component systematically

### For Production

1. **Optimal Config**: Use `max_inner_iters=1, routing_topk=0`
2. **Freeze Encoder**: Consider `--freeze_encoder` for faster training
3. **Batch Size**: Maximize within memory constraints
4. **Early Stopping**: Monitor dev UAS, stop when plateauing

### For Development

1. **Dummy Data**: Use `--data_source dummy` for rapid iteration
2. **Small Epochs**: Start with `--epochs 1` to test changes
3. **Syntax Check**: Run `python -m py_compile` before training
4. **Print Often**: Monitor loss and metrics at each epoch

---

## Next Steps

- **Architecture Details**: See `docs/architecture.md`
- **API Reference**: See `docs/api_reference.md`
- **Theory**: See `docs/GRADIENT_MODES_THEORY.md` and `docs/DEEP_SUPERVISION_GUIDE.md`
- **Examples**: Check `examples/` directory
- **Colab Notebook**: Try `notebooks/PoT_Colab.ipynb` for cloud execution

Happy parsing! 🚀

