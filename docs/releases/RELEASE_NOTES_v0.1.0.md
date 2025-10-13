# Release v0.1.0 - Pointer-over-Heads Transformer

🎉 **Initial Public Release**

We're excited to announce the first public release of the **Pointer-over-Heads (PoH) Transformer** - a novel dynamic multi-head attention architecture with adaptive routing for dependency parsing and beyond.

## 🚀 Highlights

### What is PoH?

The Pointer-over-Heads Transformer introduces **adaptive routing over attention heads** with iterative refinement, enabling:
- 🎯 **Dynamic head specialization** - Model learns which heads to use per token
- 🔄 **Iterative refinement** - Progressive improvement through 2-3 inner iterations
- ⚡ **Adaptive computation** - Entropy-based early stopping reduces compute
- 📊 **Minimal overhead** - Only +0.9% parameters vs vanilla MHA baseline

### Key Features

✅ **Parameter-Matched Comparisons** (`--param_match`)
- Fair baseline comparisons by matching total parameters
- Isolates routing benefits from model capacity

✅ **Comprehensive Logging** (`--log_csv`)
- Auto-generated CSV logs with all hyperparameters and metrics
- Perfect for ablation studies and reproducibility

✅ **Punctuation Masking** (`--ignore_punct`)
- Standard UD evaluation excluding punctuation tokens
- Proper UAS/LAS computation with `utils/metrics.py`

✅ **CoNLL-U Export** (`--emit_conllu`)
- Write predictions for official evaluation tools
- Compatible with CoNLL 2018 shared task format

✅ **Multi-Seed Support** (`--seed`)
- Reproducible results with configurable random seeds
- Automated multi-seed runner (`run_multiseed.sh`)

✅ **A100-Optimized Colab**
- One-click reproduction with complete pipeline
- Automated ablations, multi-seed runs, and visualization
- ~1-2 hour total runtime

## 📦 Installation

```bash
git clone https://github.com/Eran-BA/PoT.git
cd PoT
pip install -r requirements.txt
```

## 🎯 Quick Start

### 1. Smoke Test (30 seconds)
```bash
python ab_ud_pointer_vs_baseline.py --data_source dummy --epochs 2 --batch_size 8
```

### 2. Real Data A/B (15-20 min on A100)
```bash
python ab_ud_pointer_vs_baseline.py \
  --data_source hf --epochs 5 --batch_size 32 --lr 3e-5 \
  --param_match baseline --ignore_punct --emit_conllu \
  --log_csv results.csv
```

### 3. Multi-Seed Evaluation (20-30 min)
```bash
for seed in 42 123 456; do
  python ab_ud_pointer_vs_baseline.py \
    --data_source hf --epochs 5 --batch_size 32 \
    --halting_mode entropy --max_inner_iters 2 --routing_topk 2 \
    --param_match baseline --seed $seed --log_csv multiseed.csv
done
```

### 4. One-Click Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Eran-BA/PoT/blob/main/PoT_Colab.ipynb)

## 🧪 Ablation Studies

The framework includes comprehensive ablation support:

**Iterations:** Test 1, 2, or 3 inner loops
```bash
--max_inner_iters 2
```

**Routing:** Soft mixture or hard top-k
```bash
--routing_topk 0    # soft
--routing_topk 2    # top-2 hard
```

**Halting:** Fixed, entropy, or ACT-style
```bash
--halting_mode entropy
```

**Combination:** How to merge head outputs
```bash
--combination mask_concat
```

## 📊 Performance

**UD English EWT with DistilBERT encoder:**
- Parameter overhead: +676K (~0.9%)
- Adaptive iterations: 2-3 on average
- Training time: ~15-20 min per 5 epochs (A100)
- Expected improvement: +1-3% UAS over baseline

*Run experiments to get exact numbers for your setup*

## 🛠️ Configuration Options

| Flag | Description | Default |
|------|-------------|---------|
| `--data_source` | hf, conllu, or dummy | `hf` |
| `--param_match` | Match params (baseline/poh) | `None` |
| `--freeze_encoder` | Freeze pretrained encoder | `False` |
| `--ignore_punct` | Exclude punct from metrics | `False` |
| `--emit_conllu` | Export predictions | `False` |
| `--log_csv` | CSV output file | auto-generated |
| `--seed` | Random seed | `42` |
| `--halting_mode` | fixed/entropy/halting | `entropy` |
| `--max_inner_iters` | Max iterations | `3` |
| `--routing_topk` | Top-k routing (0=soft) | `2` |

## 📚 Documentation

- **README:** Complete usage guide and workflow
- **Colab Notebook:** Step-by-step experiments with visualization
- **CHANGELOG:** Detailed feature list and version history

## 🔬 Research Applications

While developed for dependency parsing, PoH can be applied to:
- Machine translation
- Question answering (SQuAD)
- Retrieval-augmented generation (RAG)
- Any task requiring dynamic attention patterns

## 👤 Author

**Eran Ben Artzy** - Innovative idea, architecture design, and implementation

## 📜 License

Apache License 2.0

## 🙏 Acknowledgments

Built on:
- Universal Dependencies English EWT dataset
- HuggingFace Transformers and Datasets
- PyTorch

## 📞 Support

- **Issues:** https://github.com/Eran-BA/PoT/issues
- **Discussions:** https://github.com/Eran-BA/PoT/discussions

## 🎓 Citation

```bibtex
@software{benartzy2025pot,
  title={Pointer-over-Heads Transformer: Dynamic Multi-Head Attention with Adaptive Routing},
  author={Ben Artzy, Eran},
  year={2025},
  version={0.1.0},
  url={https://github.com/Eran-BA/PoT},
  note={GitHub repository}
}
```

---

**Full Changelog:** https://github.com/Eran-BA/PoT/blob/main/CHANGELOG.md

