# Changelog

All notable changes to the Pointer-over-Heads Transformer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-11

### 🎉 Initial Release

First public release of the Pointer-over-Heads (PoH) Transformer architecture for dependency parsing.

### ✨ Features

#### Core Architecture
- **Pointer-over-Heads Transformer** - Dynamic multi-head attention with adaptive routing
- **Multiple Halting Modes** - Fixed, entropy-based, and ACT-style adaptive computation
- **Flexible Routing** - Soft mixture and hard top-k head selection
- **Iterative Refinement** - Configurable inner iterations (1-3) for progressive improvement

#### Evaluation & Comparison
- **Fair A/B Framework** - Baseline vs PoH with parameter matching (`--param_match`)
- **Multi-Seed Support** - Reproducible results with configurable random seeds
- **UAS & LAS Metrics** - Both unlabeled and labeled attachment scores
- **Punctuation Masking** - Standard evaluation excluding punctuation tokens (`--ignore_punct`)

#### Data & Export
- **Multiple Data Sources** - HuggingFace UD, local CoNLL-U, or dummy data
- **CoNLL-U Export** - Write predictions for official evaluation (`--emit_conllu`)
- **CSV Logging** - Automatic logging of all hyperparameters and metrics
- **Multi-Seed Runner** - Shell script for automated multi-seed experiments

#### Utilities
- **Comprehensive Logging** - `utils/logger.py` for structured CSV output
- **CoNLL-U Writer** - `utils/conllu_writer.py` for prediction export
- **Metrics Utilities** - `utils/metrics.py` with punctuation masking support
- **Visualization** - `plot_simple.py` for quick UAS vs iterations plots

#### Documentation & Reproducibility
- **Paper-Tight Workflow** - Complete 1-2 hour pipeline in README
- **A100-Optimized Colab** - One-click reproduction with 24-cell notebook
- **Ablation Studies Guide** - Detailed instructions for testing iterations, routing, halting
- **Expected Results** - Ballpark performance estimates for UD English EWT

### 🔧 Configuration Options

- `--data_source {hf,conllu,dummy}` - Choose data source
- `--param_match {baseline,poh}` - Match parameter counts for fair comparison
- `--freeze_encoder` - Isolate routing gains by freezing pretrained encoder
- `--ignore_punct` - Exclude punctuation from evaluation
- `--emit_conllu` - Export predictions to CoNLL-U format
- `--log_csv FILE` - Log results to CSV (auto-generated if not specified)
- `--seed INT` - Set random seed for reproducibility
- `--halting_mode {fixed,entropy,halting}` - Choose halting strategy
- `--max_inner_iters N` - Set maximum inner iterations (1-3)
- `--routing_topk K` - Top-k routing (0=soft, 1-8=hard)
- `--combination {mask_concat,mixture}` - Head combination mode

### 📊 Performance

- **Parameter Overhead:** ~676K parameters (+0.9% vs baseline)
- **Adaptive Computation:** Converges to 2-3 inner iterations on average
- **Memory Efficient:** Batch size 32 on A100 GPU
- **Training Time:** ~15-20 min per 5 epochs on A100 (UD English EWT)

### 📦 What's Included

```
PoT/
├── pointer_over_heads_transformer.py  # Core architecture
├── ab_ud_pointer_vs_baseline.py       # A/B comparison framework
├── ud_pointer_parser.py               # Standalone parser
├── plot_simple.py                     # Quick visualization
├── PoT_Colab.ipynb                    # Google Colab notebook
├── run_multiseed.sh                   # Multi-seed automation
├── requirements.txt                   # Pinned dependencies
├── LICENSE                            # Apache 2.0
├── README.md                          # Complete documentation
└── utils/
    ├── logger.py                      # CSV logging
    ├── conllu_writer.py              # CoNLL-U export
    └── metrics.py                     # Evaluation utilities
```

### 👤 Author

**Eran Ben Artzy** - Innovative idea and implementation

### 📜 License

Apache License 2.0

### 🔗 Links

- **Repository:** https://github.com/Eran-BA/PoT
- **Colab Notebook:** https://colab.research.google.com/github/Eran-BA/PoT/blob/main/PoT_Colab.ipynb

---

## How to Cite

If you use this code in your research, please cite:

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

