# Pointer-over-Heads (PoH) Transformer

**Dynamic-Routing Transformer with Iterative Refinement**

[![Tests](https://img.shields.io/badge/tests-17%2F17%20passing-brightgreen)]() [![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

> **PoH** is a modular transformer architecture that adds **head-wise routing** and **iterative refinement** to standard transformers. It's designed for tasks requiring multi-step reasoning, such as dependency parsing, with minimal parameter overhead (0.27%).

---

## 🎯 Quick Start

### Installation

```bash
git clone https://github.com/Eran-BA/PoT.git
cd PoT
pip install torch numpy matplotlib seaborn scipy pandas pytest
```

### Basic Usage

```python
from src.pot.modules import PoHConfig, PoHStack, IterRefiner
import torch

# Configure
cfg = PoHConfig(
    d_model=512,
    n_heads=8,
    route_mode="topk",      # Sparse head selection
    route_topk=2,           # Select top-2 heads per token
    pos_encoding="absolute", # Learned positional embeddings
)

# Build model
stack = PoHStack(cfg, depth=6)
refiner = IterRefiner(stack, max_inner_iters=3)

# Forward pass
x = torch.randn(2, 10, 512)  # [batch, seq_len, d_model]
out, stats = refiner(x, return_inner_stats=True)

print(f"Output shape: {out.shape}")  # [2, 10, 512]
print(f"Inner iterations: {len(stats)}")  # 3
```

**See [examples/poh_usage.py](examples/poh_usage.py) for 6 complete usage examples.**

---

## 🏗️ Architecture

### Hierarchy

```
IterRefiner                # K inner refinement steps + optional ACT halting
  ↓
PoHStack                   # N transformer blocks + positional encoding
  ↓
PoHBlock (×N)              # Head-wise routing + MHA + FFN
  ├─ HeadRouter           # Per-token, per-head routing logits
  ├─ MultiheadAttention   # Standard PyTorch MHA
  └─ FeedForward          # Standard FFN
```

### Key Features

1. **Head-Wise Routing**: Dynamically select or weight attention heads per token
   - **Soft routing**: Differentiable softmax over heads
   - **Top-k routing**: Sparse binary mask (select top-k heads)

2. **Iterative Refinement**: Apply the stack K times for multi-step reasoning
   - Optional outer residual (ReZero-style stabilization)
   - ACT halting for adaptive computation

3. **Positional Encoding**: Config-switchable (none/absolute/rotary)
   - `"none"`: Permutation-invariant tasks
   - `"absolute"`: Learned embeddings (GPT-2 style)
   - `"rotary"`: RoPE (LLaMA style, optional)

4. **Parameter Parity**: **0.27% overhead** vs baseline TransformerEncoder
   - Lightweight router: `d_model → d_model/4 → n_heads`
   - Optional bias stripping to maintain parity

---

## 📊 Results

### Dependency Parsing (Universal Dependencies)

**Coming soon:** Results on UD English, Czech, Ancient Greek

### Parameter Counts

**Configuration:** d=512, h=8, ff=2048, depth=6

| Model | Parameters | Delta | Notes |
|-------|------------|-------|-------|
| TransformerEncoder (baseline) | 18,914,304 | — | — |
| PoH (pos=none) | 18,965,680 | **+0.27%** ✅ | Routing overhead only |
| PoH (pos=absolute, L=512) | 19,227,824 | +1.66% | Includes positional embeddings |

**Breakdown:** HeadRouter (66k params) + head_gain (48 params) = **51k params (0.27%)**

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/test_poh_modules.py -v

# Expected: 17 passed in ~1.3s
```

**Test coverage:**
- ✅ Parameter parity (≤1% delta)
- ✅ Routing correctness (soft sums to 1, top-k is sparse)
- ✅ ACT halting (reduces computation)
- ✅ Gradient flow (end-to-end)
- ✅ Positional encoding modes
- ✅ Outer residual (ReZero initialization)
- ✅ Drop-in compatibility with PyTorch

---

## 🎛️ Configuration

All features are config-driven for easy ablation:

```python
cfg = PoHConfig(
    # Architecture
    d_model=512,
    n_heads=8,
    d_ff=2048,
    dropout=0.1,
    
    # Routing
    route_mode="topk",          # "soft" or "topk"
    route_topk=2,               # For topk mode
    route_temp=1.0,             # For soft mode (temperature)
    share_router=True,          # Share router across layers
    
    # Positional encoding
    pos_encoding="absolute",    # "none", "absolute", or "rotary"
    max_seq_len=512,            # For absolute mode
    
    # ACT halting
    act_halting=False,
    act_threshold=0.99,
    act_penalty=0.01,
    
    # Normalization
    norm_type="pre",            # "pre" or "post"
    param_match_baseline=True,  # Keep <1% delta
)
```

**Ablation dimensions:**
1. Routing mode (soft vs top-k)
2. Top-k heads (1, 2, ..., n_heads)
3. Inner iterations (K=1, 2, 3, ...)
4. Outer residual (on/off)
5. ReZero initialization (on/off)
6. Positional encoding (none/absolute/rotary)
7. ACT halting (on/off)
8. Shared router (on/off)

---

## 📈 Logging & Visualization

### Inner-Loop Logging

Track per-iteration dynamics during training:

```python
from src.pot.logging import InnerLoopLogger, InnerStepRow

with InnerLoopLogger("results/run1/innerloop.csv") as logger:
    for step in training:
        out, inner_stats = refiner(x, return_inner_stats=True)
        
        for s in inner_stats:
            logger.log(InnerStepRow(
                run_id="run1",
                epoch=epoch,
                global_step=step,
                inner_step=s["inner_step"],
                loss=loss.item(),
                attn_entropy_mean=s["attn_entropy_mean"],
                route_entropy_mean=s["route_entropy_mean"],
                # ... more fields
            ))
```

### Visualization

```bash
# Plot inner vs outer dynamics
python scripts/plot_inner_vs_outer.py --csv results/run1/innerloop.csv

# Auto-generate figures from experiment CSVs
python scripts/plot_results.py

# Generate Markdown tables
python scripts/make_readme_tables.py
```

---

## 📚 Documentation

### Quick Links
- **[docs/](docs/)** - Complete documentation index
- **[docs/architecture/](docs/architecture/)** - Architecture guides
- **[docs/guides/](docs/guides/)** - User & developer guides  
- **[examples/poh_usage.py](examples/poh_usage.py)** - 6 usage examples
- **[examples/synthetic/](examples/synthetic/)** - Synthetic task experiments (sorting)

### Key Documents
- **[Architecture Summary](docs/architecture/POH_ARCHITECTURE_SUMMARY.md)** - Comprehensive architecture guide
- **[Contributing Guide](docs/guides/CONTRIBUTING.md)** - Development guidelines
- **[Determinism Guide](docs/guides/DETERMINISM.md)** - Reproducibility best practices

---

## 🔬 Experiments

### Dependency Parsing

```bash
python scripts/train.py \
  --task dependency \
  --config experiments/configs/parsing/ud_en.yaml \
  --model hrm_poh \
  --epochs 50 \
  --max_inner_iters 3
```

### Synthetic Tasks

See [examples/synthetic/README.md](examples/synthetic/README.md) for partial-observability sorting experiments.

---

## 🛠️ Development

### Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy, Matplotlib, Seaborn, SciPy, pandas, pytest

**Optional:**
- `rotary-embedding-torch` (for RoPE support)

### Project Structure

```
PoT/
├── src/pot/
│   ├── modules/          # PoHBlock, PoHStack, IterRefiner, Positional Encoding
│   ├── logging/          # Inner-loop CSV logger
│   ├── core/             # HRM controller, losses, metrics
│   └── tasks/            # Task adapters (dependency parsing, etc.)
├── scripts/
│   ├── train.py          # Unified training entry point
│   ├── plot_results.py   # Auto-plotting
│   ├── plot_inner_vs_outer.py  # Inner-loop visualization
│   └── make_readme_tables.py   # Table generation
├── tests/
│   ├── test_poh_modules.py     # 17 tests (all passing)
│   └── test_core.py            # Core component tests
├── examples/
│   ├── poh_usage.py            # Usage examples
│   └── synthetic/              # Synthetic tasks (sorting)
└── experiments/
    ├── configs/                # YAML configs per task
    └── results/                # Experiment CSVs
```

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

---

## 📖 Citation

```bibtex
@misc{benartzy2025poh,
  title={Pointer-over-Heads: Iterative Refinement with Head-Wise Routing},
  author={Eran Ben Artzy},
  year={2025},
  url={https://github.com/Eran-BA/PoT}
}
```

---

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Inspired by [Hierarchical Reasoning Model (HRM)](https://arxiv.org/abs/2305.19472)
- Built on PyTorch's MultiheadAttention
- ReZero initialization from [Bachlechner et al. 2020](https://arxiv.org/abs/2003.04887)
- ACT halting from [Graves 2016](https://arxiv.org/abs/1603.08983)

---

## 🚀 Status

**v0.1.0** - Production-ready ✅

- [x] Modular architecture (PoHBlock → PoHStack → IterRefiner)
- [x] Parameter parity (0.27% overhead)
- [x] Config-switchable positional encoding
- [x] Inner-loop logging & visualization
- [x] 17/17 tests passing
- [x] Comprehensive documentation
- [ ] Baseline comparisons (Dozat-Manning, transformer+biaffine)
- [ ] Multi-language evaluation (UD)
- [ ] Publication-ready results

---

**Questions?** Open an issue or contact [Eran Ben Artzy](mailto:eran@example.com)
