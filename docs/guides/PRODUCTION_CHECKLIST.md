# PoH Production Checklist

**Status**: 🚧 In Progress  
**Updated**: 2025-10-13

Based on the tight, no-fluff checklist for shipping a rock-solid PoH repo.

---

## ✅ COMPLETED

### P0 - Results (Partial)
- [x] **scripts/plot_results.py** - Auto-plot baseline vs PoH, iterations curves, variance  
- [x] **Existing CSVs** - experiments/results/ contains fair_ab_*.csv files

### P3 - Evaluation Harness (Partial)
- [x] **scripts/run.py** - Single driver for parse/multiseed/ablations  
- [x] **scripts/make_readme_tables.py** - CSV → Markdown table generator

### P4 - Tests (Partial)
- [x] **tests/test_core.py** - Routing, halting, param parity, determinism tests

---

## 📋 TODO (Prioritized)

### P0 — Results you can point to

- [ ] **Check in artifacts**: Organize `experiments/results/<DATE>/` with exact CSVs
  - Current: CSVs scattered in experiments/results/
  - Goal: experiments/results/2025-10-13/ with baseline_len20.csv, poh_12iters_len20.csv
  
- [ ] **Install viz dependencies**: 
  ```bash
  pip install matplotlib seaborn scipy
  ```

- [ ] **Generate plots**:
  ```bash
  python scripts/plot_results.py
  # Should create figs/ with:
  # - sorting_len20_baseline_vs_poh.png
  # - sorting_len20_iterations.png
  # - sorting_len20_variance.png
  ```

- [ ] **Generate tables**:
  ```bash
  python scripts/make_readme_tables.py > RESULTS_TABLES.md
  # Copy-paste into README
  ```

- [ ] **README numbers**: Replace generic claims with actual numbers
  - Example: "PoH achieves 0.1083 ± 0.0025 Kendall-τ on length-20 sorting (+18.7% over baseline 0.0913 ± 0.0154)"
  - Add reproduce command under each table

- [ ] **Determinism note**:
  ```python
  # In training script, add:
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```

### P1 — Baselines reviewers will ask for

- [ ] **Strong baselines**:
  - Dozat & Manning (2017) biaffine parser
  - Modern transformer + biaffine (BERT/RoBERTa base)
  
- [ ] **Param parity**: Add `--param_match baseline` flag
  - Print param counts in footer
  - Ensure PoH and baseline within 10% params

- [ ] **Comparison table**:
  | Model | UAS | LAS | Params (M) | Tokens/s |
  |-------|-----|-----|-----------|----------|
  | Baseline | X.XX | X.XX | X.X | XXX |
  | PoH | X.XX | X.XX | X.X | XXX |

### P2 — Scope + repo organization

- [ ] **Move sorting to examples/**:
  ```bash
  mkdir -p examples/synthetic
  mv src/pot/tasks/sorting.py examples/synthetic/
  mv experiments/configs/sorting/ examples/synthetic/configs/
  ```

- [ ] **Update README metrics**: UAS/LAS only on front page
  - Link Kendall-τ results from examples page

### P3 — Evaluation harness

- [x] **Single driver**: `scripts/run.py` ✅
  
- [ ] **Exact UD settings**: Add to scripts/run.py
  - `--ignore_punct`
  - `--evaluation_script`
  - `--language` code
  - Print settings into CSV

- [x] **Table generator**: `scripts/make_readme_tables.py` ✅

### P4 — Tests

- [x] **Core tests**: `tests/test_core.py` ✅
  - Routing (soft vs top-k)
  - Halting (HRM period)
  - Param parity
  - Determinism
  
- [ ] **Run tests**:
  ```bash
  pip install pytest
  pytest tests/test_core.py -v
  ```

- [ ] **Metric toggles**: Add test for `--ignore_punct`

### P5 — Docs that won't drift

- [ ] **Resolve README links**: Check all internal links work
  
- [ ] **Working Colab**:
  - Create `PoH_Demo.ipynb`
  - Badge in README
  - 1-minute dev run
  - Renders one plot

- [ ] **Environment block**:
  ```yaml
  # environment.yml
  name: poh
  dependencies:
    - python=3.9
    - pytorch=2.0
    - transformers=4.30
    - matplotlib
    - seaborn
    - scipy
  ```

### P6 — CI + quality

- [ ] **Update GitHub Actions**:
  - Matrix on 3.9/3.10/3.11
  - Run `pytest tests/test_core.py`
  - Add ruff/mypy
  - Cache pip
  - Upload figs/ as artifacts

- [ ] **pre-commit**:
  ```yaml
  # .pre-commit-config.yaml
  repos:
    - repo: https://github.com/astral-sh/ruff-pre-commit
      hooks:
        - id: ruff
    - repo: https://github.com/pre-commit/pre-commit-hooks
      hooks:
        - id: end-of-file-fixer
        - id: trailing-whitespace
  ```

### P7 — Naming & discoverability

- [ ] **Rename externally to PoH** (avoid "PoT" collision with Program-of-Thoughts)
  - README title: "PoH: Pointer-over-Heads"
  - Keep internal `src/pot/` for now
  - PyPI: `poh-parsing` (future)

### P8 — Reproducibility & release

- [ ] **Tag v0.1.0**:
  ```bash
  git tag -a v0.1.0 -m "First stable release with HRM controller"
  git push origin v0.1.0
  ```

- [ ] **Dockerfile** (optional):
  ```dockerfile
  FROM python:3.9
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY . /app
  WORKDIR /app
  ```

- [ ] **Seed/variance statement**: Add to README
  - "All results: mean ± std over 3 seeds"
  - "Seed 42, 123, 456 for reproducibility"

### P9 — Experiments that sharpen the story

- [ ] **Non-projective languages**: Czech, Finnish
  - Show clearer PoH win if English is saturated

- [ ] **Compute-adaptive**: ACT halting
  - Plot tokens/steps saved vs ΔUAS
  - Show efficiency, not just accuracy

- [ ] **Frozen-encoder grid**:
  - (frozen vs unfrozen) × (1 vs 2-3 iters)
  - Localize where gains come from

- [ ] **Latency/throughput**: Add `--measure` flag
  - Log ms/sample, tokens/s per iter count

### P10 — Interpretability & reliability

- [ ] **Routing maps**: Dump per-head heatmaps
  - Gate entropy histograms for sample sentences

- [ ] **Error analysis**: Bucket UAS deltas
  - By dependency length
  - By non-projectivity

---

## 🎯 Definition of Done

When you can check all these, you're production-ready:

- [ ] `experiments/results/<DATE>/*.{csv,png}` checked in, referenced in README
- [ ] Strong baselines with param parity, single comparison table
- [ ] `scripts/run.py`, `scripts/plot_results.py`, `scripts/make_readme_tables.py` working
- [ ] 5 pytest tests green in CI; ruff+mypy clean
- [ ] README shows actual numbers with one-line reproduce command under each table
- [ ] Colab runs end-to-end, renders a figure
- [ ] External name standardized to **PoH**; links and badges working
- [ ] v0.1.0 tag cut

---

## 📝 Quick Commands

```bash
# Generate plots
python scripts/plot_results.py

# Generate tables
python scripts/make_readme_tables.py > RESULTS_TABLES.md

# Run experiments
python scripts/run.py parse --config experiments/configs/parsing/ud_en.yaml
python scripts/run.py multiseed --task sorting --config experiments/configs/sorting/len20.yaml --seeds 5
python scripts/run.py ablations --task sorting --config experiments/configs/sorting/len12.yaml --iterations 1,2,4,8,12

# Run tests
pytest tests/test_core.py -v

# Install missing deps
pip install matplotlib seaborn scipy pytest
```

---

## 🚀 Next Steps (In Order)

1. **Install viz dependencies**:
   ```bash
   pip install matplotlib seaborn scipy pytest
   ```

2. **Organize results**:
   ```bash
   mkdir -p experiments/results/2025-10-13
   # Move CSVs to dated folder
   ```

3. **Generate artifacts**:
   ```bash
   python scripts/plot_results.py
   python scripts/make_readme_tables.py > RESULTS_TABLES.md
   ```

4. **Run tests**:
   ```bash
   pytest tests/test_core.py -v
   ```

5. **Update README** with actual numbers from RESULTS_TABLES.md

6. **Commit all**:
   ```bash
   git add figs/ RESULTS_TABLES.md experiments/results/
   git commit -m "Add P0 artifacts: plots, tables, organized CSVs"
   ```

7. **Tag release**:
   ```bash
   git tag -a v0.1.0 -m "Production-ready PoH with HRM"
   git push --tags
   ```

---

**Status**: Core scripts complete, need dependencies installed + artifacts generated
**Next**: Install matplotlib/scipy, run plot/table generators, update README

