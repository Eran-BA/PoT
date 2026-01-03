---
title: PoT Sudoku Solver
emoji: 🧩
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# PoT Sudoku Solver - Interactive Demo

This Space demonstrates the **Pointer-over-Heads Transformer (PoT)** — a model that *thinks* through iterative refinement cycles to solve Sudoku puzzles.

## 📖 How PoT Works

PoT doesn't just compute — it reasons through **H × L × halt_max_steps** iterations:

- **H_cycles**: High-level reasoning sweeps (slow component)
- **L_cycles**: Low-level refinement steps (fast component)  
- **halt_max_steps**: Adaptive Computation Time depth

Increase these sliders for harder puzzles = **more thinking time** 🧠

## 🔬 Research

- **Paper**: [BERT/GPT with Inner-Thinking Cycles](https://zenodo.org/records/17959628) (DOI: 10.5281/zenodo.17959628)
- **arXiv**: [2506.21734](https://arxiv.org/abs/2506.21734)

## Model Details

| Metric | Value |
|--------|-------|
| Architecture | HybridPoHHRMSolver |
| Controller | Transformer Depth Controller |
| Accuracy | 78.9% on Sudoku-Extreme |
| Parameters | 20.8M |

## Features

- 🎯 **Solve**: Configure thinking time with sliders
- 🔄 **Auto-Tune**: Automatically find optimal config for hard puzzles
- 🧠 **Adjustable**: Change H_cycles (2-4), L_cycles (6-16), halt_max_steps (2-6)

## Links

- 📄 [Paper (Zenodo)](https://zenodo.org/records/17959628)
- 💻 [GitHub Repository](https://github.com/Eran-BA/PoT)
- 🤗 [Model Checkpoint](https://huggingface.co/Eran92/pot-sudoku-78)

