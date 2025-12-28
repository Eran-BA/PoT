# Archived Notebooks

This folder contains experimental notebooks that were archived for documentation purposes.

---

## Full Diffusion H,L Cycles (Archived: Dec 2024)

**Files:**
- `Full_Diffusion_Sudoku_Training_Colab.ipynb`
- `Diffusion_Sudoku_Training_Colab.ipynb`

### Why Archived

These notebooks implement a **fully diffusion-based architecture** where both H-level and L-level states (`z_H`, `z_L`) are progressively denoised. While architecturally interesting, this approach has a fundamental training limitation:

### The Problem

When the denoiser IS the reasoning engine (not just a routing controller), **all diffusion steps need gradients**. With the HRM-style `lastgrad` approach (only last K steps get gradients):

| Steps | Noise Level (σ) | Gradients? |
|-------|-----------------|------------|
| 0-27 | 1.0 → 0.2 | ❌ No |
| 28-31 | 0.2 → 0.05 | ✅ Yes |

The denoisers never learn to actually denoise because:
1. Most computation (steps 0-27) has no gradient flow
2. The last steps (28-31) have near-zero noise, so denoising is trivial
3. Model stays at random accuracy (~11% for Sudoku)

### Comparison with Working Approaches

| Approach | Denoiser Role | Reasoning | Works? |
|----------|---------------|-----------|--------|
| **Diffusion Controller** | Routing only | Backbone (GRU/TRM) | ✅ Yes |
| **Full Diffusion H,L** | Core reasoning | Denoiser itself | ❌ No* |

*Could work with `lastgrad = max_steps`, but requires more memory.

### Alternative: Diffusion Controller

The **Diffusion Controller** approach (see `notebooks/Diffusion_controller_Sudoku_Training_Colab.ipynb`) works because:
- The backbone (GRU/Transformer) does the reasoning
- Diffusion only controls routing weights
- Gradients flow: `loss → backbone → routing → controller`

### If You Want Full Diffusion H,L to Work

Options:
1. Set `lastgrad = max_steps` (full gradient flow, more memory)
2. Add auxiliary denoising loss (score matching)
3. Use gradient checkpointing
4. Reduce `max_steps` significantly (e.g., 4-8)

---

**See Also:**
- `docs/GRADIENT_MODES_THEORY.md` - Theory of gradient flow in PoT
- `notebooks/Diffusion_controller_Sudoku_Training_Colab.ipynb` - Working diffusion controller approach

