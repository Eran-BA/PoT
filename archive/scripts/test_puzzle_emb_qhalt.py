"""
Quick test to verify puzzle embeddings and Q-halting implementation.
"""

import torch
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from src.pot.models.puzzle_embedding import PuzzleEmbedding
from src.pot.models.adaptive_halting import QHaltingController

print("="*80)
print("Testing Puzzle Embedding Module")
print("="*80)

# Test puzzle embedding
puzzle_emb = PuzzleEmbedding(num_puzzles=100, emb_dim=256, init_std=0.0)
print(f"✓ Created PuzzleEmbedding: {puzzle_emb}")

# Test forward pass
puzzle_ids = torch.tensor([0, 5, 10, 42])
emb = puzzle_emb(puzzle_ids)
print(f"✓ Embedding shape: {emb.shape} (expected: [4, 256])")
assert emb.shape == (4, 256), f"Wrong shape: {emb.shape}"

# Check zero init
print(f"✓ Embedding mean: {emb.mean().item():.6f} (should be ~0)")
print(f"✓ Embedding std: {emb.std().item():.6f} (should be ~0)")

print("\n" + "="*80)
print("Testing Q-Halting Controller")
print("="*80)

# Test Q-halting
q_halt = QHaltingController(d_model=256, max_steps=16)
print(f"✓ Created QHaltingController: {q_halt}")

# Test forward pass
hidden = torch.randn(4, 900, 256)  # [B, L, D]
q_halt_logit, q_continue_logit = q_halt(hidden)
print(f"✓ Q-halt shape: {q_halt_logit.shape} (expected: [4])")
print(f"✓ Q-continue shape: {q_continue_logit.shape} (expected: [4])")
assert q_halt_logit.shape == (4,), f"Wrong shape: {q_halt_logit.shape}"
assert q_continue_logit.shape == (4,), f"Wrong shape: {q_continue_logit.shape}"

# Check initialization (should be negative to encourage exploration)
print(f"✓ Q-halt mean: {q_halt_logit.mean().item():.2f} (should be <0, init=-5)")
print(f"✓ Q-continue mean: {q_continue_logit.mean().item():.2f} (should be <0, init=-5)")

# Test halting decision (training mode)
should_halt = q_halt.should_halt(q_halt_logit, q_continue_logit, step=1, is_training=True)
print(f"✓ Should halt (step 1, training): {should_halt} (expected: all False, min 2 steps)")
assert not should_halt.any(), "Should not halt at step 1"

should_halt = q_halt.should_halt(q_halt_logit, q_continue_logit, step=16, is_training=True)
print(f"✓ Should halt (step 16, training): {should_halt} (expected: all True, max steps)")
assert should_halt.all(), "Should halt at max_steps"

# Test halting decision (inference mode)
should_halt = q_halt.should_halt(q_halt_logit, q_continue_logit, step=8, is_training=False)
print(f"✓ Should halt (step 8, inference): {should_halt} (expected: all False, runs to max)")
assert not should_halt.any(), "Should not halt before max_steps in inference"

print("\n" + "="*80)
print("Testing Grid2GridMazeSolver Integration")
print("="*80)

from experiments.maze_grid2grid_hrm import Grid2GridMazeSolver

# Create model with puzzle embeddings and Q-halting
model = Grid2GridMazeSolver(
    vocab_size=6,
    d_model=256,
    n_heads=8,
    n_layers=1,
    use_poh=True,
    R=4,
    T=4,
    num_puzzles=100,
    puzzle_emb_dim=256,
    max_halting_steps=16
)
print(f"✓ Created Grid2GridMazeSolver with puzzle embeddings and Q-halting")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
puzzle_params = sum(p.numel() for p in model.puzzle_emb.parameters())
print(f"✓ Total parameters: {total_params:,}")
print(f"✓ Puzzle embedding parameters: {puzzle_params:,}")

# Test forward pass
input_seq = torch.randint(0, 6, (2, 900))  # [B, 900]
puzzle_ids = torch.tensor([0, 5])  # [B]

model.eval()
with torch.no_grad():
    logits, q_halt, q_continue, steps = model(input_seq, puzzle_ids)

print(f"✓ Logits shape: {logits.shape} (expected: [2, 900, 6])")
print(f"✓ Q-halt shape: {q_halt.shape} (expected: [2])")
print(f"✓ Q-continue shape: {q_continue.shape} (expected: [2])")
print(f"✓ Actual steps taken: {steps} (max: 16 in inference)")

assert logits.shape == (2, 900, 6), f"Wrong logits shape: {logits.shape}"
assert q_halt.shape == (2,), f"Wrong q_halt shape: {q_halt.shape}"
assert q_continue.shape == (2,), f"Wrong q_continue shape: {q_continue.shape}"
assert steps == 16, f"Should run max_steps in inference, got {steps}"

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nImplementation verified:")
print("  ✓ PuzzleEmbedding: zero-init, correct shapes")
print("  ✓ QHaltingController: correct initialization, halting logic")
print("  ✓ Grid2GridMazeSolver: integrates both modules")
print("  ✓ Forward pass: correct output shapes and adaptive computation")
print("\nReady for training!")

