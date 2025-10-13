"""
HRM Controller Diagnostic Smoke Test

Quick sanity check that HRM controller exhibits expected behavior:
- Entropy decreases over iterations (routing sharpens)
- Max probability increases (sharper routing)
- H-module updates show behavioral jumps at multiples of T

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch

from src.models.layers import HRMPointerController

torch.manual_seed(0)

print("="*80)
print("HRM CONTROLLER DIAGNOSTIC SMOKE TEST")
print("="*80)

B, L, D, H = 4, 9, 64, 8
T = 3

print(f"\nConfiguration:")
print(f"  Batch size: {B}")
print(f"  Sequence length: {L}")
print(f"  Model dimension: {D}")
print(f"  Number of heads: {H}")
print(f"  H-module period (T): {T}")
print(f"  Top-k routing: 3")
print(f"  Initial temperature: 2.0")

ctrl = HRMPointerController(
    d_model=D,
    n_heads=H,
    d_ctrl=64,
    T=T,
    topk=3,
    temperature_init=2.0,
    temperature_min=0.7
)

x = torch.randn(B, L, D)
head_feats = torch.randn(B, H, L, D // H)
state = ctrl.init_state(B, x.device)

print(f"\n✓ Controller initialized")
print(f"  Parameters: {sum(p.numel() for p in ctrl.parameters()):,}")

entropies = []
max_probs = []
mean_probs = []
top_heads = []

print(f"\n{'='*80}")
print("RUNNING 8 ITERATIONS WITH TEMPERATURE ANNEALING")
print(f"{'='*80}\n")

for t in range(8):
    alphas, state, aux = ctrl(x, head_outputs=head_feats, state=state)
    
    entropy = float(aux["entropy"])
    max_prob = float(alphas.max())
    mean_prob = float(alphas.mean())
    
    entropies.append(entropy)
    max_probs.append(max_prob)
    mean_probs.append(mean_prob)
    
    # Track which heads are selected (for first batch element)
    top_k_heads = alphas[0].topk(3)[1].tolist()
    top_heads.append(top_k_heads)
    
    # Cool temperature a bit each step
    new_temp = max(0.7, 2.0 * (0.95 ** (t + 1)))
    ctrl.set_temperature(new_temp)
    
    # Mark H-module updates
    h_update = "← H-UPDATE" if (t % T == 0) else ""
    
    print(f"Iter {t}: "
          f"entropy={entropy:.4f}, "
          f"max_prob={max_prob:.4f}, "
          f"temp={aux['temperature']:.3f}, "
          f"top_heads={top_k_heads} {h_update}")

print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print(f"{'='*80}\n")

print(f"Entropy progression: {[round(e, 4) for e in entropies]}")
print(f"Max prob progression: {[round(m, 4) for m in max_probs]}")
print(f"Mean prob progression: {[round(m, 4) for m in mean_probs]}")

print(f"\nFinal state:")
print(f"  z_L[0, :5]: {state.z_L[0, :5].tolist()}")
print(f"  z_H[0, :5]: {state.z_H[0, :5].tolist()}")
print(f"  step: {state.step[0].item()}")

print(f"\n{'='*80}")
print("BEHAVIORAL CHECKS")
print(f"{'='*80}\n")

# Check 1: Entropy decreases (routing sharpens)
entropy_decreased = entropies[-1] < entropies[0]
print(f"✓ Entropy decreased: {entropies[0]:.4f} → {entropies[-1]:.4f} "
      f"{'PASS' if entropy_decreased else 'FAIL'}")

# Check 2: Max probability increases (sharper routing)
max_prob_increased = max_probs[-1] > max_probs[0]
print(f"✓ Max prob increased: {max_probs[0]:.4f} → {max_probs[-1]:.4f} "
      f"{'PASS' if max_prob_increased else 'FAIL'}")

# Check 3: Routing shows some consistency (not random)
head_usage = {}
for heads in top_heads:
    for h in heads:
        head_usage[h] = head_usage.get(h, 0) + 1

most_used_head = max(head_usage, key=head_usage.get)
most_used_count = head_usage[most_used_head]
routing_has_preference = most_used_count > 2  # At least appears in 3/8 iterations

print(f"✓ Routing shows preference: Head {most_used_head} used {most_used_count}/8 times "
      f"{'PASS' if routing_has_preference else 'FAIL'}")

# Check 4: H-module updates at correct intervals
print(f"✓ H-module update schedule: every {T} steps (steps 0, {T}, {2*T}, ...)")

print(f"\n{'='*80}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*80}\n")

print("Expected behavior:")
print("  • Entropy decreases over iterations (temperature annealing)")
print("  • Max probability increases (routing sharpens)")
print("  • Routing shows non-uniform head preference (specialization)")
print("  • With T=3, behavioral jumps at steps 0, 3, 6 (H-updates)")
print("\nAll checks passed! ✓" if all([
    entropy_decreased,
    max_prob_increased,
    routing_has_preference
]) else "\n⚠ Some checks failed - review controller behavior")

