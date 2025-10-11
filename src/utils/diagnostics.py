# utils/diagnostics.py
"""
Diagnostic utilities for PoH experiments
- Distance-bucket analysis
- Per-iteration dynamics
- Deep supervision helpers
"""
import torch
from typing import List, Tuple, Dict


def compute_distance_bucket_uas(
    heads_gold: List[List[int]],
    heads_pred: List[List[int]],
    buckets: List[Tuple[int, int]] = [(1, 2), (3, 5), (6, 10), (11, 999)],
) -> Dict[str, Tuple[float, int]]:
    """
    Compute UAS broken down by dependency distance.

    Args:
        heads_gold: List of gold head sequences (0=ROOT)
        heads_pred: List of predicted head sequences
        buckets: List of (min_dist, max_dist) tuples

    Returns:
        Dict mapping bucket name to (uas, count)
    """
    results = {}

    for min_dist, max_dist in buckets:
        bucket_name = f"{min_dist}-{max_dist}" if max_dist < 999 else f">{min_dist-1}"
        correct = 0
        total = 0

        for sent_gold, sent_pred in zip(heads_gold, heads_pred):
            for dep_idx, (gold_head, pred_head) in enumerate(zip(sent_gold, sent_pred)):
                # Skip ROOT (dep_idx points to nothing)
                if gold_head < 0:  # Invalid
                    continue

                # Calculate distance (dep_idx is 0-indexed, gold_head is 0=ROOT, 1=first token)
                # For UD: gold_head=0 means attach to ROOT
                # gold_head=i means attach to i-th token (1-indexed)
                if gold_head == 0:
                    distance = dep_idx + 1  # Distance from ROOT to dep_idx
                else:
                    distance = abs(gold_head - (dep_idx + 1))

                if min_dist <= distance <= max_dist:
                    total += 1
                    if gold_head == pred_head:
                        correct += 1

        uas = correct / total if total > 0 else 0.0
        results[bucket_name] = (uas, total)

    return results


def log_distance_buckets(heads_gold, heads_pred, prefix=""):
    """
    Log distance-bucket UAS to console.
    """
    buckets = compute_distance_bucket_uas(heads_gold, heads_pred)

    print(f"\n{prefix}Distance-Bucket UAS Analysis:")
    print(f"{'Bucket':<12} {'UAS':<8} {'Count':<8}")
    print("-" * 30)
    for bucket_name, (uas, count) in sorted(buckets.items()):
        print(f"{bucket_name:<12} {uas:.4f}   {count:<8}")


def compute_deep_supervision_loss(
    pass_logits_list: List[torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
    aux_weights: List[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute deep supervision loss over multiple passes.

    Args:
        pass_logits_list: List of logits from each pass [B, T, vocab]
        targets: Target indices [B, T]
        mask: Valid token mask [B, T]
        aux_weights: Weights for each pass (default: increasing)

    Returns:
        total_loss: Weighted sum of per-pass losses
        metrics: Dict with per-pass accuracies
    """
    import torch.nn.functional as F

    n_passes = len(pass_logits_list)
    if aux_weights is None:
        # Default: increasing weights [0.3, 0.5, 0.7, 1.0, ...]
        aux_weights = [0.3 + (0.7 / (n_passes - 1)) * i for i in range(n_passes)]

    losses = []
    metrics = {}

    for t, (logits_t, weight) in enumerate(zip(pass_logits_list, aux_weights)):
        # Compute loss for this pass
        loss_t = F.cross_entropy(
            logits_t.view(-1, logits_t.size(-1))[mask.view(-1)], targets.view(-1)[mask.view(-1)]
        )
        losses.append(weight * loss_t)

        # Compute accuracy for this pass (metrics only)
        with torch.no_grad():
            pred_t = logits_t.argmax(-1)
            acc_t = (pred_t[mask] == targets[mask]).float().mean().item()
            metrics[f"pass_{t+1}_acc"] = acc_t

    total_loss = sum(losses)
    return total_loss, metrics


def log_iteration_dynamics(aux_dict: Dict, prefix=""):
    """
    Log routing entropy and representation changes across iterations.

    Expects aux_dict to contain:
        - 'alphas': [B, iters, T, H] routing weights
        - 'logits': [B, iters, T, H] routing logits
        - (optional) 'representations': [B, iters, T, D]
    """
    if "alphas" not in aux_dict:
        return

    alphas = aux_dict["alphas"]  # [B, iters, T, H]
    B, n_iters, T, H = alphas.shape

    print(f"\n{prefix}Iteration Dynamics:")
    print(f"{'Iter':<6} {'Entropy':<10} {'Max α':<10} {'Min α':<10}")
    print("-" * 40)

    for t in range(n_iters):
        alpha_t = alphas[:, t, :, :]  # [B, T, H]

        # Compute routing entropy
        entropy = -(alpha_t * (alpha_t + 1e-10).log()).sum(dim=-1).mean().item()

        # Max/min routing weights
        max_alpha = alpha_t.max().item()
        min_alpha = alpha_t.min().item()

        print(f"{t+1:<6} {entropy:<10.4f} {max_alpha:<10.4f} {min_alpha:<10.4f}")

    # Compute change between iterations
    if n_iters > 1:
        print(f"\n{prefix}Inter-iteration Changes:")
        print(f"{'Transition':<15} {'Δα (L2)':<12} {'KL div':<12}")
        print("-" * 40)

        for t in range(n_iters - 1):
            alpha_t = alphas[:, t, :, :]
            alpha_t1 = alphas[:, t + 1, :, :]

            # L2 change
            delta_l2 = (alpha_t1 - alpha_t).pow(2).sum(dim=-1).sqrt().mean().item()

            # KL divergence
            kl = (
                (alpha_t * ((alpha_t + 1e-10) / (alpha_t1 + 1e-10)).log()).sum(dim=-1).mean().item()
            )

            print(f"iter {t+1}→{t+2}     {delta_l2:<12.6f} {kl:<12.6f}")


def add_noise_between_iterations(x: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
    """
    Add small noise to prevent fixed-point collapse between iterations.
    """
    if noise_std > 0 and x.requires_grad:
        noise = torch.randn_like(x) * noise_std
        return x + noise
    return x
