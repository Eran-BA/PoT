#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UD Dependency Parsing with Pointer-over-Heads (UAS only)
- Dataset: Universal Dependencies English EWT
- Encoder: DistilBERT -> word-level pooling
- Router: PointerMoHTransformerBlock
- Head prediction: biaffine/bilinear pointer over candidates {ROOT + words}

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import argparse
import math
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# import your block
from pointer_over_heads_transformer import PointerMoHTransformerBlock


# ----------------------------
# Utilities
# ----------------------------
def mean_pool_subwords(last_hidden, word_ids):
    """
    last_hidden: [B, S, D] subword reps
    word_ids: List[List[int or None]] length B, gives word index per subword
    returns:
        word_reps: List[Tensor] with shapes [n_words_b, D]
    """
    B, S, D = last_hidden.shape
    outs = []
    for b in range(B):
        ids = word_ids[b]
        # gather reps by word id
        # note: ids has None for special tokens; skip them
        accum = {}
        counts = {}
        for s, wid in enumerate(ids):
            if wid is None:
                continue
            if wid not in accum:
                accum[wid] = last_hidden[b, s]
                counts[wid] = 1
            else:
                accum[wid] = accum[wid] + last_hidden[b, s]
                counts[wid] += 1
        if not accum:
            outs.append(torch.zeros(0, D, device=last_hidden.device))
            continue
        n_words = max(accum.keys()) + 1
        M = torch.zeros(n_words, D, device=last_hidden.device)
        for wid, vec in accum.items():
            M[wid] = vec / counts[wid]
        outs.append(M)
    return outs


def pad_block_diagonal(word_batches: List[torch.Tensor], pad_value: float = 0.0):
    """
    Pads word-level sequences to a common max_len.
    Returns:
      X: [B, T, D], mask: [B, T] (1 for real tokens)
    """
    B = len(word_batches)
    max_len = max(w.size(0) for w in word_batches)
    D = word_batches[0].size(1) if max_len > 0 else 1
    X = word_batches[0].new_full((B, max_len, D), pad_value)
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=X.device)
    for b, w in enumerate(word_batches):
        L = w.size(0)
        if L > 0:
            X[b, :L] = w
            mask[b, :L] = True
    return X, mask


def make_pointer_targets(heads: List[List[int]], max_len: int, device):
    """
    UD heads are 0..n with 0 = ROOT; tokens indexed 1..n.
    We’ll keep that convention:
      - Our classes are [0..n] (n+1 classes), where 0 = ROOT, j in 1..n points to word j.
    Returns:
      Y: [B, T] with 0..n head index for each token position (1..n), 0 for padding positions
      pad_mask: [B, T] (True for valid tokens)
    """
    B = len(heads)
    Y = torch.zeros(B, max_len, dtype=torch.long, device=device)
    pad = torch.zeros(B, max_len, dtype=torch.bool, device=device)
    for b, h in enumerate(heads):
        n = min(len(h), max_len)
        if n > 0:
            # h already uses UD indexing (0..n); place on first n tokens
            Y[b, :n] = torch.tensor(h[:n], device=device)
            pad[b, :n] = True
    return Y, pad


# ----------------------------
# Model
# ----------------------------
class BiaffinePointer(nn.Module):
    """
    Scores heads for each dependent token:
      score[i, j] = dep_i^T W head_j + U^T head_j + b
    Includes learned ROOT embedding as candidate j=0.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(d_model, d_model))
        nn.init.xavier_uniform_(self.W)
        self.U = nn.Linear(d_model, 1, bias=True)  # on head side
        self.root = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.root, std=0.02)

    def forward(
        self, dep: torch.Tensor, head: torch.Tensor, mask_dep: torch.Tensor, mask_head: torch.Tensor
    ):
        """
        dep:  [B, T, D] dependents (tokens)
        head: [B, T, D] head candidates (tokens)
        mask_dep, mask_head: [B, T] bool masks
        Returns:
          logits: [B, T, T+1] over {ROOT + 1..T}
        """
        B, T, D = dep.shape
        # Candidate heads = [ROOT; head_tokens]
        root = self.root.view(1, 1, D).expand(B, 1, D)
        heads_all = torch.cat([root, head], dim=1)  # [B, T+1, D]

        # Bilinear term: dep (B,T,D) * W(D,D) * heads_all^T (B,D,T+1) => (B,T,T+1)
        bil = dep @ self.W
        bil = torch.matmul(bil, heads_all.transpose(1, 2))  # [B,T,T+1]

        # Linear head term (broadcast over dependents)
        u = self.U(heads_all).squeeze(-1)  # [B, T+1]
        u = u.unsqueeze(1).expand(B, T, T + 1)  # [B, T, T+1]

        logits = bil + u  # [B, T, T+1]

        # Mask invalid positions in candidate set (keep ROOT valid)
        # Build candidate mask C: [B, T+1], where C[:,0]=True for ROOT, C[:,1:]=mask_head
        C = torch.ones(B, T + 1, dtype=torch.bool, device=dep.device)
        C[:, 1:] = mask_head
        # For invalid candidates, set to -inf
        invalid = ~C
        logits = logits.masked_fill(invalid.unsqueeze(1), float("-inf"))

        # For padded dependents, set all logits to -inf (won't be used)
        dep_invalid = ~mask_dep
        logits = logits.masked_fill(dep_invalid.unsqueeze(-1), float("-inf"))

        return logits


class UDPointerParser(nn.Module):
    def __init__(
        self,
        enc_name: str = "distilbert-base-uncased",
        d_model: int = 768,
        n_heads_router: int = 8,
        d_ff_router: int = 2048,
        router_mode: str = "mask_concat",  # or 'mixture'
        halting_mode: str = "fixed",
        max_inner_iters: int = 2,
        routing_topk: int = 0,
        ent_threshold: float = 0.7,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(enc_name)
        # router expects [B,T,D] word-level features
        self.router = PointerMoHTransformerBlock(
            d_model=d_model,
            n_heads=n_heads_router,
            d_ff=d_ff_router,
            combination=router_mode,
            halting_mode=halting_mode,
            max_inner_iters=max_inner_iters,
            min_inner_iters=1,
            ent_threshold=ent_threshold,
            routing_topk=routing_topk,
            controller_recurrent=True,
            controller_summary="mean",
            use_pre_norm=True,
        )
        self.pointer = BiaffinePointer(d_model=d_model)

    def forward(
        self,
        batch_subword: Dict[str, torch.Tensor],
        word_ids_batch: List[List[int]],
        heads_gold: List[List[int]],
    ):
        """
        batch_subword: tokenizer(**, return_tensors='pt', is_split_into_words=True) dict on device
        word_ids_batch: list of word_id lists (len=B)
        heads_gold: list of gold head indices per sentence (UD convention 0..n)
        Returns:
          loss (scalar), metrics dict
        """
        device = next(self.parameters()).device
        out = self.encoder(**batch_subword)
        last_hidden = out.last_hidden_state  # [B, S, D]
        word_reps_list = mean_pool_subwords(last_hidden, word_ids_batch)  # list of [n_i, D]
        X, mask = pad_block_diagonal(word_reps_list, pad_value=0.0)  # [B,T,D], [B,T]

        # route with your block
        routed, aux = self.router(X, attn_mask=None, return_aux=True)  # [B,T,D]
        # pointer over candidates
        logits = self.pointer(routed, routed, mask_dep=mask, mask_head=mask)  # [B,T,T+1]

        # build targets
        Y, pad = make_pointer_targets(
            heads_gold, max_len=logits.size(1), device=device
        )  # [B,T], [B,T]
        # Cross-entropy over classes {0..T}
        # shift target indices: already 0..T (UD root=0)
        logits_flat = logits.view(-1, logits.size(-1))  # [B*T, T+1]
        Y_flat = Y.view(-1)  # [B*T]
        pad_flat = pad.view(-1)  # [B*T]
        loss = F.cross_entropy(logits_flat[pad_flat], Y_flat[pad_flat])

        with torch.no_grad():
            pred = logits.argmax(-1)  # [B,T]
            correct = (pred[pad] == Y[pad]).sum().item()
            total = pad.sum().item()
            uas = correct / max(1, total)
        metrics = {"uas": uas, "tokens": total}

        # Include optional routing stats
        if "inner_iters_used" in aux:
            metrics["inner_iters_used"] = aux["inner_iters_used"].item()
        return loss, metrics


# ----------------------------
# Data
# ----------------------------
def load_ud_en_ewt(split: str = "train"):
    # Use the new dataset path for Universal Dependencies
    # Try the universal-dependencies organization with hyphen
    try:
        ds = load_dataset("universal-dependencies/en_ewt", split=split)
    except:
        # Fallback to alternative UD dataset
        try:
            ds = load_dataset("UniversalDependencies/UD_English-EWT", split=split)
        except:
            # If neither works, create a small dummy dataset for testing
            print("Warning: Using dummy dataset. Please download UD_English-EWT manually.")
            from datasets import Dataset

            # Create minimal dummy data
            dummy_data = {
                "tokens": [["The", "cat", "sat"], ["Dogs", "run"]],
                "head": [[2, 3, 0], [2, 0]],
            }
            ds = Dataset.from_dict(dummy_data)
            if split == "validation":
                ds = ds.select(range(min(1, len(ds))))
            return ds

    # Keep sentences with all needed fields
    def filt(ex):
        return ex.get("tokens") is not None and ex.get("head") is not None

    return ds.filter(filt)


def collate_batch(examples, tokenizer, device):
    """
    examples: list of dicts with 'tokens', 'head' OR dict of lists
    Returns:
      subword batch dict (on device), word_ids list, heads list
    """
    # Handle both formats: dict of lists (HF dataset slice) or list of dicts
    if isinstance(examples, dict):
        tokens_list = examples["tokens"]
        heads_list = examples["head"]
    else:
        tokens_list = [ex["tokens"] for ex in examples]
        heads_list = [ex["head"] for ex in examples]  # UD heads 0..n

    enc = tokenizer(
        tokens_list,
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    # build word_id maps
    word_ids_batch = []
    for i in range(len(tokens_list)):
        word_ids_batch.append(enc.word_ids(i))

    # move tensors to device
    for k in enc:
        enc[k] = enc[k].to(device)
    return enc, word_ids_batch, heads_list


# ----------------------------
# Train / Eval loops
# ----------------------------
def run_epoch(model, ds, tokenizer, device, bs=8, train=True, lr=5e-5):
    if train:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        model.eval()
        opt = None

    from math import ceil

    n = len(ds)
    steps = ceil(n / bs)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_iters_used = 0.0
    steps_counted = 0

    for i in range(0, n, bs):
        batch = ds[i : i + bs]
        subw, word_ids, heads = collate_batch(batch, tokenizer, device)
        loss, metrics = model(subw, word_ids, heads)

        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        total_loss += loss.item()
        total_tokens += metrics["tokens"]
        total_correct += int(metrics["uas"] * metrics["tokens"])
        if "inner_iters_used" in metrics:
            total_iters_used += metrics["inner_iters_used"]
            steps_counted += 1

    avg_loss = total_loss / steps
    uas = total_correct / max(1, total_tokens)
    mean_iters = (total_iters_used / max(1, steps_counted)) if steps_counted > 0 else float("nan")
    return {"loss": avg_loss, "uas": uas, "mean_inner_iters": mean_iters}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # router config
    ap.add_argument("--router_heads", type=int, default=8)
    ap.add_argument("--router_ff", type=int, default=2048)
    ap.add_argument(
        "--router_mode", type=str, default="mask_concat", choices=["mask_concat", "mixture"]
    )
    ap.add_argument(
        "--halting_mode", type=str, default="fixed", choices=["fixed", "entropy", "halting"]
    )
    ap.add_argument("--max_inner_iters", type=int, default=2)
    ap.add_argument("--routing_topk", type=int, default=0, help="0=soft; >0=hard top-k")
    ap.add_argument("--ent_threshold", type=float, default=0.7)
    args = ap.parse_args()

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Loading UD English EWT…")
    train_ds = load_ud_en_ewt("train")
    dev_ds = load_ud_en_ewt("validation")

    d_model = AutoModel.from_pretrained(args.model_name).config.hidden_size
    model = UDPointerParser(
        enc_name=args.model_name,
        d_model=d_model,
        n_heads_router=args.router_heads,
        d_ff_router=args.router_ff,
        router_mode=args.router_mode,
        halting_mode=args.halting_mode,
        max_inner_iters=args.max_inner_iters,
        routing_topk=args.routing_topk,
        ent_threshold=args.ent_threshold,
    ).to(device)

    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    for ep in range(1, args.epochs + 1):
        tr = run_epoch(
            model, train_ds, tokenizer, device, bs=args.batch_size, train=True, lr=args.lr
        )
        ev = run_epoch(model, dev_ds, tokenizer, device, bs=args.batch_size, train=False)
        print(
            f"[Epoch {ep}] train loss {tr['loss']:.4f}  UAS {tr['uas']:.4f} "
            f"| dev UAS {ev['uas']:.4f}  mean_inner_iters {ev['mean_inner_iters']:.2f}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
