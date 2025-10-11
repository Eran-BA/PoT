#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, math, os, glob
from typing import List, Dict, Tuple
import torch, torch.nn as nn, torch.nn.functional as F

# === try HF datasets if requested ===
def try_hf_load(split):
    try:
        from datasets import load_dataset, Dataset
        # Try the new dataset paths
        try:
            ds = load_dataset("universal-dependencies/en_ewt", split=split)
        except:
            try:
                ds = load_dataset("UniversalDependencies/UD_English-EWT", split=split)
            except:
                # Fallback to dummy dataset
                print(f"Warning: Using dummy dataset for {split}. Please download UD_English-EWT manually.")
                dummy_data = {
                    "tokens": [["The", "cat", "sat"], ["Dogs", "run"], ["I", "eat", "food"]],
                    "head": [[2, 3, 0], [2, 0], [2, 3, 0]],
                }
                ds = Dataset.from_dict(dummy_data)
                if split == "validation":
                    ds = ds.select(range(min(1, len(ds))))
                return list(ds)
        
        filtered = ds.filter(lambda ex: ex.get("tokens") is not None and ex.get("head") is not None)
        return list(filtered)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# === local CoNLL-U ===
def load_conllu_dir(path):
    from conllu import parse_incr
    samples = []
    for fp in sorted(glob.glob(os.path.join(path, "*.conllu"))):
        with open(fp, "r", encoding="utf-8") as f:
            for sent in parse_incr(f):
                tokens, heads = [], []
                for tok in sent:
                    if isinstance(tok["id"], tuple):  # skip multiword token rows
                        continue
                    tokens.append(tok["form"])
                    heads.append(int(tok["head"]))
                if tokens and heads and len(tokens) == len(heads):
                    samples.append({"tokens": tokens, "head": heads})
    return samples

# === tiny dummy ===
def dummy_ds(n=64):
    import random
    rng = random.Random(0)
    sents = []
    for _ in range(n):
        L = rng.randint(5, 12)
        toks = [f"w{i}" for i in range(L)]
        # make a simple tree: each word attaches to previous, first attaches to ROOT
        heads = [0] + [i for i in range(1, L)]
        sents.append({"tokens": toks, "head": heads})
    return sents

# === tokenization & pooling ===
from transformers import AutoTokenizer, AutoModel
def mean_pool_subwords(last_hidden, word_ids):
    B, S, D = last_hidden.shape
    outs = []
    for b in range(B):
        ids = word_ids[b]
        accum, cnt = {}, {}
        for s, wid in enumerate(ids):
            if wid is None: continue
            if wid not in accum:
                accum[wid] = last_hidden[b, s]
                cnt[wid] = 1
            else:
                accum[wid] += last_hidden[b, s]
                cnt[wid] += 1
        if not accum:
            outs.append(torch.zeros(0, D, device=last_hidden.device))
            continue
        n_words = max(accum.keys()) + 1
        M = torch.zeros(n_words, D, device=last_hidden.device)
        for wid, vec in accum.items():
            M[wid] = vec / cnt[wid]
        outs.append(M)
    return outs

def pad_words(word_batches, pad_value=0.0):
    B = len(word_batches)
    max_len = max(w.size(0) for w in word_batches) if B>0 else 0
    D = word_batches[0].size(1) if max_len>0 else 1
    X = word_batches[0].new_full((B, max_len, D), pad_value)
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=X.device)
    for b, w in enumerate(word_batches):
        L = w.size(0)
        if L>0:
            X[b,:L] = w
            mask[b,:L] = True
    return X, mask

def make_targets(heads, max_len, device):
    B = len(heads)
    Y = torch.zeros(B, max_len, dtype=torch.long, device=device)
    pad = torch.zeros(B, max_len, dtype=torch.bool, device=device)
    for b, h in enumerate(heads):
        n = min(len(h), max_len)
        if n>0:
            Y[b,:n] = torch.tensor(h[:n], device=device)
            pad[b,:n] = True
    return Y, pad

# === biaffine pointer ===
class BiaffinePointer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(torch.empty(d, d)); nn.init.xavier_uniform_(self.W)
        self.U = nn.Linear(d, 1, bias=True)
        self.root = nn.Parameter(torch.zeros(d)); nn.init.normal_(self.root, std=0.02)
    def forward(self, dep, head, mask_dep, mask_head):
        B,T,D = dep.shape
        root = self.root.view(1,1,D).expand(B,1,D)
        heads_all = torch.cat([root, head], dim=1)            # [B,T+1,D]
        bil = (dep @ self.W) @ heads_all.transpose(1,2)       # [B,T,T+1]
        u = self.U(heads_all).squeeze(-1).unsqueeze(1).expand(B,T,T+1)
        logits = bil + u
        C = torch.ones(B, T+1, dtype=torch.bool, device=dep.device); C[:,1:] = mask_head
        logits = logits.masked_fill(~C.unsqueeze(1), float("-inf"))
        logits = logits.masked_fill((~mask_dep).unsqueeze(-1), float("-inf"))
        return logits

# === Baseline block (vanilla MHA) ===
class VanillaBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model), nn.Dropout(dropout))
    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        y,_ = self.attn(h, h, h, need_weights=False, attn_mask=None)  # use key_padding with batch_first?
        x = x + y
        y = self.ff(self.ln2(x))
        return x + y, {}

# === Your pointer-over-heads block ===
from pointer_over_heads_transformer import PointerMoHTransformerBlock

# === Parsers ===
class ParserBase(nn.Module):
    def __init__(self, enc_name, d_model):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(enc_name)
        self.d_model = d_model

class BaselineParser(ParserBase):
    def __init__(self, enc_name="distilbert-base-uncased", d_model=768, n_heads=8, d_ff=2048):
        super().__init__(enc_name, d_model)
        self.block = VanillaBlock(d_model, n_heads, d_ff)
        self.pointer = BiaffinePointer(d_model)
    def forward(self, subw, word_ids, heads_gold):
        device = next(self.parameters()).device
        last_hidden = self.encoder(**subw).last_hidden_state
        words = mean_pool_subwords(last_hidden, word_ids)
        X, mask = pad_words(words)
        X, _ = self.block(X)
        logits = self.pointer(X, X, mask, mask)
        Y, pad = make_targets(heads_gold, X.size(1), device)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1))[pad.view(-1)],
                               Y.view(-1)[pad.view(-1)])
        with torch.no_grad():
            pred = logits.argmax(-1); uas = (pred[pad]==Y[pad]).float().mean().item()
        return loss, {"uas": uas, "tokens": pad.sum().item()}

class PoHParser(ParserBase):
    def __init__(self, enc_name="distilbert-base-uncased", d_model=768, n_heads=8, d_ff=2048,
                 halting_mode="entropy", max_inner_iters=3, routing_topk=2, combination="mask_concat"):
        super().__init__(enc_name, d_model)
        self.block = PointerMoHTransformerBlock(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            halting_mode=halting_mode, max_inner_iters=max_inner_iters,
            min_inner_iters=1, ent_threshold=0.8,
            routing_topk=routing_topk, combination=combination,
            controller_recurrent=True, controller_summary="mean")
        self.pointer = BiaffinePointer(d_model)
    def forward(self, subw, word_ids, heads_gold):
        device = next(self.parameters()).device
        last_hidden = self.encoder(**subw).last_hidden_state
        words = mean_pool_subwords(last_hidden, word_ids)
        X, mask = pad_words(words)
        X, aux = self.block(X, attn_mask=None, return_aux=True)
        logits = self.pointer(X, X, mask, mask)
        Y, pad = make_targets(heads_gold, X.size(1), device)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1))[pad.view(-1)],
                               Y.view(-1)[pad.view(-1)])
        with torch.no_grad():
            pred = logits.argmax(-1); uas = (pred[pad]==Y[pad]).float().mean().item()
        out = {"uas": uas, "tokens": pad.sum().item()}
        if "inner_iters_used" in aux: out["inner_iters_used"] = aux["inner_iters_used"].item()
        return loss, out

# === batching ===
def collate(examples, tokenizer, device):
    # Handle both formats: dict of lists or list of dicts
    if isinstance(examples, dict):
        toks = examples["tokens"]
        heads = examples["head"]
    else:
        toks = [ex["tokens"] for ex in examples]
        heads = [ex["head"] for ex in examples]
    enc = tokenizer(toks, is_split_into_words=True, padding=True, truncation=True, return_tensors="pt", max_length=512)
    word_ids = [enc.word_ids(i) for i in range(len(toks))]
    for k in enc: enc[k] = enc[k].to(device)
    return enc, word_ids, heads

def epoch(model, data, tokenizer, device, bs=8, train=True, lr=5e-5):
    if train:
        model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        model.eval(); opt = None
    from math import ceil
    total_loss, total_tokens, total_correct, total_iters = 0.0, 0, 0, 0.0
    steps = 0
    for i in range(0, len(data), bs):
        batch = data[i:i+bs]
        subw, wids, heads = collate(batch, tokenizer, device)
        loss, m = model(subw, wids, heads)
        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total_loss += loss.item(); steps += 1
        total_tokens += m["tokens"]; total_correct += int(m["uas"]*m["tokens"])
        if "inner_iters_used" in m: total_iters += m["inner_iters_used"]
    return {
        "loss": total_loss/max(1,steps),
        "uas": total_correct/max(1,total_tokens),
        "mean_inner_iters": (total_iters/max(1, len(data)//bs)) if total_iters>0 else float("nan")
    }

def get_data(source, split, conllu_dir=None):
    if source == "hf":
        ds = try_hf_load(split)
        if ds is None: raise RuntimeError("HF load failed; use --data_source conllu or dummy.")
        return list(ds)
    if source == "conllu":
        assert conllu_dir, "Provide --conllu_dir"
        return load_conllu_dir(conllu_dir if split!="validation" else conllu_dir)  # point to your dev dir
    return dummy_ds(128 if split=="train" else 48)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data_source", type=str, default="dummy", choices=["hf","conllu","dummy"])
    ap.add_argument("--conllu_dir", type=str, default=None)
    # baseline/PoH shared
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2048)
    # PoH knobs
    ap.add_argument("--halting_mode", type=str, default="entropy", choices=["fixed","entropy","halting"])
    ap.add_argument("--max_inner_iters", type=int, default=3)
    ap.add_argument("--routing_topk", type=int, default=2)
    ap.add_argument("--combination", type=str, default="mask_concat", choices=["mask_concat","mixture"])
    args = ap.parse_args()

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    d_model = AutoModel.from_pretrained(args.model_name).config.hidden_size

    train = get_data(args.data_source, "train", args.conllu_dir)
    dev   = get_data(args.data_source, "validation", args.conllu_dir)

    baseline = BaselineParser(args.model_name, d_model, args.heads, args.d_ff).to(device)
    poh      = PoHParser(args.model_name, d_model, args.heads, args.d_ff,
                         halting_mode=args.halting_mode,
                         max_inner_iters=args.max_inner_iters,
                         routing_topk=args.routing_topk,
                         combination=args.combination).to(device)

    for ep in range(1, args.epochs+1):
        tr_b = epoch(baseline, train, tokenizer, device, bs=args.batch_size, train=True, lr=args.lr)
        dv_b = epoch(baseline, dev, tokenizer, device, bs=args.batch_size, train=False)
        tr_p = epoch(poh, train, tokenizer, device, bs=args.batch_size, train=True, lr=args.lr)
        dv_p = epoch(poh, dev, tokenizer, device, bs=args.batch_size, train=False)
        print(f"[Epoch {ep}]  BASE  loss {tr_b['loss']:.4f} UAS {tr_b['uas']:.4f} | dev {dv_b['uas']:.4f}")
        print(f"[Epoch {ep}]  PoH   loss {tr_p['loss']:.4f} UAS {tr_p['uas']:.4f} | dev {dv_p['uas']:.4f}  iters {dv_p['mean_inner_iters']:.2f}")

if __name__ == "__main__":
    main()
