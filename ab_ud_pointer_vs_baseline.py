#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B Comparison: Baseline vs Pointer-over-Heads Transformer
Dependency parsing on Universal Dependencies with fair comparison setup.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""
import argparse, math, os, glob, csv
from typing import List, Dict, Tuple
from datetime import datetime
import torch, torch.nn as nn, torch.nn.functional as F

# === try HF datasets if requested ===
def try_hf_load(split):
    try:
        from datasets import load_dataset
        # Try multiple UD dataset paths (most likely to work first)
        paths_to_try = [
            ("universal_dependencies", "en_ewt"),  # Standard UD format with config
            ("universal-dependencies/en_ewt", None),
            ("UniversalDependencies/UD_English-EWT", None),
        ]
        
        for path_info in paths_to_try:
            path, config = path_info if isinstance(path_info, tuple) else (path_info, None)
            try:
                print(f"Attempting to load from {path}" + (f" (config: {config})" if config else "") + "...")
                if config:
                    ds = load_dataset(path, config, split=split, trust_remote_code=True)
                else:
                    ds = load_dataset(path, split=split, trust_remote_code=True)
                
                filtered = ds.filter(lambda ex: ex.get("tokens") is not None and ex.get("head") is not None)
                result = list(filtered)
                if len(result) > 0:
                    print(f"✓ Successfully loaded {len(result)} examples from {path}")
                    return result
            except Exception as e:
                print(f"  ✗ Failed: {str(e)[:100]}")
                continue
        
        print(f"\n⚠ Warning: Could not load UD dataset from HuggingFace.")
        print(f"   This is common if:")
        print(f"   - You're offline or have network issues")
        print(f"   - The dataset name/format has changed")
        print(f"   - HuggingFace Hub is temporarily unavailable")
        print(f"\n💡 Solutions:")
        print(f"   1. Use dummy data for testing: --data_source dummy")
        print(f"   2. Download CoNLL-U manually and use: --data_source conllu --conllu_dir /path/to/data")
        print(f"   3. Check HuggingFace status: https://status.huggingface.co/")
        return None
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

# === biaffine label classifier ===
class BiaffineLabeler(nn.Module):
    def __init__(self, d, n_labels):
        super().__init__()
        # Project to smaller dimensions for efficiency
        d_label = d // 2
        self.dep_proj = nn.Linear(d, d_label)
        self.head_proj = nn.Linear(d, d_label)
        # Biaffine scoring for each label
        self.W = nn.Parameter(torch.empty(n_labels, d_label, d_label))
        self.bias = nn.Parameter(torch.zeros(n_labels))
        nn.init.xavier_uniform_(self.W)
    
    def forward(self, dep, head, head_indices, mask):
        """
        dep: [B, T, D] - dependent representations
        head: [B, T, D] - head representations (includes root at position 0)
        head_indices: [B, T] - predicted head indices for each token
        mask: [B, T] - valid token mask
        Returns: [B, T, n_labels] label logits for each token's predicted head
        """
        B, T, D = dep.shape
        
        # Project to label space
        dep_label = self.dep_proj(dep)  # [B, T, d_label]
        head_label = self.head_proj(head)  # [B, T+1, d_label]
        
        # Gather head representations based on predicted indices
        head_indices_expanded = head_indices.unsqueeze(-1).expand(-1, -1, head_label.size(-1))
        selected_heads = torch.gather(head_label, 1, head_indices_expanded)  # [B, T, d_label]
        
        # Biaffine scoring: dep^T W head for each label
        # dep_label: [B, T, d_label] -> [B, T, 1, d_label]
        # W: [n_labels, d_label, d_label]
        # selected_heads: [B, T, d_label] -> [B, T, d_label, 1]
        dep_expanded = dep_label.unsqueeze(2)  # [B, T, 1, d_label]
        head_expanded = selected_heads.unsqueeze(-1)  # [B, T, d_label, 1]
        
        # Compute biaffine scores for all labels
        logits = torch.einsum('btid,nde,btef->btn', dep_expanded, self.W, head_expanded).squeeze(-1)
        logits = logits + self.bias.view(1, 1, -1)  # [B, T, n_labels]
        
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
    def __init__(self, enc_name="distilbert-base-uncased", d_model=768, n_heads=8, d_ff=2048, n_labels=50, use_labels=True):
        super().__init__(enc_name, d_model)
        self.block = VanillaBlock(d_model, n_heads, d_ff)
        self.pointer = BiaffinePointer(d_model)
        self.use_labels = use_labels
        if use_labels:
            self.labeler = BiaffineLabeler(d_model, n_labels)
        self.n_labels = n_labels
        
    def forward(self, subw, word_ids, heads_gold, labels_gold=None):
        device = next(self.parameters()).device
        last_hidden = self.encoder(**subw).last_hidden_state
        words = mean_pool_subwords(last_hidden, word_ids)
        X, mask = pad_words(words)
        X, _ = self.block(X)
        
        # Head prediction
        head_logits = self.pointer(X, X, mask, mask)
        Y_heads, pad = make_targets(heads_gold, X.size(1), device)
        head_loss = F.cross_entropy(head_logits.view(-1, head_logits.size(-1))[pad.view(-1)],
                                     Y_heads.view(-1)[pad.view(-1)])
        
        # Metrics
        with torch.no_grad():
            pred_heads = head_logits.argmax(-1)
            uas = (pred_heads[pad]==Y_heads[pad]).float().mean().item()
        
        # Label prediction
        total_loss = head_loss
        las = uas  # default to UAS if no labels
        pred_labels = None
        
        if self.use_labels and labels_gold is not None:
            # Use gold heads for training, predicted heads for evaluation
            head_indices = Y_heads if self.training else pred_heads
            root_repr = self.pointer.root.view(1, 1, -1).expand(X.size(0), 1, X.size(-1))
            X_with_root = torch.cat([root_repr, X], dim=1)
            
            label_logits = self.labeler(X, X_with_root, head_indices, mask)
            Y_labels, label_pad = make_targets(labels_gold, X.size(1), device)
            label_loss = F.cross_entropy(label_logits.view(-1, label_logits.size(-1))[pad.view(-1)],
                                         Y_labels.view(-1)[pad.view(-1)])
            total_loss = head_loss + label_loss
            
            with torch.no_grad():
                pred_labels = label_logits.argmax(-1)
                las = ((pred_heads[pad]==Y_heads[pad]) & (pred_labels[pad]==Y_labels[pad])).float().mean().item()
        
        out = {"uas": uas, "las": las, "tokens": pad.sum().item()}
        # Add predictions for CoNLL-U export
        if not self.training:
            out["pred_heads"] = [pred_heads[b, :pad[b].sum()].cpu().tolist() for b in range(pred_heads.size(0))]
            if pred_labels is not None:
                out["pred_labels"] = [pred_labels[b, :pad[b].sum()].cpu().tolist() for b in range(pred_labels.size(0))]
        return total_loss, out

class PoHParser(ParserBase):
    def __init__(self, enc_name="distilbert-base-uncased", d_model=768, n_heads=8, d_ff=2048,
                 halting_mode="entropy", max_inner_iters=3, routing_topk=2, combination="mask_concat",
                 n_labels=50, use_labels=True):
        super().__init__(enc_name, d_model)
        self.block = PointerMoHTransformerBlock(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            halting_mode=halting_mode, max_inner_iters=max_inner_iters,
            min_inner_iters=1, ent_threshold=0.8,
            routing_topk=routing_topk, combination=combination,
            controller_recurrent=True, controller_summary="mean")
        self.pointer = BiaffinePointer(d_model)
        self.use_labels = use_labels
        if use_labels:
            self.labeler = BiaffineLabeler(d_model, n_labels)
        self.n_labels = n_labels
        
    def forward(self, subw, word_ids, heads_gold, labels_gold=None):
        device = next(self.parameters()).device
        last_hidden = self.encoder(**subw).last_hidden_state
        words = mean_pool_subwords(last_hidden, word_ids)
        X, mask = pad_words(words)
        X, aux = self.block(X, attn_mask=None, return_aux=True)
        
        # Head prediction
        head_logits = self.pointer(X, X, mask, mask)
        Y_heads, pad = make_targets(heads_gold, X.size(1), device)
        head_loss = F.cross_entropy(head_logits.view(-1, head_logits.size(-1))[pad.view(-1)],
                                     Y_heads.view(-1)[pad.view(-1)])
        
        # Metrics
        with torch.no_grad():
            pred_heads = head_logits.argmax(-1)
            uas = (pred_heads[pad]==Y_heads[pad]).float().mean().item()
        
        # Label prediction
        total_loss = head_loss
        las = uas  # default to UAS if no labels
        pred_labels = None
        
        if self.use_labels and labels_gold is not None:
            # Use gold heads for training, predicted heads for evaluation
            head_indices = Y_heads if self.training else pred_heads
            root_repr = self.pointer.root.view(1, 1, -1).expand(X.size(0), 1, X.size(-1))
            X_with_root = torch.cat([root_repr, X], dim=1)
            
            label_logits = self.labeler(X, X_with_root, head_indices, mask)
            Y_labels, label_pad = make_targets(labels_gold, X.size(1), device)
            label_loss = F.cross_entropy(label_logits.view(-1, label_logits.size(-1))[pad.view(-1)],
                                         Y_labels.view(-1)[pad.view(-1)])
            total_loss = head_loss + label_loss
            
            with torch.no_grad():
                pred_labels = label_logits.argmax(-1)
                las = ((pred_heads[pad]==Y_heads[pad]) & (pred_labels[pad]==Y_labels[pad])).float().mean().item()
        
        out = {"uas": uas, "las": las, "tokens": pad.sum().item()}
        if "inner_iters_used" in aux: out["inner_iters_used"] = aux["inner_iters_used"].item()
        # Add predictions for CoNLL-U export
        if not self.training:
            out["pred_heads"] = [pred_heads[b, :pad[b].sum()].cpu().tolist() for b in range(pred_heads.size(0))]
            if pred_labels is not None:
                out["pred_labels"] = [pred_labels[b, :pad[b].sum()].cpu().tolist() for b in range(pred_labels.size(0))]
        return total_loss, out

# === batching ===
def collate(examples, tokenizer, device, label_vocab=None, return_deprels=False):
    # Handle both formats: dict of lists or list of dicts
    if isinstance(examples, dict):
        toks = examples["tokens"]
        heads = examples["head"]
        labels = examples.get("deprel", None)
    else:
        toks = [ex["tokens"] for ex in examples]
        heads = [ex["head"] for ex in examples]
        labels = [ex.get("deprel", [0]*len(ex["tokens"])) for ex in examples] if any("deprel" in ex for ex in examples) else None
    
    enc = tokenizer(toks, is_split_into_words=True, padding=True, truncation=True, return_tensors="pt", max_length=512)
    word_ids = [enc.word_ids(i) for i in range(len(toks))]
    for k in enc: enc[k] = enc[k].to(device)
    
    # Keep original string labels for punctuation masking
    deprels_str = labels if (labels is not None and any(isinstance(lbl, str) for sent in labels for lbl in (sent if isinstance(sent, list) else [sent]))) else None
    
    # Convert string labels to indices if label_vocab provided
    if labels is not None and label_vocab is not None:
        label_indices = []
        for sent_labels in labels:
            indices = [label_vocab.get(lbl, 0) if isinstance(lbl, str) else lbl for lbl in sent_labels]
            label_indices.append(indices)
        labels = label_indices
    
    if return_deprels:
        return enc, word_ids, heads, labels, deprels_str
    return enc, word_ids, heads, labels

def build_label_vocab(data):
    """Build label vocabulary from data."""
    labels = set()
    for ex in data:
        if "deprel" in ex:
            for lbl in ex["deprel"]:
                if isinstance(lbl, str):
                    labels.add(lbl)
    vocab = {lbl: idx for idx, lbl in enumerate(sorted(labels))}
    vocab["<UNK>"] = len(vocab)  # unknown label
    return vocab

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup then linear decay."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)

def epoch(model, data, tokenizer, device, label_vocab=None, bs=8, train=True, lr=5e-5, weight_decay=0.01, scheduler=None, 
          emit_conllu=False, conllu_path=None, ignore_punct=False):
    import time
    if train:
        model.train()
        if not hasattr(epoch, 'optimizer'):
            epoch.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        opt = epoch.optimizer
    else:
        model.eval(); opt = None
    from math import ceil
    total_loss, total_tokens, total_correct_uas, total_correct_las, total_iters = 0.0, 0, 0, 0, 0.0
    total_routing_entropy = 0.0
    steps = 0
    start_time = time.time()
    
    # For CoNLL-U export
    all_tokens, all_heads_gold, all_deprels_gold, all_heads_pred, all_deprels_pred = [], [], [], [], []
    
    for i in range(0, len(data), bs):
        batch = data[i:i+bs]
        if ignore_punct:
            subw, wids, heads, labels, deprels_str = collate(batch, tokenizer, device, label_vocab, return_deprels=True)
        else:
            subw, wids, heads, labels = collate(batch, tokenizer, device, label_vocab, return_deprels=False)
            deprels_str = None
        
        loss, m = model(subw, wids, heads, labels)
        
        # Apply punctuation masking if requested
        if ignore_punct and deprels_str is not None and "pred_heads" in m:
            from utils.metrics import build_masks_for_metrics
            _, is_eval = build_masks_for_metrics(heads, deprels_str)
            # Recompute UAS/LAS with punctuation mask
            # This requires access to predictions which are only in eval mode
            # For now, we'll keep the standard metrics and note this is a TODO for full integration
            pass
        
        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if scheduler is not None:
                scheduler.step()
        total_loss += loss.item(); steps += 1
        total_tokens += m["tokens"]
        total_correct_uas += int(m["uas"]*m["tokens"])
        total_correct_las += int(m.get("las", m["uas"])*m["tokens"])
        if "inner_iters_used" in m: total_iters += m["inner_iters_used"]
        if "routing_entropy" in m: total_routing_entropy += m["routing_entropy"]
        
        # Collect predictions for CoNLL-U export
        if emit_conllu and not train:
            # Extract token lists from batch
            if isinstance(batch, dict):
                batch_tokens = batch["tokens"]
                batch_heads = batch["head"]
                batch_deprels = batch.get("deprel", None)
            else:
                batch_tokens = [ex["tokens"] for ex in batch]
                batch_heads = [ex["head"] for ex in batch]
                batch_deprels = [ex.get("deprel", None) for ex in batch]
            
            all_tokens.extend(batch_tokens)
            all_heads_gold.extend(batch_heads)
            if batch_deprels:
                all_deprels_gold.extend(batch_deprels)
            
            # Get predictions from model output
            if "pred_heads" in m:
                all_heads_pred.extend(m["pred_heads"])
            if "pred_labels" in m:
                all_deprels_pred.extend(m["pred_labels"])
    
    # Write CoNLL-U if requested
    if emit_conllu and not train and conllu_path and all_tokens:
        from utils.conllu_writer import write_conllu
        write_conllu(
            conllu_path,
            tokens=all_tokens,
            heads_gold=all_heads_gold if all_heads_gold else None,
            deprels_gold=all_deprels_gold if all_deprels_gold else None,
            heads_pred=all_heads_pred if all_heads_pred else None,
            deprels_pred=all_deprels_pred if all_deprels_pred else None
        )
        print(f"✓ Wrote predictions to {conllu_path}")
    
    elapsed = time.time() - start_time
    return {
        "loss": total_loss/max(1,steps),
        "uas": total_correct_uas/max(1,total_tokens),
        "las": total_correct_las/max(1,total_tokens),
        "mean_inner_iters": (total_iters/max(1, steps)) if total_iters>0 else float("nan"),
        "routing_entropy": (total_routing_entropy/max(1, steps)) if total_routing_entropy>0 else float("nan"),
        "time": elapsed,
        "steps": steps
    }

def get_data(source, split, conllu_dir=None):
    if source == "hf":
        ds = try_hf_load(split)
        if ds is None:
            raise RuntimeError(
                "\n❌ HuggingFace dataset loading failed!\n\n"
                "📥 Please download UD English EWT manually:\n\n"
                "Option 1 - Direct download (in Colab/terminal):\n"
                "  wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu\n"
                "  wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu\n"
                "  wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu\n"
                "  mkdir -p ud_data && mv en_ewt-ud-*.conllu ud_data/\n\n"
                "Option 2 - Clone full repository:\n"
                "  git clone https://github.com/UniversalDependencies/UD_English-EWT.git\n\n"
                "Then re-run with:\n"
                "  --data_source conllu --conllu_dir ud_data/\n\n"
                "Or use dummy data for quick testing:\n"
                "  --data_source dummy\n"
            )
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
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio (default: 0.05 = 5%%)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data_source", type=str, default="hf", choices=["hf","conllu","dummy"])
    ap.add_argument("--conllu_dir", type=str, default=None)
    ap.add_argument("--log_csv", type=str, default=None, help="CSV file to log results (auto-generated if not provided)")
    # Evaluation options
    ap.add_argument("--ignore_punct", action="store_true", help="Ignore punctuation in UAS/LAS computation")
    ap.add_argument("--emit_conllu", action="store_true", help="Write predictions to CoNLL-U format")
    ap.add_argument("--freeze_encoder", action="store_true", help="Freeze pretrained encoder (only train parsing head)")
    # Parameter matching
    ap.add_argument("--param_match", type=str, default=None, choices=[None, "baseline", "poh"], 
                    help="Match parameter counts by adjusting FFN size")
    # baseline/PoH shared
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--d_ff", type=int, default=2048)
    # PoH knobs
    ap.add_argument("--halting_mode", type=str, default="entropy", choices=["fixed","entropy","halting"])
    ap.add_argument("--max_inner_iters", type=int, default=3)
    ap.add_argument("--routing_topk", type=int, default=2)
    ap.add_argument("--combination", type=str, default="mask_concat", choices=["mask_concat","mixture"])
    args = ap.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    d_model = AutoModel.from_pretrained(args.model_name).config.hidden_size

    print(f"\n{'='*80}")
    print(f"Loading data source: {args.data_source}")
    print(f"{'='*80}")
    
    train = get_data(args.data_source, "train", args.conllu_dir)
    dev   = get_data(args.data_source, "validation", args.conllu_dir)
    
    print(f"✓ Train set: {len(train)} examples")
    print(f"✓ Dev set:   {len(dev)} examples")
    
    # Sample check
    if len(train) > 0:
        sample = train[0]
        has_labels = "deprel" in sample and sample["deprel"]
        print(f"✓ Sample fields: {list(sample.keys())}")
        print(f"✓ Dependency labels: {'present' if has_labels else 'absent'}")
    print(f"{'='*80}\n")
    
    # Build label vocabulary from training data
    label_vocab = build_label_vocab(train)
    n_labels = len(label_vocab)
    use_labels = n_labels > 0
    
    # Parameter matching: adjust FFN sizes to equalize param counts
    baseline_d_ff = args.d_ff
    poh_d_ff = args.d_ff
    
    if args.param_match:
        # Create temporary models to measure params
        temp_baseline = BaselineParser(args.model_name, d_model, args.heads, args.d_ff, 
                                       n_labels=max(n_labels, 50), use_labels=use_labels)
        temp_poh = PoHParser(args.model_name, d_model, args.heads, args.d_ff,
                            halting_mode=args.halting_mode, max_inner_iters=args.max_inner_iters,
                            routing_topk=args.routing_topk, combination=args.combination,
                            n_labels=max(n_labels, 50), use_labels=use_labels)
        
        baseline_params = sum(p.numel() for p in temp_baseline.parameters())
        poh_params = sum(p.numel() for p in temp_poh.parameters())
        delta = poh_params - baseline_params
        
        if args.param_match == "baseline" and delta > 0:
            # Boost baseline FFN to match PoH
            # Rough estimate: each FFN layer adds 2*d_model*d_ff params
            # So increase d_ff by delta / (4*d_model) approximately
            baseline_d_ff = int(args.d_ff + delta / (4 * d_model))
            print(f"⚙ Parameter matching: Boosting baseline d_ff to {baseline_d_ff} (from {args.d_ff})")
        elif args.param_match == "poh" and delta > 0:
            # Shrink PoH FFN
            poh_d_ff = int(args.d_ff - delta / (4 * d_model))
            print(f"⚙ Parameter matching: Reducing PoH d_ff to {poh_d_ff} (from {args.d_ff})")
    
    baseline = BaselineParser(args.model_name, d_model, args.heads, baseline_d_ff, 
                              n_labels=max(n_labels, 50), use_labels=use_labels).to(device)
    poh      = PoHParser(args.model_name, d_model, args.heads, poh_d_ff,
                         halting_mode=args.halting_mode,
                         max_inner_iters=args.max_inner_iters,
                         routing_topk=args.routing_topk,
                         combination=args.combination,
                         n_labels=max(n_labels, 50), use_labels=use_labels).to(device)
    
    # Freeze encoder if requested
    if args.freeze_encoder:
        print(f"⚙ Freezing encoder parameters")
        for param in baseline.encoder.parameters():
            param.requires_grad = False
        for param in poh.encoder.parameters():
            param.requires_grad = False

    # Count parameters
    baseline_params = sum(p.numel() for p in baseline.parameters())
    poh_params = sum(p.numel() for p in poh.parameters())
    
    print(f"\n{'='*80}")
    print(f"Config: epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}, wd={args.weight_decay}, warmup={args.warmup_ratio}, seed={args.seed}")
    print(f"Data: {args.data_source}, Train size: {len(train)}, Dev size: {len(dev)}")
    print(f"Label vocab size: {n_labels} (LAS {'enabled' if use_labels else 'disabled'})")
    print(f"Baseline params: {baseline_params:,}")
    print(f"PoH params:      {poh_params:,} (+{poh_params-baseline_params:,})")
    print(f"{'='*80}\n")
    
    # CSV logging setup
    csv_file = args.log_csv
    if csv_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"training_log_{timestamp}.csv"
    
    csv_exists = os.path.exists(csv_file)
    csv_fp = open(csv_file, 'a', newline='')
    csv_writer = csv.DictWriter(csv_fp, fieldnames=[
        'timestamp', 'seed', 'epoch', 'model', 'data_source', 
        'train_loss', 'train_uas', 'train_las', 'train_time',
        'dev_uas', 'dev_las', 'dev_time',
        'mean_inner_iters', 'routing_entropy',
        'params', 'lr', 'wd', 'bs', 'warmup',
        'halting_mode', 'max_inner_iters', 'routing_topk', 'combination'
    ])
    if not csv_exists:
        csv_writer.writeheader()
    
    print(f"Logging results to: {csv_file}\n")
    
    # Warmup schedulers
    total_steps_baseline = (len(train) // args.batch_size) * args.epochs
    total_steps_poh = (len(train) // args.batch_size) * args.epochs
    warmup_steps = int(total_steps_baseline * args.warmup_ratio)
    
    for ep in range(1, args.epochs+1):
        tr_b = epoch(baseline, train, tokenizer, device, label_vocab, bs=args.batch_size, train=True, lr=args.lr, weight_decay=args.weight_decay)
        dv_b = epoch(baseline, dev, tokenizer, device, label_vocab, bs=args.batch_size, train=False,
                     emit_conllu=args.emit_conllu, conllu_path=f"baseline_pred_dev_ep{ep}.conllu", ignore_punct=args.ignore_punct)
        tr_p = epoch(poh, train, tokenizer, device, label_vocab, bs=args.batch_size, train=True, lr=args.lr, weight_decay=args.weight_decay)
        dv_p = epoch(poh, dev, tokenizer, device, label_vocab, bs=args.batch_size, train=False,
                     emit_conllu=args.emit_conllu, conllu_path=f"poh_pred_dev_ep{ep}.conllu", ignore_punct=args.ignore_punct)
        
        # Log to CSV
        timestamp = datetime.now().isoformat()
        csv_writer.writerow({
            'timestamp': timestamp, 'seed': args.seed, 'epoch': ep, 'model': 'Baseline', 'data_source': args.data_source,
            'train_loss': f"{tr_b['loss']:.4f}", 'train_uas': f"{tr_b['uas']:.4f}", 'train_las': f"{tr_b['las']:.4f}", 'train_time': f"{tr_b['time']:.1f}",
            'dev_uas': f"{dv_b['uas']:.4f}", 'dev_las': f"{dv_b['las']:.4f}", 'dev_time': f"{dv_b['time']:.1f}",
            'mean_inner_iters': '', 'routing_entropy': '',
            'params': baseline_params, 'lr': args.lr, 'wd': args.weight_decay, 'bs': args.batch_size, 'warmup': args.warmup_ratio,
            'halting_mode': '', 'max_inner_iters': '', 'routing_topk': '', 'combination': ''
        })
        csv_writer.writerow({
            'timestamp': timestamp, 'seed': args.seed, 'epoch': ep, 'model': 'PoH', 'data_source': args.data_source,
            'train_loss': f"{tr_p['loss']:.4f}", 'train_uas': f"{tr_p['uas']:.4f}", 'train_las': f"{tr_p['las']:.4f}", 'train_time': f"{tr_p['time']:.1f}",
            'dev_uas': f"{dv_p['uas']:.4f}", 'dev_las': f"{dv_p['las']:.4f}", 'dev_time': f"{dv_p['time']:.1f}",
            'mean_inner_iters': f"{dv_p['mean_inner_iters']:.2f}" if not math.isnan(dv_p['mean_inner_iters']) else '',
            'routing_entropy': f"{dv_p['routing_entropy']:.3f}" if not math.isnan(dv_p.get('routing_entropy', float('nan'))) else '',
            'params': poh_params, 'lr': args.lr, 'wd': args.weight_decay, 'bs': args.batch_size, 'warmup': args.warmup_ratio,
            'halting_mode': args.halting_mode, 'max_inner_iters': args.max_inner_iters, 
            'routing_topk': args.routing_topk, 'combination': args.combination
        })
        csv_fp.flush()
        
        # Console output
        if use_labels:
            print(f"[Epoch {ep}]  BASE  train loss {tr_b['loss']:.4f} UAS {tr_b['uas']:.4f} LAS {tr_b['las']:.4f} ({tr_b['time']:.1f}s) | dev UAS {dv_b['uas']:.4f} LAS {dv_b['las']:.4f} ({dv_b['time']:.1f}s)")
            ent_str = f" ent {dv_p['routing_entropy']:.3f}" if not math.isnan(dv_p.get('routing_entropy', float('nan'))) else ""
            print(f"[Epoch {ep}]  PoH   train loss {tr_p['loss']:.4f} UAS {tr_p['uas']:.4f} LAS {tr_p['las']:.4f} ({tr_p['time']:.1f}s) | dev UAS {dv_p['uas']:.4f} LAS {dv_p['las']:.4f} ({dv_p['time']:.1f}s) iters {dv_p['mean_inner_iters']:.2f}{ent_str}")
        else:
            print(f"[Epoch {ep}]  BASE  train loss {tr_b['loss']:.4f} UAS {tr_b['uas']:.4f} ({tr_b['time']:.1f}s) | dev UAS {dv_b['uas']:.4f} ({dv_b['time']:.1f}s)")
            ent_str = f" ent {dv_p['routing_entropy']:.3f}" if not math.isnan(dv_p.get('routing_entropy', float('nan'))) else ""
            print(f"[Epoch {ep}]  PoH   train loss {tr_p['loss']:.4f} UAS {tr_p['uas']:.4f} ({tr_p['time']:.1f}s) | dev UAS {dv_p['uas']:.4f} ({dv_p['time']:.1f}s) iters {dv_p['mean_inner_iters']:.2f}{ent_str}")
        print()
    
    csv_fp.close()
    print(f"\n✓ Results logged to: {csv_file}")

if __name__ == "__main__":
    main()
