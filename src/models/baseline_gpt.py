"""
Baseline GPT implementation for A/B testing against PoHGPT.

Standard GPT-2-style autoregressive decoder with causal masking.

Author: Eran Ben Artzy
Year: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTBlock(nn.Module):
    """Standard GPT Transformer block (causal self-attention + FFN)."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None):
        # causal self-attn with residual
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(
            x, x, x, attn_mask=attn_mask, need_weights=False, is_causal=(attn_mask is not None)
        )
        x = residual + attn_out
        
        # feed-forward with residual
        residual = x
        x = self.ln2(x)
        x = residual + self.ff(x)
        return x


class BaselineGPT(nn.Module):
    """Baseline GPT-like autoregressive model for A/B testing vs PoHGPT."""
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        depth=6,
        dropout=0.1,
        max_seq_len=512,
        pos_encoding="absolute",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, n_heads, d_ff, dropout) for _ in range(depth)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        B, T = input_ids.size()
        assert T <= self.max_seq_len, f"seq_len {T} > {self.max_seq_len}"
        
        positions = torch.arange(T, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(positions)[None, :, :]

        attn_mask = torch.triu(
            torch.ones(T, T, device=input_ids.device), 1
        ).bool()  # causal mask

        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=20, temperature=1.0, top_k=None):
        """Autoregressive generation."""
        self.eval()
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, ix = torch.topk(next_token_logits, top_k)
                mask = next_token_logits < v[:, [-1]]
                next_token_logits[mask] = -float("Inf")
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids

