import torch
from transformers import AutoModel
from pointer_over_heads_transformer import PointerMoHTransformerBlock
# from your A/B file:
from ab_ud_pointer_vs_baseline import BaselineParser, PoHParser

def count_params(model, include_encoder=True):
    if include_encoder:
        return sum(p.numel() for p in model.parameters())
    # exclude encoder params
    enc_params = set(id(p) for p in model.encoder.parameters())
    return sum(p.numel() for p in model.parameters() if id(p) not in enc_params)

device = "cuda" if torch.cuda.is_available() else "cpu"
d_model = AutoModel.from_pretrained("distilbert-base-uncased").config.hidden_size

baseline = BaselineParser(d_model=d_model).to(device)
poh = PoHParser(d_model=d_model, max_inner_iters=3, routing_topk=2).to(device)

print("TOTAL params (Baseline):", count_params(baseline, True))
print("TOTAL params (PoH):     ", count_params(poh, True))
print("NON-ENC params (Baseline):", count_params(baseline, False))
print("NON-ENC params (PoH):     ", count_params(poh, False))

