"""
Base parser class with shared encoder.

Provides common encoder initialization for all dependency parsers.

Classes:
    ParserBase: Abstract base class for dependency parsers

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import torch.nn as nn
from transformers import AutoModel


class ParserBase(nn.Module):
    """Base class for dependency parsers.

    Provides shared encoder initialization and word pooling functionality.
    All parser implementations (baseline, PoH) inherit from this class.

    Args:
        enc_name: HuggingFace model identifier (e.g., "distilbert-base-uncased")
        d_model: Model hidden dimension

    Attributes:
        encoder: Pretrained transformer encoder from HuggingFace
        d_model: Hidden dimension size

    Example:
        >>> # Not used directly, but subclassed by specific parsers
        >>> class MyParser(ParserBase):
        ...     def __init__(self, enc_name="distilbert-base-uncased"):
        ...         super().__init__(enc_name, d_model=768)
        ...         # Add parser-specific components

    Note:
        This is an abstract base class. Subclasses must implement the
        forward() method for their specific parsing strategy.
    """

    def __init__(self, enc_name: str, d_model: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(enc_name)
        self.d_model = d_model
