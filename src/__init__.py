"""
PoT: Pointer-over-Heads Transformer for Dependency Parsing.

Main exports for easy importing.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

__version__ = "0.1.1"
__author__ = "Eran Ben Artzy"

# Lazy imports to avoid requiring transformers for basic PoH usage
def __getattr__(name):
    """Lazy import parser models only when accessed."""
    if name in ("PoHParser", "BaselineParser", "ParserBase"):
        from src.models import PoHParser, BaselineParser, ParserBase
        return {"PoHParser": PoHParser, "BaselineParser": BaselineParser, "ParserBase": ParserBase}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["PoHParser", "BaselineParser", "ParserBase"]
