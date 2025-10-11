"""
PoT: Pointer-over-Heads Transformer for Dependency Parsing.

Main exports for easy importing.

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

from src.models import PoHParser, BaselineParser, ParserBase

__version__ = "0.1.1"
__author__ = "Eran Ben Artzy"
__all__ = ["PoHParser", "BaselineParser", "ParserBase"]
