"""
Dataset loading for Universal Dependencies.

Supports multiple data sources:
- HuggingFace datasets (automatic download)
- Local CoNLL-U files
- Synthetic dummy data for testing

Functions:
    load_hf_dataset: Load UD English EWT from HuggingFace
    load_conllu_files: Load from local CoNLL-U files
    create_dummy_dataset: Generate synthetic dependency data
    get_dataset: Unified interface for all data sources
    build_label_vocab: Build dependency label vocabulary

Author: Eran Ben Artzy
Year: 2025
License: Apache 2.0
"""

import os
import glob
import random
from typing import List, Dict, Optional

def load_hf_dataset(split: str) -> Optional[List[Dict]]:
    """Load UD English EWT dataset from HuggingFace.
    
    Tries multiple dataset paths to handle different formats and naming schemes.
    Filters out examples without required fields (tokens, heads).
    
    Args:
        split: Dataset split ('train', 'validation', or 'test')
        
    Returns:
        List of examples with 'tokens', 'head', and optionally 'deprel' fields.
        Returns None if loading fails.
        
    Example:
        >>> train_data = load_hf_dataset('train')
        >>> if train_data:
        ...     print(f"Loaded {len(train_data)} training examples")
        >>> else:
        ...     print("Failed to load HuggingFace dataset")
    """
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
                print(f"Attempting to load from {path}" + 
                      (f" (config: {config})" if config else "") + "...")
                if config:
                    ds = load_dataset(path, config, split=split, trust_remote_code=True)
                else:
                    ds = load_dataset(path, split=split, trust_remote_code=True)
                
                # Filter valid examples
                filtered = ds.filter(
                    lambda ex: ex.get("tokens") is not None and ex.get("head") is not None
                )
                result = list(filtered)
                
                if len(result) > 0:
                    print(f"✓ Successfully loaded {len(result)} examples from {path}")
                    return result
            except Exception as e:
                print(f"  ✗ Failed: {str(e)[:100]}")
                continue
        
        # All paths failed
        print(f"\n⚠ Warning: Could not load UD dataset from HuggingFace.")
        print(f"   This is common if:")
        print(f"   - You're offline or have network issues")
        print(f"   - The dataset name/format has changed")
        print(f"   - HuggingFace Hub is temporarily unavailable")
        print(f"\n💡 Solutions:")
        print(f"   1. Use dummy data for testing: --data_source dummy")
        print(f"   2. Download CoNLL-U manually and use: "
              f"--data_source conllu --conllu_dir /path/to/data")
        print(f"   3. Check HuggingFace status: https://status.huggingface.co/")
        return None
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def load_conllu_files(path: str) -> List[Dict]:
    """Load dependency trees from local CoNLL-U files.
    
    Reads all .conllu files in the specified directory and parses them
    into the standard format with tokens, heads, and relation labels.
    
    Args:
        path: Directory containing .conllu files
        
    Returns:
        List of examples with 'tokens', 'head', and 'deprel' fields
        
    Example:
        >>> data = load_conllu_files('data/')
        >>> print(f"Loaded {len(data)} sentences")
        >>> print(data[0].keys())  # dict_keys(['tokens', 'head', 'deprel'])
        
    Note:
        - Skips multiword token rows (where id is a tuple)
        - Only includes complete sentences with matching tokens/heads
        - Processes files in alphabetical order
    """
    from conllu import parse_incr
    
    samples = []
    for fp in sorted(glob.glob(os.path.join(path, "*.conllu"))):
        with open(fp, "r", encoding="utf-8") as f:
            for sent in parse_incr(f):
                tokens, heads, deprels = [], [], []
                for tok in sent:
                    # Skip multiword token rows
                    if isinstance(tok["id"], tuple):
                        continue
                    tokens.append(tok["form"])
                    heads.append(int(tok["head"]))
                    deprels.append(tok.get("deprel", "dep"))  # Default to 'dep'
                
                # Only include complete sentences
                if tokens and heads and len(tokens) == len(heads):
                    samples.append({
                        "tokens": tokens,
                        "head": heads,
                        "deprel": deprels
                    })
    
    return samples


def create_dummy_dataset(n_samples: int = 64) -> List[Dict]:
    """Generate synthetic dependency data for testing.
    
    Creates simple left-branching dependency trees for rapid development
    and testing without requiring real data.
    
    Args:
        n_samples: Number of sentences to generate
        
    Returns:
        List of synthetic examples with 'tokens' and 'head' fields
        
    Example:
        >>> train_data = create_dummy_dataset(n_samples=100)
        >>> dev_data = create_dummy_dataset(n_samples=50)
        >>> print(train_data[0])
        # {'tokens': ['w0', 'w1', ...], 'head': [0, 1, 2, ...]}
        
    Note:
        Uses fixed random seed (0) for reproducibility.
        Trees are simple left-branching: each word attaches to previous,
        first word attaches to ROOT.
    """
    rng = random.Random(0)
    sents = []
    
    for _ in range(n_samples):
        L = rng.randint(5, 12)  # Sentence length
        toks = [f"w{i}" for i in range(L)]
        
        # Simple left-branching tree
        # First word attaches to ROOT (0), others attach to previous word
        heads = [0] + [i for i in range(1, L)]
        
        sents.append({"tokens": toks, "head": heads})
    
    return sents


def get_dataset(
    source: str,
    split: str,
    conllu_dir: Optional[str] = None
) -> List[Dict]:
    """Unified interface for loading dependency parsing data.
    
    Args:
        source: Data source ('hf', 'conllu', or 'dummy')
        split: Dataset split ('train', 'validation', or 'test')
        conllu_dir: Directory for CoNLL-U files (required if source='conllu')
        
    Returns:
        List of examples with dependency annotations
        
    Raises:
        RuntimeError: If HuggingFace loading fails
        AssertionError: If conllu_dir not provided for source='conllu'
        
    Example:
        >>> # Try HuggingFace first
        >>> try:
        ...     train_data = get_dataset('hf', 'train')
        ... except RuntimeError:
        ...     # Fallback to dummy data
        ...     train_data = get_dataset('dummy', 'train')
        
        >>> # Use local CoNLL-U files
        >>> train_data = get_dataset('conllu', 'train', conllu_dir='data/')
    """
    if source == "hf":
        ds = load_hf_dataset(split)
        if ds is None:
            raise RuntimeError(
                "\n❌ HuggingFace dataset loading failed!\n\n"
                "📥 Please download UD English EWT manually:\n\n"
                "Option 1 - Direct download (in Colab/terminal):\n"
                "  wget https://raw.githubusercontent.com/UniversalDependencies/"
                "UD_English-EWT/master/en_ewt-ud-train.conllu\n"
                "  wget https://raw.githubusercontent.com/UniversalDependencies/"
                "UD_English-EWT/master/en_ewt-ud-dev.conllu\n"
                "  wget https://raw.githubusercontent.com/UniversalDependencies/"
                "UD_English-EWT/master/en_ewt-ud-test.conllu\n"
                "  mkdir -p data && mv en_ewt-ud-*.conllu data/\n\n"
                "Option 2 - Clone full repository:\n"
                "  git clone https://github.com/UniversalDependencies/UD_English-EWT.git\n\n"
                "Then re-run with:\n"
                "  --data_source conllu --conllu_dir data/\n\n"
                "Or use dummy data for quick testing:\n"
                "  --data_source dummy\n"
            )
        return list(ds)
    
    if source == "conllu":
        assert conllu_dir, "Provide --conllu_dir when using source='conllu'"
        return load_conllu_files(conllu_dir)
    
    # Default: dummy data
    return create_dummy_dataset(128 if split == "train" else 48)


def build_label_vocab(data: List[Dict]) -> Dict[str, int]:
    """Build vocabulary mapping dependency labels to indices.
    
    Scans all examples for unique dependency relation labels and creates
    a sorted vocabulary. Adds a special <UNK> token for unknown labels.
    
    Args:
        data: List of examples with 'deprel' field
        
    Returns:
        Dictionary mapping label strings to integer indices
        
    Example:
        >>> train_data = load_conllu_files('data/')
        >>> label_vocab = build_label_vocab(train_data)
        >>> print(f"Found {len(label_vocab)} unique labels")
        >>> print(label_vocab['nsubj'])  # e.g., 15
        >>> print(label_vocab['<UNK>'])  # Last index
        
    Note:
        Labels are sorted alphabetically before indexing for consistency.
        The <UNK> token is always placed at the end.
    """
    labels = set()
    for ex in data:
        if "deprel" in ex:
            for lbl in ex["deprel"]:
                if isinstance(lbl, str):
                    labels.add(lbl)
    
    # Create sorted vocabulary
    vocab = {lbl: idx for idx, lbl in enumerate(sorted(labels))}
    vocab["<UNK>"] = len(vocab)  # Unknown label
    
    return vocab

