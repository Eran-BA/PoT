"""
Setup helper for running PoT experiments in Colab.
Import this at the start of any experiment script to ensure paths are correct.
"""

import sys
import os

def setup_pot_paths():
    """Configure Python paths for PoT imports."""
    
    # Find PoT root
    cwd = os.getcwd()
    pot_root = None
    
    # Check current directory
    if os.path.exists(os.path.join(cwd, 'src', 'pot')):
        pot_root = cwd
    # Check parent directory
    elif os.path.exists(os.path.join(os.path.dirname(cwd), 'src', 'pot')):
        pot_root = os.path.dirname(cwd)
    # Search up tree
    else:
        current = cwd
        for _ in range(10):
            if os.path.exists(os.path.join(current, 'src', 'pot')):
                pot_root = current
                break
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    
    if pot_root is None:
        raise RuntimeError(
            f"Cannot find PoT repository root!\n"
            f"Current directory: {cwd}\n"
            f"Please navigate to PoT directory first: %cd /content/PoT"
        )
    
    # Add to path if not already there
    if pot_root not in sys.path:
        sys.path.insert(0, pot_root)
    
    return pot_root

if __name__ == '__main__':
    root = setup_pot_paths()
    print(f"✓ PoT root configured: {root}")
    print(f"✓ Python path: {sys.path[0]}")

