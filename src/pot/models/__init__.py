"""
PoT model implementations.

This module exposes the public model surface lazily so submodules can be
imported safely when `pot` is installed as a dependency.
"""

from importlib import import_module


_EXPORTS = {
    "PoHGPT": (".poh_gpt", "PoHGPT"),
    "ReasoningModule": (".reasoning_module", "ReasoningModule"),
    "HybridHRMBase": (".hybrid_hrm", "HybridHRMBase"),
    "ACTCarry": (".hybrid_hrm", "ACTCarry"),
    "PoTAsyncCarry": (".hybrid_hrm", "PoTAsyncCarry"),
    "PuzzleEmbedding": (".puzzle_embedding", "PuzzleEmbedding"),
    "RMSNorm": (".hrm_layers", "RMSNorm"),
    "SwiGLU": (".hrm_layers", "SwiGLU"),
    "QHaltingController": (".adaptive_halting", "QHaltingController"),
    "PoHSudokuSolver": (".sudoku_solver", "PoHSudokuSolver"),
    "HybridPoHHRMSolver": (".sudoku_solver", "HybridPoHHRMSolver"),
    "BaselineSudokuSolver": (".sudoku_solver", "BaselineSudokuSolver"),
    "HybridPoHHRMForNLI": (".hybrid_nli", "HybridPoHHRMForNLI"),
    "HybridPoHARCSolver": (".arc_solver", "HybridPoHARCSolver"),
    "BaselineARCSolver": (".arc_solver", "BaselineARCSolver"),
    "HybridPoTBlocksworldSolver": (".blocksworld_solver", "HybridPoTBlocksworldSolver"),
    "BaselineBlocksworldSolver": (".blocksworld_solver", "BaselineBlocksworldSolver"),
    "SimplePoTBlocksworldSolver": (".blocksworld_solver", "SimplePoTBlocksworldSolver"),
    "BlocksworldCritic": (".blocksworld_critic", "BlocksworldCritic"),
    "BlocksworldActorCritic": (".blocksworld_critic", "BlocksworldActorCritic"),
    "create_actor_critic": (".blocksworld_critic", "create_actor_critic"),
    "PoTSokobanSolver": (".sokoban_solver", "PoTSokobanSolver"),
    "HybridPoTSokobanSolver": (".sokoban_solver", "HybridPoTSokobanSolver"),
    "BaselineSokobanSolver": (".sokoban_solver", "BaselineSokobanSolver"),
    "SokobanActorCritic": (".sokoban_solver", "SokobanActorCritic"),
    "SokobanConvEncoder": (".sokoban_solver", "SokobanConvEncoder"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value

