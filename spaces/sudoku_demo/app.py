#!/usr/bin/env python3
"""
Sudoku Solver Demo - HuggingFace Spaces Gradio App

This app loads the 78.9% accuracy PoT Sudoku solver and lets users:
1. Input Sudoku puzzles
2. See the model's solution
3. Experiment with different inference settings (thinking time)

Paper: https://zenodo.org/records/17959628
"""

import gradio as gr
import torch
import numpy as np
import pandas as pd
import os
import re
from huggingface_hub import hf_hub_download

# Optional: OpenAI for comparison
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import from local pot package (bundled with the Space)
from pot.models.sudoku_solver import HybridPoHHRMSolver


# ============================================================================
# Model Configuration (from v34 checkpoint - trained config)
# ============================================================================
MODEL_CONFIG = {
    "d_model": 512,
    "n_heads": 8,
    "H_layers": 2,
    "L_layers": 2,
    "d_ff": 2048,
    "H_cycles": 2,  # Can increase at inference
    "L_cycles": 6,  # Can increase at inference
    "T": 4,
    "dropout": 0.039,
    "hrm_grad_style": True,
    "halt_max_steps": 2,  # Can increase at inference
    "controller_type": "transformer",
    "controller_kwargs": {
        "n_ctrl_layers": 2,
        "n_ctrl_heads": 4,
        "d_ctrl": 256,
        "max_depth": 32,  # Will be adjusted dynamically
        "token_conditioned": True,
    },
    "injection_mode": "none",
    "vocab_size": 10,
    "num_puzzles": 1,
    "puzzle_emb_dim": 512,
}

# Global model cache
_model = None
_checkpoint = None


def load_model():
    """Load model from HuggingFace Hub (cached)."""
    global _model, _checkpoint
    
    if _model is not None:
        return _model, _checkpoint
    
    print("Downloading model from HuggingFace...")
    checkpoint_path = hf_hub_download("Eran92/pot-sudoku-78", "best_model.pt")
    
    print("Creating model with checkpoint config...")
    config = MODEL_CONFIG.copy()
    config["controller_kwargs"] = MODEL_CONFIG["controller_kwargs"].copy()
    # Use max_depth=32 to match checkpoint, we'll expand at runtime if needed
    config["controller_kwargs"]["max_depth"] = 32
    
    _model = HybridPoHHRMSolver(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        H_layers=config["H_layers"],
        L_layers=config["L_layers"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        H_cycles=config["H_cycles"],
        L_cycles=config["L_cycles"],
        T=config["T"],
        num_puzzles=config["num_puzzles"],
        puzzle_emb_dim=config["puzzle_emb_dim"],
        hrm_grad_style=config["hrm_grad_style"],
        halt_max_steps=config["halt_max_steps"],
        controller_type=config["controller_type"],
        controller_kwargs=config["controller_kwargs"],
        injection_mode=config["injection_mode"],
    )
    
    print("Loading weights...")
    _checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _model.load_state_dict(_checkpoint["model_state_dict"])
    _model.eval()
    
    # Expand depth position embeddings to allow larger max_depth at inference
    expand_depth_positions(_model, target_max_depth=128)
    
    print(f"Model loaded! Best grid accuracy: {_checkpoint.get('best_grid_acc', 'N/A')}%")
    return _model, _checkpoint


def expand_depth_positions(model, target_max_depth=128):
    """Expand depth position embeddings by repeating/interpolating."""
    import torch.nn.functional as F
    
    # Find and expand depth_pos in both H and L level controllers
    for level_name in ['H_level', 'L_level']:
        level = getattr(model, level_name, None)
        if level is None:
            continue
        
        controller = getattr(level, 'pointer_controller', None)
        if controller is None:
            continue
        
        if hasattr(controller, 'depth_pos'):
            old_pos = controller.depth_pos  # [old_max_depth, d_ctrl]
            old_max_depth, d_ctrl = old_pos.shape
            
            if old_max_depth >= target_max_depth:
                continue
            
            # Interpolate to new size
            # Reshape for interpolation: [1, d_ctrl, old_max_depth]
            old_pos_t = old_pos.T.unsqueeze(0)
            new_pos_t = F.interpolate(old_pos_t, size=target_max_depth, mode='linear', align_corners=True)
            new_pos = new_pos_t.squeeze(0).T  # [target_max_depth, d_ctrl]
            
            # Replace the parameter
            controller.depth_pos = torch.nn.Parameter(new_pos)
            print(f"  Expanded {level_name}.depth_pos: {old_max_depth} → {target_max_depth}")


def parse_sudoku_input(text: str) -> np.ndarray:
    """Parse various Sudoku input formats."""
    text = text.strip()
    
    # Handle 81-character string
    digits = [c for c in text if c.isdigit() or c in '.0_']
    
    if len(digits) == 81:
        puzzle = []
        for c in digits:
            if c in '.0_':
                puzzle.append(0)
            else:
                puzzle.append(int(c))
        return np.array(puzzle).reshape(9, 9)
    
    # Handle multi-line format
    lines = text.strip().split('\n')
    puzzle = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('-') or line.startswith('+'):
            continue
        
        row = []
        for c in line:
            if c.isdigit():
                row.append(int(c))
            elif c in '.0_':
                row.append(0)
        
        if len(row) >= 9:
            puzzle.extend(row[:9])
    
    if len(puzzle) == 81:
        return np.array(puzzle).reshape(9, 9)
    
    raise ValueError(f"Could not parse Sudoku. Got {len(puzzle)} cells, expected 81.")


def validate_solution(puzzle: np.ndarray, solution: np.ndarray) -> tuple[bool, str]:
    """Validate a Sudoku solution."""
    if np.any(solution == 0):
        empty_count = np.sum(solution == 0)
        return False, f"Incomplete: {empty_count} cells still empty"
    
    for i in range(9):
        if len(set(solution[i])) != 9:
            return False, f"Invalid row {i+1}"
    
    for j in range(9):
        if len(set(solution[:, j])) != 9:
            return False, f"Invalid column {j+1}"
    
    for bi in range(3):
        for bj in range(3):
            box = solution[bi*3:(bi+1)*3, bj*3:(bj+1)*3].flatten()
            if len(set(box)) != 9:
                return False, f"Invalid box ({bi+1}, {bj+1})"
    
    mask = puzzle > 0
    if not np.all(solution[mask] == puzzle[mask]):
        return False, "Givens were modified"
    
    return True, "Valid solution!"


def format_grid_html(grid: np.ndarray, puzzle: np.ndarray = None) -> str:
    """Format a 9x9 grid as HTML table with clear 3x3 blocks."""
    html = ['''<table style="
        border-collapse: collapse; 
        font-family: 'Courier New', Consolas, monospace; 
        font-size: clamp(18px, 4vw, 24px); 
        margin: 0 auto;
        border: 3px solid #1f2937;
        max-width: 100%;
    ">''']
    
    for i in range(9):
        html.append('<tr>')
        for j in range(9):
            val = grid[i, j]
            
            # Base style - responsive sizing
            style = "width: clamp(28px, 8vw, 40px); height: clamp(28px, 8vw, 40px); text-align: center; font-weight: 600; "
            
            # Border styling - thick borders for 3x3 blocks
            # Right border
            if j == 2 or j == 5:
                style += "border-right: 3px solid #1f2937; "
            elif j < 8:
                style += "border-right: 1px solid #d1d5db; "
            
            # Bottom border
            if i == 2 or i == 5:
                style += "border-bottom: 3px solid #1f2937; "
            elif i < 8:
                style += "border-bottom: 1px solid #d1d5db; "
            
            # Color styling
            if puzzle is not None and puzzle[i, j] == 0 and val > 0:
                # Solved cell - green
                style += "color: #059669; font-weight: bold; background: #d1fae5; "
            elif val > 0:
                # Given cell - dark
                style += "color: #1f2937; background: #f9fafb; "
            else:
                # Empty cell
                style += "color: #d1d5db; background: #ffffff; "
            
            # Only display valid digits 1-9 (prevent XSS)
            if isinstance(val, (int, np.integer)) and 1 <= val <= 9:
                content = str(int(val))
            else:
                content = "·"
            html.append(f'<td style="{style}">{content}</td>')
        
        html.append('</tr>')
    
    html.append('</table>')
    return '\n'.join(html)


def solve_sudoku_from_df(
    puzzle_df,
    halt_max_steps: int = 2,
    h_cycles: int = 2,
    l_cycles: int = 6,
) -> tuple[str, str, str]:
    """Solve a Sudoku puzzle from dataframe with configurable thinking time."""
    try:
        puzzle = dataframe_to_puzzle_array(puzzle_df)
    except Exception as e:
        return "❌ Error parsing board. Please check your input.", "", ""
    
    # Check if puzzle is empty
    if np.sum(puzzle > 0) == 0:
        return "❌ Please enter some numbers on the board first!", "", ""
    
    # Validate input parameters (prevent abuse)
    halt_max_steps = max(2, min(8, int(halt_max_steps)))
    h_cycles = max(2, min(4, int(h_cycles)))
    l_cycles = max(6, min(16, int(l_cycles)))
    
    # Check for valid Sudoku (no duplicate givens)
    for i in range(9):
        row_vals = [v for v in puzzle[i] if v > 0]
        if len(row_vals) != len(set(row_vals)):
            return "❌ Invalid puzzle: duplicate numbers in a row", "", ""
    
    for j in range(9):
        col_vals = [v for v in puzzle[:, j] if v > 0]
        if len(col_vals) != len(set(col_vals)):
            return "❌ Invalid puzzle: duplicate numbers in a column", "", ""
    
    return _solve_puzzle(puzzle, halt_max_steps, h_cycles, l_cycles)


def _solve_puzzle(
    puzzle: np.ndarray,
    halt_max_steps: int = 2,
    h_cycles: int = 2,
    l_cycles: int = 6,
) -> tuple[str, str, str]:
    """Core solve logic for a numpy puzzle array."""
    try:
        model, checkpoint = load_model()
    except Exception as e:
        return "❌ Error loading model. Please try again.", "", ""
    
    # Safety: ensure puzzle is valid numpy array
    if not isinstance(puzzle, np.ndarray) or puzzle.shape != (9, 9):
        return "❌ Invalid puzzle format", "", ""
    
    # Store original values
    orig_halt = model.halt_max_steps
    orig_h = model.H_cycles
    orig_l = model.L_cycles
    
    # Override for inference (more thinking time!)
    model.halt_max_steps = halt_max_steps
    model.H_cycles = h_cycles
    model.L_cycles = l_cycles
    
    # Compute effective depth
    effective_depth = h_cycles * l_cycles * halt_max_steps
    
    puzzle_html = format_grid_html(puzzle)
    
    with torch.no_grad():
        puzzle_tensor = torch.tensor(puzzle.flatten(), dtype=torch.long).unsqueeze(0)
        
        try:
            output = model(puzzle_tensor, puzzle_ids=torch.zeros(1, dtype=torch.long))
            
            if isinstance(output, tuple):
                logits = output[0]
            elif isinstance(output, dict):
                logits = output.get('logits', output.get('predictions'))
            else:
                logits = output
            
            if logits.dim() == 3:
                predictions = logits.argmax(dim=-1).squeeze(0)
            else:
                predictions = logits.squeeze(0)
            
            solution = predictions.numpy().reshape(9, 9)
            
        except Exception as e:
            # Restore original values
            model.halt_max_steps = orig_halt
            model.H_cycles = orig_h
            model.L_cycles = orig_l
            return f"❌ Inference error: {e}", puzzle_html, ""
    
    # Restore original values
    model.halt_max_steps = orig_halt
    model.H_cycles = orig_h
    model.L_cycles = orig_l
    
    # Validate
    is_valid, msg = validate_solution(puzzle, solution)
    solution_html = format_grid_html(solution, puzzle)
    
    filled_cells = np.sum(puzzle == 0)
    solved_by_model = np.sum((puzzle == 0) & (solution > 0))
    
    config_str = f"H×L×halt = {h_cycles}×{l_cycles}×{halt_max_steps} = {effective_depth} iterations"
    
    if is_valid:
        status = f"✅ {msg}\n\n📊 Solved {solved_by_model}/{filled_cells} blank cells\n⚙️ {config_str}"
    else:
        status = f"⚠️ {msg}\n\n📊 Filled {solved_by_model}/{filled_cells} blank cells\n⚙️ {config_str}"
    
    return status, puzzle_html, solution_html


def solve_sudoku(
    puzzle_text: str,
    halt_max_steps: int = 2,
    h_cycles: int = 2,
    l_cycles: int = 6,
) -> tuple[str, str, str]:
    """Solve a Sudoku puzzle with configurable thinking time."""
    try:
        puzzle = parse_sudoku_input(puzzle_text)
    except Exception as e:
        return f"❌ Error parsing input: {e}", "", ""
    
    return _solve_puzzle(puzzle, halt_max_steps, h_cycles, l_cycles)


def auto_tune_solve_from_df(puzzle_df) -> tuple[str, str, str]:
    """Automatically find the best config to solve the puzzle from dataframe."""
    try:
        puzzle = dataframe_to_puzzle_array(puzzle_df)
    except Exception as e:
        return f"❌ Error parsing board: {e}", "", ""
    
    if np.sum(puzzle > 0) == 0:
        return "❌ Please enter some numbers on the board first!", "", ""
    
    return _auto_tune_solve(puzzle)


def _auto_tune_solve(puzzle: np.ndarray) -> tuple[str, str, str]:
    """Core auto-tune logic."""
    puzzle_html = format_grid_html(puzzle)
    
    # Try progressively more thinking time
    configs = [
        (2, 6, 2),   # Default: 24 iterations
        (2, 8, 2),   # 32 iterations
        (2, 6, 3),   # 36 iterations
        (3, 8, 2),   # 48 iterations
        (2, 8, 3),   # 48 iterations
        (3, 8, 3),   # 72 iterations
        (4, 8, 3),   # 96 iterations
        (4, 12, 3),  # 144 iterations
        (4, 16, 4),  # 256 iterations
        (4, 16, 6),  # 384 iterations (max)
    ]
    
    for h_cycles, l_cycles, halt_steps in configs:
        status, _, solution_html = _solve_puzzle(
            puzzle,
            halt_max_steps=halt_steps,
            h_cycles=h_cycles,
            l_cycles=l_cycles,
        )
        
        if "✅" in status:
            effective = h_cycles * l_cycles * halt_steps
            status = f"🎯 Auto-solved with {effective} iterations!\n\n" + status
            return status, puzzle_html, solution_html
    
    return status, puzzle_html, solution_html


def auto_tune_solve(puzzle_text: str) -> tuple[str, str, str]:
    """Automatically find the best config to solve the puzzle."""
    try:
        puzzle = parse_sudoku_input(puzzle_text)
    except Exception as e:
        return f"❌ Error parsing input: {e}", "", ""
    
    return _auto_tune_solve(puzzle)


# ============================================================================
# Gradio Interface
# ============================================================================

EXAMPLE_PUZZLES = [
    ("Easy", "530070000600195000098000060800060003400803001700020006060000280000419005000080079"),
    ("Medium", "000000680000073009309000040200900000050000010000005003090000506400320000027000000"),
    ("Hard", "800000000003600000070090200050007000000045700000100030001000068008500010090000400"),
    ("Extreme", "000000010400000000020000000000050407008000300001090000300400200050100000000806000"),
]

# More puzzles for random selection
ALL_PUZZLES = [
    # Easy
    "530070000600195000098000060800060003400803001700020006060000280000419005000080079",
    "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
    "200080300060070084030500209000105408000000000402706000301007040720040060004010003",
    # Medium
    "000000680000073009309000040200900000050000010000005003090000506400320000027000000",
    "020000000000600003074000000000200090000070100600000080050007020080100000000030000",
    "000075400000000008080190000300001060000000034000068170204000603900000020530200000",
    # Hard
    "800000000003600000070090200050007000000045700000100030001000068008500010090000400",
    "000006000059000008200008000045000000003000000006003054000325006000000000000000000",
    "100007090030020008009600500005300900010080002600004000300000010040000007007000300",
    # Extreme
    "000000010400000000020000000000050407008000300001090000300400200050100000000806000",
    "000000000000003085001020000000507000004000100090000000500000073002010000000040009",
    "000000039000001005003050800008090006070002000100400000009080050020000600400700000",
]

def get_random_puzzle():
    """Return a random puzzle from the collection."""
    import random
    puzzle_str = random.choice(ALL_PUZZLES)
    df = puzzle_string_to_dataframe(puzzle_str)
    puzzle_arr = dataframe_to_puzzle_array(df)
    html = format_grid_html(puzzle_arr)
    return html, df


def format_puzzle_for_prompt(puzzle: np.ndarray) -> str:
    """Format puzzle as a copyable prompt for any AI (ChatGPT, Claude, Gemini, etc.)."""
    lines = []
    lines.append("Solve this Sudoku puzzle. Replace the dots (.) with numbers 1-9.")
    lines.append("Each row, column, and 3x3 box must contain digits 1-9 exactly once.")
    lines.append("")
    for i in range(9):
        row = ""
        for j in range(9):
            if puzzle[i, j] == 0:
                row += ". "
            else:
                row += f"{puzzle[i, j]} "
            if j in [2, 5]:
                row += "| "
        lines.append(row)
        if i in [2, 5]:
            lines.append("------+-------+------")
    lines.append("")
    lines.append("Show your solution as a formatted 9x9 grid.")
    return "\n".join(lines)


def get_copyable_prompt(puzzle_df):
    """Generate a copyable prompt from the current puzzle."""
    try:
        puzzle = dataframe_to_puzzle_array(puzzle_df)
    except:
        return "Error: Could not read puzzle"
    
    if np.sum(puzzle > 0) == 0:
        return "Please load a puzzle first!"
    
    return format_puzzle_for_prompt(puzzle)


def format_puzzle_for_gpt(puzzle: np.ndarray) -> str:
    """Format puzzle as text for GPT API."""
    return format_puzzle_for_prompt(puzzle)


def solve_with_gpt(puzzle: np.ndarray) -> tuple[str, str, float]:
    """Send puzzle to GPT-4 and get response (raw, no parsing)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return None, "❌ OpenAI API key not configured. Add OPENAI_API_KEY in Space secrets.", 0
    
    if not OPENAI_AVAILABLE:
        return None, "❌ OpenAI library not installed", 0
    
    # Validate puzzle before sending
    if not isinstance(puzzle, np.ndarray) or puzzle.shape != (9, 9):
        return None, "❌ Invalid puzzle format", 0
    
    try:
        client = OpenAI(api_key=api_key)
        
        prompt = format_puzzle_for_gpt(puzzle)
        
        import time
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Sudoku solver. Show your solution as a formatted grid."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500,
            timeout=30  # 30 second timeout
        )
        
        elapsed = time.time() - start_time
        
        answer = response.choices[0].message.content
        
        # Limit response length to prevent abuse
        if answer and len(answer) > 2000:
            answer = answer[:2000] + "\n...(truncated)"
        
        return answer, "✅ GPT responded", elapsed
        
    except Exception as e:
        # Don't expose internal error details
        error_msg = str(e)
        if "api_key" in error_msg.lower():
            return None, "❌ Invalid API key", 0
        elif "timeout" in error_msg.lower():
            return None, "❌ GPT request timed out", 0
        else:
            return None, "❌ GPT request failed. Please try again.", 0


def compare_with_gpt(puzzle_df, halt_max_steps, h_cycles, l_cycles):
    """Solve with both PoT and GPT, show comparison."""
    import time
    
    try:
        puzzle = dataframe_to_puzzle_array(puzzle_df)
    except Exception as e:
        return f"❌ Error: {e}", "", "", ""
    
    if np.sum(puzzle > 0) == 0:
        return "❌ Please load a puzzle first!", "", "", ""
    
    puzzle_html = format_grid_html(puzzle)
    
    # Solve with PoT
    pot_start = time.time()
    pot_status, _, pot_solution_html = _solve_puzzle(puzzle, halt_max_steps, h_cycles, l_cycles)
    pot_time = time.time() - pot_start
    
    # Solve with GPT (raw response, no parsing)
    gpt_response, gpt_msg, gpt_time = solve_with_gpt(puzzle)
    
    if gpt_response is not None:
        # Sanitize GPT response to prevent XSS (escape HTML)
        import html
        safe_response = html.escape(gpt_response)
        gpt_solution_html = f"""<div style="font-family: monospace; font-size: 14px; background: #f8f8f8; padding: 12px; border-radius: 8px; white-space: pre-wrap;">{safe_response}</div>"""
        gpt_status = f"⏱️ {gpt_time:.1f}s"
    else:
        gpt_solution_html = f"<p style='text-align:center; color:#888;'>{gpt_msg}</p>"
        gpt_status = gpt_msg
    
    # Build comparison status
    comparison = f"""### 🧠 PoT Model (20.8M params)
{pot_status}
⏱️ {pot_time:.1f}s

---

### 🤖 GPT-4o-mini
{gpt_status}
"""
    
    return comparison, puzzle_html, pot_solution_html, gpt_solution_html

def format_puzzle_for_display(puzzle_str: str) -> str:
    """Convert 81-char string to readable 9x9 grid format."""
    lines = []
    for i in range(9):
        row = puzzle_str[i*9:(i+1)*9]
        # Replace 0 with dot for readability
        row = row.replace('0', '.')
        # Add spacing between 3x3 blocks
        formatted = f"{row[0:3]} {row[3:6]} {row[6:9]}"
        lines.append(formatted)
        # Add separator after every 3 rows
        if i in [2, 5]:
            lines.append("")
    return "\n".join(lines)

def puzzle_to_string(puzzle_text: str) -> str:
    """Convert any format back to 81-char string."""
    # Remove all non-digit characters except dots
    chars = []
    for c in puzzle_text:
        if c.isdigit():
            chars.append(c)
        elif c == '.':
            chars.append('0')
    return ''.join(chars)

DESCRIPTION = """
# 🧩 PoT Sudoku Solver

AI that **thinks step-by-step** to solve Sudoku puzzles.

📄 [Paper](https://zenodo.org/records/17959628) • 💻 [GitHub](https://github.com/Eran-BA/PoT) • 🤗 [Model](https://huggingface.co/Eran92/pot-sudoku-78)
"""

def create_empty_board():
    """Create an empty 9x9 Sudoku board as a dataframe."""
    data = [["" for _ in range(9)] for _ in range(9)]
    return pd.DataFrame(data, columns=[str(i+1) for i in range(9)])

def puzzle_string_to_dataframe(puzzle_str: str):
    """Convert 81-char puzzle string to dataframe."""
    data = []
    for i in range(9):
        row = []
        for j in range(9):
            val = puzzle_str[i*9 + j]
            if val == '0' or val == '.':
                row.append("")
            else:
                row.append(val)
        data.append(row)
    return pd.DataFrame(data, columns=[str(i+1) for i in range(9)])

def dataframe_to_puzzle_array(df) -> np.ndarray:
    """Convert dataframe to numpy array with input validation."""
    puzzle = np.zeros((9, 9), dtype=int)
    
    # Safety check: ensure dataframe has expected shape
    if df is None or len(df) < 9:
        return puzzle
    
    for i in range(min(9, len(df))):
        for j in range(min(9, len(df.columns))):
            try:
                val = df.iloc[i, j]
                if val is None:
                    continue
                # Convert to string and sanitize
                val_str = str(val).strip()[:10]  # Limit length
                # Only accept single digits 1-9
                if val_str.isdigit() and len(val_str) == 1:
                    num = int(val_str)
                    if 1 <= num <= 9:
                        puzzle[i, j] = num
            except (IndexError, ValueError, TypeError):
                # Ignore invalid cells
                pass
    return puzzle

FOOTER = """
---

<div style="text-align: center; padding: 20px; color: #666;">

**How it works**: The model (GPT/BERT with Inner-Thinking Cycles) runs through multiple "thinking cycles" — 
refining its understanding of the puzzle with each pass. Like a human solver, it starts with obvious 
deductions and progressively tackles harder cells through iterative self-correction.

📄 [Paper](https://zenodo.org/records/17959628) • 💻 [Code](https://github.com/Eran-BA/PoT) • 🤗 [Model](https://huggingface.co/Eran92/pot-sudoku-78)

*80%+ accuracy on Sudoku-Extreme benchmark • 20.8M parameters*

</div>
"""

with gr.Blocks(
    title="PoT Sudoku Solver",
    theme=gr.themes.Default(primary_hue="green"),
    css="""
    /* General styling */
    .gradio-container { 
        max-width: 1200px !important; 
    }
    
    /* Bigger buttons for mobile */
    button {
        min-height: 44px !important;
        font-size: 16px !important;
    }
    
    /* Style the dataframe as a Sudoku grid */
    .sudoku-grid table { 
        font-size: 18px !important; 
        font-family: monospace !important;
    }
    .sudoku-grid td, .sudoku-grid th { 
        text-align: center !important; 
        width: 36px !important;
        height: 36px !important;
        padding: 6px !important;
    }
    .sudoku-grid input {
        text-align: center !important;
        font-size: 16px !important;
        font-weight: bold !important;
    }
    
    /* Mobile responsive styles */
    @media (max-width: 768px) {
        /* Stack columns vertically on mobile */
        .gr-row {
            flex-direction: column !important;
        }
        
        /* Bigger touch targets */
        button {
            min-height: 52px !important;
            font-size: 18px !important;
            padding: 12px 16px !important;
        }
        
        /* Center the puzzle grid */
        table {
            margin: 0 auto !important;
        }
        
        /* Larger grid cells on mobile */
        .sudoku-grid td, .sudoku-grid th {
            width: 32px !important;
            height: 32px !important;
            font-size: 16px !important;
        }
        
        /* Better spacing */
        .gr-box {
            padding: 8px !important;
        }
        
        /* Full width buttons */
        .gr-button {
            width: 100% !important;
        }
    }
    
    /* Tablet styles */
    @media (max-width: 1024px) and (min-width: 769px) {
        button {
            min-height: 48px !important;
        }
    }
    """
) as demo:
    
    gr.Markdown(DESCRIPTION)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Your Puzzle")
            
            # Visual display of the puzzle
            puzzle_preview = gr.HTML(
                value=format_grid_html(dataframe_to_puzzle_array(puzzle_string_to_dataframe(EXAMPLE_PUZZLES[0][1]))),
                label=""
            )
            
            gr.Markdown("*Click an example below or edit the grid:*")
            
            # Hidden dataframe for editing (will show if user wants to manually edit)
            with gr.Accordion("✏️ Manual Edit (click to expand)", open=False):
                gr.Markdown("*Type numbers 1-9 in cells. Leave empty for blanks.*")
                puzzle_grid = gr.Dataframe(
                    value=puzzle_string_to_dataframe(EXAMPLE_PUZZLES[0][1]),
                    label="",
                    interactive=True,
                    row_count=(9, "fixed"),
                    col_count=(9, "fixed"),
                    headers=None,  # No header row
                    datatype="str",
                    elem_classes=["sudoku-grid"],
                )
            
            with gr.Accordion("🧠 Thinking Time (adjust for harder puzzles)", open=False):
                gr.Markdown("""
                *Increase these to give the AI more time to think. Higher = slower but better for hard puzzles.*
                
                ⚡ **Note**: High settings (e.g. 4×16×8 = 512 iterations) may take 1-2 min on free CPU. 
                For faster inference, upgrade to GPU in Space settings.
                """)
                
                halt_steps = gr.Slider(
                    minimum=2,
                    maximum=8,
                    value=2,
                    step=1,
                    label="🔄 Reasoning Depth",
                    info="How many times to reconsider (2-8). Higher = slower but better."
                )
                
                with gr.Row():
                    h_cycles = gr.Slider(
                        minimum=2,
                        maximum=4,
                        value=2,
                        step=1,
                        label="🌳 Big Picture Passes",
                        info="High-level thinking (2-4)"
                    )
                    l_cycles = gr.Slider(
                        minimum=6,
                        maximum=16,
                        value=6,
                        step=1,
                        label="🔍 Detail Refinements",
                        info="Fine-grained reasoning (6-16)"
                    )
                
                depth_display = gr.Markdown(
                    value="💭 **24 thinking steps** *(default)*",
                    elem_classes=["thinking-indicator"]
                )
            
            def update_depth(h, l, halt):
                total = int(h) * int(l) * int(halt)
                if total <= 24:
                    emoji, note = "💭", "*(default)*"
                elif total <= 48:
                    emoji, note = "🧠", "*(moderate)*"
                elif total <= 96:
                    emoji, note = "🤔", "*(deep thinking)*"
                else:
                    emoji, note = "🧙", "*(maximum reasoning)*"
                return f"{emoji} **{total} thinking steps** {note}"
            
            for slider in [halt_steps, h_cycles, l_cycles]:
                slider.change(
                    update_depth,
                    inputs=[h_cycles, l_cycles, halt_steps],
                    outputs=depth_display,
                )
            
            # Main solve button
            solve_btn = gr.Button("🎯 Solve with PoT AI", variant="primary", size="lg")
            
            with gr.Row():
                auto_btn = gr.Button("🧙 Try Harder", size="lg")
                copy_prompt_btn = gr.Button("📋 Copy Prompt", size="lg")
            
            compare_btn = gr.Button("🆚 Compare PoT vs GPT", variant="secondary", size="sm")
            
            # Textbox to show the copyable prompt
            prompt_output = gr.Textbox(
                label="📋 Prompt (copy this to ChatGPT, Claude, Gemini...)",
                lines=12,
                show_copy_button=True,
                visible=False,
            )
            
            # Main action button - big and prominent
            random_btn = gr.Button("🎲 New Random Puzzle", size="lg", variant="primary")
            
            with gr.Accordion("📋 Choose Difficulty", open=False):
                with gr.Row():
                    easy_btn = gr.Button("🟢 Easy", size="lg")
                    medium_btn = gr.Button("🟡 Medium", size="lg")
                with gr.Row():
                    hard_btn = gr.Button("🟠 Hard", size="lg")
                    extreme_btn = gr.Button("🔴 Extreme", size="lg")
                
                clear_btn = gr.Button("🗑️ Clear Board", size="sm", variant="secondary")
            
            # Helper functions to update both preview and grid
            def load_puzzle(puzzle_str):
                df = puzzle_string_to_dataframe(puzzle_str)
                puzzle_arr = dataframe_to_puzzle_array(df)
                html = format_grid_html(puzzle_arr)
                return html, df
            
            def clear_puzzle():
                df = create_empty_board()
                puzzle_arr = np.zeros((9, 9), dtype=int)
                html = format_grid_html(puzzle_arr)
                return html, df
            
            def update_preview_from_grid(df):
                puzzle_arr = dataframe_to_puzzle_array(df)
                return format_grid_html(puzzle_arr)
            
            # Connect buttons to update BOTH preview AND grid
            random_btn.click(
                fn=get_random_puzzle,
                outputs=[puzzle_preview, puzzle_grid]
            )
            easy_btn.click(
                fn=lambda: load_puzzle(EXAMPLE_PUZZLES[0][1]),
                outputs=[puzzle_preview, puzzle_grid]
            )
            medium_btn.click(
                fn=lambda: load_puzzle(EXAMPLE_PUZZLES[1][1]),
                outputs=[puzzle_preview, puzzle_grid]
            )
            hard_btn.click(
                fn=lambda: load_puzzle(EXAMPLE_PUZZLES[2][1]),
                outputs=[puzzle_preview, puzzle_grid]
            )
            extreme_btn.click(
                fn=lambda: load_puzzle(EXAMPLE_PUZZLES[3][1]),
                outputs=[puzzle_preview, puzzle_grid]
            )
            clear_btn.click(
                fn=clear_puzzle,
                outputs=[puzzle_preview, puzzle_grid]
            )
            
            # Update preview when grid is edited
            puzzle_grid.change(
                fn=update_preview_from_grid,
                inputs=[puzzle_grid],
                outputs=[puzzle_preview]
            )
        
        with gr.Column(scale=2):
            status_output = gr.Markdown(value="*Click 🎯 Solve to see the AI solution*")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📥 Input")
                    puzzle_display = gr.HTML()
                
                with gr.Column():
                    gr.Markdown("### 🧠 PoT Solution")
                    solution_display = gr.HTML()
            
            with gr.Row(visible=True):
                with gr.Column():
                    gr.Markdown("### 🤖 GPT Response")
                    gpt_solution_display = gr.HTML(value="<p style='color:#888; text-align:center;'>Click 'Compare with GPT' to see GPT's attempt</p>")
            
            gr.Markdown("*Green = solved by AI*")
    
    gr.Markdown(FOOTER)
    
    # Event handlers
    solve_btn.click(
        fn=solve_sudoku_from_df,
        inputs=[puzzle_grid, halt_steps, h_cycles, l_cycles],
        outputs=[status_output, puzzle_display, solution_display],
    )
    
    auto_btn.click(
        fn=auto_tune_solve_from_df,
        inputs=[puzzle_grid],
        outputs=[status_output, puzzle_display, solution_display],
    )
    
    compare_btn.click(
        fn=compare_with_gpt,
        inputs=[puzzle_grid, halt_steps, h_cycles, l_cycles],
        outputs=[status_output, puzzle_display, solution_display, gpt_solution_display],
    )
    
    def show_prompt(puzzle_df):
        prompt = get_copyable_prompt(puzzle_df)
        return gr.update(value=prompt, visible=True)
    
    copy_prompt_btn.click(
        fn=show_prompt,
        inputs=[puzzle_grid],
        outputs=[prompt_output],
    )

if __name__ == "__main__":
    demo.launch()
