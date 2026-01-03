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
from huggingface_hub import hf_hub_download

# Import from pot-transformer package (installed from GitHub)
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
    
    print("Creating model with max capacity for inference...")
    # Create with larger max_depth to allow runtime adjustments
    config = MODEL_CONFIG.copy()
    config["controller_kwargs"] = MODEL_CONFIG["controller_kwargs"].copy()
    config["controller_kwargs"]["max_depth"] = 128  # Large enough for any inference config
    
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
    _checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Load with strict=False to allow max_depth mismatch in position embeddings
    _model.load_state_dict(_checkpoint["model_state_dict"], strict=False)
    _model.eval()
    
    print(f"Model loaded! Best grid accuracy: {_checkpoint.get('best_grid_acc', 'N/A')}%")
    return _model, _checkpoint


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
    """Format a 9x9 grid as HTML table."""
    html = ['<table style="border-collapse: collapse; font-family: monospace; font-size: 24px; margin: auto;">']
    
    for i in range(9):
        html.append('<tr>')
        for j in range(9):
            val = grid[i, j]
            
            style = "width: 40px; height: 40px; text-align: center; "
            
            if j % 3 == 0:
                style += "border-left: 3px solid black; "
            else:
                style += "border-left: 1px solid gray; "
            
            if j == 8:
                style += "border-right: 3px solid black; "
            
            if i % 3 == 0:
                style += "border-top: 3px solid black; "
            else:
                style += "border-top: 1px solid gray; "
            
            if i == 8:
                style += "border-bottom: 3px solid black; "
            
            if puzzle is not None and puzzle[i, j] == 0 and val > 0:
                style += "color: #22c55e; font-weight: bold; background: #f0fdf4; "
            elif val > 0:
                style += "color: #000; background: #f8f8f8; "
            else:
                style += "background: #fff; "
            
            content = str(val) if val > 0 else ""
            html.append(f'<td style="{style}">{content}</td>')
        
        html.append('</tr>')
    
    html.append('</table>')
    return '\n'.join(html)


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
    
    try:
        model, checkpoint = load_model()
    except Exception as e:
        return f"❌ Error loading model: {e}", "", ""
    
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


def auto_tune_solve(puzzle_text: str) -> tuple[str, str, str]:
    """Automatically find the best config to solve the puzzle."""
    try:
        puzzle = parse_sudoku_input(puzzle_text)
    except Exception as e:
        return f"❌ Error parsing input: {e}", "", ""
    
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
        status, _, solution_html = solve_sudoku(
            puzzle_text,
            halt_max_steps=halt_steps,
            h_cycles=h_cycles,
            l_cycles=l_cycles,
        )
        
        if "✅" in status:
            effective = h_cycles * l_cycles * halt_steps
            status = f"🎯 Auto-solved with {effective} iterations!\n\n" + status
            return status, puzzle_html, solution_html
    
    return status, puzzle_html, solution_html


# ============================================================================
# Gradio Interface
# ============================================================================

EXAMPLE_PUZZLES = [
    ("Easy", "530070000600195000098000060800060003400803001700020006060000280000419005000080079"),
    ("Medium", "000000680000073009309000040200900000050000010000005003090000506400320000027000000"),
    ("Hard", "800000000003600000070090200050007000000045700000100030001000068008500010090000400"),
    ("Extreme", "000000010400000000020000000000050407008000300001090000300400200050100000000806000"),
]

DESCRIPTION = """
# 🧩 PoT Sudoku Solver

An AI that **thinks step-by-step** to solve Sudoku puzzles using iterative refinement.

> *"PoT doesn't just compute — it thinks within its embeddings, iteratively reorganizing its representation space."*

📄 **[Read the Paper](https://zenodo.org/records/17959628)** • 💻 **[GitHub](https://github.com/Eran-BA/PoT)** • 🤗 **[Model](https://huggingface.co/Eran92/pot-sudoku-78)**

---
"""

FOOTER = """
---

<div style="text-align: center; padding: 20px; color: #666;">

**How it works**: The model runs through multiple "thinking cycles" — refining its understanding of the puzzle 
with each pass. Like a human solver, it starts with obvious deductions and progressively tackles harder cells.

📄 [Paper](https://zenodo.org/records/17959628) • 💻 [Code](https://github.com/Eran-BA/PoT) • 🤗 [Model](https://huggingface.co/Eran92/pot-sudoku-78)

*78.9% accuracy on Sudoku-Extreme benchmark • 20.8M parameters*

</div>
"""

with gr.Blocks(
    title="PoT Sudoku Solver",
    theme=gr.themes.Soft(primary_hue="green"),
    css="""
    .thinking-indicator { 
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); 
        border-radius: 8px; 
        padding: 12px 16px;
        margin: 8px 0;
        text-align: center;
        border: 1px solid #bbf7d0;
    }
    .gradio-container { max-width: 1200px !important; }
    """
) as demo:
    
    gr.Markdown(DESCRIPTION)
    
    with gr.Row():
        with gr.Column(scale=1):
            puzzle_input = gr.Textbox(
                label="📝 Input Puzzle",
                placeholder="Paste your Sudoku here (0 or . for empty cells)",
                lines=10,
                value=EXAMPLE_PUZZLES[0][1],
            )
            
            with gr.Accordion("🧠 Thinking Time (adjust for harder puzzles)", open=False):
                gr.Markdown("""
                *Increase these to give the AI more time to think. Higher = slower but better for hard puzzles.*
                """)
                
                halt_steps = gr.Slider(
                    minimum=2,
                    maximum=6,
                    value=2,
                    step=1,
                    label="🔄 Reasoning Depth",
                    info="How many times to reconsider the solution (2-6)"
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
            
            with gr.Row():
                solve_btn = gr.Button("🎯 Solve Puzzle", variant="primary", size="lg")
                auto_btn = gr.Button("🧙 Try Harder (Auto)", variant="secondary", size="lg")
            
            with gr.Accordion("📋 Example Puzzles", open=True):
                gr.Examples(
                    examples=[
                        [EXAMPLE_PUZZLES[0][1]],  # Easy
                        [EXAMPLE_PUZZLES[1][1]],  # Medium
                        [EXAMPLE_PUZZLES[2][1]],  # Hard
                        [EXAMPLE_PUZZLES[3][1]],  # Extreme
                    ],
                    inputs=puzzle_input,
                    label="",
                    examples_per_page=4,
                )
                gr.Markdown("*Click any example above, or paste your own puzzle*")
        
        with gr.Column(scale=2):
            status_output = gr.Textbox(label="📊 Result", lines=5, show_copy_button=True)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Input Puzzle")
                    puzzle_display = gr.HTML()
                
                with gr.Column():
                    gr.Markdown("### Solution")
                    solution_display = gr.HTML()
    
    gr.Markdown(FOOTER)
    
    # Event handlers
    solve_btn.click(
        fn=solve_sudoku,
        inputs=[puzzle_input, halt_steps, h_cycles, l_cycles],
        outputs=[status_output, puzzle_display, solution_display],
    )
    
    auto_btn.click(
        fn=auto_tune_solve,
        inputs=[puzzle_input],
        outputs=[status_output, puzzle_display, solution_display],
    )

if __name__ == "__main__":
    demo.launch()
