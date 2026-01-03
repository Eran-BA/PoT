#!/usr/bin/env python3
"""
W&B Checkpoint Compatibility Test

Tests whether the 78.9% Sudoku-Extreme checkpoint from W&B is compatible
with the current codebase.

This script:
1. Downloads the W&B artifact (if not cached)
2. Creates a model with the EXACT configuration from the W&B run
3. Compares state dict keys
4. Attempts to load the checkpoint
5. Runs a quick inference test

Artifact: eranbt92-open-university-of-israel/sudoku-controller-transformer-finetune/sudoku-finetune-best:v34

Author: Eran Ben Artzy
Year: 2025
"""

import os
import sys
from pathlib import Path

# Load .env file if it exists
def load_dotenv():
    """Load environment variables from .env file."""
    env_paths = [
        Path(__file__).parent.parent / ".env",  # Project root
        Path.cwd() / ".env",  # Current directory
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            print(f"Loading environment from: {env_path}")
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value:
                            os.environ[key] = value
            return True
    return False

# Load .env before any imports that might use env vars
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

# ============================================================
# EXACT CONFIG FROM W&B RUN
# ============================================================
# From: https://wandb.ai/eranbt92-open-university-of-israel/sudoku-controller-transformer-finetune

WANDB_CONFIG = {
    # Model architecture - EXTRACTED FROM CHECKPOINT config
    "d_model": 512,
    "n_heads": 8,
    "H_layers": 2,  # h_layers in checkpoint
    "L_layers": 2,  # l_layers in checkpoint
    "d_ff": 2048,
    "H_cycles": 2,  # h_cycles in checkpoint (NOT 4!)
    "L_cycles": 6,  # l_cycles in checkpoint (NOT 4!)
    "T": 4,
    "dropout": 0.039,  # From HPO
    "hrm_grad_style": True,  # IMPORTANT: True in this run
    "halt_max_steps": 2,  # Gradually increased during training
    "halt_exploration_prob": 0.075,  # From HPO
    
    # Controller
    "controller_type": "transformer",
    
    # Controller kwargs - from checkpoint config
    "controller_kwargs": {
        "n_ctrl_layers": 2,
        "n_ctrl_heads": 4,
        "d_ctrl": 256,
        "max_depth": 32,
        "token_conditioned": True,
        "temperature": 1.0,
        "topk": None,
        "entropy_reg": 1e-3,
        "use_learned_depth_pe": False,
    },
    
    # Feature injection (CRITICAL: was "none" in W&B run!)
    "injection_mode": "none",
    "injection_kwargs": {},
    
    # RoPE and Flash Attention (defaults at time of training)
    "use_rope": False,
    "use_flash_attn": True,
    
    # Sudoku-specific
    "vocab_size": 10,
    "num_puzzles": 1,
    "puzzle_emb_dim": 512,
    
    # Training config (for reference)
    "_training_info": {
        "epoch": 2001,
        "best_grid_acc": 78.9,
        "batch_size": 768,
        "lr": 3.7e-4,
        "weight_decay": 0.108,
        "warmup_steps": 2000,
        "async_batch": True,
        "augment": True,
    },
}


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def download_wandb_artifact(artifact_path: str, download_dir: str = "./wandb_artifacts"):
    """Download W&B artifact if not already cached."""
    import wandb
    
    os.makedirs(download_dir, exist_ok=True)
    cache_path = Path(download_dir) / artifact_path.replace("/", "_").replace(":", "_")
    
    # Check if already downloaded
    if cache_path.exists():
        pt_files = list(cache_path.glob("*.pt"))
        if pt_files:
            print(f"✓ Using cached artifact: {cache_path}")
            return pt_files[0]
    
    # Check for API key in environment
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        print("✓ Found WANDB_API_KEY in environment")
        wandb.login(key=api_key)
    
    # Download from W&B
    print(f"Downloading artifact: {artifact_path}")
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download(root=str(cache_path))
    
    pt_files = list(Path(artifact_dir).glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in artifact: {artifact_dir}")
    
    print(f"✓ Downloaded to: {pt_files[0]}")
    return pt_files[0]


def create_model_from_config(config: dict) -> nn.Module:
    """Create HybridPoHHRMSolver with the exact W&B config."""
    from src.pot.models.sudoku_solver import HybridPoHHRMSolver
    
    model = HybridPoHHRMSolver(
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
        injection_kwargs=config["injection_kwargs"],
    )
    
    return model


def compare_state_dicts(model_keys: set, checkpoint_keys: set):
    """Compare model and checkpoint state dict keys."""
    
    # Find differences
    missing_in_model = checkpoint_keys - model_keys
    missing_in_checkpoint = model_keys - checkpoint_keys
    common_keys = model_keys & checkpoint_keys
    
    print(f"\n📊 State Dict Comparison:")
    print(f"   Model keys:      {len(model_keys)}")
    print(f"   Checkpoint keys: {len(checkpoint_keys)}")
    print(f"   Common keys:     {len(common_keys)}")
    
    if missing_in_model:
        print(f"\n⚠️  Keys in CHECKPOINT but not in MODEL ({len(missing_in_model)}):")
        for key in sorted(missing_in_model)[:20]:
            print(f"      - {key}")
        if len(missing_in_model) > 20:
            print(f"      ... and {len(missing_in_model) - 20} more")
    
    if missing_in_checkpoint:
        print(f"\n⚠️  Keys in MODEL but not in CHECKPOINT ({len(missing_in_checkpoint)}):")
        for key in sorted(missing_in_checkpoint)[:20]:
            print(f"      - {key}")
        if len(missing_in_checkpoint) > 20:
            print(f"      ... and {len(missing_in_checkpoint) - 20} more")
    
    return len(missing_in_model) == 0 and len(missing_in_checkpoint) == 0


def compare_tensor_shapes(model_state: dict, checkpoint_state: dict):
    """Compare tensor shapes for common keys."""
    common = set(model_state.keys()) & set(checkpoint_state.keys())
    mismatches = []
    
    for key in common:
        m_shape = model_state[key].shape
        c_shape = checkpoint_state[key].shape
        if m_shape != c_shape:
            mismatches.append((key, m_shape, c_shape))
    
    if mismatches:
        print(f"\n❌ Shape Mismatches ({len(mismatches)}):")
        for key, m_shape, c_shape in mismatches[:10]:
            print(f"   {key}:")
            print(f"      Model:      {m_shape}")
            print(f"      Checkpoint: {c_shape}")
        return False
    else:
        print(f"\n✓ All {len(common)} common tensors have matching shapes")
        return True


def run_inference_test(model: nn.Module, device: str = "cpu"):
    """Run a quick inference test with random input."""
    print(f"\n🔬 Running inference test on {device}...")
    
    model = model.to(device)
    model.eval()
    
    # Create random Sudoku-like input
    batch_size = 2
    seq_len = 81  # 9x9 grid
    
    # Input: 0 = empty, 1-9 = given digits
    input_seq = torch.randint(0, 10, (batch_size, seq_len), device=device)
    puzzle_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    with torch.no_grad():
        try:
            output = model(input_seq, puzzle_ids)
            
            # Handle different output formats
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            print(f"   Input shape:  {input_seq.shape}")
            print(f"   Output shape: {logits.shape}")
            
            # Check output values
            if torch.isnan(logits).any():
                print("   ❌ Output contains NaN values!")
                return False
            if torch.isinf(logits).any():
                print("   ❌ Output contains Inf values!")
                return False
            
            print(f"   Output range: [{logits.min():.3f}, {logits.max():.3f}]")
            print("   ✓ Inference successful!")
            return True
            
        except Exception as e:
            print(f"   ❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_architecture_only():
    """Test just the model architecture (no checkpoint needed)."""
    print_section("Architecture-Only Test (No Checkpoint)")
    
    print("Creating model with W&B config...")
    model = create_model_from_config(WANDB_CONFIG)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    
    # Expected parameter count from W&B run (for verification)
    # Checkpoint has 120 keys, ~20M with d_ctrl=256, n_ctrl_layers=2
    EXPECTED_PARAMS_RANGE = (20_000_000, 22_000_000)
    if EXPECTED_PARAMS_RANGE[0] <= total_params <= EXPECTED_PARAMS_RANGE[1]:
        print(f"✓ Parameter count in expected range: {EXPECTED_PARAMS_RANGE}")
    else:
        print(f"⚠️ Parameter count {total_params:,} outside expected range {EXPECTED_PARAMS_RANGE}")
    
    # Print parameter breakdown
    print("\n📊 Parameter Breakdown:")
    param_groups = {}
    for name, param in model.named_parameters():
        group = name.split('.')[0]
        if group not in param_groups:
            param_groups[group] = 0
        param_groups[group] += param.numel()
    
    for group, count in sorted(param_groups.items(), key=lambda x: -x[1]):
        print(f"   {group}: {count:,} params")
    
    # Verify expected components exist
    print("\n🔍 Verifying Model Components:")
    expected_components = [
        ("L_level", "L-level reasoning module"),
        ("H_level", "H-level reasoning module"),
        ("L_level.pointer_controller", "L-level transformer depth controller"),
        ("H_level.pointer_controller", "H-level transformer depth controller"),
        ("input_embed", "Input embedding"),
        ("output_proj", "Output projection"),
        ("final_norm", "Final normalization"),
        ("q_head", "Q-halting head"),
    ]
    
    for attr_path, desc in expected_components:
        obj = model
        found = True
        for part in attr_path.split('.'):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                found = False
                break
        status = "✓" if found else "❌"
        print(f"   {status} {desc} ({attr_path})")
    
    # Verify controller type
    print("\n🔍 Verifying Controller Type:")
    L_ctrl = model.L_level.pointer_controller
    H_ctrl = model.H_level.pointer_controller
    ctrl_type = type(L_ctrl).__name__
    print(f"   Controller class: {ctrl_type}")
    
    from src.pot.core.depth_transformer_controller import CausalDepthTransformerRouter
    if isinstance(L_ctrl, CausalDepthTransformerRouter):
        print("   ✓ Using CausalDepthTransformerRouter (matches W&B run)")
    else:
        print(f"   ⚠️ Expected CausalDepthTransformerRouter, got {ctrl_type}")
    
    # Verify injection mode is "none"
    print("\n🔍 Verifying Injection Mode:")
    L_inj = model.L_level.injector
    print(f"   Injection mode: {L_inj.mode}")
    if L_inj.mode == "none":
        print("   ✓ Injection mode is 'none' (matches W&B run)")
    else:
        print(f"   ⚠️ Expected 'none', got '{L_inj.mode}'")
    
    # Verify key parameters
    print("\n🔍 Verifying Key Parameters:")
    checks = [
        ("d_model", model.d_model, WANDB_CONFIG["d_model"]),
        ("n_heads", model.n_heads, WANDB_CONFIG["n_heads"]),
        ("H_cycles", model.H_cycles, WANDB_CONFIG["H_cycles"]),
        ("L_cycles", model.L_cycles, WANDB_CONFIG["L_cycles"]),
        ("seq_len", model.seq_len, 81),  # Sudoku grid
    ]
    
    all_match = True
    for name, actual, expected in checks:
        if actual == expected:
            print(f"   ✓ {name}: {actual}")
        else:
            print(f"   ❌ {name}: {actual} (expected {expected})")
            all_match = False
    
    # Run inference test
    print_section("Inference Test")
    inference_success = run_inference_test(model)
    
    # Summary
    print_section("Summary")
    if inference_success and all_match:
        print("🎉 Architecture test PASSED!")
        print("   Model architecture matches W&B run configuration.")
        print("   Inference works correctly.")
        print("\n   The checkpoint from W&B should be compatible with this model.")
        print("\n   To verify actual checkpoint loading:")
        print("   1. Run 'wandb login' to authenticate")
        print("   2. Run this script without --architecture-only")
        print("   3. Or use --checkpoint <path> with a local checkpoint")
        return 0
    else:
        print("❌ Architecture test FAILED!")
        return 1


def main():
    """Run full compatibility test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test W&B checkpoint compatibility")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to local checkpoint file")
    parser.add_argument("--architecture-only", action="store_true",
                        help="Only test architecture, skip checkpoint loading")
    parser.add_argument("--wandb-login", action="store_true",
                        help="Force W&B login prompt")
    args = parser.parse_args()
    
    # Architecture-only mode
    if args.architecture_only:
        return test_architecture_only()
    
    print_section("W&B Checkpoint Compatibility Test")
    
    # 1. Create model with W&B config
    print_section("1. Creating Model with W&B Config")
    print(f"Config:")
    for key, value in WANDB_CONFIG.items():
        if key not in ("controller_kwargs", "injection_kwargs"):
            print(f"   {key}: {value}")
    
    model = create_model_from_config(WANDB_CONFIG)
    model_state = model.state_dict()
    print(f"\n✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 2. Get checkpoint (local or W&B)
    print_section("2. Getting Checkpoint")
    
    ARTIFACT_PATH = "eranbt92-open-university-of-israel/sudoku-controller-transformer-finetune/sudoku-finetune-best:v34"
    
    if args.checkpoint:
        # Use local checkpoint
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return 1
        print(f"✓ Using local checkpoint: {checkpoint_path}")
    else:
        # Try W&B download
        if args.wandb_login:
            import wandb
            wandb.login()
        
        try:
            checkpoint_path = download_wandb_artifact(ARTIFACT_PATH)
        except Exception as e:
            print(f"❌ Failed to download artifact: {e}")
            print("\nOptions:")
            print("  1. Run 'wandb login' and retry")
            print("  2. Use --architecture-only to test just the model")
            print("  3. Use --checkpoint <path> with a local checkpoint file")
            print("\nRunning architecture-only test as fallback...")
            return test_architecture_only()
    
    # 3. Load checkpoint
    print_section("3. Loading Checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint_state = checkpoint["model_state_dict"]
        print("   Checkpoint format: dict with 'model_state_dict'")
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint_state = checkpoint["state_dict"]
        print("   Checkpoint format: dict with 'state_dict'")
    else:
        checkpoint_state = checkpoint
        print("   Checkpoint format: raw state_dict")
    
    print(f"✓ Checkpoint loaded: {len(checkpoint_state)} keys")
    
    # 4. Compare state dicts
    print_section("4. Comparing State Dicts")
    model_keys = set(model_state.keys())
    checkpoint_keys = set(checkpoint_state.keys())
    
    keys_match = compare_state_dicts(model_keys, checkpoint_keys)
    shapes_match = compare_tensor_shapes(model_state, checkpoint_state)
    
    # 5. Try loading checkpoint
    print_section("5. Loading Checkpoint into Model")
    
    try:
        # Try strict load first
        model.load_state_dict(checkpoint_state, strict=True)
        print("✓ Strict load successful!")
        load_success = True
    except RuntimeError as e:
        print(f"❌ Strict load failed: {e}")
        
        # Try non-strict load
        try:
            incompatible = model.load_state_dict(checkpoint_state, strict=False)
            print(f"\n⚠️  Non-strict load:")
            if incompatible.missing_keys:
                print(f"   Missing keys: {len(incompatible.missing_keys)}")
                for k in incompatible.missing_keys[:5]:
                    print(f"      - {k}")
            if incompatible.unexpected_keys:
                print(f"   Unexpected keys: {len(incompatible.unexpected_keys)}")
                for k in incompatible.unexpected_keys[:5]:
                    print(f"      - {k}")
            load_success = True
        except Exception as e2:
            print(f"❌ Non-strict load also failed: {e2}")
            load_success = False
    
    # 6. Inference test
    if load_success:
        print_section("6. Inference Test")
        inference_success = run_inference_test(model)
    else:
        inference_success = False
    
    # Summary
    print_section("SUMMARY")
    
    results = {
        "Keys match": keys_match,
        "Shapes match": shapes_match,
        "Load successful": load_success,
        "Inference works": inference_success,
    }
    
    all_pass = all(results.values())
    
    for test, passed in results.items():
        status = "✓" if passed else "❌"
        print(f"   {status} {test}")
    
    print()
    if all_pass:
        print("🎉 COMPATIBILITY TEST PASSED!")
        print("   The W&B checkpoint is fully compatible with the current codebase.")
        return 0
    else:
        print("⚠️  COMPATIBILITY ISSUES DETECTED")
        print("   See details above for specific issues.")
        return 1


def export_expected_state_dict():
    """Export expected state dict keys for documentation."""
    print_section("Exporting Expected State Dict Keys")
    
    model = create_model_from_config(WANDB_CONFIG)
    state = model.state_dict()
    
    output_file = Path(__file__).parent / "expected_sudoku_state_dict.txt"
    
    with open(output_file, "w") as f:
        f.write("# Expected State Dict Keys for Sudoku-Extreme 78.9% Model\n")
        f.write(f"# Generated from W&B config\n")
        f.write(f"# Total keys: {len(state)}\n")
        f.write(f"# Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write("#\n")
        f.write("# Key\tShape\tNumel\n")
        f.write("#" + "="*80 + "\n")
        
        for key in sorted(state.keys()):
            tensor = state[key]
            f.write(f"{key}\t{list(tensor.shape)}\t{tensor.numel()}\n")
    
    print(f"✓ Exported {len(state)} keys to: {output_file}")
    print(f"\nFirst 20 keys:")
    for i, key in enumerate(sorted(state.keys())[:20]):
        tensor = state[key]
        print(f"   {key}: {list(tensor.shape)}")
    print(f"   ... and {len(state) - 20} more")
    
    return 0


if __name__ == "__main__":
    import argparse
    
    # If called directly without main's argparse, handle --export-keys
    if len(sys.argv) > 1 and sys.argv[1] == "--export-keys":
        sys.exit(export_expected_state_dict())
    
    sys.exit(main())

