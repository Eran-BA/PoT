"""
Basic import tests to validate restructured codebase.

Author: Eran Ben Artzy
License: Apache 2.0
"""


def test_model_imports():
    """Test that all model classes can be imported."""
    from src.models import (
        ParserBase,
        BaselineParser,
        PoHParser,
        PointerMoHTransformerBlock,
        BiaffinePointer,
        BiaffineLabeler,
    )

    assert ParserBase is not None
    assert BaselineParser is not None
    assert PoHParser is not None


def test_data_imports():
    """Test that data loading functions can be imported."""
    from src.data.loaders import (
        load_hf_dataset,
        load_conllu_files,
        create_dummy_dataset,
        get_dataset,
        build_label_vocab,
    )
    from src.data.collate import collate_batch

    assert create_dummy_dataset is not None
    assert collate_batch is not None


def test_training_imports():
    """Test that training components can be imported."""
    from src.training.trainer import Trainer
    from src.training.schedulers import get_linear_schedule_with_warmup

    assert Trainer is not None
    assert get_linear_schedule_with_warmup is not None


def test_utils_imports():
    """Test that utility functions can be imported."""
    from src.utils.helpers import mean_pool_subwords, pad_words, make_targets
    from src.utils.logger import append_row, flatten_cfg
    from src.utils.metrics import build_masks_for_metrics, compute_uas_las
    from src.utils.conllu_writer import write_conllu
    from src.utils.iterative_losses import (
        deep_supervision_loss,
        act_expected_loss,
        act_deep_supervision_loss,
    )

    assert mean_pool_subwords is not None
    assert append_row is not None
    assert compute_uas_las is not None


def test_basic_model_creation():
    """Test that models can be instantiated."""
    from src.models import BaselineParser, PoHParser

    baseline = BaselineParser(d_model=768, n_heads=8, d_ff=2048)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    assert baseline_params > 0

    poh = PoHParser(d_model=768, n_heads=8, d_ff=2048, max_inner_iters=2)
    poh_params = sum(p.numel() for p in poh.parameters())
    assert poh_params > 0


if __name__ == "__main__":
    print("=" * 80)
    print("PoT Project Structure Validation")
    print("=" * 80 + "\n")

    try:
        test_model_imports()
        test_data_imports()
        test_training_imports()
        test_utils_imports()
        test_basic_model_creation()

        print("\n" + "=" * 80)
        print("✅ All tests passed! Project structure is valid.")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
