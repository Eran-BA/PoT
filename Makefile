# Makefile for PoT (Pointer-over-Heads Transformer)
# Author: Eran Ben Artzy
# Year: 2025
# License: Apache 2.0

.PHONY: help install test test-hrm test-ud smoke-hrm smoke-ud hrm-ab hrm-quick format lint clean

help:
	@echo "PoT Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run all tests"
	@echo "  make test-hrm     - Run HRM controller tests"
	@echo "  make test-ud      - Run UD dependency parser tests"
	@echo "  make smoke-hrm    - Run HRM diagnostic smoke test"
	@echo "  make smoke-ud     - Run UD parser smoke test"
	@echo "  make hrm-quick    - Quick HRM smoke training (3 epochs)"
	@echo "  make hrm-ab       - Full HRM vs Baseline A/B comparison"
	@echo "  make format       - Format code with black"
	@echo "  make lint         - Run linters (ruff)"
	@echo "  make clean        - Clean temporary files"

install:
	pip install -e .
	pip install -r requirements.txt

test:
	PYTHONPATH=. pytest tests/ -v

test-hrm:
	PYTHONPATH=. pytest tests/test_hrm_pointer_controller.py -v

test-ud:
	PYTHONPATH=. pytest tests/test_ud_parser.py -v

smoke-hrm:
	PYTHONPATH=. python tools/hrm_diag_smoke.py

smoke-ud:
	PYTHONPATH=. python tools/test_ud_parser_smoke.py

format:
	black src/ tests/ examples/ scripts/
	ruff check --fix src/ tests/ examples/ scripts/

lint:
	ruff check src/ tests/ examples/ scripts/
	black --check src/ tests/ examples/ scripts/

hrm-quick:
	@echo "Running quick HRM smoke training (3 epochs, 100 samples)..."
	PYTHONPATH=. python experiments/fair_ab_comparison.py \
		--model pot \
		--array_len 12 \
		--mask_rate 0.5 \
		--train_samples 100 \
		--epochs 3 \
		--batch_size 16 \
		--lr 3e-4 \
		--max_inner_iters 2 \
		--seeds 1 \
		--output_csv experiments/results/smoke_hrm.csv
	@echo "✓ Quick smoke training complete. Check experiments/results/smoke_hrm.csv"

hrm-ab:
	@echo "================================================================================"
	@echo "HRM vs BASELINE A/B COMPARISON"
	@echo "================================================================================"
	@echo ""
	@echo "Configuration:"
	@echo "  Task: Partial observability sorting (50% masked)"
	@echo "  Array length: 12"
	@echo "  Training: 1000 samples, 40 epochs"
	@echo "  Seeds: 1, 2, 3, 4, 5"
	@echo ""
	@echo "Running Baseline (Standard Controller)..."
	PYTHONPATH=. python experiments/fair_ab_comparison.py \
		--model baseline \
		--array_len 12 \
		--mask_rate 0.5 \
		--train_samples 1000 \
		--epochs 40 \
		--batch_size 64 \
		--lr 3e-4 \
		--seeds 1 2 3 4 5 \
		--output_csv experiments/results/ab_baseline.csv
	@echo ""
	@echo "Running PoT with HRM Controller (T=4, top-k=3, 4 iterations)..."
	PYTHONPATH=. python experiments/fair_ab_comparison.py \
		--model pot \
		--array_len 12 \
		--mask_rate 0.5 \
		--train_samples 1000 \
		--epochs 40 \
		--batch_size 64 \
		--lr 3e-4 \
		--max_inner_iters 4 \
		--seeds 1 2 3 4 5 \
		--output_csv experiments/results/ab_pot_hrm.csv
	@echo ""
	@echo "================================================================================"
	@echo "A/B COMPARISON COMPLETE"
	@echo "================================================================================"
	@echo ""
	@echo "Results:"
	@echo "  Baseline: experiments/results/ab_baseline.csv"
	@echo "  HRM:      experiments/results/ab_pot_hrm.csv"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Compare results: python experiments/compare_ab_results.py"
	@echo "  2. Plot curves: python experiments/plot_sorting_results.py"
	@echo "  3. Check significance: bootstrap CI analysis"

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .ruff_cache/

