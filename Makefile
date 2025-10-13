# Makefile for PoT (Pointer-over-Heads Transformer)
# Author: Eran Ben Artzy
# Year: 2025
# License: Apache 2.0

.PHONY: help install test test-hrm smoke-hrm format lint clean

help:
	@echo "PoT Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run all tests"
	@echo "  make test-hrm     - Run HRM controller tests"
	@echo "  make smoke-hrm    - Run HRM diagnostic smoke test"
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

smoke-hrm:
	PYTHONPATH=. python tools/hrm_diag_smoke.py

format:
	black src/ tests/ examples/ scripts/
	ruff check --fix src/ tests/ examples/ scripts/

lint:
	ruff check src/ tests/ examples/ scripts/
	black --check src/ tests/ examples/ scripts/

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .ruff_cache/

