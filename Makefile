.PHONY: help install test format lint clean smoke analyze

help:  ## Show this help message
	@echo "PoT: Pointer-over-Heads Transformer"
	@echo "===================================="
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

test:  ## Run all tests
	pytest tests/ -v --cov=src/pot --cov-report=term-missing

test-fast:  ## Run tests without coverage
	pytest tests/ -v

format:  ## Format code with black and isort
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

lint:  ## Lint code with flake8
	flake8 src/ tests/ scripts/ --max-line-length=100

clean:  ## Clean cache and temp files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov coverage.xml

# Task-specific smoke tests
smoke-sorting:  ## Quick smoke test for sorting task
	@echo "Running sorting smoke test (1 epoch, CPU)..."
	@python -c "print('✅ Sorting smoke test placeholder - implement in scripts/train.py')"

smoke-parsing:  ## Quick smoke test for dependency parsing
	@echo "Running parsing smoke test (1 epoch, CPU)..."
	@python -c "print('✅ Parsing smoke test placeholder - implement in scripts/train.py')"

smoke:  ## Run all smoke tests
	@echo "Running all smoke tests..."
	$(MAKE) smoke-sorting
	$(MAKE) smoke-parsing

# Analysis
analyze:  ## Analyze all experiment results
	python scripts/analyze.py

analyze-sorting:  ## Analyze sorting results only
	python scripts/analyze.py --task sorting

analyze-parsing:  ## Analyze parsing results only
	python scripts/analyze.py --task dependency

# Training shortcuts
train-sorting-len12:  ## Train sorting (length 12)
	python scripts/train.py --task sorting --config experiments/configs/sorting/len12.yaml

train-sorting-len20:  ## Train sorting (length 20)
	python scripts/train.py --task sorting --config experiments/configs/sorting/len20.yaml

train-parsing:  ## Train dependency parsing
	python scripts/train.py --task dependency --config experiments/configs/parsing/ud_en.yaml

# CI checks
ci:  ## Run all CI checks
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) smoke

# Development
dev-setup:  ## Set up development environment
	$(MAKE) install
	pip install -e ".[dev]"
	pre-commit install || echo "pre-commit not configured yet"

.DEFAULT_GOAL := help
