# Contributing to PoH

Thank you for your interest in contributing! This guide will help you get started.

---

## 📋 Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Code Style](#code-style)
5. [Testing](#testing)
6. [Submitting Changes](#submitting-changes)
7. [Adding New Features](#adding-new-features)

---

## Code of Conduct

Be respectful, constructive, and professional. We're all here to learn and improve.

---

## Getting Started

### Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/PoT.git
cd PoT

# Add upstream remote
git remote add upstream https://github.com/Eran-BA/PoT.git
```

### Create a Branch

```bash
# Create a feature branch
git checkout -b feature/my-new-feature

# Or a bugfix branch
git checkout -b fix/issue-123
```

---

## Development Setup

### Install Dependencies

```bash
# Install core dependencies
pip install torch numpy matplotlib seaborn scipy pandas pytest

# Install development tools
pip install black ruff mypy pre-commit

# Optional: RoPE support
pip install rotary-embedding-torch
```

### Pre-commit Hooks (Optional)

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Code Style

### Python Style

We follow **PEP 8** with a few modifications:

- **Line length:** 100 characters (not 79)
- **Imports:** Use absolute imports from `src.pot.*`
- **Type hints:** Encouraged for public APIs
- **Docstrings:** Google style for all public functions/classes

### Formatting

```bash
# Auto-format with black
black src/ tests/ examples/ scripts/

# Lint with ruff
ruff check src/ tests/

# Type check with mypy (optional)
mypy src/pot/
```

### Example

```python
"""Module docstring."""

from typing import Optional, Tuple
import torch
import torch.nn as nn

def my_function(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Short one-line summary.
    
    Longer description if needed.
    
    Args:
        x: Input tensor [B, T, D]
        mask: Optional attention mask [B, T]
    
    Returns:
        out: Output tensor [B, T, D]
        stats: Statistics dictionary
    """
    # Implementation
    return out, stats
```

---

## Testing

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_poh_modules.py -v

# Run specific test
pytest tests/test_poh_modules.py::TestRouting::test_soft_routing_sums_to_one -v

# With coverage
pytest tests/ --cov=src/pot --cov-report=term-missing
```

### Writing Tests

Place tests in `tests/` with filename `test_*.py`:

```python
# tests/test_my_feature.py
import torch
import pytest
from src.pot.modules import PoHConfig, PoHBlock

class TestMyFeature:
    """Test suite for my feature."""
    
    def test_basic_functionality(self):
        """Test that basic functionality works."""
        cfg = PoHConfig(d_model=64, n_heads=4)
        block = PoHBlock(cfg)
        
        x = torch.randn(2, 10, 64)
        out, stats = block(x)
        
        assert out.shape == x.shape
        assert "route_entropy_mean" in stats
    
    def test_edge_case(self):
        """Test edge case behavior."""
        # Test implementation
        pass
```

**Guidelines:**
- One test class per feature
- Descriptive test names (test_*_does_what)
- Test both success and failure cases
- Use small tensors (faster tests)
- Add docstrings for complex tests

---

## Submitting Changes

### Before You Submit

1. ✅ **Tests pass:** `pytest tests/ -v`
2. ✅ **Code formatted:** `black src/ tests/`
3. ✅ **No linter errors:** `ruff check src/`
4. ✅ **Documentation updated:** Docstrings, README, etc.
5. ✅ **Commit messages are clear:** See style guide below

### Commit Message Style

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Adding or updating tests
- `refactor:` Code refactoring (no behavior change)
- `perf:` Performance improvement
- `chore:` Maintenance (dependencies, tooling)

**Example:**

```
feat: Add sinusoidal positional encoding

- Implement SinusoidalPositionalEncoding class
- Add tests for fixed vs learned embeddings
- Update PoHConfig with pos_encoding="sinusoidal" option

Closes #42
```

### Pull Request

1. **Push your branch:**
   ```bash
   git push origin feature/my-new-feature
   ```

2. **Open PR on GitHub:**
   - Title: Clear, concise description
   - Description: What, why, how
   - Link to related issues

3. **PR checklist:**
   - [ ] Tests added/updated
   - [ ] Documentation updated
   - [ ] All tests pass
   - [ ] Code formatted and linted
   - [ ] Linked to issue (if applicable)

4. **Wait for review:**
   - Address review comments
   - Push updates to the same branch
   - Squash commits before merge (if requested)

---

## Adding New Features

### Architecture: New Module

1. **Create module in `src/pot/modules/`:**
   ```python
   # src/pot/modules/my_module.py
   import torch.nn as nn
   
   class MyModule(nn.Module):
       """My new module."""
       def __init__(self, cfg):
           super().__init__()
           # Implementation
       
       def forward(self, x):
           # Implementation
           return x
   ```

2. **Export in `src/pot/modules/__init__.py`:**
   ```python
   from .my_module import MyModule
   
   __all__ = [..., "MyModule"]
   ```

3. **Add tests:**
   ```python
   # tests/test_my_module.py
   from src.pot.modules import MyModule
   
   def test_my_module():
       module = MyModule(cfg)
       # Test implementation
   ```

4. **Add usage example:**
   ```python
   # examples/my_module_usage.py
   # Usage example
   ```

### Task: New Task Adapter

1. **Create task in `src/pot/tasks/`:**
   ```python
   # src/pot/tasks/my_task.py
   from .base import TaskAdapter
   
   class MyTask(TaskAdapter):
       """My task adapter."""
       # Implementation following base interface
   ```

2. **Add config:**
   ```yaml
   # experiments/configs/my_task/default.yaml
   task: my_task
   model: poh
   # ... task-specific config
   ```

3. **Add to task registry:**
   ```python
   # src/pot/tasks/__init__.py
   TASK_REGISTRY = {
       "dependency": DependencyTask,
       "sorting": SortingTask,
       "my_task": MyTask,  # Add here
   }
   ```

---

## Documentation

### Docstrings

Use **Google style**:

```python
def function(arg1: int, arg2: str, flag: bool = False) -> dict:
    """
    Short one-line summary.
    
    Longer description if needed. Can span multiple lines and include
    examples, references, etc.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        flag: Description of flag (default: False)
    
    Returns:
        Dictionary with results:
        - "key1": Description
        - "key2": Description
    
    Raises:
        ValueError: If arg1 < 0
    
    Examples:
        >>> result = function(42, "hello")
        >>> print(result["key1"])
        42
    """
    # Implementation
```

### README Updates

If your PR changes user-facing behavior, update:
- `README.md` (main documentation)
- Relevant docs in `docs/architecture/` or `docs/guides/`
- `docs/architecture/ARCHITECTURE_COMPLETE.md` (if architecture changed)
- Usage examples

---

## Issue Reporting

### Bug Reports

Include:
- **Description:** What went wrong?
- **Reproduction:** Minimal code to reproduce
- **Expected behavior:** What should happen?
- **Actual behavior:** What actually happened?
- **Environment:** OS, Python version, PyTorch version, GPU/CPU

**Template:**
```markdown
**Bug description:**
Routing entropy is NaN after 10 iterations.

**Reproduction:**
```python
cfg = PoHConfig(route_mode="soft", route_temp=0.0)  # Bug: temp=0!
stack = PoHStack(cfg, depth=6)
# ...
```

**Expected:** Routing entropy should be a valid float.
**Actual:** NaN after iteration 10.

**Environment:**
- OS: Ubuntu 22.04
- Python: 3.10.12
- PyTorch: 2.0.1+cu118
- GPU: RTX 3090
```

### Feature Requests

Include:
- **Use case:** Why is this useful?
- **Proposed solution:** How should it work?
- **Alternatives:** Other ways to solve it?

---

## Development Tips

### Debugging

```python
# Add temporary debug prints
print(f"[DEBUG] x.shape={x.shape}, stats={stats.keys()}")

# Use pdb for interactive debugging
import pdb; pdb.set_trace()

# Profile performance
import time
t0 = time.perf_counter()
out = model(x)
print(f"Forward took {time.perf_counter() - t0:.3f}s")
```

### Profiling

```python
# PyTorch profiler
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    out = model(x)

print(prof.key_averages().table(sort_by="cpu_time_total"))
```

### Memory Debugging

```python
# Check memory usage
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Find memory leaks
import gc
gc.collect()
torch.cuda.empty_cache()
```

---

## Questions?

- **General questions:** Open a GitHub discussion
- **Bug reports:** Open a GitHub issue
- **Feature requests:** Open a GitHub issue
- **Security issues:** Email directly (do not open public issue)

---

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

**Thank you for contributing!** 🎉
