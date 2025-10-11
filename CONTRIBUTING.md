# Contributing to Pointer-over-Heads Transformer

Thank you for your interest in contributing to the Pointer-over-Heads Transformer (PoT) project!

## How to Contribute

### Reporting Issues

- Use the [issue tracker](https://github.com/Eran-BA/PoT/issues) to report bugs or suggest features
- Check existing issues before creating a new one
- Provide a clear description, reproduction steps, and expected vs actual behavior

### Pull Requests

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/PoT.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Make your changes** following our code style
5. **Run tests**: `pytest`
6. **Run linting**: `ruff check .` and `black --check .`
7. **Commit**: Use clear, descriptive commit messages
8. **Push**: `git push origin feature/your-feature-name`
9. **Open a Pull Request** with a clear description of your changes

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Eran-BA/PoT.git
cd PoT

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Code Style

- We use **Black** for code formatting (line length: 88)
- We use **Ruff** for linting
- We use **mypy** for type checking
- Add type hints to all function signatures
- Write docstrings for public functions and classes
- Pre-commit hooks will automatically check formatting

### Testing

- Write tests for new features
- Ensure all tests pass before submitting a PR
- Run tests with: `pytest`
- Check coverage with: `pytest --cov=src`

### Documentation

- Update relevant documentation for any changes
- Add docstrings to new functions and classes
- Update `README.md` if adding major features
- Add examples to `examples/` for new functionality

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat: add new routing mode`
- `fix: correct gradient flow in TRM mode`
- `docs: update architecture diagram`
- `test: add tests for deep supervision`
- `refactor: simplify loss computation`
- `chore: update dependencies`

### Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

### Questions?

Feel free to open an issue for questions or discussions. We're happy to help!

---

**Author:** Eran Ben Artzy  
**License:** Apache 2.0

