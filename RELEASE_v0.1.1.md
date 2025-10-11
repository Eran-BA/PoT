# Release v0.1.1 - CI & Documentation Polish

## 🎉 Highlights

This release focuses on CI/CD improvements, comprehensive testing, and project documentation.

## ✨ New Features

- **Full CI/CD Pipeline**: GitHub Actions workflow with linting, formatting, type checking, and tests
- **Test Suite**: Import tests and model tests with pytest coverage reporting
- **Community Guidelines**: CODE_OF_CONDUCT.md, SECURITY.md, CONTRIBUTING.md
- **GitHub Templates**: Issue templates (bug report, feature request) and PR template
- **README Badges**: CI status, license, Python version, and release version badges

## 🔧 Improvements

- **Black Formatting**: All code formatted with Black (line-length: 100)
- **Ruff Linting**: Configured with appropriate ignore rules for the project
- **Mypy Type Checking**: Type hints validated with relaxed settings for compatibility
- **Package Structure**: Fixed setup.py for proper package discovery
- **Test Discovery**: Added conftest.py and proper pytest configuration

## 🐛 Bug Fixes

- Fixed circular import issues in `src/__init__.py`
- Fixed `src/data` package not being tracked in git
- Fixed mypy unpacking errors in trainer.py
- Fixed Black formatting for Python 3.9-3.11 compatibility
- Removed `test_*.py` from .gitignore to allow test files

## 📦 CI/CD

- **Platforms**: Tests run on Python 3.9, 3.10, and 3.11
- **Checks**: Ruff linting, Black formatting, Mypy type checking, Pytest with coverage
- **Coverage**: Coverage reports uploaded as CI artifacts
- **Release Workflow**: Automated release artifact upload

## 📚 Documentation

- Enhanced README with badges and better structure
- Added comprehensive CONTRIBUTING guide
- Added training and loss flow diagram documentation
- Added API reference and usage guide

## 🙏 Acknowledgments

**Author:** Eran Ben Artzy - Creator of the Pointer-over-Heads innovation

---

**Full Changelog**: https://github.com/Eran-BA/PoT/compare/v0.1.0...v0.1.1
