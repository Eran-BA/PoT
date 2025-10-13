# Contributing to PoT

Thank you for your interest in contributing to Pointer-over-Heads Transformer!

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Eran-BA/PoT.git
   cd PoT
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Editable install
   ```

3. **Run tests:**
   ```bash
   make test
   ```

## Project Structure

```
PoT/
├── src/pot/
│   ├── core/          # Task-agnostic architecture
│   ├── tasks/         # Task adapters (sorting, parsing, etc.)
│   └── utils/         # Utilities
├── scripts/           # Training/analysis scripts
├── experiments/       # Configs and results
├── tests/             # Unit tests
└── docs/              # Documentation
```

## Adding a New Task

To add a new task (e.g., `my_task`):

1. **Create task adapter:** `src/pot/tasks/my_task.py`
   ```python
   from .base import TaskAdapter
   
   class MyTask(TaskAdapter):
       def prepare_data(self, config):
           # Load datasets
           pass
       
       def build_model(self, config):
           # Build model
           pass
       
       def compute_loss(self, model_output, batch, config):
           # Compute loss
           pass
       
       def compute_metrics(self, model_output, batch, config):
           # Compute metrics
           return {'metric_name': value}
       
       def collate_fn(self, batch):
           # Collate batch
           pass
   ```

2. **Register task:** Add to `src/pot/tasks/__init__.py`
   ```python
   from .my_task import MyTask
   __all__ = [..., "MyTask"]
   ```

3. **Add config:** `experiments/configs/my_task/default.yaml`
   ```yaml
   task: my_task
   # ... task-specific config
   ```

4. **Update registry:** `experiments/registry.json`
   ```json
   {
     "tasks": {
       "my_task": {
         "default": {
           "config": "experiments/configs/my_task/default.yaml",
           "baseline": null,
           "hrm_poh": null,
           "description": "My task description"
         }
       }
     }
   }
   ```

5. **Add tests:** `tests/test_my_task.py`

6. **Run your task:**
   ```bash
   python scripts/train.py --task my_task --config experiments/configs/my_task/default.yaml
   ```

## Code Style

- Use **Black** for formatting: `make format`
- Use **isort** for imports: `isort src/ tests/`
- Use **flake8** for linting: `make lint`
- Follow PEP 8 conventions
- Add docstrings to all public functions/classes

## Testing

- Write unit tests for new code
- Ensure all tests pass: `pytest tests/`
- Aim for >80% code coverage

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `make test`
5. Format code: `make format`
6. Commit: `git commit -m "Add: my feature"`
7. Push: `git push origin feature/my-feature`
8. Open a Pull Request

## Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Keep first line under 72 characters
- Reference issues: "Fix #123: bug description"

## Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples to `docs/` for new features

## Questions?

Open an issue or discussion on GitHub!

---

**Happy coding! 🚀**
