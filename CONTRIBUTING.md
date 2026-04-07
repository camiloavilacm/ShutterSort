# Contributing to ShutterSort

Thank you for your interest in contributing! This guide will help you get started.

## Quick Start

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/ShutterSort.git
cd ShutterSort

# 2. Create a virtual environment (isolates dependencies)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install in development mode with dev dependencies
pip install -e ".[dev]"

# 4. Run tests to verify everything works
pytest -m "not integration"
```

## Branching Model (Git-Flow)

We use Git-Flow for branch management:

```
main           ← Production releases (auto-publishes to PyPI)
develop        ← Staging branch (auto-publishes to TestPyPI)
feature/*      ← New features (PR → develop)
bugfix/*       ← Bug fixes (PR → develop)
hotfix/*       ← Urgent fixes (PR → main)
```

### Branch Naming

- `feature/add-batch-processing` — new features
- `bugfix/fix-arw-extraction` — bug fixes
- `docs/update-readme` — documentation changes
- `refactor/simplify-agent-base` — code improvements

## Pull Request Process

1. **Create a branch** from `develop` (or `main` for hotfixes)
2. **Make your changes** with type hints and inline comments
3. **Run the quality checks** locally before pushing:
   ```bash
   ruff check media_pruner/ tests/
   ruff format media_pruner/ tests/
   mypy media_pruner/
   pytest -m "not integration"
   ```
4. **Push your branch** and open a PR against `develop`
5. **Fill out the PR template** — it helps reviewers understand your changes
6. **Wait for review** — at least 1 approval is required
7. **Address feedback** — push additional commits to the same branch
8. **Merge** — once approved and CI passes, the branch can be merged

## Code Style

- **Type hints everywhere**: Every function parameter and return value must be typed
- **Inline comments**: Explain the "why" not the "what"
- **Docstrings**: All public classes and methods must have docstrings
- **Ruff**: We use Ruff for linting and formatting (configured in `pyproject.toml`)
- **mypy**: Strict type checking is enforced

### Example of Expected Code Style

```python
def compute_file_signature(file_path: Path) -> str:
    """Compute a unique signature for duplicate detection.

    The signature combines an MD5 hash of the first 1MB with the file size.
    This is faster than hashing the entire file while still being reliable.

    Args:
        file_path: Path to the file to hash.

    Returns:
        A string like "a1b2c3d4..._15728640".
    """
    file_size = file_path.stat().st_size
    with open(file_path, "rb") as f:
        chunk = f.read(1048576)
        file_hash = hashlib.md5(chunk).hexdigest()
    return f"{file_hash}_{file_size}"
```

## Testing

### Test Categories

| Type | Location | Purpose |
|------|----------|---------|
| Unit | `tests/test_*.py` | Test individual functions with mocks |
| Functional | `tests/test_functional.py` | Test full pipeline with mocked Ollama |
| Integration | `tests/test_integration.py` | Test with real Ollama (skippable) |

### Running Tests

```bash
# All tests except integration (the default CI run)
pytest -m "not integration"

# Only unit tests
pytest tests/test_librarian.py tests/test_curator.py tests/test_decision.py

# Only functional tests
pytest tests/test_functional.py

# With coverage report
pytest --cov=media_pruner --cov-report=html
```

### Writing Tests

- Use fixtures from `conftest.py` for common setup
- Mock external services (Ollama, cv2, rawpy) in unit tests
- Use `@pytest.mark.integration` for tests requiring real Ollama

## CI/CD Pipeline

Every PR triggers:

1. **Lint** — `ruff check`
2. **Format check** — `ruff format --check`
3. **Type check** — `mypy --strict`
4. **Tests** — `pytest` on Python 3.10–3.13, macOS + Ubuntu

Merges to `develop` auto-publish to **TestPyPI**.
Merges to `main` auto-publish to **PyPI**.

## Project Structure

```
media_pruner/
  __init__.py              # Package metadata
  models.py                # Typed dataclasses
  utils.py                 # Shared utilities
  agent_base.py            # MediaAgent ABC with retry loop
  agent_librarian.py       # File system operations
  agent_curator.py         # Vision model analysis
  agent_decision.py        # Interactive review UI
  cli.py                   # CLI entry point
tests/
  conftest.py              # Shared pytest fixtures
  test_librarian.py        # LibrarianAgent unit tests
  test_curator.py          # CuratorAgent unit tests
  test_decision.py         # DecisionAgent unit tests
  test_functional.py       # End-to-end pipeline tests
```

## Need Help?

- Open an [issue](https://github.com/camiloavilacm/ShutterSort/issues) for bugs or feature requests
- Read `LEARNING.md` for a deep dive into the architecture and Python concepts used
