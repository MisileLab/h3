# AGENTS.md

## Build & Test Commands
- **Install**: `uv sync` (with dev deps: `uv sync --dev`)
- **Lint**: `ruff check src/` and `ruff format --check src/`
- **Type check**: `mypy src/tsgb/`
- **Run all tests**: `pytest`
- **Run single test**: `pytest tests/test_file.py::test_function -v`
- **CLI**: `tsgb --help`

## Code Style
- **Python**: 3.11+ required, use union syntax `X | None` not `Optional[X]`
- **Line length**: 100 chars (ruff configured)
- **Imports**: Use `from tsgb.module import X`, sorted by ruff (isort rules)
- **Types**: Strict mypy enabled; use type hints on all functions
- **Dataclasses**: Prefer `@dataclass` for simple data containers
- **Settings**: Use pydantic-settings `BaseSettings` for config, load via `get_settings()`
- **Logging**: Use `structlog` via `from tsgb.logging import get_logger; logger = get_logger(__name__)`
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Errors**: Let exceptions propagate; log with structured context before raising
- **Docstrings**: Google style with Args/Returns sections
