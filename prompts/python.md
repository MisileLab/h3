# Python AI Agent Guidelines

## Code Standards
- Use modern type hints: `dict[str, int]` not `typing.Dict[str, int]`
- Use union syntax: `str | None` not `typing.Union[str, None]` 
- Use `list[T]`, `tuple[T, ...]`, `set[T]` over typing equivalents
- Use `@typing.override` decorator when overriding methods
- Use `@typing.final` decorator for general classes by default
- If a class needs to be overridden, explicitly type hint ALL attributes (even if inferable)
- Use Pydantic models with runtime validation instead of multiple type unions
- **Do not write tests for Pydantic validation** - trust Pydantic's built-in validation
- Python 3.9+ syntax preferred

## Best Practices
- Write clean, readable code with descriptive variable names
- Include type hints for function parameters and return values
- Use f-strings for string formatting
- Handle exceptions appropriately with specific exception types
- Add docstrings for complex functions
- Use `pathlib.Path` for file operations
- Use Polars for dataframe operations (not pandas)
- Prefer comprehensions over loops where readable

## Code Structure
- Import standard library first, then third-party, then local imports
- Use `if __name__ == "__main__":` for script entry points
- Keep functions focused and single-purpose
- Use constants for magic numbers/strings

## Error Handling
- Use specific exception types
- Provide meaningful error messages
- Log errors when appropriate
- Fail fast with clear feedback
- **Do not catch ImportError** - let import failures bubble up immediately

Check the main prompt file for additional context and specific requirements.