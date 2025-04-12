# VT.ai Testing Guide

This directory contains tests for the VT.ai project, organized into unit and integration tests.

## Test Structure

- `unit/`: Contains unit tests that test individual components in isolation
- `integration/`: Contains integration tests that verify how components work together
- `conftest.py`: Contains shared pytest fixtures and configuration

## Running Tests

You can run tests using pytest:

```bash
# Run all tests
python -m pytest

# Run only unit tests
python -m pytest tests/unit

# Run only integration tests
python -m pytest tests/integration

# Run a specific test file
python -m pytest tests/unit/test_config.py

# Run with verbose output
python -m pytest -v

# Run with test coverage
python -m pytest --cov=vtai
```

## Adding New Tests

Follow these guidelines when adding new tests:

1. Place unit tests in the `unit/` directory
2. Place integration tests in the `integration/` directory
3. Use descriptive names for test files (e.g., `test_feature_name.py`)
4. Add common fixtures to `conftest.py`
5. Follow the naming convention `test_*` for test functions

## Test Dependencies

The required dependencies for testing are included in the project's optional dev dependencies.
You can install them with:

```bash
pip install -e ".[dev]"
```

or

```bash
pip install pytest pytest-cov
```
