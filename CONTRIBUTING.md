# Contributing to Contextify

Thank you for your interest in contributing to Contextify! This document provides guidelines and instructions for contributing.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/CocoRoF/Contextify.git
cd contextify
```

2. Create a virtual environment and install dependencies:
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Or using pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

3. Run tests:
```bash
python test_all_handlers.py
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for public functions and classes
- Keep functions focused and modular

## Testing

- Add tests for new features
- Ensure all tests pass before submitting a PR
- Test with multiple document formats when applicable

## Pull Request Process

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure everything works
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Document format being processed
- Minimal code to reproduce the issue
- Error messages and stack traces

## Feature Requests

We welcome feature requests! Please:
- Check if the feature already exists or is planned
- Provide a clear description of the feature
- Explain the use case and benefits

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
