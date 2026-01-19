# Building and Publishing Contextify

This guide explains how to build and publish Contextify to PyPI.

## Prerequisites

1. Install build tools:
```bash
pip install build twine
# or
uv pip install build twine
```

2. Create a PyPI account:
   - Production: https://pypi.org/account/register/
   - Test: https://test.pypi.org/account/register/

3. Generate API token:
   - Go to Account Settings → API tokens
   - Create a token for the project
   - Save it securely (you'll only see it once)

## Building the Package

1. Clean previous builds:
```bash
# Windows
Remove-Item -Recurse -Force dist, build, *.egg-info

# Linux/Mac
rm -rf dist build *.egg-info
```

2. Build the package:
```bash
python -m build
```

This creates:
- `dist/contextify-X.Y.Z-py3-none-any.whl` (wheel distribution)
- `dist/contextify-X.Y.Z.tar.gz` (source distribution)

3. Verify the build:
```bash
twine check dist/*
```

## Testing Locally

Install the built package locally:
```bash
pip install dist/contextify-X.Y.Z-py3-none-any.whl
```

Test it:
```python
from libs.core.document_processor import DocumentProcessor

processor = DocumentProcessor()
text = processor.extract_text("test.pdf")
print(text)
```

## Publishing to Test PyPI (Recommended First)

1. Upload to Test PyPI:
```bash
twine upload --repository testpypi dist/*
```

2. Enter your Test PyPI credentials or token:
   - Username: `__token__`
   - Password: Your Test PyPI API token

3. Install from Test PyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ contextify
```

## Publishing to PyPI (Production)

1. Upload to PyPI:
```bash
twine upload dist/*
```

2. Enter your PyPI credentials or token:
   - Username: `__token__`
   - Password: Your PyPI API token

3. Verify on PyPI:
   - Visit: https://pypi.org/project/contextify/

4. Install from PyPI:
```bash
pip install contextify
```

## Using GitHub Actions (Automated Publishing)

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install build twine
      - name: Build package
        run: python -m build
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

Add your PyPI API token to GitHub Secrets:
- Go to repository Settings → Secrets → Actions
- Add secret: `PYPI_API_TOKEN`

## Version Bumping

Update version in `pyproject.toml`:
```toml
[project]
version = "X.Y.Z"
```

Follow [Semantic Versioning](https://semver.org/):
- MAJOR: Breaking changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes

## Checklist Before Publishing

- [ ] All tests pass
- [ ] Version number updated in `pyproject.toml`
- [ ] CHANGELOG.md updated
- [ ] README.md is current
- [ ] Build succeeds without warnings
- [ ] Tested installation locally
- [ ] Tested on Test PyPI
- [ ] Git tag created: `git tag -a v1.0.0 -m "Release 1.0.0"`
- [ ] Git tag pushed: `git push origin v1.0.0`

## Troubleshooting

### Import Error After Installation

Make sure the package name matches:
```python
# Correct
from libs.core.document_processor import DocumentProcessor

# If you want simpler imports, add to libs/__init__.py:
from libs.core.document_processor import DocumentProcessor
__all__ = ['DocumentProcessor']

# Then you can use:
from libs import DocumentProcessor
```

### Build Fails

- Check all dependencies are in `pyproject.toml`
- Ensure `__init__.py` files exist in all package directories
- Verify no syntax errors

### Upload Fails

- Check credentials/token
- Verify package name is available on PyPI
- Ensure version number hasn't been used before

## References

- [Python Packaging Guide](https://packaging.python.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [PyPI Help](https://pypi.org/help/)
