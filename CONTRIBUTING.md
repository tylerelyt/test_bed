# Contributing Guide

Thank you for your interest in contributing! Please read this guide to ensure efficient collaboration and high-quality changes.

## Environment
- Python 3.9+
- Recommended: virtualenv

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -r requirements.txt
```

Optional (recommended):
- Formatter: black
- Linter: ruff
- Type checks: mypy (gradual adoption)
- Hooks: pre-commit

```bash
pip install black ruff mypy pre-commit
pre-commit install
```

## Run & Test
```bash
# Quick start
bash quick_start.sh  # or see docs/QUICK_START.md

# Unit tests
pytest -q
```

Examples and tools:
```bash
python examples/basic_usage.py
python tools/demo_data_generator.py
```

## Code Style
- Follow PEP 8 and existing project style.
- Use black for formatting and ruff for linting.
- Prefer meaningful names; handle edge cases and fail fast.
- Before submitting:
  - Lint/format/test should pass
  - Add/update tests for new behavior

## Commits & Branches
- Conventional Commits (e.g., `feat: ...`, `fix: ...`, `docs: ...`).
- Create feature branches from `main` (e.g., `feat/...`, `fix/...`).

## Pull Requests
1. Branch from the latest `main`.
2. Ensure local checks pass and docs/tests are updated.
3. Open a PR and include motivation, change summary, impact, and test notes.
4. At least 1 reviewer approval and green CI required.
5. Prefer Squash & Merge to keep history tidy.

## Data & Model Artifacts
- Large files (e.g., `models/`, `image_embeddings.npy`) should use Git LFS or DVC.
- Document how to obtain and reproduce datasets/models under `docs/`.

## Security
- Do not commit secrets (API keys, tokens, etc.).
- For security issues, contact maintainers privately rather than opening a public issue.

## Community
- Use GitHub Discussions for proposals or Q&A, or open a feature issue.
- All participants must follow the Code of Conduct (CODE_OF_CONDUCT.md).

Happy contributing!

