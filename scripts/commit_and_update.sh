#!/usr/bin/env bash
set -euo pipefail

# Reformat
black .
# Run tests
python -m pytest -q
# Lint
flake8 .
# Add changes and show git status (do not commit automatically)
echo "Local updates applied; review and commit changes as needed"
