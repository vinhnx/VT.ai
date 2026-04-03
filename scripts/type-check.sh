#!/usr/bin/env bash
# Type checking with ty
set -e

echo "Running ty type checker..."
uv run ty check vtai

echo ""
echo "Type checking complete!"
