#!/usr/bin/env bash
set -euo pipefail

echo "Running MassFlow quickstart smoke test..."
uv run massflow annotate --config tutorial/tutorial_config.yaml

echo "Quickstart finished. Results in tutorial/results"
