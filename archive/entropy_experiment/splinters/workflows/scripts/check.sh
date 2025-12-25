#!/usr/bin/env bash
set -euo pipefail

python3 -m compileall yogimass scripts
python3 -m pytest "$@"
