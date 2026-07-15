#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# protoc_gen.sh – Compile the MassFlow .proto definitions into Python stubs.
#
# Usage:
#   uv run scripts/protoc_gen.sh
#
# Requires grpcio-tools (added in dev dependencies).  The script discovers
# the well-known protos bundled with grpcio-tools and generates both the
# message (pb2) and service (pb2_grpc) stubs.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROTO_DIR="$PROJECT_ROOT/protos"
OUT_DIR="$PROJECT_ROOT/src/MassFlow/streaming/generated"

# Locate grpcio-tools' bundled google/protobuf includes.
GRPC_TOOLS_DIR="$(uv run python -c 'import grpc_tools; print(grpc_tools.__path__[0])')"
WELL_KNOWN_PROTOS="$GRPC_TOOLS_DIR/_proto"

echo "==> Generating gRPC stubs from $PROTO_DIR"
echo "    Output: $OUT_DIR"
mkdir -p "$OUT_DIR"

uv run python -m grpc_tools.protoc \
    -I "$PROTO_DIR" \
    -I "$WELL_KNOWN_PROTOS" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    "$PROTO_DIR/massflow/v1/streaming.proto"

# Ensure the generated directory is importable.
touch "$OUT_DIR/__init__.py"
touch "$OUT_DIR/massflow/__init__.py"
touch "$OUT_DIR/massflow/v1/__init__.py"

# Patch the gRPC stub to use a relative import instead of absolute.
# The generated code writes ``from massflow.v1 import streaming_pb2`` but
# our package lives under ``MassFlow.streaming.generated.massflow.v1``.
GRPC_STUB="$OUT_DIR/massflow/v1/streaming_pb2_grpc.py"
if [[ -f "$GRPC_STUB" ]]; then
    sed -i '' 's/^from massflow\.v1 import streaming_pb2/from MassFlow.streaming.generated.massflow.v1 import streaming_pb2/' "$GRPC_STUB"
fi

echo "==> Done. Stubs written to $OUT_DIR"
