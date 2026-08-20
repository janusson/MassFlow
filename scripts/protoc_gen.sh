#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# protoc_gen.sh – Compile the MassFlow .proto definitions into Python stubs.
#
# Usage:
#   uv run scripts/protoc_gen.sh
#
# Requires grpcio-tools (added in dev dependencies).  The script discovers
# the well-known protos bundled with grpcio-tools and generates both the
# message (pb2) and service (pb2_grpc) stubs for:
#
#   * protos/massflow/v1/streaming.proto
#       -> src/MassFlow/streaming/generated  (gRPC real-time server)
#   * protos/massflow/v1/ml.proto
#       -> src/MassFlow/generated            (massflow-ml satellite boundary)
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROTO_DIR="$PROJECT_ROOT/protos"

# Locate grpcio-tools' bundled google/protobuf includes.
GRPC_TOOLS_DIR="$(uv run python -c 'import grpc_tools; print(grpc_tools.__path__[0])')"
WELL_KNOWN_PROTOS="$GRPC_TOOLS_DIR/_proto"

generate() {
    local proto_file="$1"
    local out_dir="$2"
    local import_path="$3"

    echo "==> Generating gRPC stubs from $proto_file"
    echo "    Output: $out_dir"
    mkdir -p "$out_dir"

    uv run python -m grpc_tools.protoc \
        -I "$PROTO_DIR" \
        -I "$WELL_KNOWN_PROTOS" \
        --python_out="$out_dir" \
        --grpc_python_out="$out_dir" \
        "$proto_file"

    # Ensure the generated directory is importable.
    touch "$out_dir/__init__.py"
    touch "$out_dir/massflow/__init__.py"
    touch "$out_dir/massflow/v1/__init__.py"

    # Patch the gRPC stub to use an absolute MassFlow import instead of the
    # bare `massflow.v1` package path emitted by protoc.
    local pb_name
    pb_name="$(basename "$proto_file" .proto)"
    local grpc_stub="$out_dir/massflow/v1/${pb_name}_pb2_grpc.py"
    if [[ -f "$grpc_stub" ]]; then
        sed -i '' "s|^from massflow\.v1 import ${pb_name}_pb2|from ${import_path} import ${pb_name}_pb2|" "$grpc_stub"
    fi
}

generate \
    "$PROTO_DIR/massflow/v1/streaming.proto" \
    "$PROJECT_ROOT/src/MassFlow/streaming/generated" \
    "MassFlow.streaming.generated.massflow.v1"

generate \
    "$PROTO_DIR/massflow/v1/ml.proto" \
    "$PROJECT_ROOT/src/MassFlow/generated" \
    "MassFlow.generated.massflow.v1"

echo "==> Done."
