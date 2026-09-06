# Streaming gRPC Server

The `MassFlow.streaming.server` module implements the asynchronous gRPC
server for real-time spectral annotation. It hosts the
`massflow.v1.streaming` service (`StreamSpectra` bidirectional RPC plus
the `GetStatus` health probe) and emits structural annotations back to
instrument clients as soon as they are computed.

Incoming spectra are validated through a Pydantic/matchms gate, buffered
in a bounded async queue with quality-gated high-water-mark backpressure
(low-quality spectra are shed under overrun and reported as
`spectra_dropped_low_quality`), micro-batched, and routed through the
`ConsensusEngine` for weighted multi-engine scoring.

Start the server with the `stream-server` CLI command:

```bash
uv run massflow stream-server --config massflow_config.yaml
```

(`massflow serve` remains as a deprecated alias and prints a deprecation
notice.)

or directly with `uv run python -m MassFlow.streaming.server --config massflow_config.yaml` (protobuf stubs must be compiled first via `scripts/protoc_gen.sh`).

!!! note "LSP language server removed"

    The standalone LSP language server module `MassFlow.server` (IDE
    integration) is not part of the current codebase. Real-time
    server functionality is now provided exclusively by the gRPC
    streaming server in `MassFlow.streaming`.

::: MassFlow.streaming.server
