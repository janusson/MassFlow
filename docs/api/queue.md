# Streaming Queue

The `MassFlow.streaming.queue` module implements the quality-gated bounded
queue used by the gRPC streaming server for backpressure: when the queue
reaches its high-water mark, low-quality spectra are shed to prevent memory
exhaustion and latency collapse under instrument overrun.

## Security model

The streaming server (`MassFlow.streaming.server`) is a security-sensitive
subsystem. Its invariants:

- **Bind address**: loopback-only (`127.0.0.1`) by default. Non-loopback
  binds require TLS, or an explicit `--allow-insecure-remote` override that
  emits a prominent warning and an audit-log entry.
- **Transport**: `--tls-cert` / `--tls-key` enable TLS (both required
  together).
- **Control plane**: every `ControlMessage` requires the admin token
  (`--admin-token` / `MASSFLOW_ADMIN_TOKEN`, sent as
  `authorization: Bearer <token>` metadata). With no token configured the
  control plane is disabled entirely. `SET_CONFIG` / `LOAD_LIBRARY`
  additionally require `--allow-remote-control`; mutations apply
  transactionally (a failed reload leaves the running state intact).
- **Data plane**: spectrum ingestion is open to any client that can reach
  the port; responses are routed strictly back to the connection that sent
  the packet (annotations never cross clients).
- **Audit**: every administrative action (attempted or successful) is logged
  to `massflow.streaming.security` with peer, command, and outcome.
- **Limits**: configurable gRPC message size (`max_message_size_mb`,
  default 16 MiB), 1 MiB control-payload cap, and a per-packet peak-count
  cap; queue rejects are always reported to the client as explicit errors.

::: MassFlow.streaming.queue
