"""
Security tests for the MassFlow streaming (gRPC) subsystem.

Covers the security model documented in ``MassFlow.streaming.server``:

- bind address: loopback-only default; remote binds refused without TLS
  (or an explicit ``allow_insecure_remote`` override)
- transport: TLS support (server cert/key) with a real handshake test
- authentication: control-plane operations require the admin token
  (``authorization: Bearer <token>`` metadata); no token configured means
  the control plane is disabled entirely
- authorization: config/library *mutation* commands additionally require
  ``allow_remote_control``; mutations apply transactionally
- data plane: spectrum ingestion stays open and is independent of the
  control plane; responses never cross connections
- malformed / oversized / peak-count-capped messages
- queue exhaustion survival
- server restart restores the file-config state (control mutations are
  ephemeral)
- every administrative action is audit-logged
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from pathlib import Path

import pytest
import pytest_asyncio

try:
    from MassFlow.streaming.generated.massflow.v1 import (  # type: ignore[import-untyped]
        streaming_pb2 as pb,
        streaming_pb2_grpc as pb_grpc,
    )

    _HAS_GRPC_STUBS = True
except ImportError:  # pragma: no cover
    _HAS_GRPC_STUBS = False

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not _HAS_GRPC_STUBS, reason="gRPC stubs not generated"),
]

ADMIN_TOKEN = "test-admin-token"
SECURITY_LOGGER = "massflow.streaming.security"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_library(path: Path, names: list[str]) -> None:
    """Write a minimal 2-peak MSP library."""
    lines: list[str] = []
    for i, name in enumerate(names):
        lines += [
            f"NAME: {name}",
            f"PRECURSORMZ: {400.0 + i}",
            "IONMODE: Positive",
            "CHARGE: 1",
            "Num Peaks: 2",
            "150.0 800.0",
            "250.0 400.0",
            "",
        ]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def config_yaml(tmp_path: Path) -> str:
    """Minimal MassFlow config pointing at library A."""
    library_a = tmp_path / "library_a.msp"
    _write_library(library_a, ["Compound A1", "Compound A2"])
    dummy_input = tmp_path / "dummy.mzML"
    dummy_input.write_text("<!-- dummy -->")
    config_path = tmp_path / "stream_config.yaml"
    config_path.write_text(
        f"""project:
  name: "Security Test"
  output_directory: "{tmp_path}/output"

input:
  input_path: "{dummy_input}"
  library_path: "{library_a}"
  format: "msp"

processing:
  min_peaks: 1
  filter_min_peaks: false

similarity:
  algorithm: "cosine"
  ms1_tolerance: 0.5
  ms2_tolerance: 0.5
  min_score: 0.0
  min_matched_peaks: 1
"""
    )
    return str(config_path)


@pytest_asyncio.fixture
async def server_factory(config_yaml):
    """Factory creating a running server with configurable security params."""
    from MassFlow.config import MassFlowConfig
    from MassFlow.streaming.server import serve

    servers = []

    async def _make(
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        admin_token: str | None = ADMIN_TOKEN,
        allow_remote_control: bool = False,
        tls_cert_path: Path | None = None,
        tls_key_path: Path | None = None,
        max_message_size_mb: int = 16,
        max_control_message_bytes: int = 1024 * 1024,
        max_spectrum_peaks: int = 1_000_000,
        queue_capacity: int = 32,
        queue_put_timeout: float = 5.0,
    ):
        config = MassFlowConfig.from_yaml(config_yaml)
        if port is None:
            import socket

            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
        grpc_server = await serve(
            config,
            host=host,
            port=port,
            queue_capacity=queue_capacity,
            queue_put_timeout=queue_put_timeout,
            top_n=3,
            admin_token=admin_token,
            allow_remote_control=allow_remote_control,
            tls_cert_path=tls_cert_path,
            tls_key_path=tls_key_path,
            max_message_size_mb=max_message_size_mb,
            max_control_message_bytes=max_control_message_bytes,
            max_spectrum_peaks=max_spectrum_peaks,
        )
        servers.append(grpc_server)
        return grpc_server, port

    yield _make

    for grpc_server in servers:
        await grpc_server.stop(grace=1.0)


def _spectrum_request(spectrum_id: str, n_peaks: int = 5):
    """A valid spectrum request."""
    return pb.StreamRequest(  # type: ignore[attr-defined]
        spectrum=pb.SpectrumPacket(  # type: ignore[attr-defined]
            spectrum_id=spectrum_id,
            mz_array=[100.0 + i for i in range(n_peaks)],
            intensity_array=[float(1000 - i) for i in range(n_peaks)],
            precursor_mz=400.0,
            charge=1,
            ion_mode="positive",
            adduct="[M+H]+",
        )
    )


def _control_request(command, **kwargs):
    return pb.StreamRequest(  # type: ignore[attr-defined]
        control=pb.ControlMessage(command=command, **kwargs)  # type: ignore[attr-defined]
    )


def _metadata(token: str | None) -> tuple | None:
    if token is None:
        return None
    return (("authorization", f"Bearer {token}"),)


async def _collect_responses(stub, requests, *, token=ADMIN_TOKEN, timeout=15.0):
    """Stream *requests* and collect all responses."""

    async def gen():
        for req in requests:
            yield req

    call = stub.StreamSpectra(gen(), timeout=timeout, metadata=_metadata(token))
    responses = []
    async for resp in call:
        responses.append(resp)
    return responses


# ---------------------------------------------------------------------------
# Control-plane authentication & authorization
# ---------------------------------------------------------------------------


async def test_control_plane_disabled_without_token(server_factory):
    """No admin token configured → every control command is rejected and the
    data plane keeps working."""
    grpc_server, port = await server_factory(admin_token=None)
    import grpc

    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)
        responses = await _collect_responses(
            stub,
            [_control_request(pb.ControlMessage.COMMAND_STOP)],
            token=None,
        )
        assert len(responses) == 1
        assert responses[0].status == "error"
        assert "no admin token" in responses[0].error_message

        # The data plane is unaffected.
        responses = await _collect_responses(
            stub, [_spectrum_request("data_ok")], token=None
        )
        assert any(r.spectrum_id == "data_ok" for r in responses)


async def test_unauthorized_config_change_rejected(server_factory, tmp_path):
    """SET_CONFIG without a token (or with a wrong token) is rejected and the
    running configuration is untouched."""
    grpc_server, port = await server_factory(allow_remote_control=True)
    servicer = grpc_server._massflow_servicer
    library_b = tmp_path / "library_b.msp"
    _write_library(library_b, ["Compound B1"])
    evil_config = (
        f'input:\n  library_path: "{library_b}"\n'
        'similarity:\n  algorithm: "modified_cosine"\n'
    ).encode()

    import grpc

    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)

        # No token.
        responses = await _collect_responses(
            stub,
            [
                _control_request(
                    pb.ControlMessage.COMMAND_SET_CONFIG, config_yaml=evil_config
                )
            ],
            token=None,
        )
        assert responses[0].status == "error"
        assert "admin token" in responses[0].error_message

        # Wrong token.
        responses = await _collect_responses(
            stub,
            [
                _control_request(
                    pb.ControlMessage.COMMAND_SET_CONFIG, config_yaml=evil_config
                )
            ],
            token="wrong-token",
        )
        assert responses[0].status == "error"
        assert "admin token" in responses[0].error_message

    # The running config still points at library A.
    assert "library_a" in str(servicer._config.input.library_path)


async def test_unauthorized_library_change_rejected(server_factory, tmp_path):
    """LOAD_LIBRARY with a valid token but without allow_remote_control is
    rejected; the loaded library is untouched."""
    grpc_server, port = await server_factory(allow_remote_control=False)
    servicer = grpc_server._massflow_servicer
    library_b = tmp_path / "library_b.msp"
    _write_library(library_b, ["Compound B1"])
    original_path = servicer._config.input.library_path

    import grpc

    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)
        responses = await _collect_responses(
            stub,
            [
                _control_request(
                    pb.ControlMessage.COMMAND_LOAD_LIBRARY, library_path=str(library_b)
                )
            ],
            token=ADMIN_TOKEN,
        )
        assert responses[0].status == "error"
        assert "allow-remote-control" in responses[0].error_message

    assert servicer._config.input.library_path == original_path


async def test_authorized_config_and_library_change_applied(server_factory, tmp_path):
    """With a valid token AND allow_remote_control, mutations apply and are
    acknowledged."""
    import yaml

    grpc_server, port = await server_factory(allow_remote_control=True)
    servicer = grpc_server._massflow_servicer
    library_b = tmp_path / "library_b.msp"
    _write_library(library_b, ["Compound B1"])

    # A fully valid config (same shape as the fixture YAML) with the
    # library path replaced.  ``mode="json"`` stringifies Path values.
    new_config = servicer._config.model_copy(deep=True)
    new_config.input.library_path = library_b
    new_config_yaml = yaml.safe_dump(new_config.model_dump(mode="json")).encode()

    import grpc

    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)

        responses = await _collect_responses(
            stub,
            [
                _control_request(
                    pb.ControlMessage.COMMAND_SET_CONFIG, config_yaml=new_config_yaml
                )
            ],
            token=ADMIN_TOKEN,
        )
        assert responses[0].status == "control", responses[0].error_message
        assert str(servicer._config.input.library_path) == str(library_b)
        assert servicer._engine is not None

        # LOAD_LIBRARY back to library A.
        responses = await _collect_responses(
            stub,
            [
                _control_request(
                    pb.ControlMessage.COMMAND_LOAD_LIBRARY,
                    library_path=str(servicer._config.input.library_path),
                )
            ],
            token=ADMIN_TOKEN,
        )
        assert responses[0].status == "control"
        assert "library reloaded" in responses[0].error_message


async def test_invalid_config_rejected_transactionally(server_factory):
    """A SET_CONFIG that fails validation/reload leaves the old state intact."""
    grpc_server, port = await server_factory(allow_remote_control=True)
    servicer = grpc_server._massflow_servicer
    original_config = servicer._config

    import grpc

    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)

        # Malformed YAML.
        responses = await _collect_responses(
            stub,
            [
                _control_request(
                    pb.ControlMessage.COMMAND_SET_CONFIG,
                    config_yaml=b"::: not yaml :::",
                )
            ],
            token=ADMIN_TOKEN,
        )
        assert responses[0].status == "error"
        assert "SET_CONFIG rejected" in responses[0].error_message

        # Valid YAML pointing at a nonexistent library (engine build fails).
        responses = await _collect_responses(
            stub,
            [
                _control_request(
                    pb.ControlMessage.COMMAND_SET_CONFIG,
                    config_yaml=b'input:\n  library_path: "/nonexistent/nowhere.msp"\n',
                )
            ],
            token=ADMIN_TOKEN,
        )
        assert responses[0].status == "error"

    # Nothing changed.
    assert servicer._config is original_config


async def test_load_library_missing_path_rejected(server_factory):
    grpc_server, port = await server_factory(allow_remote_control=True)
    import grpc

    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)
        responses = await _collect_responses(
            stub,
            [
                _control_request(
                    pb.ControlMessage.COMMAND_LOAD_LIBRARY,
                    library_path="/nonexistent/library.msp",
                )
            ],
            token=ADMIN_TOKEN,
        )
        assert responses[0].status == "error"
        assert "library not found" in responses[0].error_message


async def test_unknown_control_command_rejected(server_factory):
    grpc_server, port = await server_factory()
    import grpc

    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)
        responses = await _collect_responses(
            stub,
            [_control_request(pb.ControlMessage.COMMAND_UNSPECIFIED)],
            token=ADMIN_TOKEN,
        )
        assert responses[0].status == "error"
        assert "Unknown control command" in responses[0].error_message


async def test_oversized_control_payload_rejected(server_factory):
    grpc_server, port = await server_factory(
        allow_remote_control=True, max_control_message_bytes=1024
    )
    import grpc

    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)
        responses = await _collect_responses(
            stub,
            [
                _control_request(
                    pb.ControlMessage.COMMAND_SET_CONFIG,
                    config_yaml=b"x" * 4096,
                )
            ],
            token=ADMIN_TOKEN,
        )
        assert responses[0].status == "error"
        assert "control message limit" in responses[0].error_message


async def test_data_plane_open_without_token(server_factory):
    """The data plane is independent of the control plane: spectra stream
    without a token even when the control plane is protected."""
    grpc_server, port = await server_factory(admin_token=ADMIN_TOKEN)
    import grpc

    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)
        responses = await _collect_responses(
            stub, [_spectrum_request("plain_spectrum")], token=None
        )
        assert any(r.spectrum_id == "plain_spectrum" for r in responses)


# ---------------------------------------------------------------------------
# Bind address & transport security
# ---------------------------------------------------------------------------


async def test_remote_binding_refused_without_tls(server_factory, config_yaml):
    """A non-loopback bind without TLS raises SecurityConfigurationError."""
    from MassFlow.config import MassFlowConfig
    from MassFlow.streaming.server import SecurityConfigurationError, serve

    config = MassFlowConfig.from_yaml(config_yaml)
    with pytest.raises(SecurityConfigurationError, match="Refusing to bind"):
        await serve(config, host="0.0.0.0", port=0)

    with pytest.raises(SecurityConfigurationError, match="Refusing to bind"):
        await serve(config, host="[::]", port=0)


async def test_insecure_remote_requires_explicit_override(server_factory):
    """With allow_insecure_remote=True the validation passes (the prominent
    warning is emitted at startup)."""
    from MassFlow.streaming.server import _validate_bind_config

    assert _validate_bind_config("0.0.0.0", None, None, True) is False


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl not available")
async def test_tls_requires_both_cert_and_key(tmp_path, config_yaml):
    from MassFlow.streaming.server import (
        SecurityConfigurationError,
        _validate_bind_config,
    )

    cert = tmp_path / "cert.pem"
    cert.write_text("not a real cert")
    with pytest.raises(SecurityConfigurationError, match="both --tls-cert"):
        _validate_bind_config("0.0.0.0", cert, None, False)
    with pytest.raises(SecurityConfigurationError, match="both --tls-cert"):
        _validate_bind_config("127.0.0.1", cert, None, False)


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl not available")
def _self_signed_cert(tmp_path: Path) -> tuple[Path, Path]:
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=127.0.0.1",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl not available")
async def test_tls_handshake(server_factory, tmp_path):
    """A TLS server accepts secure clients and rejects plaintext ones."""
    cert, key = _self_signed_cert(tmp_path)
    grpc_server, port = await server_factory(tls_cert_path=cert, tls_key_path=key)
    import grpc

    target = f"127.0.0.1:{port}"
    # Secure channel trusting the self-signed cert.
    root_cert = cert.read_bytes()
    creds = grpc.ssl_channel_credentials(root_certificates=root_cert)
    async with grpc.aio.secure_channel(target, creds) as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)
        from google.protobuf import empty_pb2

        status = await stub.GetStatus(empty_pb2.Empty())
        assert status.is_active is True

    # Plaintext client must fail against a TLS-only port.
    async with grpc.aio.insecure_channel(target) as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)
        from google.protobuf import empty_pb2

        with pytest.raises(grpc.RpcError):
            await asyncio.wait_for(stub.GetStatus(empty_pb2.Empty()), timeout=5.0)


# ---------------------------------------------------------------------------
# Malformed / oversized messages
# ---------------------------------------------------------------------------


async def test_oversized_message_rejected(server_factory):
    """Messages above the configured gRPC limit are rejected at the wire."""
    grpc_server, port = await server_factory(max_message_size_mb=1)
    import grpc

    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)
        # ~1.6 MB of float64 peaks > 1 MiB server cap.
        big = pb.StreamRequest(
            spectrum=pb.SpectrumPacket(
                spectrum_id="big",
                mz_array=[100.0] * 200_000,
                intensity_array=[1.0] * 200_000,
                precursor_mz=400.0,
            )
        )
        with pytest.raises(grpc.RpcError) as excinfo:
            await _collect_responses(stub, [big], token=None, timeout=10.0)
        assert excinfo.value.code() in (
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            grpc.StatusCode.INTERNAL,
        )


async def test_spectrum_peak_cap_enforced(server_factory):
    """Packets above max_spectrum_peaks are rejected with an error response
    before they can occupy queue memory."""
    grpc_server, port = await server_factory(max_spectrum_peaks=100)
    import grpc

    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)
        responses = await _collect_responses(
            stub, [_spectrum_request("too_many_peaks", n_peaks=500)], token=None
        )
        assert responses[0].status == "error"
        assert "peak-count limit" in responses[0].error_message


# ---------------------------------------------------------------------------
# Queue exhaustion
# ---------------------------------------------------------------------------


async def test_queue_exhaustion_server_survives(server_factory):
    """A full queue rejects spectra explicitly and the server stays healthy."""
    grpc_server, port = await server_factory(queue_capacity=4, queue_put_timeout=0.05)
    servicer = grpc_server._massflow_servicer

    # Fill the queue without a consumer running.  High-quality packets
    # (10 peaks, ion mode set) so the high-water-mark gate does not shed
    # them before the queue reaches capacity.
    from MassFlow.streaming.queue import QueuedPacket

    for i in range(4):
        await servicer._queue.put(
            QueuedPacket(
                spectrum_id=f"fill_{i}",
                mz_array=[100.0 + j for j in range(10)],
                intensity_array=[float(1000 - j) for j in range(10)],
                precursor_mz=400.0,
                retention_time_seconds=0.0,
                charge=1,
                ion_mode="positive",
                adduct="[M+H]+",
                collision_energy=0.0,
                acquisition_timestamp_ns=0,
            )
        )
    assert servicer._queue.is_full

    # One more packet through the servicer path → explicit rejection.
    response_queue: asyncio.Queue = asyncio.Queue()
    await servicer._ingest_spectrum(
        _spectrum_request("overflow").spectrum, "test-conn", response_queue
    )
    response = response_queue.get_nowait()
    assert response.status == "error"
    assert "queue" in response.error_message
    assert servicer._queue.stats.total_dropped >= 1

    # The gRPC server is still healthy.
    import grpc

    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)
        from google.protobuf import empty_pb2

        status = await stub.GetStatus(empty_pb2.Empty())
        assert status.spectra_dropped >= 1


# ---------------------------------------------------------------------------
# Server restart
# ---------------------------------------------------------------------------


async def test_server_restart_restores_file_config(
    server_factory, config_yaml, tmp_path
):
    """Control-plane mutations are ephemeral: a restart returns to the
    file-config state."""
    from MassFlow.config import MassFlowConfig
    from MassFlow.streaming.server import serve

    library_b = tmp_path / "library_b.msp"
    _write_library(library_b, ["Compound B1"])

    import grpc
    import socket

    # Server 1: mutate via LOAD_LIBRARY.
    grpc_server, port = await server_factory(allow_remote_control=True)
    servicer = grpc_server._massflow_servicer
    original_path = servicer._config.input.library_path

    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)
        responses = await _collect_responses(
            stub,
            [
                _control_request(
                    pb.ControlMessage.COMMAND_LOAD_LIBRARY, library_path=str(library_b)
                )
            ],
            token=ADMIN_TOKEN,
        )
        assert responses[0].status == "control"
    assert servicer._config.input.library_path == library_b
    await grpc_server.stop(grace=1.0)

    # Server 2: fresh start from the same config file → library A.
    config = MassFlowConfig.from_yaml(config_yaml)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port2 = sock.getsockname()[1]
    grpc_server2 = await serve(
        config,
        host="127.0.0.1",
        port=port2,
        admin_token=ADMIN_TOKEN,
        allow_remote_control=True,
    )
    try:
        servicer2 = grpc_server2._massflow_servicer
        assert servicer2._config.input.library_path == original_path
    finally:
        await grpc_server2.stop(grace=1.0)


# ---------------------------------------------------------------------------
# Cross-client isolation (data-plane integrity)
# ---------------------------------------------------------------------------


async def test_cross_client_response_isolation(server_factory):
    """Two concurrent clients each receive ONLY their own annotations —
    responses never cross connections."""
    grpc_server, port = await server_factory()
    import grpc

    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)

        async def client_a():
            async def gen():
                for i in range(5):
                    yield _spectrum_request(f"client_a_{i}")

            call = stub.StreamSpectra(gen(), timeout=20.0)
            return [r.spectrum_id async for r in call]

        async def client_b():
            async def gen():
                for i in range(5):
                    yield _spectrum_request(f"client_b_{i}")

            call = stub.StreamSpectra(gen(), timeout=20.0)
            return [r.spectrum_id async for r in call]

        ids_a, ids_b = await asyncio.gather(client_a(), client_b())

    assert {f"client_a_{i}" for i in range(5)} <= set(ids_a)
    assert {f"client_b_{i}" for i in range(5)} <= set(ids_b)
    assert not any("client_b" in sid for sid in ids_a), (
        f"Client A received client B's responses: {ids_a}"
    )
    assert not any("client_a" in sid for sid in ids_b), (
        f"Client B received client A's responses: {ids_b}"
    )


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


async def test_control_actions_are_audit_logged(server_factory, caplog):
    """Every administrative action (accepted or rejected) is recorded in the
    audit logger with command, peer, and outcome."""
    grpc_server, port = await server_factory(allow_remote_control=True)
    import grpc

    caplog.set_level(logging.INFO, logger=SECURITY_LOGGER)
    async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
        stub = pb_grpc.MassFlowStreamingStub(channel)

        # Accepted STOP with a valid token.
        await _collect_responses(
            stub,
            [_control_request(pb.ControlMessage.COMMAND_STOP)],
            token=ADMIN_TOKEN,
        )
        # Rejected SET_CONFIG without a token.
        await _collect_responses(
            stub,
            [
                _control_request(
                    pb.ControlMessage.COMMAND_SET_CONFIG, config_yaml=b"x: 1"
                )
            ],
            token=None,
        )

    records = [r for r in caplog.records if r.name == SECURITY_LOGGER]
    accepted = [r for r in records if r.getMessage().startswith("control accepted")]
    rejected = [r for r in records if r.getMessage().startswith("control rejected")]
    assert any(
        "COMMAND_STOP" in r.getMessage() and "peer=" in r.getMessage() for r in accepted
    )
    assert any(
        "COMMAND_SET_CONFIG" in r.getMessage() and "peer=" in r.getMessage()
        for r in rejected
    )
    assert any("admin token" in r.getMessage() for r in rejected)
