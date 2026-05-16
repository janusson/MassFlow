from yogimass.similarity.backends import (
    create_index_backend,
    NaiveSpectrumIndex,
    logger,
)


def test_faiss_backend_falls_back_to_naive_and_logs(monkeypatch):
    recorded: list[str] = []

    def _record_warning(msg, *args, **kwargs):
        recorded.append(str(msg))

    monkeypatch.setattr(logger, "warning", _record_warning)

    backend = create_index_backend("faiss", entries=[])
    assert isinstance(backend, NaiveSpectrumIndex)
    # Assert that a warning about falling back was emitted
    assert any("falling back to naive" in msg.lower() for msg in recorded)
