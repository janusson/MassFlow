"""
Tests for MassFlow.tui.pipeline — the bridge to the core annotation modules.
"""

import logging
from pathlib import Path

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.io import save_spectra_to_mgf, save_spectra_to_msp
from MassFlow.tui.diagnostics import TuiError
from MassFlow.tui.pipeline import (
    capture_quarantine_records,
    inspect_library,
    load_query_preview,
    run_identification,
)
from MassFlow.tui.state import IdentificationRequest


def make_spectrum(
    spec_id: str,
    precursor_mz: float,
    seed: int,
    n_peaks: int = 20,
    **extra_metadata,
) -> Spectrum:
    """Deterministic synthetic spectrum with float64 peaks."""
    rng = np.random.default_rng(seed)
    mz = np.unique(
        np.round(np.sort(rng.uniform(40.0, precursor_mz * 0.9, size=n_peaks)), 2)
    )
    intensities = rng.uniform(1.0, 1000.0, size=mz.size)
    metadata = {
        "id": spec_id,
        "precursor_mz": float(precursor_mz),
        "compound_name": f"compound_{spec_id}",
        "adduct": "[M+H]+",
        "ionmode": "positive",
        "charge": 1,
        **extra_metadata,
    }
    return Spectrum(
        mz=mz.astype(np.float64),
        intensities=intensities.astype(np.float64),
        metadata=metadata,
    )


@pytest.fixture
def library_spectra() -> list[Spectrum]:
    return [make_spectrum(f"lib_{i}", 200.0 + 10.0 * i, seed=100 + i) for i in range(6)]


@pytest.fixture
def query_spectra() -> list[Spectrum]:
    reference = make_spectrum("lib_0", 200.0, seed=100)
    rng = np.random.default_rng(3)
    shifted = reference.peaks.mz + rng.uniform(-0.005, 0.005, reference.peaks.mz.size)
    return [
        Spectrum(
            mz=shifted.astype(np.float64),
            intensities=reference.peaks.intensities.copy(),
            metadata={
                "id": "query_1",
                "precursor_mz": 200.0,
                "adduct": "[M+H]+",
                "ionmode": "positive",
            },
        )
    ]


@pytest.fixture
def query_file(tmp_path: Path, query_spectra) -> Path:
    path = tmp_path / "query.mgf"
    save_spectra_to_mgf(query_spectra, path)
    return path


@pytest.fixture
def library_file(tmp_path: Path, library_spectra) -> Path:
    path = tmp_path / "library.msp"
    save_spectra_to_msp(library_spectra, path)
    return path


class TestCaptureQuarantineRecords:
    def test_captures_quarantine_log_records(self):
        with capture_quarantine_records() as captured:
            logging.getLogger("quarantine").warning("bad spectrum #1")
        assert captured == ["bad spectrum #1"]

    def test_caps_records(self):
        with capture_quarantine_records(max_records=2) as captured:
            for index in range(5):
                logging.getLogger("quarantine").warning(f"bad #{index}")
        assert captured == ["bad #0", "bad #1"]

    def test_handler_removed_after(self):
        logger = logging.getLogger("quarantine")
        handlers_before = list(logger.handlers)
        with capture_quarantine_records():
            pass
        assert list(logger.handlers) == handlers_before


class TestLoadQueryPreview:
    def test_valid_mgf(self, query_file: Path):
        result = load_query_preview(query_file)
        assert result.format_hint == "mgf"
        assert len(result.summaries) == 1
        assert result.summaries[0].spectrum_id == "query_1"
        assert result.summaries[0].precursor_mz == pytest.approx(200.0)
        assert result.quarantined_messages == []

    def test_max_spectra_cap(self, tmp_path: Path, library_spectra):
        path = tmp_path / "three.mgf"
        save_spectra_to_mgf(library_spectra[:3], path)
        result = load_query_preview(path, max_spectra=2)
        assert len(result.summaries) == 2

    def test_missing_file_raises_tui_error(self, tmp_path: Path):
        with pytest.raises(TuiError) as error:
            load_query_preview(tmp_path / "nope.mgf")
        assert error.value.stage == "load-query"
        assert error.value.hint is not None

    def test_directory_raises_tui_error(self, tmp_path: Path):
        with pytest.raises(TuiError) as error:
            load_query_preview(tmp_path)
        assert "directory" in str(error.value)
        assert error.value.hint is not None

    def test_vendor_format_raises_tui_error(self, tmp_path: Path):
        vendor = tmp_path / "run.raw"
        vendor.write_text("")
        with pytest.raises(TuiError) as error:
            load_query_preview(vendor)
        assert error.value.stage == "load-query"
        assert "convert" in (error.value.hint or "")

    def test_quarantined_spectra_reported(self, tmp_path: Path, query_spectra):
        bad = Spectrum(
            mz=np.array([50.0, 60.0], dtype=np.float64),
            intensities=np.array([1.0, 2.0], dtype=np.float64),
            metadata={"id": "no_precursor"},
        )
        path = tmp_path / "mixed.mgf"
        save_spectra_to_mgf([bad, *query_spectra], path)
        result = load_query_preview(path)
        assert len(result.summaries) == 1
        assert len(result.quarantined_messages) == 1
        assert (
            "no_precursor" in result.quarantined_messages[0]
            or "Missing" in result.quarantined_messages[0]
        )


class TestInspectLibrary:
    def test_text_library(self, library_file: Path):
        info = inspect_library(library_file)
        assert info.backend == "text"
        assert info.total_spectra == 6
        assert info.error is None
        assert info.precursor_mz_range == pytest.approx((200.0, 250.0))

    def test_missing_library_reports_error(self, tmp_path: Path):
        info = inspect_library(tmp_path / "nope.msp")
        assert info.error is not None
        assert info.total_spectra is None

    def test_sqlite_library(self, tmp_path: Path, library_spectra):
        from MassFlow.storage import create_spectral_store

        db_path = tmp_path / "library.db"
        store = create_spectral_store(db_path, backend="sqlite")
        store.add_spectra(library_spectra, category="test")
        store.close()

        info = inspect_library(db_path)
        assert info.backend == "sqlite"
        assert info.total_spectra == 6
        assert info.categories == {"test": 6}

    def test_zarr_library(self, tmp_path: Path, library_spectra):
        from MassFlow.storage import create_spectral_store

        store_path = tmp_path / "library.zarr"
        store = create_spectral_store(store_path, backend="zarr")
        store.add_spectra(library_spectra, category="ztest")
        store.close()

        info = inspect_library(store_path)
        assert info.backend == "zarr"
        assert info.total_spectra == 6


class TestRunIdentification:
    def test_end_to_end(self, query_file: Path, library_file: Path):
        request = IdentificationRequest(
            query_path=query_file,
            library_path=library_file,
            algorithm="modified_cosine",
            min_score=0.3,
            top_n=5,
            fdr_threshold=1.0,
        )
        outcome = run_identification(request)
        assert outcome.engine_used == "modified_cosine"
        assert outcome.num_queries == 1
        assert outcome.num_references == 6
        assert outcome.num_hits >= 1
        top = outcome.hits[0]
        assert top.query_id == "query_1"
        assert top.reference_name == "compound_lib_0"
        assert top.score > 0.9
        assert top.matched_peaks > 0
        # Mirror-plot peaks are included for the top hit.
        assert "query_1" in outcome.query_peaks
        assert "lib_0" in outcome.hit_reference_peaks

    def test_small_library_warning(self, query_file: Path, library_file: Path):
        request = IdentificationRequest(
            query_path=query_file,
            library_path=library_file,
            algorithm="cosine",
            min_score=0.3,
            top_n=5,
            fdr_threshold=0.05,
        )
        outcome = run_identification(request)
        assert any(
            "small" in warning.lower() or "FDR" in warning
            for warning in outcome.warnings
        )

    def test_unknown_algorithm_falls_back(
        self, query_file: Path, library_file: Path, monkeypatch
    ):
        import MassFlow.similarity

        real_factory = MassFlow.similarity.get_similarity_engine
        calls = {"count": 0}

        def flaky_factory(config):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError(
                    "This scoring engine requires the machine-learning extras. "
                    "Install them with: pip install massflow[ml]"
                )
            return real_factory(config)

        monkeypatch.setattr(MassFlow.similarity, "get_similarity_engine", flaky_factory)

        request = IdentificationRequest(
            query_path=query_file,
            library_path=library_file,
            algorithm="spec2vec",
            min_score=0.3,
            top_n=5,
            fdr_threshold=1.0,
        )
        outcome = run_identification(request)
        assert "fallback" in outcome.engine_used
        assert any(
            "fell back" in warning or "fallback" in warning
            for warning in outcome.warnings
        )

    def test_search_failure_raises_staged_error(
        self, query_file: Path, library_file: Path, monkeypatch
    ):
        import MassFlow.similarity

        class ExplodingEngine:
            def search(self, *args, **kwargs):
                raise ValueError("scoring blew up")

        monkeypatch.setattr(
            MassFlow.similarity,
            "get_similarity_engine",
            lambda config: ExplodingEngine(),
        )
        request = IdentificationRequest(
            query_path=query_file,
            library_path=library_file,
            algorithm="cosine",
            min_score=0.3,
            top_n=5,
        )
        with pytest.raises(TuiError) as error:
            run_identification(request)
        assert error.value.stage == "search"
        assert error.value.hint is None or "scoring blew up" in str(error.value)

    def test_empty_query_file_raises(self, tmp_path: Path, library_file: Path):
        empty = tmp_path / "empty.mgf"
        empty.write_text("")
        request = IdentificationRequest(query_path=empty, library_path=library_file)
        with pytest.raises(TuiError) as error:
            run_identification(request)
        assert error.value.stage in {"load-query", "search"}

    def test_missing_library_raises(self, query_file: Path, tmp_path: Path):
        request = IdentificationRequest(
            query_path=query_file, library_path=tmp_path / "nope.msp"
        )
        with pytest.raises(TuiError) as error:
            run_identification(request)
        assert error.value.stage == "load-library"

    def test_hits_respect_top_n(self, query_file: Path, tmp_path: Path):
        # A library of many near-identical references produces many hits;
        # top_n must bound the returned list.
        references = []
        for index in range(5):
            base = make_spectrum(f"dup_{index}", 200.0, seed=100 + index)
            rng = np.random.default_rng(100 + index)
            jittered_mz = base.peaks.mz + rng.uniform(-0.002, 0.002, base.peaks.mz.size)
            references.append(
                Spectrum(
                    mz=jittered_mz.astype(np.float64),
                    intensities=base.peaks.intensities.copy(),
                    metadata=dict(base.metadata),
                )
            )
        library_path = tmp_path / "dups.msp"
        save_spectra_to_msp(references, library_path)

        request = IdentificationRequest(
            query_path=query_file,
            library_path=library_path,
            algorithm="modified_cosine",
            min_score=0.9,
            top_n=2,
            fdr_threshold=1.0,
        )
        outcome = run_identification(request)
        assert outcome.num_hits <= 2
