"""
Targeted tests to push coverage above 92%.
Covers remaining gaps in config.py, io.py, processing.py, database.py, similarity.py.
"""

from unittest.mock import patch

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow import config, io, processing
from MassFlow.database import (
    SpectralDatabase,
    _decode_legacy_peaks_payload,
    _serialize_peak_arrays,
)


# ==============================================================================
# config.py - precursor_mz validator
# ==============================================================================


def test_processing_config_precursor_mz_negative():
    with pytest.raises(ValueError, match="must be non-negative"):
        config.ProcessingConfig(precursor_mz=-1.0)


# ==============================================================================
# io.py - edge cases
# ==============================================================================


def test_validate_iterator_non_numeric_precursor_quarantine(tmp_path):
    src = tmp_path / "test.mgf"
    src.touch()
    s = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensities=np.array([1.0, 1.0], dtype=np.float64),
        metadata={"precursor_mz": "NOT_A_NUMBER", "id": "bad_pmz"},
    )
    result = list(io._validate_spectra_iterator([s], src))
    assert len(result) == 0


def test_validate_iterator_non_tuple_pepmass(tmp_path):
    src = tmp_path / "test.mgf"
    src.touch()
    s = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensities=np.array([1.0, 1.0], dtype=np.float64),
        metadata={"pepmass": 200.0, "id": "scalar_pepmass"},
    )
    result = list(io._validate_spectra_iterator([s], src))
    assert len(result) == 1
    assert result[0].get("precursor_mz") == 200.0


def test_save_match_results_empty(tmp_path):
    with patch("MassFlow.io.logger") as mock_logger:
        io.save_match_results([], tmp_path / "empty.csv", query_spectra=None)
        mock_logger.warning.assert_called()


def test_save_match_results_to_mztab_empty(tmp_path):
    with patch("MassFlow.io.logger") as mock_logger:
        io.save_match_results_to_mztab([], tmp_path / "empty.mztab", query_spectra=None)
        mock_logger.warning.assert_called()


def test_build_results_dataframe_join_dedup():
    results = [
        {"query_id": "q1", "score": 0.95, "query_precursor_mz": 200.0},
    ]
    s = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"id": "q1", "precursor_mz": 200.0},
    )
    df = io._build_results_dataframe(results, query_spectra=[s])
    assert df is not None
    assert "query_precursor_mz_right" not in df.columns


# ==============================================================================
# processing.py - edge cases
# ==============================================================================


def test_metadata_processing_repair_inchi_returns_none():
    s = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"id": "fail_repair"},
    )
    with patch("matchms.filtering.repair_inchi_inchikey_smiles", return_value=None):
        result = processing.metadata_processing(s)
        assert result is None


def test_peak_processing_mz_range_filters_all():
    """When all peaks are outside m/z range, peaks are empty after filtering."""
    cfg = config.ProcessingConfig(
        filter_by_intensity=False,
        filter_by_mz=True,
        mz_min=0.0,
        mz_max=50.0,
        filter_min_peaks=True,
        min_peaks=1,
    )
    s = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensities=np.array([100.0, 200.0], dtype=np.float64),
        metadata={"id": "all_out_of_range", "precursor_mz": 150.0},
    )
    result = processing.peak_processing(s, cfg)
    # Result is either None or a spectrum with 0 peaks
    if result is not None:
        assert len(result.peaks.mz) == 0


def test_peak_processing_top_n_preserves_peaks():
    cfg = config.ProcessingConfig(
        reduce_to_top_n_peaks=True,
        n_max=100,
        filter_by_intensity=False,
        filter_min_peaks=False,
        filter_by_mz=False,
    )
    s = Spectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensities=np.array([100.0], dtype=np.float64),
        metadata={"id": "topn", "precursor_mz": 150.0},
    )
    result = processing.peak_processing(s, cfg)
    assert result is not None
    assert len(result.peaks.mz) == 1


def test_peak_processing_normalize_skipped_when_disabled():
    cfg = config.ProcessingConfig(
        normalize_intensity=False,
        filter_by_intensity=False,
        filter_min_peaks=False,
        filter_by_mz=False,
    )
    s = Spectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensities=np.array([100.0], dtype=np.float64),
        metadata={"id": "no_norm", "precursor_mz": 150.0},
    )
    result = processing.peak_processing(s, cfg)
    assert result is not None
    assert result.peaks.intensities[0] == 100.0  # Not normalized


def test_process_spectra_batch_exception_in_metadata_processing():
    s = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensities=np.array([1.0, 1.0], dtype=np.float64),
        metadata={"id": "will_fail", "precursor_mz": 200.0, "charge": 1},
    )
    cfg = config.ProcessingConfig(noise_threshold=0.0, min_peaks=1)

    with patch(
        "MassFlow.processing.metadata_processing", side_effect=RuntimeError("fail")
    ):
        result = processing.process_spectra_batch([s], cfg)
        assert len(result) == 0


def test_process_spectra_batch_exception_in_peak_processing():
    s = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensities=np.array([1.0, 1.0], dtype=np.float64),
        metadata={"id": "will_fail_peaks", "precursor_mz": 200.0, "charge": 1},
    )
    cfg = config.ProcessingConfig(noise_threshold=0.0, min_peaks=1)

    with patch("MassFlow.processing.peak_processing", side_effect=RuntimeError("fail")):
        result = processing.process_spectra_batch([s], cfg)
        assert len(result) == 0


# ==============================================================================
# database.py - edge cases
# ==============================================================================


def test_decode_legacy_dict_unknown_shape():
    with pytest.raises(ValueError, match="not recognized"):
        _decode_legacy_peaks_payload({"bad_key": "data"})


def test_decode_legacy_list_of_tuples():
    """Cover list-of-tuple-pairs format."""
    mz, intensity = _decode_legacy_peaks_payload([(100.0, 1.0), (200.0, 2.0)])
    np.testing.assert_array_almost_equal(mz, [100.0, 200.0])
    np.testing.assert_array_almost_equal(intensity, [1.0, 2.0])


def test_serialize_peak_arrays_preserves_shape():
    mz = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    intensity = np.array([0.5, 1.0, 0.3], dtype=np.float64)
    mz_blob, int_blob = _serialize_peak_arrays(mz, intensity)
    mz_restored = np.frombuffer(mz_blob, dtype=np.float64)
    int_restored = np.frombuffer(int_blob, dtype=np.float64)
    assert len(mz_restored) == 3
    assert len(int_restored) == 3


def test_add_spectra_with_triage_in_metadata(tmp_path):
    """Cover triage_flags storage during add_spectra."""
    db = SpectralDatabase(tmp_path / "test.db")
    s = Spectrum(
        mz=np.array([136.0, 200.0], dtype=np.float64),
        intensities=np.array([100.0, 50.0], dtype=np.float64),
        metadata={
            "id": "tyrosine_test",
            "precursor_mz": 200.0,
            "triage_flags": {"has_tyrosine_fragment": True},
        },
    )
    db.add_spectra([s], category="triage")
    retrieved = list(db.get_spectra(category="triage"))
    assert len(retrieved) == 1
    assert retrieved[0].get("has_tyrosine_fragment") is True
    db.close()


def test_add_spectra_batch_boundary(tmp_path):
    """Test batch that hits exactly batch_size boundary."""
    db = SpectralDatabase(tmp_path / "test.db")
    spectra = []
    for i in range(60):
        s = Spectrum(
            mz=np.array([100.0], dtype=np.float64),
            intensities=np.array([1.0], dtype=np.float64),
            metadata={"id": f"batch_{i}", "precursor_mz": 200.0},
        )
        spectra.append(s)

    count = db.add_spectra(iter(spectra), batch_size=30)
    assert count == 60
    total = db.get_total_spectra_count()
    assert total == 60
    db.close()


# ==============================================================================
# similarity.py - edge cases
# ==============================================================================


def test_yield_fixed_chunks_empty():
    from MassFlow.similarity import yield_fixed_chunks

    chunks = list(yield_fixed_chunks([], chunk_size=10000))
    assert len(chunks) == 0


def test_search_exception_handler():
    from MassFlow.similarity import SimilarityEngine, SimilarityConfig
    from matchms import Spectrum

    cfg = SimilarityConfig(
        algorithm="modified_cosine", min_score=0.0, ms2_tolerance=0.1
    )
    engine = SimilarityEngine(cfg)

    q = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensities=np.array([1.0, 1.0], dtype=np.float64),
        metadata={"id": "q", "precursor_mz": 200.0},
    )
    ref = Spectrum(
        mz=np.array([100.0, 200.0], dtype=np.float64),
        intensities=np.array([1.0, 1.0], dtype=np.float64),
        metadata={"id": "r", "precursor_mz": 200.0},
    )

    with patch(
        "MassFlow.similarity.calculate_scores",
        side_effect=RuntimeError("Vectorized fail"),
    ):
        with pytest.raises(RuntimeError, match="Vectorized fail"):
            engine.search([q], [ref], include_decoys=False)
