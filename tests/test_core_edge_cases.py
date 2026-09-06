import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import ProcessingConfig, SimilarityConfig
from MassFlow.processing import (
    compute_spectral_metrics,
    metadata_processing,
    peak_processing,
    process_spectra,
    process_spectra_batch,
)
from MassFlow.similarity import SimilarityEngine, generate_decoys


def test_generate_decoys_edge_cases():
    # Spectrum with 1 peak -> hits < 2 unique peaks branch
    spec1 = Spectrum(
        mz=np.array([100.0]), intensities=np.array([1.0]), metadata={"id": "spec1"}
    )

    # Spectrum with 3 identical peaks -> hits < 2 unique peaks branch
    spec2 = Spectrum(
        mz=np.array([100.0, 200.0, 300.0]),
        intensities=np.array([1.0, 1.0, 1.0]),
        metadata={"id": "spec2", "compound_name": "Test"},
    )

    decoys = generate_decoys([spec1, spec2])
    assert len(decoys) == 2
    assert decoys[0].metadata["is_decoy"] is True
    assert decoys[0].metadata["id"] == "spec1_decoy"
    assert len(decoys[0].peaks.intensities) == 1

    assert decoys[1].metadata["compound_name"] == "Test_decoy"
    # Intensities should be altered (randomised taper for uniform-intensity spectra).
    # The old deterministic linspace taper is replaced with shuffled uniform
    # multipliers, so no individual position is guaranteed a fixed value.
    d2_ints = decoys[1].peaks.intensities
    assert not np.array_equal(d2_ints, spec2.peaks.intensities), (
        "Decoy intensities must differ from target intensities"
    )
    assert len(np.unique(d2_ints.round(decimals=4))) > 1, (
        "Tapered intensities should not all be equal"
    )


def test_similarity_engine_empty_search():
    config = SimilarityConfig(algorithm="cosine")
    engine = SimilarityEngine(config)

    # Empty queries
    assert (
        engine.search(
            [], [Spectrum(mz=np.array([1.0]), intensities=np.array([1.0]), metadata={})]
        )
        == []
    )

    # Empty references
    assert (
        engine.search(
            [Spectrum(mz=np.array([1.0]), intensities=np.array([1.0]), metadata={})], []
        )
        == []
    )


@pytest.fixture
def empty_spectrum():
    return Spectrum(
        mz=np.array([], dtype=float),
        intensities=np.array([], dtype=float),
        metadata={"id": "empty_spec", "precursor_mz": 100.0, "charge": 1},
    )


@pytest.fixture
def zero_intensity_spectrum():
    return Spectrum(
        mz=np.array([100.0, 200.0, 300.0]),
        intensities=np.array([0.0, 0.0, 0.0]),
        metadata={"id": "zero_spec", "precursor_mz": 150.0, "charge": 1},
    )


@pytest.fixture
def malformed_metadata_spectrum():
    return Spectrum(
        mz=np.array([100.0, 200.0]),
        intensities=np.array([1.0, 1.0]),
        metadata={"id": "malformed_spec"},  # Missing precursor_mz, charge, etc.
    )


def test_compute_spectral_metrics_edge_cases():
    mz_arr = np.array([100.0, 200.0])

    # Missing precursor mz
    nl, offset = compute_spectral_metrics(mz_arr, None)
    assert len(nl) == 0
    assert len(offset) == 0

    # Negative precursor mz
    nl, offset = compute_spectral_metrics(mz_arr, -10.0)
    assert len(nl) == 0
    assert len(offset) == 0


def test_metadata_processing_edge_cases(empty_spectrum, malformed_metadata_spectrum):
    config = ProcessingConfig()

    # None spectrum
    assert metadata_processing(None, config) is None

    # Normal spectrum
    res = metadata_processing(empty_spectrum, config)
    assert res is not None

    # Malformed metadata usually passes but matchms filters might inject defaults or fail
    res = metadata_processing(malformed_metadata_spectrum, config)
    # add_precursor_mz requires precursor_mz
    assert res is not None

    # Config options toggle coverage
    config_none = None
    assert metadata_processing(empty_spectrum, config_none) is not None


def test_metadata_processing_instrument_injection():
    config = ProcessingConfig(instrument="Orbitrap", mode="positive")
    spec = Spectrum(mz=np.array([100.0]), intensities=np.array([1.0]), metadata={})
    res = metadata_processing(spec, config)
    assert res.get("instrument") == "Orbitrap"
    assert res.get("ionmode") == "positive"


def test_peak_processing_edge_cases(empty_spectrum, zero_intensity_spectrum):
    config = ProcessingConfig(filter_min_peaks=True, min_peaks=1, noise_threshold=0.0)

    # None spectrum
    assert peak_processing(None, config) is None

    # Zero intensity spectrum -> should drop because all peaks are 0.0 and fail threshold
    # Note: matchms select_by_intensity might keep them if threshold is exactly 0.0,
    # but we usually set threshold > 0 or they drop on min_peaks.
    config_noise = ProcessingConfig(
        noise_threshold=0.1,
        filter_min_peaks=True,
        min_peaks=1,
        filter_by_intensity=True,
    )
    assert peak_processing(zero_intensity_spectrum, config_noise) is None

    # Empty spectrum -> fails min_peaks
    assert peak_processing(empty_spectrum, config) is None

    # Out of bounds mz range -> returns spectrum with 0 peaks because count check is BEFORE mz truncation
    config_mz = ProcessingConfig(
        mz_min=500.0,
        mz_max=600.0,
        filter_min_peaks=True,
        min_peaks=1,
        noise_threshold=0.0,
        filter_by_mz=True,
    )
    spec = Spectrum(
        mz=np.array([100.0, 200.0]), intensities=np.array([1.0, 1.0]), metadata={}
    )
    res_mz = peak_processing(spec, config_mz)
    assert res_mz is not None
    assert len(res_mz.peaks.mz) == 0


def test_process_spectra_batch_edge_cases():
    config = ProcessingConfig(min_peaks=1, noise_threshold=0.0, filter_min_peaks=True)

    # Empty list
    assert process_spectra_batch([], config) == []

    # Batch with None (should have been filtered by orchestrator, but testing safety)
    # Actually process_spectra_batch assumes non-None, but let's test if one fails peak processing
    spec1 = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"id": "good", "precursor_mz": 200.0},
    )
    spec2 = Spectrum(
        mz=np.array([]),
        intensities=np.array([]),
        metadata={"id": "bad", "precursor_mz": 200.0},
    )

    batch = [spec1, spec2]
    res = process_spectra_batch(batch, config)
    assert len(res) == 1
    assert res[0].get("id") == "good"


def test_process_spectra_generator_edge_cases():
    config = ProcessingConfig(min_peaks=1, noise_threshold=0.0)

    def spec_gen():
        yield None
        yield Spectrum(
            mz=np.array([100.0]),
            intensities=np.array([1.0]),
            metadata={"id": "good", "precursor_mz": 200.0},
        )
        yield None

    res = list(process_spectra(spec_gen(), config))
    assert len(res) == 1
    assert res[0].get("id") == "good"
