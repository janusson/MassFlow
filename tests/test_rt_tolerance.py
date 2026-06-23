import numpy as np
from matchms import Spectrum
from MassFlow.config import SimilarityConfig
from MassFlow.similarity import SimilarityEngine

def test_rt_tolerance():
    query = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"id": "query1", "precursor_mz": 400.0, "retention_time": 5.0}
    )
    ref = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"id": "ref1", "precursor_mz": 400.0, "retention_time": 5.5}
    )

    config = SimilarityConfig(
        algorithm="cosine", ms1_tolerance=0.0, min_matched_peaks=1, min_score=0.0, rt_tolerance=0.2
    )
    engine = SimilarityEngine(config)
    results = engine.search(query_spectra=[query], reference_spectra=[ref])
    results = [r for r in results if not r.get("is_decoy")]
    assert len(results) == 0

    config2 = SimilarityConfig(
        algorithm="cosine", ms1_tolerance=0.0, min_matched_peaks=1, min_score=0.0, rt_tolerance=1.0
    )
    engine2 = SimilarityEngine(config2)
    results2 = engine2.search(query_spectra=[query], reference_spectra=[ref])
    results2 = [r for r in results2 if not r.get("is_decoy")]
    assert len(results2) == 1

def test_rt_tolerance_missing():
    query = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"id": "query1", "precursor_mz": 400.0, "retention_time": 5.0}
    )
    ref = Spectrum(
        mz=np.array([100.0]),
        intensities=np.array([1.0]),
        metadata={"id": "ref1", "precursor_mz": 400.0}
    )

    config = SimilarityConfig(
        algorithm="cosine", ms1_tolerance=0.0, min_matched_peaks=1, min_score=0.0, rt_tolerance=0.2
    )
    engine = SimilarityEngine(config)
    results = engine.search(query_spectra=[query], reference_spectra=[ref])
    results = [r for r in results if not r.get("is_decoy")]
    assert len(results) == 1

if __name__ == "__main__":
    test_rt_tolerance()
    test_rt_tolerance_missing()
    print("All tests passed!")
