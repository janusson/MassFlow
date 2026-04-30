"""Performance-oriented tests for cascade routing behavior."""

from __future__ import annotations

import time

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.config import SimilarityConfig
from MassFlow.similarity import CascadeEngine

pytestmark = pytest.mark.experimental


def make_spectrum(spec_id: str, precursor_mz: float = 100.0) -> Spectrum:
    return Spectrum(
        mz=np.array([precursor_mz], dtype="float"),
        intensities=np.array([1.0], dtype="float"),
        metadata={"id": spec_id, "precursor_mz": precursor_mz},
    )


def make_result(query_id: str, reference_id: str, score: float) -> dict[str, object]:
    return {
        "query_id": query_id,
        "query_precursor_mz": 100.0,
        "reference_id": reference_id,
        "reference_name": reference_id,
        "reference_precursor_mz": 100.0,
        "score": score,
        "matched_peaks": 5,
        "smiles": None,
        "inchikey": None,
        "is_decoy": False,
        "q_value": 1.0,
        "annotation_tier": None,
    }


def test_cascade_benchmark_reduces_expensive_tier2_work(monkeypatch):
    """
    Benchmark-like check that cascade routing materially reduces Tier 2 work.

    The test uses mocked engines with deterministic per-query Tier 2 cost so the
    performance comparison is stable: Tier 1 labels exact, gray-zone, and noise
    queries, while Tier 2 sleeps in proportion to the number of routed queries.
    """

    class FakeSimilarityEngine:
        def __init__(self, config: SimilarityConfig):
            self.config = config
            self.last_query_count = 0

        def search(
            self,
            query_spectra: list[Spectrum],
            reference_spectra: list[Spectrum],
            min_score: float | None = None,
            top_n: int | None = None,
            include_decoys: bool = True,
        ) -> list[dict[str, object]]:
            self.last_query_count = len(query_spectra)

            if self.config.algorithm == "cosine":
                results = []
                for query in query_spectra:
                    query_id = str(query.get("id"))
                    if query_id.startswith("exact_"):
                        score = 0.95
                    elif query_id.startswith("gray_"):
                        score = 0.6
                    else:
                        score = 0.2
                    results.append(make_result(query_id, "ref_1", score))
                return results

            if self.config.algorithm == "ms2deepscore":
                time.sleep(0.001 * len(query_spectra))
                return [
                    make_result(str(query.get("id")), "ref_1", 0.97)
                    for query in query_spectra
                ]

            raise AssertionError(
                f"Unexpected algorithm in test: {self.config.algorithm}"
            )

    monkeypatch.setattr("MassFlow.similarity.SimilarityEngine", FakeSimilarityEngine)

    query_spectra = (
        [make_spectrum(f"exact_{i}", precursor_mz=100.0 + i) for i in range(60)]
        + [make_spectrum(f"gray_{i}", precursor_mz=200.0 + i) for i in range(20)]
        + [make_spectrum(f"noise_{i}", precursor_mz=300.0 + i) for i in range(40)]
    )
    reference_spectra = [make_spectrum("ref_1", precursor_mz=100.0)]

    cascade_config = SimilarityConfig(
        algorithm="cascade",
        cascade_tier1="cosine",
        cascade_tier2="ms2deepscore",
        cascade_lower_bound=0.4,
        cascade_upper_bound=0.85,
    )

    cascade_engine = CascadeEngine(cascade_config)

    cascade_start = time.perf_counter()
    cascade_results = cascade_engine.search(
        query_spectra, reference_spectra, min_score=0.0, top_n=1
    )
    cascade_elapsed = time.perf_counter() - cascade_start

    baseline_engine = FakeSimilarityEngine(
        cascade_config.model_copy(update={"algorithm": "ms2deepscore"})
    )
    baseline_start = time.perf_counter()
    baseline_engine.search(query_spectra, reference_spectra, min_score=0.0, top_n=1)
    baseline_elapsed = time.perf_counter() - baseline_start

    assert cascade_engine.tier2_engine.last_query_count == 20
    assert baseline_engine.last_query_count == 120
    assert len(cascade_results) == 80
    assert baseline_elapsed > cascade_elapsed * 2

    annotation_tiers = {result["annotation_tier"] for result in cascade_results}
    assert "Tier 1 (cosine)" in annotation_tiers
    assert "Tier 2 (ms2deepscore)" in annotation_tiers
