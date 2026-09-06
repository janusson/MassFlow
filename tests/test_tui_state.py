"""
Tests for MassFlow.tui.state — plain-data containers.
"""

import math
from pathlib import Path

from MassFlow.tui.state import (
    IdentificationOutcome,
    IdentificationRequest,
    SearchHit,
)


class TestSearchHit:
    def test_full_result(self):
        hit = SearchHit.from_search_result(
            {
                "query_id": "q1",
                "query_precursor_mz": 100.0,
                "reference_id": "r1",
                "reference_name": "Caffeine",
                "reference_precursor_mz": 100.01,
                "score": 0.93,
                "matched_peaks": 12,
                "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
                "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
                "is_decoy": False,
                "q_value": 0.01,
                "p_value": 0.005,
                "annotation_tier": "level_2a",
                "structural_similarity": 0.8,
                "mass_error_ppm": 3.2,
                "score_breakdown": {"cosine": 0.9, "modified_cosine": 0.96},
            }
        )
        assert hit.query_id == "q1"
        assert hit.reference_name == "Caffeine"
        assert hit.score == 0.93
        assert hit.q_value == 0.01
        assert hit.score_breakdown == {"cosine": 0.9, "modified_cosine": 0.96}

    def test_missing_fields_become_none_or_defaults(self):
        hit = SearchHit.from_search_result({"query_id": "q1", "score": 0.5})
        assert hit.reference_id == ""
        assert hit.reference_name is None
        assert hit.matched_peaks == 0
        assert hit.q_value is None
        assert hit.score_breakdown is None

    def test_malformed_values_do_not_raise(self):
        hit = SearchHit.from_search_result(
            {
                "query_id": 123,
                "score": "not-a-float",
                "matched_peaks": "nope",
                "q_value": "high",
                "score_breakdown": ["not", "a", "dict"],
            }
        )
        assert hit.query_id == "123"
        assert math.isnan(hit.score)
        assert hit.matched_peaks == 0
        assert hit.q_value is None
        assert hit.score_breakdown is None

    def test_nan_values_become_none(self):
        hit = SearchHit.from_search_result(
            {"query_id": "q1", "score": 0.5, "q_value": float("nan")}
        )
        assert hit.q_value is None


class TestIdentificationOutcome:
    def test_derived_properties(self):
        request = IdentificationRequest(
            query_path=Path("q.mgf"), library_path=Path("lib.msp")
        )
        outcome = IdentificationOutcome(
            request=request,
            engine_used="cosine",
            hits=[SearchHit.from_search_result({"query_id": "a", "score": 1.0})],
            num_queries=2,
            num_references=10,
            duration_seconds=0.5,
            fdr_threshold=0.05,
        )
        assert outcome.num_hits == 1
        assert outcome.queries_with_hits == 1
