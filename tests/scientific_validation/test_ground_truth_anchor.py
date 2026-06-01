"""
This test serves as a scientific "Ground Truth" anchor for MassFlow.

It uses a small, curated subset of a reference library and a specific
experimental spectrum. It runs a consensus of multiple similarity engines
and compares the final, aggregated scores against a "frozen" result file.

Any significant deviation from these frozen results indicates a potential
regression in the core scientific scoring logic of the platform. This test
must always pass before any new changes are merged.
"""

import json
from pathlib import Path

import pytest
from matchms.importing import load_from_mgf

from MassFlow import processing
from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    SimilarityConfig,
)
from MassFlow.consensus import ConsensusEngine
from MassFlow.models import (
    AnnotationHit,
    ConsensusInput,
    MassFlowSpectrum,
    SpectralPeaks,
    SpectrumMetadata,
)
from MassFlow.models import (
    ConsensusConfig as MFConsensusConfig,
)
from MassFlow.similarity import SimilarityEngine


@pytest.fixture
def ground_truth_library_path() -> Path:
    return Path(__file__).parent / "ground_truth_library.mgf"


@pytest.fixture
def ground_truth_experiment_path() -> Path:
    return Path(__file__).parent / "ground_truth_experiment.mgf"


@pytest.fixture
def ground_truth_results_path() -> Path:
    return Path(__file__).parent / "ground_truth_results.json"


def test_ground_truth_anchor(
    ground_truth_library_path,
    ground_truth_experiment_path,
    ground_truth_results_path,
):
    """
    Verify that the consensus engine produces consistent scientific results.
    """
    # 1. Load data directly with matchms to bypass MassFlow's I/O validation layer,
    # making the test more focused on the processing and consensus logic.
    lib_spectra = list(load_from_mgf(str(ground_truth_library_path)))
    exp_spectra = list(load_from_mgf(str(ground_truth_experiment_path)))

    # Manually set precursor_mz from pepmass to ensure data is ready for processing
    for spec in lib_spectra:
        if spec.get("precursor_mz") is None and spec.get("pepmass") is not None:
            spec.set("precursor_mz", spec.get("pepmass")[0])
    for spec in exp_spectra:
        if spec.get("precursor_mz") is None and spec.get("pepmass") is not None:
            spec.set("precursor_mz", spec.get("pepmass")[0])

    # Manually set spectrum IDs from scan numbers for consistent testing
    for spec in lib_spectra:
        spec.set("id", spec.get("scans"))
    for spec in exp_spectra:
        spec.set("id", spec.get("scans"))

    with open(ground_truth_results_path, "r") as f:
        frozen_results = json.load(f)

    # 2. Configure the engines for the consensus pipeline
    processing_config = ProcessingConfig(
        default_filters=["default_filters"],
        min_peaks=1,
        filter_min_peaks=False,
        reduce_to_top_n_peaks=True,
        n_max=2,
    )
    base_config = MassFlowConfig(
        input=InputConfig(input_path="fake_input.mgf", library_path="fake_library.msp"),
        processing=processing_config,
    )
    sim_config_cosine = SimilarityConfig(
        algorithm="cosine", min_score=0.7, ms2_tolerance=0.1, min_matched_peaks=1
    )
    sim_config_modcosine = SimilarityConfig(
        algorithm="modified_cosine",
        min_score=0.7,
        ms2_tolerance=0.1,
        min_matched_peaks=1,
    )

    # 3. Process spectra using default MassFlow parameters
    processed_lib = list(
        processing.process_spectra(lib_spectra, base_config.processing)
    )
    processed_exp = list(
        processing.process_spectra(exp_spectra, base_config.processing)
    )

    # 4. Run similarity searches for two independent "engines"
    engine_cosine = SimilarityEngine(sim_config_cosine)
    results_cosine = engine_cosine.search(
        processed_exp, processed_lib, include_decoys=False
    )

    engine_modcosine = SimilarityEngine(sim_config_modcosine)
    results_modcosine = engine_modcosine.search(
        processed_exp, processed_lib, include_decoys=False
    )

    # 5. Aggregate and rank hits to prepare for the ConsensusEngine
    all_engine_results = {
        "cosine": results_cosine,
        "modified_cosine": results_modcosine,
    }
    all_annotation_hits = {}  # Dict[query_id, List[AnnotationHit]]

    for engine_id, results in all_engine_results.items():
        # Group results by query to determine rank
        hits_by_query = {}
        for r in results:
            qid = r["query_id"]
            if qid not in hits_by_query:
                hits_by_query[qid] = []
            hits_by_query[qid].append(r)

        # Sort by score to determine rank, then convert to AnnotationHit
        for qid, q_results in hits_by_query.items():
            q_results.sort(key=lambda x: x["score"], reverse=True)
            if qid not in all_annotation_hits:
                all_annotation_hits[qid] = []
            for i, r in enumerate(q_results):
                all_annotation_hits[qid].append(
                    AnnotationHit(
                        engine_id=engine_id,
                        reference_id=r["reference_id"],
                        score=r["score"],
                        rank=i + 1,
                        inchikey=r["inchikey"],
                        smiles=r["smiles"],
                    )
                )

    # 6. Run the ConsensusEngine
    consensus_config = MFConsensusConfig(
        engine_weights={"cosine": 0.5, "modified_cosine": 0.5}
    )
    consensus_engine = ConsensusEngine(consensus_config)

    actual_results = {}
    for q_spec in processed_exp:
        q_id = q_spec.get("id")
        if q_id in all_annotation_hits:
            # Convert matchms.Spectrum to MassFlowSpectrum for Pydantic validation
            mf_spec = MassFlowSpectrum(
                metadata=SpectrumMetadata(
                    spectrum_id=str(q_id),
                    precursor_mz=float(q_spec.get("precursor_mz")),
                ),
                peaks=SpectralPeaks(
                    mz_array=list(q_spec.peaks.mz),
                    intensity_array=list(q_spec.peaks.intensities),
                ),
            )
            consensus_input = ConsensusInput(
                query_id=q_id,
                hits=all_annotation_hits[q_id],
                experimental_spectrum=mf_spec,
            )
            result = consensus_engine.resolve(consensus_input)
            actual_results[q_id] = {
                "best_reference_id": result.best_reference_id,
                "best_consensus_score": result.best_consensus_score,
                "flagged_for_review": result.flagged_for_review,
                "review_reason": result.review_reason,
            }

    # 7. Compare actual results to the "frozen" ground truth
    for query_id, expected in frozen_results.items():
        assert query_id in actual_results, f"Query ID {query_id} not in results"
        actual = actual_results[query_id]

        assert actual["best_reference_id"] == expected["best_reference_id"]
        assert actual["flagged_for_review"] == expected["flagged_for_review"]
        assert actual["review_reason"] == expected["review_reason"]

        # The core scientific anchor: assert score is within a tight tolerance
        assert (
            abs(actual["best_consensus_score"] - expected["best_consensus_score"])
            < 1e-6
        )
