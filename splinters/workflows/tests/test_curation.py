import json

import numpy as np
import pytest

pytest.importorskip("matchms", reason="Curation tests require matchms.")

from matchms import Spectrum

from yogimass import workflow
from yogimass.curation import detect_duplicate_groups
from yogimass.similarity.library import LocalSpectralLibrary


def _spectrum(peaks, intensities, name, precursor_mz):
    return Spectrum(
        mz=np.asarray(peaks, dtype="float32"),
        intensities=np.asarray(intensities, dtype="float32"),
        metadata={"name": name, "precursor_mz": precursor_mz},
    )


def test_detect_duplicate_groups_pairs_by_precursor_and_similarity(tmp_path):
    # Create simple entries with overlapping precursor m/z and identical vectors.
    lib = LocalSpectralLibrary(tmp_path / "curation_dummy.json")
    lib.add_spectrum(_spectrum([50, 75], [100, 80], "a", 100.0), identifier="a")
    lib.add_spectrum(_spectrum([50.0005, 75.0], [95, 70], "b", 100.005), identifier="b")
    lib.add_spectrum(_spectrum([10], [1], "c", 200.0), identifier="c")

    groups = detect_duplicate_groups(
        list(lib.iter_entries()),
        precursor_tolerance=0.01,
        similarity_threshold=0.9,
    )

    assert ["a", "b"] in groups
    assert all("c" not in group for group in groups)


def test_curate_library_merges_duplicates_and_drops_low_quality(tmp_path):
    raw_library = tmp_path / "raw.json"
    curated_library = tmp_path / "curated.json"
    qc_report = tmp_path / "qc.json"

    library = LocalSpectralLibrary(raw_library)
    library.add_spectrum(
        _spectrum([50, 75, 90], [100, 80, 10], "dup1", 100.0), identifier="dup-1"
    )
    library.add_spectrum(
        _spectrum([50, 75, 91], [95, 70, 5], "dup2", 100.004), identifier="dup-2"
    )
    library.add_spectrum(
        _spectrum([150, 175], [50, 40], "keeper", 200.0), identifier="keep-1"
    )
    # Low-quality: single dominant peak and missing precursor should trigger drop.
    library.add_spectrum(_spectrum([10], [1], "low", None), identifier="low-quality")

    result = workflow.curate_library(
        raw_library,
        curated_library,
        config={
            "min_peaks": 2,
            "precursor_tolerance": 0.01,
            "similarity_threshold": 0.9,
            "max_single_peak_fraction": 0.95,
        },
        qc_report_path=qc_report,
    )

    curated_entries = list(LocalSpectralLibrary(curated_library).iter_entries())
    assert len(curated_entries) == 2
    identifiers = sorted(entry.identifier for entry in curated_entries)
    assert identifiers == ["dup-1", "keep-1"] or identifiers == ["dup-2", "keep-1"]

    # Duplicate metadata should keep merged IDs.
    merged_entry = next(
        entry for entry in curated_entries if entry.identifier.startswith("dup")
    )
    assert "merged_ids" in merged_entry.metadata
    assert set(merged_entry.metadata["merged_ids"]) == {"dup-1", "dup-2"}

    summary = result.summary
    assert summary["kept"] == 2
    assert summary["dropped"] == 1
    assert summary["merged_groups"] == 1

    report = json.loads(qc_report.read_text())
    assert report["summary"]["dropped"] == 1
    assert qc_report.exists()
