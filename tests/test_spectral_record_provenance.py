"""Adversarial synthetic records: structure identity never identifies a scan."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from matchms import Spectrum
from pyteomics.mass import calculate_mass

from MassFlow import io, processing
from MassFlow.cheminformatics import compute_adduct_offset
from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)
from MassFlow.library import RawFileLibraryStore, open_library, prepare_library
from MassFlow.similarity import SimilarityEngine
from MassFlow.storage import create_spectral_store

pytestmark = [pytest.mark.core, pytest.mark.scientific]
INCHIKEY = "RYYVLZVUVIJVGH-UHFFFAOYSA-N"


def _records(identifiers: tuple[str | None, str | None]) -> list[Spectrum]:
    """Construct two formula-valid caffeine acquisitions with conflicting metadata."""
    spectra = []
    # This is a loop over two acquisition records, not spectral peak arrays.
    for index, identifier in enumerate(identifiers):
        adduct = ("[M+H]+", "[M+Na]+")[index]
        offset = compute_adduct_offset(adduct)
        assert offset is not None
        metadata = {
            "compound_name": "caffeine",
            "formula": "C8H10N4O2",
            "inchikey": INCHIKEY,
            "precursor_mz": calculate_mass(formula="C8H10N4O2") + offset,
            "charge": 1,
            "adduct": adduct,
            "ionmode": "positive",
            "collision_energy": (20, 45)[index],
            "instrument": ("QTOF-A", "Orbitrap-B")[index],
            "instrument_type": ("QTOF", "Orbitrap")[index],
            "library_id": ("LIBRARY-A", "LIBRARY-B")[index],
            "accession": ("ACCESSION-A", "ACCESSION-B")[index],
            "source_file": ("acquisition-A.mzML", "acquisition-B.mzML")[index],
        }
        if identifier is not None:
            metadata["id"] = identifier
        spectrum = Spectrum(
            mz=np.array([42.033826, 83.060374, 138.066188], dtype=np.float64),
            intensities=np.array((1.0, 0.5, 0.25), dtype=np.float64),
            metadata=metadata,
            metadata_harmonization=False,
        )
        assert processing.physical_integrity_reason(spectrum) is None
        spectra.append(spectrum)
    return spectra


def _write_library(path: Path, identifiers: tuple[str | None, str | None]) -> None:
    """Write explicit synthetic acquisitions, preserving every scalar metadata field."""
    blocks = []
    for spectrum in _records(identifiers):
        lines = [f"{key}: {value}" for key, value in spectrum.metadata.items()]
        lines.extend(
            ["Num Peaks: 3", "42.033826 1.0", "83.060374 0.5", "138.066188 0.25"]
        )
        blocks.append("\n".join(lines))
    path.write_text("\n\n".join(blocks) + "\n\n")


def _assert_acquisitions(spectra: list[Spectrum]) -> None:
    """Check the association of every experimental field with its source record."""
    assert len(spectra) == 2
    for index, spectrum in enumerate(spectra):
        assert spectrum.get("inchikey") == INCHIKEY
        assert float(spectrum.get("collision_energy")) == (20.0, 45.0)[index]
        assert spectrum.get("adduct") == ("[M+H]+", "[M+Na]+")[index]
        assert spectrum.get("instrument") == ("QTOF-A", "Orbitrap-B")[index]
        assert spectrum.get("instrument_type") == ("QTOF", "Orbitrap")[index]
        assert spectrum.get("library_id") == ("LIBRARY-A", "LIBRARY-B")[index]
        # matchms canonicalizes the accession key to spectrum_id on import.
        assert spectrum.get("spectrum_id") == ("ACCESSION-A", "ACCESSION-B")[index]
        assert (
            spectrum.get("source_file")
            == ("acquisition-A.mzML", "acquisition-B.mzML")[index]
        )
        assert spectrum.peaks.mz.dtype == np.float64
        assert processing.physical_integrity_reason(spectrum) is None


@pytest.mark.parametrize("reverse", [False, True])
def test_same_structure_preserves_acquisition_metadata(reverse: bool) -> None:
    """No metadata crossover is reproduced in main's actual metadata/peak pipeline."""
    spectra = _records(("record-A", "record-B"))
    original = [json.dumps(spectrum.metadata, sort_keys=True) for spectrum in spectra]
    inputs = list(reversed(spectra)) if reverse else spectra
    result = list(processing.process_spectra(inputs, ProcessingConfig()))
    result.sort(key=lambda spectrum: spectrum.get("id"))
    _assert_acquisitions(result)
    assert [
        json.dumps(spectrum.metadata, sort_keys=True) for spectrum in spectra
    ] == original


@pytest.mark.parametrize(
    "identifiers",
    [
        (None, None),
        ("", ""),
        ("duplicate", "duplicate"),
        (INCHIKEY, INCHIKEY),
        ("record-A", "record-B"),
    ],
)
@pytest.mark.parametrize("backend", ["sqlite", "zarr", "hybrid"])
def test_library_record_identity_roundtrip(
    tmp_path: Path, identifiers: tuple[str | None, str | None], backend: str
) -> None:
    """Raw library processing and native serialization retain independent records."""
    source = tmp_path / "library.msp"
    _write_library(source, identifiers)
    adapter = RawFileLibraryStore(source, ProcessingConfig())
    spectra = list(adapter.get_spectra())
    _assert_acquisitions(spectra)
    record_ids = [spectrum.get("id") for spectrum in spectra]
    assert all(
        isinstance(identifier, str) and identifier.strip() for identifier in record_ids
    )
    assert len(set(record_ids)) == 2
    if identifiers == ("record-A", "record-B"):
        assert record_ids == list(identifiers)

    # IDs are deterministic, even when the file is moved without byte changes.
    copy = tmp_path / "copy.msp"
    copy.write_bytes(source.read_bytes())
    assert [
        s.get("id") for s in RawFileLibraryStore(copy, ProcessingConfig()).get_spectra()
    ] == record_ids

    suffix = ".zarr" if backend == "zarr" else ".db"
    destination = tmp_path / f"roundtrip{suffix}"
    store = create_spectral_store(destination, backend=backend)
    try:
        assert store.add_spectra(iter(spectra), category="library") == 2
    finally:
        store.close()
    store = create_spectral_store(destination, backend=backend)
    try:
        restored = list(store.get_spectra())
        _assert_acquisitions(restored)
        # matchms drops empty strings on reconstruction; retain explicit nulls.
        assert [s.metadata for s in restored] == [
            {key: value for key, value in s.metadata.items() if value != ""}
            for s in spectra
        ]
        for expected in restored:
            fetched = store.get_spectrum_by_id(str(expected.get("id")))
            assert fetched is not None
            assert fetched.metadata == expected.metadata
            np.testing.assert_array_equal(fetched.peaks.mz, expected.peaks.mz)
        for algorithm in ("cosine", "modified_cosine"):
            engine = SimilarityEngine(
                SimilarityConfig(
                    algorithm=algorithm, min_score=0.9, min_matched_peaks=3
                )
            )
            query = restored[0].clone()
            query.set("id", "query")
            hits = engine.search([query], restored, include_decoys=False)
            expected_ids = record_ids[:1] if algorithm == "cosine" else record_ids
            assert {hit["reference_id"] for hit in hits} == set(expected_ids)
            assert all(
                hit["score"] == pytest.approx(1.0) and hit["matched_peaks"] == 3
                for hit in hits
            )
            for extension in ("csv", "mztab"):
                output = tmp_path / f"{algorithm}.{extension}"
                if extension == "csv":
                    io.save_match_results(hits, output)
                    with output.open() as handle:
                        rows = list(csv.DictReader(handle))
                else:
                    io.save_match_results_to_mztab(hits, output)
                    lines = [
                        line
                        for line in output.read_text().splitlines()
                        if line.startswith(("SMH\t", "SML\t"))
                    ]
                    rows = list(
                        csv.DictReader(
                            [line.split("\t", 1)[1] for line in lines], delimiter="\t"
                        )
                    )
                assert {row["reference_id"] for row in rows} == set(expected_ids)
                metadata_by_id = {
                    str(spectrum.get("id")): spectrum.metadata for spectrum in restored
                }
                for row in rows:
                    exported_metadata = json.loads(row["reference_metadata"])
                    source_metadata = metadata_by_id[row["reference_id"]]
                    assert (
                        exported_metadata["massflow_source_sha256"]
                        == source_metadata["massflow_source_sha256"]
                    )
                    assert (
                        exported_metadata["massflow_source_index"]
                        == source_metadata["massflow_source_index"]
                    )
                    assert (
                        exported_metadata["massflow_source_path"]
                        == source_metadata["massflow_source_path"]
                    )
                    assert exported_metadata.get(
                        "massflow_original_id"
                    ) == source_metadata.get("massflow_original_id")
                    assert exported_metadata["id"] == row["reference_id"]
                    assert (
                        exported_metadata["spectrum_id"]
                        == source_metadata["spectrum_id"]
                    )
                    assert (
                        exported_metadata["collision_energy"]
                        == source_metadata["collision_energy"]
                    )
                    assert (
                        exported_metadata["instrument"] == source_metadata["instrument"]
                    )
    finally:
        store.close()


@pytest.mark.parametrize("backend", ["sqlite", "zarr", "hybrid"])
def test_prepared_library_adds_missing_record_identity(
    tmp_path: Path, backend: str
) -> None:
    """Managed annotation caches must carry new per-record identity to workers."""
    source = tmp_path / "library.msp"
    _write_library(source, (None, None))
    config = MassFlowConfig(
        project=ProjectConfig(name="provenance", output_directory=tmp_path / "output"),
        input=InputConfig(
            input_path=source, library_path=source, storage_backend=backend
        ),
        processing=ProcessingConfig(),
    )
    spec, count = prepare_library(config, tmp_path / "prepared")
    assert count == 2
    store = open_library(spec, config.processing)
    try:
        spectra = list(store.get_spectra())
        _assert_acquisitions(spectra)
        assert len({s.get("id") for s in spectra}) == 2
        assert all(s.get("id") for s in spectra)
    finally:
        store.close()


def test_source_locator_does_not_reindex_quarantined_records(tmp_path: Path) -> None:
    """A rejected precursor before valid records must not shift source ordinals."""
    source = tmp_path / "library.msp"
    _write_library(source, (None, None))
    source.write_text("Name: invalid\nNum Peaks: 1\n42.0 1\n\n" + source.read_text())
    spectra = list(RawFileLibraryStore(source, ProcessingConfig()).get_spectra())
    _assert_acquisitions(spectra)
    assert [s.get("massflow_source_index") for s in spectra] == [1, 2]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert all(s.get("massflow_source_sha256") == digest for s in spectra)


@pytest.mark.parametrize("chemical_identifier", [None, ""])
def test_missing_structure_identifier_never_borrows_metadata(
    chemical_identifier: str | None,
) -> None:
    """Absent chemistry and acquisition fields stay absent in the second record."""
    spectra = _records(("record-A", "record-B"))
    for spectrum in spectra:
        spectrum.set("inchikey", chemical_identifier)
    spectra[1].set("collision_energy", None)
    spectra[1].set("instrument", None)
    result = list(processing.process_spectra(spectra, ProcessingConfig()))
    assert len(result) == 2
    assert not result[0].get("inchikey") and not result[1].get("inchikey")
    assert float(result[0].get("collision_energy")) == 20.0
    assert result[1].get("collision_energy") is None
    assert result[0].get("instrument") == "QTOF-A"
    assert result[1].get("instrument") is None
    assert [s.get("library_id") for s in result] == ["LIBRARY-A", "LIBRARY-B"]


def test_query_loading_remains_raw(tmp_path: Path) -> None:
    """Generic/query loading does not acquire the library identity policy."""
    source = tmp_path / "query.msp"
    _write_library(source, (None, None))
    assert all(s.get("id") is None for s in io.load_spectra(source))


@pytest.mark.parametrize("format_name", ["msp", "mgf"])
def test_open_format_roundtrip_preserves_record_metadata(
    tmp_path: Path, format_name: str
) -> None:
    """Open-format exports keep the two records and their source/acquisition fields."""
    source = tmp_path / "library.msp"
    _write_library(source, (None, None))
    spectra = list(RawFileLibraryStore(source, ProcessingConfig()).get_spectra())
    output = tmp_path / f"export.{format_name}"
    if format_name == "msp":
        io.save_spectra_to_msp(spectra, output)
    else:
        io.save_spectra_to_mgf(spectra, output)
    restored = list(
        processing.process_spectra(io.load_spectra(output), ProcessingConfig())
    )
    _assert_acquisitions(restored)
    assert [s.get("id") for s in restored] == [s.get("id") for s in spectra]
    assert [str(s.get("massflow_source_index")) for s in restored] == ["0", "1"]
    assert all(
        s.get("massflow_source_sha256")
        == hashlib.sha256(source.read_bytes()).hexdigest()
        for s in restored
    )


@pytest.mark.parametrize("backend", ["sqlite", "zarr", "hybrid"])
def test_legacy_managed_cache_rebuilds_once(tmp_path: Path, backend: str) -> None:
    """Config-only caches containing unidentified rows rebuild without appending."""
    source = tmp_path / "library.msp"
    _write_library(source, (None, None))
    config = MassFlowConfig(
        project=ProjectConfig(name="provenance", output_directory=tmp_path / "output"),
        input=InputConfig(
            input_path=source, library_path=source, storage_backend=backend
        ),
        processing=ProcessingConfig(),
    )
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    destination = prepared / (
        "library_library.zarr" if backend == "zarr" else "library_library.db"
    )
    store = create_spectral_store(destination, backend=backend)
    try:
        raw = io.load_spectra(source)
        assert (
            store.add_spectra(processing.process_spectra(raw, config.processing)) == 2
        )
    finally:
        store.close()
    source_stat = source.stat()
    sidecar = destination.with_suffix(destination.suffix + ".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "source": str(source),
                "source_mtime_ns": source_stat.st_mtime_ns,
                "source_size": source_stat.st_size,
                "backend": backend,
                "fingerprint": hashlib.sha256(
                    config.processing.model_dump_json().encode()
                ).hexdigest(),
            }
        )
    )
    spec, count = prepare_library(config, prepared)
    assert count == 2
    store = open_library(spec, config.processing)
    try:
        records = list(store.get_spectra())
        assert len({s.get("id") for s in records}) == 2
        assert all(s.get("massflow_source_sha256") for s in records)
    finally:
        store.close()
    modification_time = sidecar.stat().st_mtime_ns
    assert prepare_library(config, prepared) == (spec, 2)
    assert sidecar.stat().st_mtime_ns == modification_time


@pytest.mark.parametrize("identifier", [None, "", "None", "null", "nan"])
def test_classical_search_rejects_missing_reference_ids(identifier: str | None) -> None:
    """Classical results must not identify multiple acquisitions as None/empty."""
    references = _records((identifier, "record-B"))
    engine = SimilarityEngine(
        SimilarityConfig(algorithm="modified_cosine", min_score=0.0)
    )
    with pytest.raises(ValueError, match="non-empty unique record id"):
        engine.search([references[1]], references, include_decoys=False)


def test_classical_search_rejects_duplicate_reference_ids() -> None:
    """Two acquisitions cannot silently share one search-result identifier."""
    references = _records(("duplicate", "duplicate"))
    engine = SimilarityEngine(
        SimilarityConfig(algorithm="modified_cosine", min_score=0.0)
    )
    with pytest.raises(ValueError, match="unique record id"):
        engine.search([references[0]], references, include_decoys=False)


def test_classical_search_rejects_generated_decoy_id_collision() -> None:
    """A target identifier cannot collide with another target's generated decoy."""
    references = _records(("x", "x_decoy"))
    engine = SimilarityEngine(
        SimilarityConfig(algorithm="modified_cosine", min_score=0.0)
    )
    with pytest.raises(ValueError, match="generated decoy id"):
        engine.search([references[0]], references, include_decoys=True)


def test_tied_search_results_have_order_independent_record_order() -> None:
    """Equal scores use acquisition IDs rather than input/store iteration order."""
    references = _records(("record-z", "record-a"))
    query = references[0].clone()
    query.set("id", "query")
    engine = SimilarityEngine(
        SimilarityConfig(algorithm="modified_cosine", min_score=0.0)
    )

    forward = engine.search([query], references, include_decoys=False)
    reverse = engine.search([query], list(reversed(references)), include_decoys=False)

    assert [hit["reference_id"] for hit in forward] == ["record-a", "record-z"]
    assert [hit["reference_id"] for hit in reverse] == ["record-a", "record-z"]


def test_textual_missing_library_ids_receive_source_record_ids(tmp_path: Path) -> None:
    """Textual null sentinels cannot become apparently stable record IDs."""
    source = tmp_path / "library.msp"
    _write_library(source, ("None", "null"))

    records = list(io.load_spectra(source, assign_record_ids=True))

    assert len(records) == 2
    assert all(record.get("id") for record in records)
    assert len({record.get("id") for record in records}) == 2
    assert [record.get("massflow_original_id") for record in records] == [
        "None",
        "null",
    ]
