"""Adversarial, explicitly synthetic caffeine metadata with independent masses.

The neutral formula is C8H10N4O2. Pyteomics supplies neutral/proton masses;
ion m/z is (M + n * proton_mass) / abs(z), with n fixed by the adduct.
See https://pyteomics.readthedocs.io/en/stable/api/mass.html.
No measured spectrum or empirical identification accuracy is claimed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from matchms import Spectrum
from matchms.exporting import save_as_msp
from pyteomics import mass

from MassFlow import library
from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    ProjectConfig,
    SimilarityConfig,
)
from MassFlow.models import MolecularStructure, SpectrumMetadata
from MassFlow.processing import PhysicalIntegrityError, process_spectra_batch
from MassFlow.similarity import SimilarityEngine
from MassFlow.storage import create_spectral_store

try:
    from rdkit import Chem  # noqa: F401

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

pytestmark = pytest.mark.scientific

CAFFEINE_FORMULA = "C8H10N4O2"
CAFFEINE_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
NEUTRAL_MASS = mass.calculate_mass(formula=CAFFEINE_FORMULA)
PROTON_MASS = mass.nist_mass["H+"][0][0]
SINGLY_PROTONATED_MZ = NEUTRAL_MASS + PROTON_MASS
DOUBLY_PROTONATED_MZ = (NEUTRAL_MASS + 2 * PROTON_MASS) / 2
DEPROTONATED_MZ = NEUTRAL_MASS - PROTON_MASS


@pytest.fixture(params=["formula", "smiles"])
def molecule(request: pytest.FixtureRequest) -> MolecularStructure:
    """Exercise both real structure-resolution paths, with the optional skip."""
    if request.param == "smiles":
        if not _HAS_RDKIT:
            pytest.skip("RDKit not installed")
        return MolecularStructure(smiles=CAFFEINE_SMILES)
    return MolecularStructure(formula=CAFFEINE_FORMULA)


@pytest.mark.parametrize(
    ("adduct", "charge", "precursor_mz"),
    [
        ("[M+2H]2+", 1, 2 * DOUBLY_PROTONATED_MZ),
        ("[M+H]+", 2, SINGLY_PROTONATED_MZ / 2),
        ("[M+H]+", -1, SINGLY_PROTONATED_MZ),
        ("[M-H]-", 1, DEPROTONATED_MZ),
    ],
    ids=[
        "double-as-single",
        "single-as-double",
        "positive-as-negative",
        "negative-as-positive",
    ],
)
def test_model_rejects_charge_contradicting_adduct(
    molecule: MolecularStructure, adduct: str, charge: int, precursor_mz: float
) -> None:
    """A mass fitting the wrong charge must not validate a contradictory ion."""
    metadata = SpectrumMetadata(
        spectrum_id="contradictory-caffeine",
        precursor_mz=precursor_mz,
        charge=charge,
        adduct=adduct,
        molecule=molecule,
    )
    assert metadata.is_physically_valid is False


@pytest.mark.parametrize(
    ("adduct", "charge", "precursor_mz"),
    [
        ("M+H", 1, SINGLY_PROTONATED_MZ),
        ("[M+2H]2+", 2, DOUBLY_PROTONATED_MZ),
        ("[M-H]-", -1, DEPROTONATED_MZ),
    ],
)
@pytest.mark.parametrize(
    ("ppm_error", "expected_valid"), [(0.0, True), (4.75, True), (5.25, False)]
)
def test_consistent_ions_retain_five_ppm_boundary(
    molecule: MolecularStructure,
    adduct: str,
    charge: int,
    precursor_mz: float,
    ppm_error: float,
    expected_valid: bool,
) -> None:
    """Consistent single/double/negative ions still use the real 5 ppm gate."""
    metadata = SpectrumMetadata(
        spectrum_id="consistent-caffeine",
        precursor_mz=precursor_mz * (1 + ppm_error / 1e6),
        charge=charge,
        adduct=adduct,
        molecule=molecule,
    )
    assert metadata.is_physically_valid is expected_valid


def _spectrum(
    spectrum_id: str,
    precursor_mz: float,
    adduct: str,
    charge: int,
    *,
    formula: str | None,
) -> Spectrum:
    """Build synthetic, element-compatible fragments with fully stated metadata."""
    metadata: dict[str, object] = {
        "id": spectrum_id,
        "compound_name": spectrum_id,
        "precursor_mz": precursor_mz,
        "adduct": adduct,
        "charge": charge,
    }
    if formula is not None:
        metadata["formula"] = formula
    # C2H4N+, C3H3N2+, C3H5N2+: synthetic ion subformulas, not observed peaks.
    return Spectrum(
        mz=np.array([42.033826, 67.029075, 69.044725], dtype=np.float64),
        intensities=np.array([0.4, 1.0, 0.6], dtype=np.float64),
        metadata=metadata,
    )


@pytest.mark.parametrize(
    ("adduct", "charge", "precursor_mz"),
    [("[M+2H]2+", 1, 2 * DOUBLY_PROTONATED_MZ), ("[M-H]-", 1, DEPROTONATED_MZ)],
)
def test_batch_quarantines_contradictory_charge(
    adduct: str, charge: int, precursor_mz: float
) -> None:
    """Harmonisation must not let the contradiction bypass the processing gate."""
    reasons: list[str] = []
    retained = process_spectra_batch(
        [
            _spectrum(
                "contradictory", precursor_mz, adduct, charge, formula=CAFFEINE_FORMULA
            )
        ],
        ProcessingConfig(min_peaks=3, noise_threshold=0.0),
        rejection_reporter=reasons.append,
    )
    assert retained == []
    assert len(reasons) == 1
    assert "charge" in reasons[0].lower()
    assert adduct in reasons[0]


@pytest.mark.parametrize(
    ("adduct", "charge", "expected_retained"),
    [
        ("[M+2H]2+", 1, False),
        ("[M-H]-", 1, False),
        ("[M+H]+", 1.5, False),
        ("[M+2H]2+", 2, True),
    ],
)
def test_explicit_adduct_charge_is_checked_without_precursor(
    adduct: str, charge: float, expected_retained: bool
) -> None:
    """Missing precursor data cannot bypass a known charge/adduct mismatch."""
    spectrum = Spectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        metadata={
            "id": "adduct-charge-record",
            "precursor_mz": None,
            "adduct": adduct,
            "charge": charge,
        },
        metadata_harmonization=False,
    )
    reasons: list[str] = []
    retained = process_spectra_batch(
        [spectrum],
        ProcessingConfig(
            min_peaks=1,
            filter_min_peaks=False,
            filter_by_intensity=False,
            filter_by_mz=False,
            normalize_intensity=False,
        ),
        rejection_reporter=reasons.append,
    )

    assert bool(retained) is expected_retained
    if expected_retained:
        assert retained[0].get("precursor_mz") is None
        assert reasons == []
    else:
        assert retained == []
        assert len(reasons) == 1
        assert "charge" in reasons[0].lower()


@pytest.mark.parametrize(
    ("charge", "precursor_mz"),
    [(1, 2 * DOUBLY_PROTONATED_MZ), (2, DOUBLY_PROTONATED_MZ)],
)
def test_impossible_library_claim_cannot_produce_perfect_cosine_hit(
    charge: int, precursor_mz: float
) -> None:
    """The wrong-charge library row previously returned cosine=1, three matches."""
    query = _spectrum(
        "synthetic-query", 2 * DOUBLY_PROTONATED_MZ, "[M+H]+", 1, formula=None
    )
    reference = _spectrum(
        "synthetic-caffeine-claim",
        precursor_mz,
        "[M+2H]2+",
        charge,
        formula=CAFFEINE_FORMULA,
    )
    references = process_spectra_batch(
        [reference], ProcessingConfig(min_peaks=3, noise_threshold=0.0)
    )
    # Correctly labelled z=2 caffeine survives processing but cannot match the
    # query at twice its precursor m/z under the ordinary cosine MS1 gate.
    assert len(references) == (1 if charge == 2 else 0)
    hits = SimilarityEngine(
        SimilarityConfig(
            algorithm="cosine",
            ms1_tolerance=0.02,
            ms2_tolerance=0.02,
            min_score=0.7,
            min_matched_peaks=3,
        )
    ).search([query], references, include_decoys=False)
    assert hits == []


def test_processing_revision_changes_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validator change must invalidate caches even if config stays identical."""
    config = ProcessingConfig()
    before = library.processing_fingerprint(config)
    assert library.processing_fingerprint(config.model_copy()) == before
    monkeypatch.setattr(
        library,
        "PROCESSING_SEMANTICS_REVISION",
        getattr(library, "PROCESSING_SEMANTICS_REVISION", 0) + 1,
        raising=False,
    )
    assert library.processing_fingerprint(config) != before


@pytest.mark.parametrize("backend_name", ["sqlite", "zarr", "hybrid"])
@pytest.mark.parametrize(
    "valid_source", [True, False], ids=["rebuild-once", "reject-raw-library"]
)
def test_legacy_cache_cannot_bypass_charge_validation(
    tmp_path: Path,
    backend_name: Literal["sqlite", "zarr", "hybrid"],
    valid_source: bool,
) -> None:
    """Revalidate legacy caches; reject contradictions or replace without mixing."""
    library_path = tmp_path / "library.msp"
    source = _spectrum(
        "valid-source" if valid_source else "invalid-source",
        SINGLY_PROTONATED_MZ if valid_source else 2 * DOUBLY_PROTONATED_MZ,
        "[M+H]+" if valid_source else "[M+2H]2+",
        1,
        formula=CAFFEINE_FORMULA,
    )
    save_as_msp([source], str(library_path))
    config = MassFlowConfig(
        project=ProjectConfig(output_directory=tmp_path / "results"),
        input=InputConfig(
            input_path=tmp_path / "query.mgf",
            library_path=library_path,
            storage_backend=backend_name,
        ),
        processing=ProcessingConfig(min_peaks=3, noise_threshold=0.0),
    )
    output_directory = config.project.output_directory
    output_directory.mkdir()
    store_path = output_directory / (
        "library_library.zarr" if backend_name == "zarr" else "library_library.db"
    )
    # Construct the actual pre-fix store/sidecar format. Direct insertion is
    # intentional: this entry was admitted by the old processing validator.
    legacy_store = create_spectral_store(store_path, backend=backend_name)
    try:
        legacy_store.add_spectra(
            iter(
                [
                    _spectrum(
                        "legacy-invalid",
                        2 * DOUBLY_PROTONATED_MZ,
                        "[M+2H]2+",
                        1,
                        formula=CAFFEINE_FORMULA,
                    )
                ]
            ),
            category="library",
        )
    finally:
        legacy_store.close()
    legacy_fingerprint = hashlib.sha256(
        config.processing.model_dump_json().encode("utf-8")
    ).hexdigest()
    metadata_path = store_path.with_suffix(store_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(
            {
                "source": str(library_path),
                "source_mtime_ns": library_path.stat().st_mtime_ns,
                "source_size": library_path.stat().st_size,
                "fingerprint": legacy_fingerprint,
                "backend": backend_name,
            }
        )
    )
    if not valid_source:
        with pytest.raises(PhysicalIntegrityError, match="charge"):
            library.prepare_library(config, output_directory)
        # A failed staged rebuild must leave the existing cache untouched;
        # its legacy fingerprint ensures future runs still cannot reuse it.
        assert (
            json.loads(metadata_path.read_text())["fingerprint"] == legacy_fingerprint
        )
        return

    spec, count = library.prepare_library(config, output_directory)
    assert count == 1
    store = library.open_library(spec, config.processing)
    try:
        assert [s.get("compound_name") for s in store.iter_spectra()] == [
            "valid-source"
        ]
    finally:
        store.close()
    assert json.loads(metadata_path.read_text())["fingerprint"] != legacy_fingerprint
    first_mtime_ns = metadata_path.stat().st_mtime_ns
    reused_spec, reused_count = library.prepare_library(config, output_directory)
    assert reused_spec == spec
    assert reused_count == 1
    assert metadata_path.stat().st_mtime_ns == first_mtime_ns
