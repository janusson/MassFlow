"""
Deterministic identifier tests for the MassFlow graph data layer.

These tests prove that node/relationship identifiers are content-addressed,
stable across processes (hash-seed independent), and sensitive to the inputs
that define their identity. No file I/O or network access is involved.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
from matchms import Spectrum

from MassFlow.network import (
    feature_node_id,
    library_node_id,
    relationship_id,
    spectrum_node_id,
)

pytestmark = pytest.mark.unit


def _spectrum(
    precursor_mz: float = 300.0,
    mz: tuple[float, ...] = (100.0, 200.0),
    intensities: tuple[float, ...] = (0.5, 1.0),
    spec_id: str = "q1",
    **extra: object,
) -> Spectrum:
    metadata: dict[str, object] = {
        "id": spec_id,
        "precursor_mz": precursor_mz,
        "retention_time": 12.5,
    }
    metadata.update(extra)
    return Spectrum(
        mz=np.array(mz, dtype=np.float64),
        intensities=np.array(intensities, dtype=np.float64),
        metadata=metadata,
    )


def test_spectrum_node_id_is_deterministic() -> None:
    spectrum = _spectrum()
    assert spectrum_node_id(spectrum) == spectrum_node_id(spectrum)
    assert spectrum_node_id(spectrum).startswith("nid1:query:")


def test_spectrum_node_id_changes_with_content() -> None:
    base = spectrum_node_id(_spectrum())
    assert spectrum_node_id(_spectrum(mz=(100.0, 201.0))) != base
    assert spectrum_node_id(_spectrum(precursor_mz=301.0)) != base
    assert spectrum_node_id(_spectrum(intensities=(0.5, 0.9))) != base


def test_spectrum_node_id_namespace_disambiguates() -> None:
    spectrum = _spectrum()
    assert spectrum_node_id(spectrum, namespace="A") != spectrum_node_id(spectrum)
    assert spectrum_node_id(spectrum, namespace="A") != spectrum_node_id(
        spectrum, namespace="B"
    )


def test_spectrum_node_id_missing_is_distinct_from_zero() -> None:
    # A missing precursor (NaN / absent) must not collide with a real 0 value.
    missing = Spectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        metadata={"id": "m1", "precursor_mz": float("nan")},
    )
    zero = Spectrum(
        mz=np.array([100.0], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        metadata={"id": "z1", "precursor_mz": 0.0},
    )
    assert spectrum_node_id(missing) != spectrum_node_id(zero)


def test_adduct_spelling_does_not_change_node_id() -> None:
    # "M+H" and "[M+H]+" are the same ion and must canonicalise identically.
    assert spectrum_node_id(_spectrum(adduct="M+H")) == spectrum_node_id(
        _spectrum(adduct="[M+H]+")
    )


def test_library_node_id_stability_and_sensitivity() -> None:
    node = library_node_id(7, "ref_001")
    assert node == library_node_id(7, "ref_001")
    assert node.startswith("nid1:library:")
    assert node != library_node_id(8, "ref_001")
    assert node != library_node_id(7, "ref_002")


def test_feature_node_id_determinism() -> None:
    first = feature_node_id(
        precursor_mz=195.0877,
        retention_time_seconds=120.0,
        charge=1,
        adduct="[M+H]+",
        ion_mode="positive",
    )
    assert first == feature_node_id(
        precursor_mz=195.0877,
        retention_time_seconds=120.0,
        charge=1,
        adduct="M+H",
        ion_mode="positive",
    )
    assert first.startswith("nid1:feature:")
    assert first != feature_node_id(precursor_mz=195.0877, retention_time_seconds=121.0)


def test_relationship_id_determinism_and_sensitivity() -> None:
    base = relationship_id(
        relationship_type="spectral",
        source_node_id="nid1:query:abc",
        target_node_id="nid1:library:def",
        key_parameters={"algorithm": "cosine"},
    )
    assert base == relationship_id(
        relationship_type="spectral",
        source_node_id="nid1:query:abc",
        target_node_id="nid1:library:def",
        key_parameters={"algorithm": "cosine"},
    )
    assert base.startswith("rid1:spectral:")
    # Endpoint, direction and parameters all affect identity.
    assert base != relationship_id(
        relationship_type="spectral",
        source_node_id="nid1:library:def",
        target_node_id="nid1:query:abc",
        key_parameters={"algorithm": "cosine"},
    )
    assert base != relationship_id(
        relationship_type="spectral",
        source_node_id="nid1:query:abc",
        target_node_id="nid1:library:def",
        key_parameters={"algorithm": "modified_cosine"},
    )
    assert base != relationship_id(
        relationship_type="spectral",
        source_node_id="nid1:query:abc",
        target_node_id="nid1:library:def",
        directed=False,
        key_parameters={"algorithm": "cosine"},
    )


def test_relationship_id_rejects_non_primitive_parameters() -> None:
    with pytest.raises(TypeError):
        relationship_id(
            relationship_type="spectral",
            source_node_id="nid1:query:abc",
            target_node_id="nid1:library:def",
            key_parameters={"bad": {"nested": 1}},
        )


_SUBPROCESS_ID_SNIPPET = (
    "import numpy as np\n"
    "from matchms import Spectrum\n"
    "from MassFlow.network import spectrum_node_id, relationship_id\n"
    "s = Spectrum(mz=np.array([100.0, 200.0]), intensities=np.array([0.5, 1.0]),"
    " metadata={'id': 'q1', 'precursor_mz': 300.0, 'retention_time': 12.5})\n"
    "print(spectrum_node_id(s))\n"
    "print(relationship_id(relationship_type='spectral',"
    " source_node_id='nid1:query:abc', target_node_id='nid1:library:def',"
    " key_parameters={'algorithm': 'cosine'}))\n"
)


def _run_with_hash_seed(seed: str) -> list[str]:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    completed = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_ID_SNIPPET],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return completed.stdout.strip().splitlines()


def test_identifiers_are_hash_seed_independent() -> None:
    """Identifiers must be identical regardless of PYTHONHASHSEED."""
    first = _run_with_hash_seed("1")
    second = _run_with_hash_seed("2")
    assert first == second
    # And identical to the in-process computation of the same payload.
    assert first[0] == spectrum_node_id(_spectrum())
