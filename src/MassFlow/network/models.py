"""
Graph-compatible data layer for MassFlow.

This module defines the minimal, strongly typed, serializable vocabulary that
lets future molecular-networking code consume MassFlow results **without**
changing the current annotation workflow. It implements the *data* layer only:
there is no graph construction, no clustering, no ion-identity discovery, no
GraphML/FBMN export, and no visualization here. Every algorithm belongs to a
future, experiment-gated subsystem that sits strictly downstream of the
annotation engine.

Design contract (see :mod:`MassFlow.workflow` for the producer side)
--------------------------------------------------------------------

The annotation engine already emits, per experimental file, a
:class:`~MassFlow.workflow.FileExecutionResult` whose ``results`` are
:class:`~MassFlow.similarity.SearchResult` rows and whose ``fdr_summary``
carries the per-query target-decoy statistics. This module projects those
artifacts into layered, immutable types:

* **L0 Measurement** - raw spectra; represented by ``matchms.Spectrum``
  (never duplicated here). :class:`GraphNode` stores only *metadata* (no peak
  arrays), so float64 m/z/intensity precision lives in the spectral store, not
  in the graph.
* **L1 Direct identification evidence** - :class:`AnnotationEvidence`
  (similarity ``score``, ``matched_peaks``, ``mass_error_ppm``, tier). This is
  *not* a statistical-confidence carrier.
* **L2 Primary statistical confidence** - :class:`FdrAssessment`
  (per-query ``q_value``/``p_value``). This is the **only** FDR carrier, it is
  keyed by query node in :attr:`MolecularGraph.fdr_assessments`, and it is
  copy-only downstream.
* **L3 Spectral similarity** - :class:`SpectralRelationship`.
* **L4 Chemical / ion-identity relationships** - :class:`ChemicalRelationship`
  and :class:`IonIdentityRelationship`.
* **L5 Network context** - deliberately absent; future family analysis lives
  in its own (post-1.0) layer.

Hard invariants enforced by construction:

1. No relationship type exposes a field named ``q_value``, ``p_value``,
   ``fdr`` or ``confidence``. FDR can *never* be attached to an edge.
2. :class:`FdrAssessment` carries the unchanged, query-scoped semantics of the
   existing target-decoy competition; this module never recomputes it.
3. Missing scientific values are ``None`` (JSON ``null``); they are never
   encoded as ``0``.
4. Numeric m/z and mass-difference values are Python ``float`` (float64).
5. Retention time always carries its unit in the field name:
   ``retention_time_seconds``.

No module on the stable annotation path imports this module, and this module
imports nothing from :mod:`MassFlow.workflow`, :mod:`MassFlow.cli`,
:mod:`MassFlow.database` or :mod:`MassFlow.io` (file writes are performed by
:mod:`MassFlow.io`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import struct
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    NewType,
    Optional,
    Sequence,
    Union,
)

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from MassFlow.cheminformatics import normalize_adduct

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matchms import Spectrum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema and identifier-format constants
# ---------------------------------------------------------------------------

#: Version of the molecular-graph document schema written by this module.
SCHEMA_VERSION = "1"

#: Version of the deterministic identifier scheme (embedded in every id prefix).
_ID_SCHEME_VERSION = 1

#: Number of hex characters kept from the SHA-256 digest of a canonical payload.
_DIGEST_LENGTH = 32

_NODE_ID_PATTERN = rf"^nid{_ID_SCHEME_VERSION}:[a-z_]+:[0-9a-f]{{{_DIGEST_LENGTH}}}$"
_RELATIONSHIP_ID_PATTERN = (
    rf"^rid{_ID_SCHEME_VERSION}:[a-z_]+:[0-9a-f]{{{_DIGEST_LENGTH}}}$"
)

IonMode = Literal["positive", "negative", "neutral"]
MoleculeRelationship = Literal[
    "same_molecule", "same_molecule_candidate", "related_molecule", "unknown"
]

#: A stable, deterministic spectral/feature node identifier (``nid1:<kind>:<hex>``).
NodeId = NewType("NodeId", str)

#: A stable, deterministic relationship identifier (``rid1:<type>:<hex>``).
RelationshipId = NewType("RelationshipId", str)

# Field aliases: validate the identifier *format* on every model.
NodeIdField = Annotated[str, StringConstraints(pattern=_NODE_ID_PATTERN)]
RelationshipIdField = Annotated[
    str, StringConstraints(pattern=_RELATIONSHIP_ID_PATTERN)
]


# ---------------------------------------------------------------------------
# Deterministic identifier primitives
# ---------------------------------------------------------------------------


def _frame(payload: bytes) -> bytes:
    """Length-prefix *payload* so concatenated fields cannot be ambiguous."""
    return struct.pack(">I", len(payload)) + payload


def _encode_text(value: str) -> bytes:
    return b"t" + value.encode("utf-8")


def _encode_optional_text(value: Optional[str]) -> bytes:
    if value is None:
        return b"\x00"
    return b"\x01" + value.encode("utf-8")


def _encode_number(value: Optional[float]) -> bytes:
    """Encode a float64, distinguishing "missing" from any real value.

    ``None`` and ``NaN`` both mean "not measured" (``AGENTS.md`` §1.2) and map
    to a dedicated missing tag, so they never collide with a genuine value.
    """
    if value is None:
        return b"\x00"
    number = float(value)
    if math.isnan(number):
        return b"\x00"
    return b"\x01" + struct.pack(">d", number)


def _encode_integer(value: Optional[int]) -> bytes:
    if value is None:
        return b"\x00"
    return b"\x01" + struct.pack(">q", int(value))


def _encode_array(array: Any) -> bytes:
    """Encode a numeric array as a length-prefixed big-endian float64 block."""
    values = np.asarray(array, dtype=np.float64)
    return struct.pack(">Q", values.size) + values.astype(">f8", copy=False).tobytes()


def _encode_scalar(value: Any) -> bytes:
    """Encode a JSON-primitive relationship parameter deterministically."""
    if value is None:
        return b"\x00"
    if isinstance(value, bool):
        return b"\x01" + (b"1" if value else b"0")
    if isinstance(value, int):
        return b"\x02" + struct.pack(">q", int(value))
    if isinstance(value, float):
        return b"\x03" + struct.pack(">d", float(value))
    if isinstance(value, str):
        return b"\x04" + value.encode("utf-8")
    raise TypeError(
        f"Relationship key parameters must be JSON primitives; got {type(value)!r}."
    )


def _encode_mapping(mapping: Mapping[str, Any]) -> bytes:
    chunks: List[bytes] = []
    for key in sorted(mapping):
        chunks.append(_frame(_encode_text(str(key))))
        chunks.append(_frame(_encode_scalar(mapping[key])))
    return b"".join(chunks)


def _digest(parts: Sequence[bytes]) -> str:
    """SHA-256 over length-framed parts, truncated to ``_DIGEST_LENGTH`` hex.

    Uses ``hashlib`` (never the built-in ``hash``) so identifiers are identical
    across processes and independent of ``PYTHONHASHSEED``.
    """
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(_frame(part))
    return hasher.hexdigest()[:_DIGEST_LENGTH]


def _canonical_adduct(adduct: Any) -> Optional[str]:
    """Return the canonical adduct spelling, falling back to the raw text."""
    if adduct is None:
        return None
    text = str(adduct)
    normalized = normalize_adduct(text)
    return normalized if normalized is not None else text


def spectrum_node_id(
    spectrum: "Spectrum", *, namespace: Optional[str] = None
) -> NodeId:
    """
    Compute the stable node identifier of a spectrum.

    The identifier is a content hash of the precursor m/z, retention time,
    charge, (canonical) adduct and the full float64 peak arrays. Two spectra
    with identical content therefore share a node; pass ``namespace`` (for
    example the source file stem) when identical measurements from different
    samples must remain distinct nodes.

    Parameters
    ----------
    spectrum : matchms.Spectrum
        The spectrum to identify.
    namespace : str or None, optional
        Optional disambiguating namespace mixed into the digest. ``None`` (the
        default) yields a purely content-addressed identifier.

    Returns
    -------
    NodeId
        A stable identifier of the form ``nid1:query:<hex>``.

    Examples
    --------
    >>> node_id = spectrum_node_id(spectrum, namespace="sample_A")  # doctest: +SKIP
    """
    mz_array = np.asarray(spectrum.peaks.mz, dtype=np.float64)
    intensity_array = np.asarray(spectrum.peaks.intensities, dtype=np.float64)
    parts = [
        _encode_text(f"nid{_ID_SCHEME_VERSION}"),
        _encode_text("query"),
        _encode_optional_text(namespace),
        _encode_number(spectrum.get("precursor_mz")),
        _encode_number(spectrum.get("retention_time")),
        _encode_integer(_coerce_int(spectrum.get("charge"))),
        _encode_optional_text(_canonical_adduct(spectrum.get("adduct"))),
        _encode_array(mz_array),
        _encode_array(intensity_array),
    ]
    return NodeId(f"nid{_ID_SCHEME_VERSION}:query:{_digest(parts)}")


def library_node_id(library_build_id: int, original_id: str) -> NodeId:
    """
    Compute the stable node identifier of a reference-library spectrum.

    The identifier is derived from the library build row id and the spectrum's
    original identifier only, so it never requires re-hashing a large library.

    Parameters
    ----------
    library_build_id : int
        The ``library_builds.id`` row that produced the spectrum.
    original_id : str
        The spectrum's stored ``original_id``/``id`` metadata value.

    Returns
    -------
    NodeId
        A stable identifier of the form ``nid1:library:<hex>``.
    """
    parts = [
        _encode_text(f"nid{_ID_SCHEME_VERSION}"),
        _encode_text("library"),
        _encode_integer(int(library_build_id)),
        _encode_text(str(original_id)),
    ]
    return NodeId(f"nid{_ID_SCHEME_VERSION}:library:{_digest(parts)}")


def feature_node_id(
    *,
    precursor_mz: Optional[float],
    retention_time_seconds: Optional[float] = None,
    charge: Optional[int] = None,
    adduct: Optional[str] = None,
    ion_mode: Optional[str] = None,
) -> NodeId:
    """
    Compute the stable node identifier of an LC-MS feature.

    Parameters
    ----------
    precursor_mz : float or None
        Feature precursor m/z (float64 semantics).
    retention_time_seconds : float or None, optional
        Feature-centered retention time in **seconds**.
    charge : int or None, optional
        Ion charge state.
    adduct : str or None, optional
        Ionization adduct; canonicalised before hashing.
    ion_mode : str or None, optional
        ``"positive"``, ``"negative"`` or ``"neutral"``.

    Returns
    -------
    NodeId
        A stable identifier of the form ``nid1:feature:<hex>``.
    """
    parts = [
        _encode_text(f"nid{_ID_SCHEME_VERSION}"),
        _encode_text("feature"),
        _encode_number(precursor_mz),
        _encode_number(retention_time_seconds),
        _encode_integer(charge),
        _encode_optional_text(_canonical_adduct(adduct)),
        _encode_optional_text(ion_mode),
    ]
    return NodeId(f"nid{_ID_SCHEME_VERSION}:feature:{_digest(parts)}")


def relationship_id(
    *,
    relationship_type: str,
    source_node_id: str,
    target_node_id: str,
    directed: bool = True,
    key_parameters: Optional[Mapping[str, Any]] = None,
) -> RelationshipId:
    """
    Compute the deterministic identifier of a relationship.

    The identifier hashes the relationship type, endpoints, direction and the
    type-specific ``key_parameters`` (never the provenance), so two
    relationships derived from identical inputs are the same edge.

    Parameters
    ----------
    relationship_type : str
        One of ``"spectral"``, ``"chemical"`` or ``"ion_identity"``.
    source_node_id, target_node_id : str
        Endpoint node identifiers.
    directed : bool, optional
        Whether the relationship is directed (default ``True``).
    key_parameters : mapping or None, optional
        Type-specific parameters (JSON primitives only).

    Returns
    -------
    RelationshipId
        A stable identifier of the form ``rid1:<type>:<hex>``.
    """
    parts = [
        _encode_text(f"rid{_ID_SCHEME_VERSION}"),
        _encode_text(relationship_type),
        _encode_text(str(source_node_id)),
        _encode_text(str(target_node_id)),
        _encode_text("1" if directed else "0"),
        _encode_mapping(key_parameters or {}),
    ]
    return RelationshipId(
        f"rid{_ID_SCHEME_VERSION}:{relationship_type}:{_digest(parts)}"
    )


# ---------------------------------------------------------------------------
# Coercion helpers (None/Nan == missing; units explicit; float64 preserved)
# ---------------------------------------------------------------------------


def _coerce_float(value: Any) -> Optional[float]:
    """Return a float64 value, or ``None`` for missing/NaN/non-numeric input."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _coerce_int(value: Any) -> Optional[int]:
    """Return an int, or ``None`` for missing/NaN/non-numeric input."""
    number = _coerce_float(value)
    if number is None:
        return None
    return int(number)


def _optional_text(value: Any) -> Optional[str]:
    """Return stripped text, or ``None`` when empty/absent."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_ion_mode(value: Any) -> Optional[IonMode]:
    text = _optional_text(value)
    if text in ("positive", "negative", "neutral"):
        return text  # type: ignore[return-value]
    return None


def _coerce_breakdown(value: Any) -> Optional[Dict[str, float]]:
    """Normalize a score breakdown (dict or JSON string) to ``dict[str, float]``."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(value, Mapping):
        return None
    breakdown: Dict[str, float] = {}
    for key, raw in value.items():
        number = _coerce_float(raw)
        if number is not None:
            breakdown[str(key)] = number
    return breakdown or None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _installed_massflow_version() -> str:
    try:
        from MassFlow import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - defensive provenance default
        return "unknown"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class NetworkModel(BaseModel):
    """Base class for the graph data layer (extra fields forbidden).

    ``allow_inf_nan`` is disabled so a required numeric field can never carry
    ``NaN``/``inf``: missing scientific values are represented as ``None``
    (``null`` on the wire), never as a non-finite float.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class GraphNode(NetworkModel):
    """
    A graph node: a serializable metadata projection of a spectrum.

    Peak arrays are deliberately **not** copied here (``matchms.Spectrum`` is
    not duplicated); retrieve the underlying spectrum by identifier through a
    :class:`~MassFlow.storage.SpectralStore`. Numeric m/z is float64 and
    retention time is always expressed in **seconds**.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: NodeIdField = Field(
        ..., description="Stable, deterministic node identifier (nid1:<kind>:<hex>)."
    )
    node_kind: Literal["query_spectrum", "library_spectrum"] = Field(
        ..., description="Whether this node is an experimental query or a library hit."
    )
    precursor_mz: float = Field(..., gt=0.0, description="Precursor m/z (float64).")
    retention_time_seconds: Optional[float] = Field(
        None, ge=0.0, description="Retention time in seconds (None = not measured)."
    )
    charge: Optional[int] = Field(None, description="Ion charge state.")
    ion_mode: Optional[IonMode] = Field(None, description="Ionization polarity.")
    adduct: Optional[str] = Field(
        None, description="Canonical ionization adduct (e.g. [M+H]+)."
    )
    name: Optional[str] = Field(None, description="Compound name, when known.")
    smiles: Optional[str] = Field(None, description="Structure SMILES, when known.")
    inchikey: Optional[str] = Field(None, description="InChIKey, when known.")
    formula: Optional[str] = Field(None, description="Chemical formula, when known.")
    library_build_id: Optional[int] = Field(
        None, ge=0, description="library_builds.id this library node came from."
    )
    feature_id: Optional[NodeIdField] = Field(
        None, description="Parent feature node identifier, when assigned."
    )
    num_peaks: Optional[int] = Field(
        None,
        ge=0,
        description="Number of fragment peaks (peaks themselves are not stored).",
    )
    physical_valid: Optional[bool] = Field(
        None, description="Result of the 5 ppm physical-integrity gate, when evaluated."
    )
    physical_integrity_reason: Optional[str] = Field(
        None, description="Human-readable reason when physical validation failed."
    )

    @classmethod
    def from_query_spectrum(
        cls, spectrum: "Spectrum", *, namespace: Optional[str] = None
    ) -> "GraphNode":
        """
        Build a query node from a :class:`matchms.Spectrum`.

        Parameters
        ----------
        spectrum : matchms.Spectrum
            The experimental spectrum (metadata is projected; peaks are not stored).
        namespace : str or None, optional
            Optional identifier namespace (e.g. source file stem).

        Returns
        -------
        GraphNode
            A node populated from the spectrum's metadata.
        """
        return cls._from_spectrum(
            spectrum,
            node_id=spectrum_node_id(spectrum, namespace=namespace),
            node_kind="query_spectrum",
        )

    @classmethod
    def from_library_spectrum(
        cls, spectrum: "Spectrum", *, library_build_id: int
    ) -> "GraphNode":
        """
        Build a library node from a :class:`matchms.Spectrum`.

        The node identifier is derived from the library build row and the
        spectrum's stored id (never from a full-library re-hash).

        Parameters
        ----------
        spectrum : matchms.Spectrum
            A reference-library spectrum.
        library_build_id : int
            The ``library_builds.id`` row that owns the spectrum.

        Returns
        -------
        GraphNode
            A library node populated from the spectrum's metadata.
        """
        original_id = str(spectrum.get("id", ""))
        return cls._from_spectrum(
            spectrum,
            node_id=library_node_id(library_build_id, original_id),
            node_kind="library_spectrum",
            library_build_id=library_build_id,
        )

    @classmethod
    def _from_spectrum(
        cls,
        spectrum: "Spectrum",
        *,
        node_id: str,
        node_kind: Literal["query_spectrum", "library_spectrum"],
        library_build_id: Optional[int] = None,
    ) -> "GraphNode":
        precursor_mz = _coerce_float(spectrum.get("precursor_mz"))
        if precursor_mz is None or precursor_mz <= 0.0:
            raise ValueError(
                "A graph node requires a positive precursor m/z; the spectrum "
                "has no usable precursor (missing values are never encoded as 0)."
            )
        peaks = spectrum.peaks
        num_peaks = int(np.asarray(peaks.mz).size) if peaks is not None else None
        return cls(
            node_id=node_id,
            node_kind=node_kind,
            precursor_mz=float(precursor_mz),
            retention_time_seconds=_coerce_float(spectrum.get("retention_time")),
            charge=_coerce_int(spectrum.get("charge")),
            ion_mode=_optional_ion_mode(
                spectrum.get("ionmode") or spectrum.get("ion_mode")
            ),
            adduct=_canonical_adduct(spectrum.get("adduct")),
            name=_optional_text(spectrum.get("compound_name") or spectrum.get("name")),
            smiles=_optional_text(spectrum.get("smiles")),
            inchikey=_optional_text(spectrum.get("inchikey")),
            formula=_optional_text(spectrum.get("formula")),
            library_build_id=library_build_id,
            num_peaks=num_peaks,
        )


class Feature(NetworkModel):
    """
    An LC-MS feature: a group of one or more spectra sharing an ion identity.

    This is a *data* type only. Feature detection and representative/consensus
    spectrum selection are future (post-1.0) work; the corresponding field is
    typed now but left unset.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_id: NodeIdField = Field(..., description="Stable feature identifier.")
    precursor_mz: float = Field(
        ..., gt=0.0, description="Feature precursor m/z (float64)."
    )
    retention_time_seconds: Optional[float] = Field(
        None, ge=0.0, description="Feature retention time in seconds."
    )
    ion_mode: Optional[IonMode] = None
    charge: Optional[int] = None
    adduct: Optional[str] = None
    spectrum_node_ids: tuple[NodeIdField, ...] = Field(
        ..., min_length=1, description="Member spectrum node identifiers."
    )
    representative_spectrum_node_id: Optional[NodeIdField] = Field(
        None,
        description="Representative/consensus spectrum node (post-1.0; unset for now).",
    )
    sample_id: Optional[str] = Field(None, description="Originating sample identifier.")
    abundance: Optional[float] = Field(
        None, ge=0.0, description="Relative abundance (None = not measured)."
    )

    @model_validator(mode="after")
    def _validate_members(self) -> "Feature":
        if len(set(self.spectrum_node_ids)) != len(self.spectrum_node_ids):
            raise ValueError("spectrum_node_ids must be unique.")
        if (
            self.representative_spectrum_node_id is not None
            and self.representative_spectrum_node_id not in self.spectrum_node_ids
        ):
            raise ValueError(
                "representative_spectrum_node_id must be one of spectrum_node_ids."
            )
        return self


class FdrAssessment(NetworkModel):
    """
    **L2 - primary statistical confidence.** The only FDR carrier in the graph.

    This type preserves, unchanged, the semantics of MassFlow's per-query
    target-decoy competition (:func:`MassFlow.similarity.calibrate_query_level_fdr`):
    ``q_value`` is scoped to the *query* competition unit, and ``p_value`` is a
    diagnostic only (never compared against an FDR threshold). Downstream
    network layers may read these values but must never recompute or adjust
    them, and network context must never alter them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    competition_unit: Literal["query"] = Field(
        "query",
        description="TDC competition unit; currently always the query spectrum.",
    )
    q_value: float = Field(..., ge=0.0, le=1.0, description="Query-scoped q-value.")
    p_value: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Diagnostic empirical p-value (not a threshold).",
    )
    calibrated: bool = Field(
        ...,
        description="False when the decoy null was empty and q_value is the 1/N bound.",
    )
    method: str = Field(
        "target_decoy_competition", description="Confidence method identifier."
    )
    library_size: int = Field(
        ..., ge=0, description="True target-library size used for calibration."
    )
    n_decoy_competitions: Optional[int] = Field(
        None, ge=0, description="Number of decoy competitions observed."
    )


class AnnotationEvidence(NetworkModel):
    """
    **L1 - direct identification evidence** for one (query, reference) hit.

    Carries the similarity score, matched-peak count and physical mass error of
    a single match. It intentionally has **no** q-value/p-value/fdr field: the
    query's statistical confidence is a separate :class:`FdrAssessment` stored
    once on the graph, so an edge can never be mistaken for a confidence claim.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_node_id: NodeIdField = Field(..., description="Query node identifier.")
    reference_node_id: NodeIdField = Field(
        ..., description="Reference node identifier."
    )
    algorithm: str = Field(
        ..., description="Similarity algorithm that produced the hit."
    )
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Spectral similarity score (not an FDR)."
    )
    matched_peaks: int = Field(..., ge=0, description="Number of matched peaks.")
    mass_error_ppm: Optional[float] = Field(
        None, description="Precursor mass error in ppm (direct physical evidence)."
    )
    annotation_tier: Optional[str] = Field(
        None, description="Annotation tier label, when assigned."
    )
    structural_similarity: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Structural (e.g. Tanimoto) similarity, when available.",
    )
    score_breakdown: Optional[Dict[str, float]] = Field(
        None, description="Per-component score breakdown (e.g. consensus engines)."
    )
    physical_valid: Optional[bool] = Field(
        None, description="Whether the precursor passed the 5 ppm gate."
    )

    @classmethod
    def from_search_result(
        cls,
        row: Mapping[str, Any],
        *,
        query_node_id: str,
        reference_node_id: str,
        algorithm: str,
    ) -> "AnnotationEvidence":
        """
        Project a :class:`~MassFlow.similarity.SearchResult` row into L1 evidence.

        Parameters
        ----------
        row : mapping
            A single ``SearchResult`` dict produced by the annotation engine.
        query_node_id : str
            Stable node identifier of the query spectrum.
        reference_node_id : str
            Stable node identifier of the matched reference spectrum.
        algorithm : str
            Similarity algorithm (``SearchResult`` rows do not carry it).

        Returns
        -------
        AnnotationEvidence
            The direct identification evidence for this hit.
        """
        score = _coerce_float(row.get("score"))
        if score is None:
            raise ValueError("A search result row must carry a numeric 'score'.")
        return cls(
            query_node_id=query_node_id,
            reference_node_id=reference_node_id,
            algorithm=algorithm,
            score=score,
            matched_peaks=_coerce_int(row.get("matched_peaks")) or 0,
            mass_error_ppm=_coerce_float(row.get("mass_error_ppm")),
            annotation_tier=_optional_text(row.get("annotation_tier")),
            structural_similarity=_coerce_float(row.get("structural_similarity")),
            score_breakdown=_coerce_breakdown(row.get("score_breakdown")),
        )


class NeutralLoss(NetworkModel):
    """A neutral-loss value object (label/formula plus measured mass)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: Optional[str] = Field(None, description="Common label, e.g. 'H2O', 'NH3'.")
    formula: Optional[str] = Field(
        None, description="Neutral-loss formula, when known."
    )
    mass_da: float = Field(..., description="Neutral-loss mass in Da (float64).")
    mass_error_ppm: Optional[float] = Field(
        None, description="Deviation from the theoretical loss mass, in ppm."
    )


class RelationshipProvenance(NetworkModel):
    """Per-relationship lineage, mirroring the ``library_builds`` JSON shape."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: str = Field(
        ..., description="Algorithm/derivation that produced the edge."
    )
    algorithm_version: Optional[str] = Field(
        None, description="Algorithm version, if known."
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Algorithm parameters (JSON-serializable)."
    )
    library_build_id: Optional[int] = Field(
        None, ge=0, description="Reference library build row, when applicable."
    )
    config_digest_sha256: Optional[str] = Field(
        None, description="SHA-256 of the effective MassFlow configuration."
    )
    massflow_version: str = Field(
        default_factory=_installed_massflow_version,
        description="MassFlow version that produced the edge.",
    )
    created_at: str = Field(
        default_factory=_utc_now_iso, description="ISO-8601 UTC creation timestamp."
    )
    source_module: str = Field(
        "MassFlow.network", description="Module that produced the edge."
    )


class RelationshipBase(NetworkModel):
    """
    Shared edge semantics for all typed relationships.

    Subclasses declare the ``relationship_type`` discriminator and a
    ``key_parameters()`` implementation; the shared validator recomputes and
    verifies :attr:`relationship_id`, so an edge's identity is always
    consistent with its content. Construct relationships via each subclass'
    ``create(...)`` classmethod.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    relationship_id: RelationshipIdField = Field(
        ..., description="Stable, deterministic edge identifier (rid1:<type>:<hex>)."
    )
    relationship_type: str = Field(
        ..., description="Discriminator for the edge variant."
    )
    source_node_id: NodeIdField = Field(
        ..., description="Source endpoint node identifier."
    )
    target_node_id: NodeIdField = Field(
        ..., description="Target endpoint node identifier."
    )
    directed: bool = Field(True, description="Whether the relationship is directed.")
    molecule_relationship: MoleculeRelationship = Field(
        "unknown",
        description=(
            "'same_molecule' / 'same_molecule_candidate' (same analyte), "
            "'related_molecule' (transformation/related), or 'unknown'."
        ),
    )
    provenance: RelationshipProvenance = Field(
        ..., description="Per-edge lineage (algorithm, build, config, version)."
    )

    def key_parameters(self) -> Dict[str, Any]:
        """Return the type-specific parameters that define this edge's identity."""
        raise NotImplementedError

    @model_validator(mode="after")
    def _validate_identifier(self) -> "RelationshipBase":
        if type(self) is RelationshipBase:  # abstract base: nothing to verify
            return self
        expected = relationship_id(
            relationship_type=self.relationship_type,
            source_node_id=self.source_node_id,
            target_node_id=self.target_node_id,
            directed=self.directed,
            key_parameters=self.key_parameters(),
        )
        if str(self.relationship_id) != str(expected):
            raise ValueError(
                "relationship_id is not consistent with the relationship content; "
                "construct edges with <RelationshipType>.create(...)."
            )
        return self


class SpectralRelationship(RelationshipBase):
    """**L3** similarity edge between two spectra (score, never FDR)."""

    relationship_type: Literal["spectral"] = Field(
        "spectral", description="Edge discriminator."
    )
    score: float = Field(..., ge=0.0, le=1.0, description="Spectral similarity score.")
    matched_peaks: int = Field(..., ge=0, description="Number of matched peaks.")
    algorithm: str = Field(..., description="Similarity algorithm.")
    ms1_tolerance: Optional[float] = Field(
        None, ge=0.0, description="Precursor tolerance."
    )
    ms2_tolerance: Optional[float] = Field(
        None, ge=0.0, description="Fragment tolerance."
    )
    tolerance_unit: Literal["Da", "ppm"] = Field("Da", description="Tolerance unit.")
    library_build_id: Optional[int] = Field(
        None, ge=0, description="Reference library build row, when applicable."
    )

    _KEY_FIELDS = (
        "algorithm",
        "ms1_tolerance",
        "ms2_tolerance",
        "tolerance_unit",
        "library_build_id",
    )

    def key_parameters(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self._KEY_FIELDS}

    @classmethod
    def create(
        cls,
        *,
        source_node_id: str,
        target_node_id: str,
        score: float,
        matched_peaks: int,
        algorithm: str,
        provenance: RelationshipProvenance,
        directed: bool = True,
        molecule_relationship: MoleculeRelationship = "unknown",
        ms1_tolerance: Optional[float] = None,
        ms2_tolerance: Optional[float] = None,
        tolerance_unit: Literal["Da", "ppm"] = "Da",
        library_build_id: Optional[int] = None,
    ) -> "SpectralRelationship":
        """Construct a spectral edge, deriving its deterministic identifier."""
        params: Dict[str, Any] = {
            "algorithm": algorithm,
            "ms1_tolerance": ms1_tolerance,
            "ms2_tolerance": ms2_tolerance,
            "tolerance_unit": tolerance_unit,
            "library_build_id": library_build_id,
        }
        return cls(
            relationship_id=relationship_id(
                relationship_type="spectral",
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                directed=directed,
                key_parameters=params,
            ),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            directed=directed,
            molecule_relationship=molecule_relationship,
            provenance=provenance,
            score=score,
            matched_peaks=matched_peaks,
            **params,
        )


class ChemicalRelationship(RelationshipBase):
    """
    **L4** mass-difference edge between two molecules.

    A chemical relationship is an explicit *hypothesis* with uncertainty; it is
    never an automatic structural claim (``interpretation_status`` defaults to
    ``"hypothesis"``).
    """

    relationship_type: Literal["chemical"] = Field(
        "chemical", description="Edge discriminator."
    )
    delta_mass_da: float = Field(
        ...,
        description=(
            "Neutral-mass difference (Da, float64): target - source, "
            "adduct-resolved. A hypothesis quantity, not a structural claim."
        ),
    )
    neutral_loss: Optional[NeutralLoss] = Field(
        None, description="Interpreted neutral loss, when assigned."
    )
    transformation: Optional[str] = Field(
        None, description="Hypothesized chemical transformation, when assigned."
    )
    interpretation_status: Literal["hypothesis", "unassigned"] = Field(
        "hypothesis", description="Explicit uncertainty of the interpretation."
    )
    notes: Optional[str] = Field(None, description="Free-text annotation.")

    _KEY_FIELDS = ("delta_mass_da", "transformation", "interpretation_status")

    def key_parameters(self) -> Dict[str, Any]:
        params = {name: getattr(self, name) for name in self._KEY_FIELDS}
        params["neutral_loss_mass_da"] = (
            self.neutral_loss.mass_da if self.neutral_loss is not None else None
        )
        return params

    @classmethod
    def create(
        cls,
        *,
        source_node_id: str,
        target_node_id: str,
        delta_mass_da: float,
        provenance: RelationshipProvenance,
        directed: bool = True,
        molecule_relationship: MoleculeRelationship = "related_molecule",
        neutral_loss: Optional[NeutralLoss] = None,
        transformation: Optional[str] = None,
        interpretation_status: Literal["hypothesis", "unassigned"] = "hypothesis",
        notes: Optional[str] = None,
    ) -> "ChemicalRelationship":
        """Construct a chemical edge, deriving its deterministic identifier."""
        params: Dict[str, Any] = {
            "delta_mass_da": delta_mass_da,
            "transformation": transformation,
            "interpretation_status": interpretation_status,
            "neutral_loss_mass_da": (
                neutral_loss.mass_da if neutral_loss is not None else None
            ),
        }
        return cls(
            relationship_id=relationship_id(
                relationship_type="chemical",
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                directed=directed,
                key_parameters=params,
            ),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            directed=directed,
            molecule_relationship=molecule_relationship,
            provenance=provenance,
            delta_mass_da=delta_mass_da,
            neutral_loss=neutral_loss,
            transformation=transformation,
            interpretation_status=interpretation_status,
            notes=notes,
        )


class IonIdentityRelationship(RelationshipBase):
    """
    **L4** ion-identity edge: two ions that are the *same* molecule.

    Separates "same molecule" relationships (adducts, isotopes, multimers)
    from "related molecule" relationships (in-source fragments), using the
    inherited :attr:`molecule_relationship` field.
    """

    relationship_type: Literal["ion_identity"] = Field(
        "ion_identity", description="Edge discriminator."
    )
    relationship_kind: Literal[
        "adduct", "isotope", "in_source_fragment", "multimer"
    ] = Field(..., description="Ion-identity mechanism.")
    adduct_source: Optional[str] = Field(None, description="Source ion adduct.")
    adduct_target: Optional[str] = Field(None, description="Target ion adduct.")
    charge_source: Optional[int] = Field(None, description="Source ion charge.")
    charge_target: Optional[int] = Field(None, description="Target ion charge.")
    delta_mass_da: float = Field(
        ..., description="Observed m/z difference (Da, float64)."
    )
    expected_delta_mass_da: Optional[float] = Field(
        None, description="Theoretical m/z difference (Da, float64)."
    )
    delta_mass_error_ppm: Optional[float] = Field(
        None, description="Deviation from the theoretical difference, in ppm."
    )

    _KEY_FIELDS = (
        "relationship_kind",
        "adduct_source",
        "adduct_target",
        "charge_source",
        "charge_target",
        "expected_delta_mass_da",
    )

    def key_parameters(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self._KEY_FIELDS}

    @classmethod
    def create(
        cls,
        *,
        source_node_id: str,
        target_node_id: str,
        relationship_kind: Literal[
            "adduct", "isotope", "in_source_fragment", "multimer"
        ],
        delta_mass_da: float,
        provenance: RelationshipProvenance,
        directed: bool = True,
        molecule_relationship: Optional[MoleculeRelationship] = None,
        adduct_source: Optional[str] = None,
        adduct_target: Optional[str] = None,
        charge_source: Optional[int] = None,
        charge_target: Optional[int] = None,
        expected_delta_mass_da: Optional[float] = None,
        delta_mass_error_ppm: Optional[float] = None,
    ) -> "IonIdentityRelationship":
        """Construct an ion-identity edge, deriving its deterministic identifier."""
        if molecule_relationship is None:
            molecule_relationship = (
                "related_molecule"
                if relationship_kind == "in_source_fragment"
                else "same_molecule"
            )
        params: Dict[str, Any] = {
            "relationship_kind": relationship_kind,
            "adduct_source": adduct_source,
            "adduct_target": adduct_target,
            "charge_source": charge_source,
            "charge_target": charge_target,
            "expected_delta_mass_da": expected_delta_mass_da,
        }
        return cls(
            relationship_id=relationship_id(
                relationship_type="ion_identity",
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                directed=directed,
                key_parameters=params,
            ),
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            directed=directed,
            molecule_relationship=molecule_relationship,
            provenance=provenance,
            relationship_kind=relationship_kind,
            delta_mass_da=delta_mass_da,
            adduct_source=adduct_source,
            adduct_target=adduct_target,
            charge_source=charge_source,
            charge_target=charge_target,
            expected_delta_mass_da=expected_delta_mass_da,
            delta_mass_error_ppm=delta_mass_error_ppm,
        )


#: Discriminated union of every relationship variant.
Relationship = Annotated[
    Union[SpectralRelationship, ChemicalRelationship, IonIdentityRelationship],
    Field(discriminator="relationship_type"),
]


class GraphProvenance(NetworkModel):
    """Run/build provenance for a whole :class:`MolecularGraph` document."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    massflow_version: str = Field(
        default_factory=_installed_massflow_version, description="MassFlow version."
    )
    created_at: str = Field(
        default_factory=_utc_now_iso, description="ISO-8601 UTC creation timestamp."
    )
    builder: str = Field(
        "MassFlow.network", description="Component that built the graph."
    )
    builder_version: Optional[str] = Field(
        None, description="Builder version, if known."
    )
    config_digest_sha256: Optional[str] = Field(
        None, description="SHA-256 of the effective MassFlow configuration."
    )
    library_build_id: Optional[int] = Field(
        None, ge=0, description="Reference library build row, when applicable."
    )
    library_sha256: Optional[str] = Field(
        None, description="Content hash of the reference library, when available."
    )
    run_provenance: Optional[Dict[str, Any]] = Field(
        None, description="Embedded run-provenance payload, when available."
    )


class NetworkContext(NetworkModel):
    """
    **L5 — network context.** A descriptive, non-statistical family label.

    A component may inherit a descriptive label from its confidently annotated
    members. This object is *informational only*: it carries no q-value, score or
    tier, it can never modify L1/L2 confidence, and it is never a target-decoy
    competition unit (see the anti-circularity rules in
    ``docs/network-aware-annotation-spec.md`` §6, Stage 5).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: NodeIdField = Field(
        ..., description="Deterministic connected-component identifier."
    )
    member_node_ids: tuple[NodeIdField, ...] = Field(
        ..., min_length=1, description="Member node/feature identifiers."
    )
    member_count: int = Field(..., ge=1, description="Number of members.")
    label: str = Field(
        ..., description="Descriptive family label inherited from seeds."
    )
    label_source: Literal["seed_annotation"] = Field(
        "seed_annotation", description="Where the label came from."
    )
    seed_annotation_ids: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="Seeds (direct high-confidence annotations) used.",
    )
    provenance: RelationshipProvenance = Field(
        ..., description="How this context was produced."
    )

    @model_validator(mode="after")
    def _validate_context(self) -> "NetworkContext":
        if self.member_count != len(self.member_node_ids):
            raise ValueError("member_count must equal the number of member_node_ids.")
        if len(set(self.member_node_ids)) != len(self.member_node_ids):
            raise ValueError("member_node_ids must be unique.")
        return self


class AnnotationInference(NetworkModel):
    """
    **L5 — network-inferred annotation.** Never a direct, FDR-calibrated hit.

    An inference proposes a label for a *poorly annotated* node that belongs to a
    family seeded by direct, calibrated, high-confidence annotations. It is kept
    strictly separate from :class:`AnnotationEvidence` (L1) and
    :class:`FdrAssessment` (L2): it carries **no** q-value/p-value/score, and it
    must never be placed in the target-decoy pool (its null is undefined).

    Inference is bounded: it is derived only from direct seed annotations (never
    from other inferences), and every inference records the exact seeds and
    incident edges that justify it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: NodeIdField = Field(..., description="The poorly annotated node.")
    label: str = Field(..., description="Inherited family label.")
    status: Literal["network_inferred"] = Field(
        "network_inferred", description="Explicit marker: this is not a library hit."
    )
    component_id: NodeIdField = Field(..., description="Context component identifier.")
    supporting_annotation_ids: tuple[str, ...] = Field(
        ..., min_length=1, description="Seed annotation ids that justify the inference."
    )
    inferred_from_edge_ids: tuple[RelationshipIdField, ...] = Field(
        default_factory=tuple,
        description="Relationship ids incident to the inferred node.",
    )
    provenance: RelationshipProvenance = Field(
        ..., description="How this inference was produced."
    )


class MolecularFamily(NetworkModel):
    """
    **L5 — a molecular family.** A deterministic connected component with its
    member nodes, internal edges, and (when available) an inherited label.

    A family is descriptive: it labels and groups ions using the typed edges, but
    it carries no similarity score and no statistical confidence, and it never
    alters L1/L2. ``community`` refinement is deferred (ML-optional).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    family_id: NodeIdField = Field(
        ..., description="Deterministic family identifier (the component id)."
    )
    member_node_ids: tuple[NodeIdField, ...] = Field(
        ..., min_length=1, description="Member node/feature identifiers."
    )
    member_count: int = Field(..., ge=1, description="Number of members.")
    edge_ids: tuple[RelationshipIdField, ...] = Field(
        default_factory=tuple, description="Relationships internal to the family."
    )
    seed_annotation_ids: tuple[str, ...] = Field(
        default_factory=tuple, description="Seed annotations backing the label."
    )
    label: Optional[str] = Field(None, description="Inherited family label, if any.")
    label_source: Optional[Literal["seed_annotation"]] = Field(
        None, description="Source of the label, when present."
    )
    provenance: RelationshipProvenance = Field(
        ..., description="How this family was produced."
    )

    @model_validator(mode="after")
    def _validate_family(self) -> "MolecularFamily":
        if self.member_count != len(self.member_node_ids):
            raise ValueError("member_count must equal the number of member_node_ids.")
        if len(set(self.member_node_ids)) != len(self.member_node_ids):
            raise ValueError("member_node_ids must be unique.")
        if len(set(self.edge_ids)) != len(self.edge_ids):
            raise ValueError("edge_ids must be unique.")
        return self


def _check_schema_version(version: str) -> None:
    """Reject documents written by an incompatible major schema version."""
    major = str(version).split(".", 1)[0]
    if major != SCHEMA_VERSION.split(".", 1)[0]:
        raise ValueError(
            f"Unsupported graph schema_version {version!r}; this build reads and "
            f"writes '{SCHEMA_VERSION}'."
        )


class MolecularGraph(NetworkModel):
    """
    A serializable, data-only container of nodes, features and relationships.

    This type performs **no** analysis: no search, no clustering, no family
    detection. It exists so future networking code has a stable, typed,
    round-trippable document to read and write. :class:`FdrAssessment` values
    are keyed by query node in :attr:`fdr_assessments` and are the only FDR
    carrier; edges never carry confidence.
    """

    schema_version: str = Field(
        SCHEMA_VERSION, description="Molecular-graph document schema version."
    )
    nodes: List[GraphNode] = Field(default_factory=list, description="Graph nodes.")
    features: List[Feature] = Field(default_factory=list, description="LC-MS features.")
    relationships: List[Relationship] = Field(
        default_factory=list, description="Typed relationships (edges)."
    )
    fdr_assessments: Dict[str, FdrAssessment] = Field(
        default_factory=dict,
        description="Query-scoped FDR assessments keyed by query node id.",
    )
    network_contexts: List[NetworkContext] = Field(
        default_factory=list,
        description="L5 descriptive family contexts (non-statistical).",
    )
    inferences: List[AnnotationInference] = Field(
        default_factory=list,
        description="L5 network-inferred annotations (never in the FDR pool).",
    )
    families: List[MolecularFamily] = Field(
        default_factory=list,
        description="L5 deterministic molecular families.",
    )
    provenance: GraphProvenance = Field(..., description="Graph-level provenance.")

    @model_validator(mode="after")
    def _validate_graph(self) -> "MolecularGraph":
        node_ids = [node.node_id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Graph node identifiers must be unique.")
        feature_ids = [feature.feature_id for feature in self.features]
        if len(set(feature_ids)) != len(feature_ids):
            raise ValueError("Feature identifiers must be unique.")
        relationship_ids = [rel.relationship_id for rel in self.relationships]
        if len(set(relationship_ids)) != len(relationship_ids):
            raise ValueError("Relationship identifiers must be unique.")

        known_endpoints = set(node_ids) | set(feature_ids)
        for rel in self.relationships:
            if rel.source_node_id not in known_endpoints:
                raise ValueError(
                    f"Relationship {rel.relationship_id} references unknown source "
                    f"node {rel.source_node_id!r}."
                )
            if rel.target_node_id not in known_endpoints:
                raise ValueError(
                    f"Relationship {rel.relationship_id} references unknown target "
                    f"node {rel.target_node_id!r}."
                )

        query_node_ids = {
            node.node_id for node in self.nodes if node.node_kind == "query_spectrum"
        }
        for key in self.fdr_assessments:
            if key not in query_node_ids:
                raise ValueError(
                    f"fdr_assessments key {key!r} is not a query-spectrum node."
                )

        context_ids = [context.component_id for context in self.network_contexts]
        if len(set(context_ids)) != len(context_ids):
            raise ValueError("Network-context identifiers must be unique.")
        for context in self.network_contexts:
            for member in context.member_node_ids:
                if member not in known_endpoints:
                    raise ValueError(
                        f"Network context {context.component_id} references "
                        f"unknown member {member!r}."
                    )
        context_id_set = set(context_ids)
        inference_node_ids = [inference.node_id for inference in self.inferences]
        if len(set(inference_node_ids)) != len(inference_node_ids):
            raise ValueError("Inferences must target distinct nodes.")
        for inference in self.inferences:
            if inference.node_id not in query_node_ids:
                raise ValueError(
                    f"Inference targets {inference.node_id!r}, which is not a "
                    f"query-spectrum node."
                )
            if inference.component_id not in context_id_set:
                raise ValueError(
                    f"Inference for {inference.node_id!r} references unknown "
                    f"context {inference.component_id!r}."
                )

        family_ids = [family.family_id for family in self.families]
        if len(set(family_ids)) != len(family_ids):
            raise ValueError("Family identifiers must be unique.")
        relationship_id_set = {rel.relationship_id for rel in self.relationships}
        for family in self.families:
            for member in family.member_node_ids:
                if member not in known_endpoints:
                    raise ValueError(
                        f"Family {family.family_id} references unknown member "
                        f"{member!r}."
                    )
            for edge_id in family.edge_ids:
                if edge_id not in relationship_id_set:
                    raise ValueError(
                        f"Family {family.family_id} references unknown edge {edge_id!r}."
                    )
        return self

    # -- Serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-compatible document (missing values are ``null``)."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MolecularGraph":
        """Reconstruct a graph from :meth:`to_dict` output."""
        _check_schema_version(str(data.get("schema_version", "")))
        return cls.model_validate(dict(data))

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Serialize the graph to a deterministic JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "MolecularGraph":
        """Deserialize a graph from a JSON string produced by :meth:`to_json`."""
        return cls.from_dict(json.loads(text))

    def iter_relationship_jsonl(self) -> Iterator[str]:
        """Yield one JSON object per relationship (for streaming large graphs)."""
        for relationship in self.relationships:
            yield relationship.model_dump_json()

    def to_relationship_jsonl(self) -> str:
        """Return all relationships as newline-delimited JSON."""
        return "\n".join(self.iter_relationship_jsonl())

    def iter_family_jsonl(self) -> Iterator[str]:
        """Yield one JSON object per family (for streaming exports)."""
        for family in self.families:
            yield family.model_dump_json()

    def to_family_jsonl(self) -> str:
        """Return all families as newline-delimited JSON (with a trailing newline)."""
        lines = list(self.iter_family_jsonl())
        return "\n".join(lines) + ("\n" if lines else "")

    def query_fdr(self, node_id: str) -> Optional[FdrAssessment]:
        """Return the query-scoped FDR assessment for *node_id*, if present."""
        return self.fdr_assessments.get(str(node_id))


__all__ = [
    "SCHEMA_VERSION",
    "IonMode",
    "MoleculeRelationship",
    "NodeId",
    "RelationshipId",
    "NodeIdField",
    "RelationshipIdField",
    "spectrum_node_id",
    "library_node_id",
    "feature_node_id",
    "relationship_id",
    "NetworkModel",
    "GraphNode",
    "Feature",
    "FdrAssessment",
    "AnnotationEvidence",
    "NetworkContext",
    "AnnotationInference",
    "MolecularFamily",
    "NeutralLoss",
    "RelationshipProvenance",
    "RelationshipBase",
    "SpectralRelationship",
    "ChemicalRelationship",
    "IonIdentityRelationship",
    "Relationship",
    "GraphProvenance",
    "MolecularGraph",
]
