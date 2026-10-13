"""
Experimental network-aware annotation **propagation** (Stage 5-lite).

This module is a deliberately small, DataFrame-oriented sibling of the
object-graph Stage 5 contextualizer (:mod:`MassFlow.network.context`). It consumes
two Polars tables — an explicit **molecular-network edge list** and the existing
**library search results** — and returns a new results table enriched with
*network-derived* evidence. It never owns spectrum ingestion, library storage,
similarity search, or workflow orchestration.

Scientifically conservative by construction
-------------------------------------------
* **One-hop, non-recursive.** Only *direct* library hits (from the ``matches``
  table) act as seeds. A candidate that this function derives is never fed back
  as an independent seed, so propagated annotations cannot amplify themselves.
* **Direct evidence is immutable.** The original ``score``, ``q_value``,
  ``p_value``, ``is_decoy`` and all other direct-match fields are copied through
  verbatim. Network evidence is written only to *new* columns.
* **Decoys never propagate.** A decoy row is a target-decoy-competition device,
  not a structure claim, so it contributes no seed weight.
* **No score-scale mixing.** Edge weights must be similarity-scale values in
  ``[0, 1]`` (the ``SpectralRelationship`` scale). Neutral-loss or other
  incompatible scales are rejected rather than silently combined.
* **No probability reinterpretation.** Match scores are used as similarity
  *magnitudes* (documented), never read as probabilities; existing ``q_value`` /
  ``p_value`` semantics are preserved untouched.
* **Candidate identity.** The compound identity is the full (27-character)
  InChIKey when present, otherwise the library ``reference_id`` (a unique
  per-entry proxy). A truncated 14-character "skeleton" key is *not* used as
  identity — it does not distinguish stereoisomers — so such rows fall back to
  ``reference_id`` (with a one-time warning). Names are never used as identity,
  so structures with similar names are not silently merged.
* **Network support is a hypothesis, not an identification.** A network-derived
  candidate carries ``evidence_source == "network"``, null direct-match fields,
  and a ``network_status`` that distinguishes it from a direct library hit.

Canonical edge schema
---------------------
A *network edge* is an undirected, weighted pair of node identifiers:
``source_id``, ``target_id`` (``Utf8``) and ``weight`` (``Float64`` in
``[0, 1]``). An optional ``edge_id`` (``Utf8``) may be carried for provenance.
:func:`validate_network_edges` normalises this schema (see its docstring for the
exact rules). Node identifiers in the edge table share one namespace with the
``query_id`` column of the ``matches`` table.

See ``docs/api/network.md`` for the user-facing contract.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import polars as pl
import scipy.sparse as sp
from pydantic import ConfigDict, Field

from MassFlow.network.models import NetworkModel

logger = logging.getLogger(__name__)

__all__ = [
    "NetworkPropagationConfig",
    "validate_network_edges",
    "propagate_annotations",
    "network_seed_id",
    "EDGE_SCHEMA",
    "PROPAGATION_OUTPUT_SCHEMA",
]

#: Canonical edge column names (overridable per call).
DEFAULT_SOURCE_COL = "source_id"
DEFAULT_TARGET_COL = "target_id"
DEFAULT_WEIGHT_COL = "weight"
#: ``matches`` column naming the query node identifier.
DEFAULT_NODE_COL = "query_id"

#: Distinct ``network_status`` values (documented contract).
STATUS_DIRECT_MATCH = "direct_match"
STATUS_DIRECT_CONFLICT = "direct_match_conflict"
STATUS_DECOY = "decoy_match"
STATUS_NOT_IN_NETWORK = "not_in_network"
STATUS_NETWORK_SUPPORTED = "network_supported"
STATUS_NETWORK_CONFLICTED = "network_conflicted"

#: Column names carried through from a ``SearchResult`` row unchanged.
_DIRECT_PASSTHROUGH: tuple[str, ...] = (
    "reference_id",
    "reference_name",
    "reference_precursor_mz",
    "query_precursor_mz",
    "score",
    "matched_peaks",
    "smiles",
    "inchikey",
    "is_decoy",
    "q_value",
    "p_value",
    "annotation_tier",
    "structural_similarity",
    "mass_error_ppm",
    "score_breakdown",
)

#: Output dtypes (a stable, documented schema).
PROPAGATION_OUTPUT_SCHEMA: dict[str, Any] = {
    "query_id": pl.Utf8,
    "evidence_source": pl.Utf8,
    "is_direct_match": pl.Boolean,
    "candidate_id": pl.Utf8,
    "candidate_inchikey": pl.Utf8,
    "reference_id": pl.Utf8,
    "reference_name": pl.Utf8,
    "reference_precursor_mz": pl.Float64,
    "query_precursor_mz": pl.Float64,
    "score": pl.Float64,
    "matched_peaks": pl.Int64,
    "smiles": pl.Utf8,
    "inchikey": pl.Utf8,
    "is_decoy": pl.Boolean,
    "q_value": pl.Float64,
    "p_value": pl.Float64,
    "annotation_tier": pl.Utf8,
    "structural_similarity": pl.Float64,
    "mass_error_ppm": pl.Float64,
    "score_breakdown": pl.Utf8,
    "network_support_score": pl.Float64,
    "supporting_neighbor_count": pl.UInt32,
    "supporting_seed_ids": pl.List(pl.Utf8),
    "network_status": pl.Utf8,
    "network_conflict": pl.Boolean,
    "network_conflicting_candidates": pl.List(pl.Utf8),
    "network_rank": pl.UInt32,
}

#: Alias kept for documentation/back-compat in tests and docs.
EDGE_SCHEMA: tuple[str, str, str] = (
    DEFAULT_SOURCE_COL,
    DEFAULT_TARGET_COL,
    DEFAULT_WEIGHT_COL,
)


class NetworkPropagationConfig(NetworkModel):
    """
    Explicit, validated configuration for one-hop annotation propagation.

    Attributes
    ----------
    seed_max_q_value : float or None
        Maximum query-scoped ``q_value`` for a direct hit to act as a seed. When
        ``None`` the q-value gate is disabled and eligibility is decided by
        ``seed_min_score`` alone (documented as less conservative). Rows whose
        ``q_value`` is missing are **not** seeds while this gate is active.
    seed_min_score : float
        Minimum direct match ``score`` for a seed (similarity scale, ``[0, 1]``).
    min_support_score : float
        Minimum weighted neighbour support for a candidate to be reported as a
        network hypothesis.
    conflict_min_support_share : float
        A node whose leading candidate holds less than this share of the total
        qualifying support is treated as *conflicted* rather than resolved.
    conflict_tie_tolerance : float
        Two leading candidates within this absolute support distance are treated
        as a tie (and therefore a conflict).
    weight_seed_by_score : bool
        When ``True`` (default) a neighbour's seed contributes
        ``edge_weight * seed_score``; when ``False`` it contributes only
        ``edge_weight``. The score is used as a similarity magnitude, never as a
        probability.
    """

    seed_max_q_value: Optional[float] = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Max q-value for a direct hit to seed propagation (None disables).",
    )
    seed_min_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum seed similarity score."
    )
    min_support_score: float = Field(
        default=0.05,
        ge=0.0,
        description="Minimum weighted neighbour support to report a candidate.",
    )
    conflict_min_support_share: float = Field(
        default=0.6,
        gt=0.0,
        le=1.0,
        description="Below this share of qualifying support the node is conflicted.",
    )
    conflict_tie_tolerance: float = Field(
        default=1e-9,
        ge=0.0,
        description="Absolute support distance under which two candidates tie.",
    )
    weight_seed_by_score: bool = Field(
        default=True,
        description="Weight seed evidence by its similarity score (not a probability).",
    )

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


# ---------------------------------------------------------------------------
# Small, dependency-free helpers
# ---------------------------------------------------------------------------


def network_seed_id(node_id: str, candidate_id: str) -> str:
    """Return a deterministic identifier for a (node, candidate) seed.

    Uses ``hashlib`` (never the built-in ``hash``) so the identifier is stable
    across processes and independent of ``PYTHONHASHSEED``.

    Parameters
    ----------
    node_id : str
        Query node identifier.
    candidate_id : str
        Compound identity used by propagation (InChIKey or reference identifier).

    Returns
    -------
    str
        Content-derived seed identifier prefixed with ``"nseed1:"``.

    Examples
    --------
    >>> seed_id = network_seed_id("query-1", "caffeine-reference")
    >>> seed_id == network_seed_id("query-1", "caffeine-reference")
    True
    """
    digest = hashlib.sha256(f"{node_id}\n{candidate_id}".encode("utf-8")).hexdigest()[
        :32
    ]
    return f"nseed1:{digest}"


def _clean_text(value: Any) -> Optional[str]:
    """Return stripped text or ``None`` when the value is empty/absent/NaN."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _finite_float(value: Any) -> Optional[float]:
    """Return a finite float or ``None`` for missing/NaN/non-numeric input."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


#: Canonical full InChIKey: 14-10-1 uppercase blocks (27 chars). The 14-char
#: first block ("skeleton") is deliberately *not* accepted as identity: it
#: collapses stereoisomers and isotopic variants into one candidate.
_FULL_INCHIKEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


@lru_cache(maxsize=4096)
def _warn_non_full_inchikey(value: str) -> None:
    """Log once per distinct value that a non-full InChIKey was ignored."""
    logger.warning(
        "Ignoring non-full InChIKey %r as candidate identity; only the 27-char "
        "full key is stereochemically complete. Falling back to reference_id.",
        value,
    )


def _is_full_inchikey(value: str) -> bool:
    """Return whether *value* is a canonical full (27-character) InChIKey."""
    return _FULL_INCHIKEY_PATTERN.match(value) is not None


def _record_min(mapping: dict[str, Any], key: str, value: Any) -> None:
    """Keep the deterministic smallest non-``None`` *value* seen for *key*."""
    if value is None:
        return
    current = mapping.get(key)
    if current is None or value < current:
        mapping[key] = value


def _resolve_candidate_id(row: dict[str, Any]) -> Optional[str]:
    """
    Resolve the compound identity of a search-result row.

    The *full* (27-character) InChIKey is preferred; otherwise the library
    ``reference_id`` — a unique per-entry proxy — is used. A non-full InChIKey
    (e.g. the 14-character skeleton block) is rejected as identity because it
    merges stereoisomers, and the row falls back to ``reference_id``. Names are
    deliberately never used, so two different structures with similar names are
    not merged.
    """
    inchikey = _clean_text(row.get("inchikey"))
    if inchikey is not None:
        if _is_full_inchikey(inchikey):
            return inchikey
        _warn_non_full_inchikey(inchikey)
    return _clean_text(row.get("reference_id"))


def _row_is_decoy(row: dict[str, Any]) -> bool:
    value = row.get("is_decoy")
    return bool(value) if value is not None else False


def _row_is_seed(row: dict[str, Any], cfg: NetworkPropagationConfig) -> bool:
    """Return whether a direct-match row may act as a propagation seed."""
    if _row_is_decoy(row):
        return False
    score = _finite_float(row.get("score"))
    if score is None or score < cfg.seed_min_score:
        return False
    if cfg.seed_max_q_value is not None:
        q_value = _finite_float(row.get("q_value"))
        if q_value is None or q_value > cfg.seed_max_q_value:
            return False
    return True


def _empty_output() -> pl.DataFrame:
    """Return an empty DataFrame carrying the canonical output schema."""
    return pl.DataFrame(schema=PROPAGATION_OUTPUT_SCHEMA)


def _empty_edges(
    source_col: str, target_col: str, weight_col: str, edge_id_col: Optional[str]
) -> pl.DataFrame:
    cols = [source_col, target_col, weight_col]
    schema: dict[str, Any] = {
        source_col: pl.Utf8,
        target_col: pl.Utf8,
        weight_col: pl.Float64,
    }
    if edge_id_col is not None:
        cols.append(edge_id_col)
        schema[edge_id_col] = pl.Utf8
    return pl.DataFrame(schema={c: schema[c] for c in cols})


# ---------------------------------------------------------------------------
# Edge validation
# ---------------------------------------------------------------------------


def validate_network_edges(
    edges: pl.DataFrame,
    *,
    source_col: str = DEFAULT_SOURCE_COL,
    target_col: str = DEFAULT_TARGET_COL,
    weight_col: str = DEFAULT_WEIGHT_COL,
    edge_id_col: Optional[str] = None,
) -> pl.DataFrame:
    """
    Validate and normalise a molecular-network edge table.

    Rules (all documented so behaviour is never implicit):

    * ``source_col``, ``target_col`` and ``weight_col`` must exist.
    * Missing (``null``) or empty-string endpoints raise :class:`ValueError` —
      a dangling endpoint is malformed input, not something to drop silently.
    * ``weight`` is cast to ``float64``; rows whose weight is missing/``NaN``/
      ``±inf``, negative, or greater than ``1.0`` are **dropped** (their count is
      logged). Only similarity-scale (``[0, 1]``) weights are comparable, so
      cosine and neutral-loss scales can never be combined by accident.
    * **Self-loops** (``source == target``) are removed (they contribute no
      cross-node evidence) and the count is logged.
    * **Directionality**: edges are treated as **undirected**. Endpoints are
      canonicalised so that ``source_col <= target_col`` lexicographically.
    * **Duplicate edges** collapsing to the same undirected pair are
      de-duplicated deterministically, keeping the **maximum** weight; ties are
      broken by the lexicographically smallest ``edge_id`` (when supplied).
    * Output is sorted by ``(source_col, target_col)`` for reproducibility.

    Parameters
    ----------
    edges : polars.DataFrame
        Raw edge table.
    source_col, target_col, weight_col : str
        Column names for the endpoints and similarity weight.
    edge_id_col : str or None, optional
        Optional column carried through for provenance and dedup tie-breaking.

    Returns
    -------
    polars.DataFrame
        Canonical, validated edges (empty input yields an empty canonical table).

    Raises
    ------
    TypeError
        If ``edges`` is not a :class:`polars.DataFrame`.
    ValueError
        If required columns are missing, or endpoints are null/empty.

    Examples
    --------
    >>> import polars as pl
    >>> edges = pl.DataFrame({
    ...     "source_id": ["b", "a"], "target_id": ["a", "b"],
    ...     "weight": [0.75, 0.5],
    ... })
    >>> validate_network_edges(edges).rows()
    [('a', 'b', 0.75)]
    """
    if not isinstance(edges, pl.DataFrame):
        raise TypeError(
            f"edges must be a polars.DataFrame, got {type(edges).__name__}."
        )

    required = [source_col, target_col, weight_col]
    missing = [name for name in required if name not in edges.columns]
    if missing:
        raise ValueError(
            f"edges is missing required column(s): {missing}. "
            f"Expected at least {required}."
        )
    if edge_id_col is not None and edge_id_col not in edges.columns:
        raise ValueError(f"edges is missing edge_id column {edge_id_col!r}.")

    selected = list(required) + ([edge_id_col] if edge_id_col else [])
    df = edges.select(selected)
    if df.height == 0:
        return _empty_edges(source_col, target_col, weight_col, edge_id_col)

    # Endpoints: enforce Utf8 (string identifiers) and reject null/empty.
    df = df.with_columns(
        pl.col(source_col).cast(pl.Utf8, strict=False),
        pl.col(target_col).cast(pl.Utf8, strict=False),
    )
    has_null_endpoint = df.select(
        (pl.col(source_col).is_null() | pl.col(target_col).is_null()).any()
    ).item()
    if has_null_endpoint:
        raise ValueError(
            "edges contains null endpoints; every edge needs a source and target id."
        )
    has_blank_endpoint = df.select(
        (
            (pl.col(source_col).str.strip_chars() == "")
            | (pl.col(target_col).str.strip_chars() == "")
        ).any()
    ).item()
    if has_blank_endpoint:
        raise ValueError(
            "edges contains empty-string endpoints; every edge needs a source "
            "and target id."
        )

    # Weights: cast to float64 and drop rows with invalid (missing/non-finite/
    # out-of-range) weights. Only similarity-scale values in [0, 1] are
    # comparable, so cosine and neutral-loss scales are never mixed.
    df = df.with_columns(pl.col(weight_col).cast(pl.Float64, strict=False))
    invalid_weight = (
        pl.col(weight_col).is_null()
        | ~pl.col(weight_col).is_finite()
        | (pl.col(weight_col) < 0.0)
        | (pl.col(weight_col) > 1.0)
    )
    invalid_weight_count = df.filter(invalid_weight).height
    if invalid_weight_count:
        logger.warning(
            "Dropping %d edge(s) with invalid weight (missing/non-finite/outside "
            "[0, 1]); only similarity-scale weights are comparable.",
            invalid_weight_count,
        )
        df = df.filter(~invalid_weight)
    if df.height == 0:
        return _empty_edges(source_col, target_col, weight_col, edge_id_col)

    # Drop self-loops (no cross-node evidence).
    self_loop_count = df.filter(pl.col(source_col) == pl.col(target_col)).height
    if self_loop_count:
        logger.debug("Dropping %d self-loop edge(s) from the network.", self_loop_count)
        df = df.filter(pl.col(source_col) != pl.col(target_col))
    if df.height == 0:
        return _empty_edges(source_col, target_col, weight_col, edge_id_col)

    # Undirected canonicalisation (lexicographic on the string identifiers).
    lower = (
        pl.when(pl.col(source_col) <= pl.col(target_col))
        .then(pl.col(source_col))
        .otherwise(pl.col(target_col))
    )
    upper = (
        pl.when(pl.col(source_col) <= pl.col(target_col))
        .then(pl.col(target_col))
        .otherwise(pl.col(source_col))
    )
    df = df.with_columns(
        lower.alias("_canonical_source"), upper.alias("_canonical_target")
    )

    # Dedup: keep max weight per undirected pair; tie-break on edge_id ascending.
    if edge_id_col is not None:
        df = df.with_columns(pl.col(edge_id_col).cast(pl.Utf8, strict=False))
        order_cols = ["_canonical_source", "_canonical_target", weight_col, edge_id_col]
        descending = [False, False, True, False]
    else:
        order_cols = ["_canonical_source", "_canonical_target", weight_col]
        descending = [False, False, True]
    df = (
        df.sort(by=order_cols, descending=descending)
        .unique(
            subset=["_canonical_source", "_canonical_target"],
            keep="first",
            maintain_order=True,
        )
        .with_columns(
            pl.col("_canonical_source").alias(source_col),
            pl.col("_canonical_target").alias(target_col),
        )
        .select(selected)
        .sort(by=[source_col, target_col])
    )
    return df


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


def _build_adjacency(
    node_index: dict[str, int],
    source: list[str],
    target: list[str],
    weights: np.ndarray,
) -> sp.csr_matrix:
    """Build a symmetric CSR adjacency matrix from (already canonical) edges."""
    node_count = len(node_index)
    left = np.fromiter(
        (node_index[s] for s in source), dtype=np.int64, count=len(source)
    )
    right = np.fromiter(
        (node_index[t] for t in target), dtype=np.int64, count=len(target)
    )
    rows = np.concatenate((left, right))
    cols = np.concatenate((right, left))
    data = np.concatenate((weights, weights))
    matrix = sp.coo_matrix(
        (data, (rows, cols)), shape=(node_count, node_count), dtype=np.float64
    ).tocsr()
    matrix.sort_indices()
    return matrix


def _contributing_seed_ids(
    row_start: int,
    row_end: int,
    indices: np.ndarray,
    candidate_column: np.ndarray,
    node_ids: list[str],
    candidate_id: str,
) -> list[str]:
    """Return sorted seed ids among a node's neighbours supporting *candidate_id*."""
    neighbours = indices[row_start:row_end]
    supporting = [
        network_seed_id(node_ids[int(neighbour)], candidate_id)
        for neighbour in neighbours
        if candidate_column[int(neighbour)]
    ]
    return sorted(supporting)


def propagate_annotations(
    edges: pl.DataFrame,
    matches: pl.DataFrame,
    *,
    config: Optional[NetworkPropagationConfig] = None,
    source_col: str = DEFAULT_SOURCE_COL,
    target_col: str = DEFAULT_TARGET_COL,
    weight_col: str = DEFAULT_WEIGHT_COL,
    edge_id_col: Optional[str] = None,
    node_col: str = DEFAULT_NODE_COL,
) -> pl.DataFrame:
    """
    Propagate direct library annotations one hop across a molecular network.

    The result is a superset of ``matches``: every input row is preserved with
    its direct-match fields **unchanged**, and network-derived columns are added.
    Nodes with no direct (non-decoy) match may additionally receive
    ``evidence_source == "network"`` hypothesis rows. Newly derived annotations
    are never reused as seeds, so the operation is strictly one hop.

    Parameters
    ----------
    edges : polars.DataFrame
        Raw molecular-network edges (see :func:`validate_network_edges`).
    matches : polars.DataFrame
        Library search results. Requires at least ``query_id`` (or ``node_col``),
        ``reference_id`` and ``score``; ``is_decoy``, ``q_value``, ``inchikey``
        and the other :class:`~MassFlow.similarity.SearchResult` fields are used
        when present and treated as missing otherwise.
    config : NetworkPropagationConfig or None, optional
        Thresholds and scoring behaviour. Defaults to a conservative
        configuration.
    source_col, target_col, weight_col, edge_id_col : str, optional
        Edge column overrides forwarded to :func:`validate_network_edges`.
    node_col : str, optional
        ``matches`` column holding the query node identifier (default
        ``"query_id"``). It shares one namespace with the edge endpoints.

    Returns
    -------
    polars.DataFrame
        Direct rows with added network columns plus any network-hypothesis rows,
        deterministically ordered by ``(query_id, evidence_source, network_rank,
        candidate_id)`` and conforming to :data:`PROPAGATION_OUTPUT_SCHEMA`.

    Raises
    ------
    TypeError
        If ``edges`` or ``matches`` is not a :class:`polars.DataFrame`.
    ValueError
        If ``matches`` is missing a required column, or edge validation fails.

    Examples
    --------
    >>> import polars as pl
    >>> edges = pl.DataFrame({
    ...     "source_id": ["seed"], "target_id": ["target"], "weight": [0.8],
    ... })
    >>> matches = pl.DataFrame({
    ...     "query_id": ["seed"], "reference_id": ["caffeine-reference"],
    ...     "score": [0.9], "q_value": [0.001], "is_decoy": [False],
    ... })
    >>> result = propagate_annotations(edges, matches)
    >>> result.filter(pl.col("query_id") == "target")["evidence_source"].to_list()
    ['network']
    >>> result.filter(pl.col("query_id") == "target")["q_value"].to_list()
    [None]
    """
    cfg = config or NetworkPropagationConfig()

    if not isinstance(matches, pl.DataFrame):
        raise TypeError(
            f"matches must be a polars.DataFrame, got {type(matches).__name__}."
        )
    required_match_cols = [node_col, "reference_id", "score"]
    missing = [name for name in required_match_cols if name not in matches.columns]
    if missing:
        raise ValueError(f"matches is missing required column(s): {missing}.")

    edges_df = validate_network_edges(
        edges,
        source_col=source_col,
        target_col=target_col,
        weight_col=weight_col,
        edge_id_col=edge_id_col,
    )

    match_rows: list[dict[str, Any]] = matches.to_dicts()
    if not match_rows:
        return _empty_output()

    if edges_df.height == 0:
        # No network: direct rows pass through, annotated as not-in-network.
        return _assemble_without_network(match_rows, node_col)

    node_ids = sorted(
        set(edges_df[source_col].to_list()) | set(edges_df[target_col].to_list())
    )
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    node_count = len(node_ids)

    weights = np.asarray(edges_df[weight_col].to_list(), dtype=np.float64)
    adjacency = _build_adjacency(
        node_index,
        edges_df[source_col].to_list(),
        edges_df[target_col].to_list(),
        weights,
    )
    boolean_adjacency = adjacency.copy()
    boolean_adjacency.data = np.ones_like(boolean_adjacency.data)

    # ---- Seeds: (node, candidate) -> maximum direct score, from target hits ---
    seed_scores: dict[tuple[str, str], float] = {}
    candidate_names: dict[str, Optional[str]] = {}
    candidate_inchikeys: dict[str, Optional[str]] = {}
    candidate_reference_ids: dict[str, str] = {}
    candidate_precursor_mzs: dict[str, float] = {}
    candidate_smiles: dict[str, str] = {}
    for row in match_rows:
        node_id = _clean_text(row.get(node_col))
        if node_id is None or node_id not in node_index:
            continue
        if not _row_is_seed(row, cfg):
            continue
        candidate_id = _resolve_candidate_id(row)
        if candidate_id is None:
            continue
        score = _finite_float(row.get("score"))
        if score is None:
            continue
        _record_min(
            candidate_names, candidate_id, _clean_text(row.get("reference_name"))
        )
        _record_min(candidate_inchikeys, candidate_id, _clean_text(row.get("inchikey")))
        _record_min(
            candidate_reference_ids, candidate_id, _clean_text(row.get("reference_id"))
        )
        _record_min(
            candidate_precursor_mzs,
            candidate_id,
            _finite_float(row.get("reference_precursor_mz")),
        )
        _record_min(candidate_smiles, candidate_id, _clean_text(row.get("smiles")))
        key = (node_id, candidate_id)
        seed_scores[key] = max(seed_scores.get(key, 0.0), score)

    candidates = sorted({candidate for _, candidate in seed_scores})
    candidate_index = {candidate: index for index, candidate in enumerate(candidates)}
    candidate_count = len(candidates)

    support = np.zeros((node_count, candidate_count), dtype=np.float64)
    neighbour_count = np.zeros((node_count, candidate_count), dtype=np.int64)
    seed_masks: list[np.ndarray] = [
        np.zeros(node_count, dtype=bool) for _ in range(candidate_count)
    ]
    for candidate in candidates:
        column = candidate_index[candidate]
        seed_vector = np.zeros(node_count, dtype=np.float64)
        for (node_id, candidate_id), score in seed_scores.items():
            if candidate_id != candidate:
                continue
            seed_vector[node_index[node_id]] = score
        seed_masks[column] = seed_vector > 0.0
        contribution = (
            seed_vector if cfg.weight_seed_by_score else (seed_vector > 0.0) * 1.0
        )
        support[:, column] = adjacency @ contribution
        neighbour_count[:, column] = np.rint(
            boolean_adjacency @ seed_masks[column].astype(np.float64)
        ).astype(np.int64)

    # ---- Classify nodes -------------------------------------------------------
    direct_target_nodes: set[str] = set()
    for row in match_rows:
        node_id = _clean_text(row.get(node_col))
        if node_id is None or _row_is_decoy(row):
            continue
        if _resolve_candidate_id(row) is not None:
            direct_target_nodes.add(node_id)

    def _node_support(node_id: str) -> np.ndarray:
        return support[node_index[node_id]]

    def _qualifying(node_id: str) -> list[tuple[str, float]]:
        if node_id not in node_index:
            return []
        row = _node_support(node_id)
        result: list[tuple[str, float]] = []
        for candidate in candidates:
            value = float(row[candidate_index[candidate]])
            if value > 0.0 and value >= cfg.min_support_score:
                result.append((candidate, value))
        result.sort(key=lambda item: (-item[1], item[0]))
        return result

    def _is_conflicted(qualifying: list[tuple[str, float]]) -> bool:
        if len(qualifying) < 2:
            return False
        total = sum(value for _, value in qualifying)
        share = qualifying[0][1] / total if total > 0 else 0.0
        tie = (qualifying[0][1] - qualifying[1][1]) <= cfg.conflict_tie_tolerance
        return tie or share < cfg.conflict_min_support_share

    def _candidate_metadata(candidate_id: str) -> dict[str, Any]:
        """Deterministic structure/provenance metadata for a candidate identity."""
        return {
            "candidate_name": candidate_names.get(candidate_id),
            "candidate_inchikey": candidate_inchikeys.get(candidate_id),
            "reference_id": candidate_reference_ids.get(candidate_id) or candidate_id,
            "reference_precursor_mz": candidate_precursor_mzs.get(candidate_id),
            "smiles": candidate_smiles.get(candidate_id),
        }

    output_rows: list[dict[str, Any]] = []

    for row in match_rows:
        node_id = _clean_text(row.get(node_col))
        candidate_id = _resolve_candidate_id(row)
        is_decoy = _row_is_decoy(row)
        in_graph = node_id is not None and node_id in node_index

        base = _direct_row(row, node_id, candidate_id)
        support_value = 0.0
        count_value = 0
        seed_ids: list[str] = []
        conflict = False
        conflicting: list[str] = []

        if in_graph and not is_decoy and candidate_id is not None:
            assert node_id is not None  # implied by in_graph; narrows for mypy
            candidate_column = candidate_index.get(candidate_id)
            if candidate_column is not None:
                node_position = node_index[node_id]
                support_value = float(support[node_position, candidate_column])
                count_value = int(neighbour_count[node_position, candidate_column])
                if count_value > 0:
                    seed_ids = _contributing_seed_ids(
                        int(adjacency.indptr[node_position]),
                        int(adjacency.indptr[node_position + 1]),
                        adjacency.indices,
                        seed_masks[candidate_column],
                        node_ids,
                        candidate_id,
                    )
                qualifying = _qualifying(node_id)
                if qualifying:
                    network_top = qualifying[0][0]
                    if network_top != candidate_id:
                        conflict = True
                        conflicting = sorted(
                            candidate
                            for candidate, _ in qualifying
                            if candidate != candidate_id
                        )

        base["network_support_score"] = support_value
        base["supporting_neighbor_count"] = count_value
        base["supporting_seed_ids"] = seed_ids
        base["network_conflict"] = conflict
        base["network_conflicting_candidates"] = conflicting
        base["network_rank"] = None
        if is_decoy:
            base["network_status"] = STATUS_DECOY
        elif not in_graph:
            base["network_status"] = STATUS_NOT_IN_NETWORK
        elif conflict:
            base["network_status"] = STATUS_DIRECT_CONFLICT
        else:
            base["network_status"] = STATUS_DIRECT_MATCH
        output_rows.append(base)

    # ---- Network hypothesis rows for nodes without a direct target match ------
    for node_id in node_ids:
        if node_id in direct_target_nodes:
            continue
        qualifying = _qualifying(node_id)
        if not qualifying:
            continue
        conflicted = _is_conflicted(qualifying)
        if conflicted:
            others = [candidate for candidate, _ in qualifying]
            for rank, (candidate, value) in enumerate(qualifying, start=1):
                output_rows.append(
                    _network_row(
                        node_id=node_id,
                        candidate_id=candidate,
                        support_value=value,
                        neighbour_count=int(
                            neighbour_count[
                                node_index[node_id], candidate_index[candidate]
                            ]
                        ),
                        seed_ids=_contributing_seed_ids(
                            int(adjacency.indptr[node_index[node_id]]),
                            int(adjacency.indptr[node_index[node_id] + 1]),
                            adjacency.indices,
                            seed_masks[candidate_index[candidate]],
                            node_ids,
                            candidate,
                        ),
                        status=STATUS_NETWORK_CONFLICTED,
                        conflict=True,
                        conflicting=[c for c in others if c != candidate],
                        rank=rank,
                        **_candidate_metadata(candidate),
                    )
                )
        else:
            candidate, value = qualifying[0]
            output_rows.append(
                _network_row(
                    node_id=node_id,
                    candidate_id=candidate,
                    support_value=value,
                    neighbour_count=int(
                        neighbour_count[node_index[node_id], candidate_index[candidate]]
                    ),
                    seed_ids=_contributing_seed_ids(
                        int(adjacency.indptr[node_index[node_id]]),
                        int(adjacency.indptr[node_index[node_id] + 1]),
                        adjacency.indices,
                        seed_masks[candidate_index[candidate]],
                        node_ids,
                        candidate,
                    ),
                    status=STATUS_NETWORK_SUPPORTED,
                    conflict=False,
                    conflicting=[],
                    rank=1,
                    **_candidate_metadata(candidate),
                )
            )

    output_rows.sort(key=_output_sort_key)
    return pl.DataFrame(output_rows, schema=PROPAGATION_OUTPUT_SCHEMA)


# ---------------------------------------------------------------------------
# Row assembly helpers
# ---------------------------------------------------------------------------


def _direct_row(
    row: dict[str, Any], node_id: Optional[str], candidate_id: Optional[str]
) -> dict[str, Any]:
    """Copy a direct-match row into the output schema, preserving every value."""
    is_decoy = _row_is_decoy(row)
    output: dict[str, Any] = {name: None for name in PROPAGATION_OUTPUT_SCHEMA}
    output["query_id"] = node_id
    output["evidence_source"] = "direct"
    output["is_direct_match"] = bool(node_id is not None and not is_decoy)
    output["candidate_id"] = candidate_id
    output["candidate_inchikey"] = _clean_text(row.get("inchikey"))
    for name in _DIRECT_PASSTHROUGH:
        output[name] = _coerce_direct_value(name, row.get(name))
    return output


def _coerce_direct_value(name: str, value: Any) -> Any:
    """Coerce a carried-through value to the output schema dtype."""
    if name == "is_decoy":
        return bool(value) if value is not None else None
    if name == "matched_peaks":
        number = _finite_float(value)
        return int(number) if number is not None else None
    if name in (
        "reference_id",
        "reference_name",
        "smiles",
        "inchikey",
        "annotation_tier",
        "score_breakdown",
    ):
        if name == "score_breakdown" and isinstance(value, dict):
            # Mirror io._build_results_dataframe: nested dicts -> compact JSON.
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        return _clean_text(value)
    return _finite_float(value)


def _network_row(
    *,
    node_id: str,
    candidate_id: str,
    candidate_name: Optional[str],
    candidate_inchikey: Optional[str],
    reference_id: Optional[str],
    reference_precursor_mz: Optional[float],
    smiles: Optional[str],
    support_value: float,
    neighbour_count: int,
    seed_ids: list[str],
    status: str,
    conflict: bool,
    conflicting: list[str],
    rank: int,
) -> dict[str, Any]:
    """Build a network-hypothesis row (no direct-match fields set)."""
    output: dict[str, Any] = {name: None for name in PROPAGATION_OUTPUT_SCHEMA}
    output["query_id"] = node_id
    output["evidence_source"] = "network"
    output["is_direct_match"] = False
    output["candidate_id"] = candidate_id
    output["candidate_inchikey"] = candidate_inchikey
    # The candidate's *library* reference id and its structure/state are carried
    # through so a hypothesis keeps the same provenance as a direct hit. The
    # precursor m/z travels with the candidate but is never used as a hard gate:
    # network edges (e.g. neutral-loss / chemical edges) may legitimately link
    # nodes whose precursors differ.
    output["reference_id"] = reference_id if reference_id is not None else candidate_id
    output["reference_name"] = candidate_name
    output["reference_precursor_mz"] = reference_precursor_mz
    output["smiles"] = smiles
    output["network_support_score"] = support_value
    output["supporting_neighbor_count"] = neighbour_count
    output["supporting_seed_ids"] = seed_ids
    output["network_status"] = status
    output["network_conflict"] = conflict
    output["network_conflicting_candidates"] = conflicting
    output["network_rank"] = rank
    return output


def _assemble_without_network(
    match_rows: list[dict[str, Any]], node_col: str
) -> pl.DataFrame:
    """Passthrough used when the edge table is empty (no network to propagate on)."""
    rows: list[dict[str, Any]] = []
    for row in match_rows:
        node_id = _clean_text(row.get(node_col))
        candidate_id = _resolve_candidate_id(row)
        is_decoy = _row_is_decoy(row)
        base = _direct_row(row, node_id, candidate_id)
        base["network_support_score"] = 0.0
        base["supporting_neighbor_count"] = 0
        base["supporting_seed_ids"] = []
        base["network_conflict"] = False
        base["network_conflicting_candidates"] = []
        base["network_rank"] = None
        base["network_status"] = STATUS_DECOY if is_decoy else STATUS_NOT_IN_NETWORK
        rows.append(base)
    rows.sort(key=_output_sort_key)
    return pl.DataFrame(rows, schema=PROPAGATION_OUTPUT_SCHEMA)


def _output_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Deterministic row ordering: direct rows first, then ranked hypotheses."""
    evidence_rank = 0 if row["evidence_source"] == "direct" else 1
    rank = row["network_rank"]
    rank_sort = rank if rank is not None else 0
    return (
        row["query_id"] or "",
        evidence_rank,
        rank_sort,
        row["candidate_id"] or "",
        row["reference_id"] or "",
    )
