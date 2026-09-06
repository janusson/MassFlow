"""
Pydantic configuration models for YAML-driven MassFlow execution.

This module defines the nested schema used by the CLI and workflow layers to
validate pipeline configuration loaded from YAML. The models capture project
paths, input sources, processing toggles, similarity-engine settings, optional
workflow features, and declared export preferences.

Validation in this layer is intentionally focused on local structural
correctness such as field types and simple physical constraints. Broader runtime
assumptions, such as whether a reference library is present for annotation or
whether certain workflow toggles are currently implemented, are enforced in the
orchestrating modules.
"""

import difflib
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, List, Literal, Optional, Union, get_args

import pyteomics.mass as pmass
import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)

# Environment variable that restores the legacy CWD-relative path resolution
# (see :meth:`MassFlowConfig.from_yaml`).
_COMPAT_CWD_PATHS_ENV = "MASSFLOW_COMPAT_CWD_PATHS"


class MassFlowBaseModel(BaseModel):
    """Strict configuration base model.

    * ``extra="forbid"`` — unknown YAML keys (e.g. a misspelled
      ``ms2_tolerence``) are rejected with a human-readable error instead of
      being silently ignored.
    * ``populate_by_name=True`` — fields with validation aliases accept both
      the canonical name and the alias (e.g. ``library_path`` /
      ``reference_library``).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


def register_custom_modifications(modifications: dict) -> None:
    """Register user-defined chemical modifications into pyteomics' internal registries.

    This single-initialization function injects custom residues, adducts, or
    non-standard offsets into ``pyteomics.mass.std_aa_comp`` and/or
    ``pyteomics.mass.std_ion_comp`` so that all downstream mass calculations
    (including ``pmass.calculate_mass``) recognise them natively.

    Each entry in *modifications* must be a dict with at least a ``formula``
    key (chemical formula string, e.g. ``"C2H3O"``). An optional ``type`` key
    controls the target registry: ``"aa"`` (default) for amino-acid residues
    stored in ``std_aa_comp``, or ``"ion"`` for ion-fragment offsets stored
    in ``std_ion_comp``.

    Parameters
    ----------
    modifications : dict
        Mapping of modification names to definition dicts. Each definition
        dict may contain:

        - **formula** (*str*, required): pyteomics-compatible chemical formula.
        - **type** (*str*, optional): ``"aa"`` (default) or ``"ion"``.

    Examples
    --------
    >>> register_custom_modifications({
    ...     "pS": {"formula": "HO3P", "type": "aa"},
    ...     "Ac": {"formula": "C2H2O", "type": "aa"},
    ...     "custom_loss": {"formula": "H-2O-1", "type": "ion"},
    ... })
    """
    if not modifications:
        return

    for name, definition in modifications.items():
        if not isinstance(definition, dict):
            logger.warning(
                "Skipping modification '%s': definition must be a dict, got %s",
                name,
                type(definition).__name__,
            )
            continue

        formula = definition.get("formula")
        if not formula:
            logger.warning(
                "Skipping modification '%s': missing required 'formula' key.", name
            )
            continue

        mod_type = definition.get("type", "aa")

        try:
            comp = pmass.Composition(formula=formula)
        except Exception as exc:
            logger.error(
                "Cannot register modification '%s': invalid formula '%s' — %s",
                name,
                formula,
                exc,
            )
            continue

        if mod_type == "ion":
            pmass.std_ion_comp[name] = comp
            logger.debug("Registered ion modification '%s' → %s", name, dict(comp))
        else:
            pmass.std_aa_comp[name] = comp
            logger.debug("Registered aa modification '%s' → %s", name, dict(comp))


class LineNumberLoader(yaml.SafeLoader):
    """Custom YAML loader that extracts line numbers for configuration keys."""

    def construct_mapping(self, node, deep=False):
        mapping = {}
        lines = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
            lines[key] = key_node.start_mark.line + 1
        mapping["__lines__"] = lines
        return mapping


class ProjectConfig(MassFlowBaseModel):
    """Project metadata and output locations shared across a MassFlow run."""

    name: str = "MassFlow_Project"
    output_directory: Path = Path("results")


class InputConfig(MassFlowBaseModel):
    """
    Input path and format hints for annotation.

    The ``input_path`` can point to either a single spectral file or a
    directory containing multiple files.
    """

    input_path: Path = Field(
        ...,
        description="Path to the input spectral file or directory of files.",
    )
    format: Optional[Literal["mgf", "msp", "mzml", "mzxml", "db", "sqlite"]] = Field(
        default=None,
        description="Optional explicit format hint. If omitted, MassFlow infers the format from the file extension.",
    )
    library_path: Optional[Path] = Field(
        default=None,
        validation_alias=AliasChoices("library_path", "reference_library"),
    )
    streaming_threshold_mb: int = Field(
        default=500,
        description="Threshold (in MB) above which the library is streamed from disk instead of loaded into memory.",
    )
    storage_backend: Literal["sqlite", "zarr", "hybrid"] = Field(
        default="sqlite",
        description=(
            "Storage backend for spectral libraries. "
            "'sqlite' (default) uses SQLite BLOBs for v0.1 stable workflows. "
            "'zarr' uses compressed Zarr arrays for cloud-optimized horizontal "
            "scaling. 'hybrid' stores metadata in SQLite and peak arrays in "
            "a chunked Zarr store referenced by zarr_ref/zarr_index."
        ),
    )

    @property
    def reference_library(self) -> Optional[Path]:
        return self.library_path

    @reference_library.setter
    def reference_library(self, value: Optional[Path]) -> None:
        self.library_path = value


class SolventConfig(MassFlowBaseModel):
    """
    Named solvent/adduct mass used as optional contextual processing metadata.

    The monoisotopic mass is always derived from ``pyteomics``. When a
    ``formula`` is provided it supersedes any raw ``mz`` value; when only
    ``mz`` is given the model stores it as-is (typically for solvents whose
    exact composition is not a simple chemical formula).

    These values are stored in the configuration schema for downstream
    extensions and user metadata, but they are not currently consumed directly
    by the core annotation workflow in this repository snapshot.
    """

    name: str
    formula: Optional[str] = None
    mz: Optional[float] = None

    @model_validator(mode="after")
    def derive_mass_from_formula(self) -> "SolventConfig":
        """Derive ``mz`` from ``formula`` via pyteomics when a formula is present.

        If both ``formula`` and ``mz`` are provided and the computed value
        disagrees with the user-supplied ``mz`` beyond ~10 mDa, we raise an
        error to catch transcription mistakes in configuration files.
        """
        if self.formula:
            computed = pmass.calculate_mass(formula=self.formula)
            if self.mz is not None:
                if abs(self.mz - computed) > 0.01:
                    raise ValueError(
                        f"Solvent '{self.name}': provided mz ({self.mz}) disagrees "
                        f"with pyteomics mass ({computed}) for formula '{self.formula}'."
                    )
            self.mz = computed
        elif self.mz is None:
            raise ValueError(
                f"Solvent '{self.name}': either 'formula' or 'mz' must be provided."
            )
        return self

    @field_validator("mz")
    @classmethod
    def validate_mz(cls, v: float | None) -> float | None:
        """Ensure solvent m/z is not negative."""
        if v is not None and v < 0:
            raise ValueError(f"Solvent m/z cannot be negative. Received: {v}")
        return v


class ProcessingConfig(MassFlowBaseModel):
    """
    Parameters for metadata harmonization and peak-level processing.

    The fields in this model map onto the operations implemented in
    :mod:`MassFlow.processing`, including optional ``matchms`` metadata repairs,
    m/z truncation, intensity filtering, top-N peak reduction, normalization,
    and injection of instrument context into spectra.
    """

    # Standard peak filters
    min_peaks: int = 5
    min_intensity: float = 0.0
    normalize_intensity: bool = True

    # Metadata filtering toggles
    clean_metadata: bool = Field(
        default=True, description="Apply matchms default metadata cleaning."
    )
    add_retention_time: bool = Field(
        default=True, description="Extract and format retention time."
    )
    repair_inchi_inchikey_smiles: bool = Field(
        default=True, description="Repair structural identifiers."
    )
    derive_adduct_from_name: bool = Field(
        default=True, description="Derive adducts from compound names."
    )
    derive_formula_from_name: bool = Field(
        default=True, description="Derive formulas from compound names."
    )
    clean_compound_name: bool = Field(
        default=True, description="Standardize compound names."
    )
    derive_ionmode: bool = Field(
        default=True, description="Derive ion mode from metadata."
    )
    make_charge_int: bool = Field(
        default=True, description="Ensure charge is an integer."
    )

    # Peak filtering toggles
    filter_by_intensity: bool = Field(
        default=False, description="Filter peaks by minimum intensity/noise threshold."
    )
    filter_min_peaks: bool = Field(
        default=False, description="Require a minimum number of peaks."
    )
    filter_by_mz: bool = Field(
        default=False, description="Truncate peaks outside of an m/z range."
    )
    reduce_to_top_n_peaks: bool = Field(
        default=False, description="Reduce spectrum to top N most intense peaks."
    )

    # HLD v0.9 Pre-Processing
    mz_min: float = Field(default=0.0, ge=0.0, description="Minimum m/z")
    mz_max: float = Field(default=1000.0, description="Maximum m/z")
    n_max: int | None = Field(default=None, gt=0, description="Top N peaks")

    @field_validator("mz_max")
    @classmethod
    def validate_mz_range(cls, v: float, info: ValidationInfo) -> float:
        """
        Validate that the maximum m/z value is greater than the minimum m/z value.

        This validator ensures logical consistency in the m/z range settings by
        comparing ``mz_max`` against ``mz_min`` (if present in the validation context).

        Parameters
        ----------
        v : float
            The value of ``mz_max`` being validated.
        info : ValidationInfo
            The validation context containing other field values, specifically ``mz_min``.

        Returns
        -------
        float
            The validated ``mz_max`` value.

        Raises
        ------
        ValueError
            If ``mz_max`` is less than or equal to ``mz_min``.
        """
        if "mz_min" in info.data and v <= info.data["mz_min"]:
            raise ValueError(
                f"mz_max ({v}) must be greater than mz_min ({info.data['mz_min']})"
            )
        return v

    noise_threshold: float = Field(
        default=1000.0, description="Minimum intensity threshold"
    )

    # --- Entropy-based decoy generation (FDR calibration) ---
    decoy_min_relative_intensity: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description=(
            "Baseline noise floor for entropy-based decoy generation, as a "
            "fraction of the base peak (the most intense fragment): peaks "
            "below this threshold are excluded before the sqrt-weighted "
            "spectral entropy is computed and before decoys are constructed. "
            "Strict noise thresholding prevents chemical noise from "
            "inflating entropy estimates and biasing FDR calibration."
        ),
    )
    decoy_mz_shift_da: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Uniform per-peak m/z jitter (Da) applied to decoy fragment "
            "positions, randomizing fragmentation pathways so decoys share "
            "no fragment positions with their source spectra at scoring "
            "tolerance."
        ),
    )

    @field_validator("min_intensity", "noise_threshold")
    @classmethod
    def validate_non_negative_intensities(cls, v: float, info: ValidationInfo) -> float:
        """Ensure intensity thresholds are non-negative."""
        if v < 0:
            raise ValueError(f"{info.field_name} cannot be negative. Received: {v}")
        return v

    @field_validator("min_peaks")
    @classmethod
    def validate_non_negative_peaks(cls, v: int, info: ValidationInfo) -> int:
        """Ensure min_peaks is non-negative."""
        if v < 0:
            raise ValueError(f"{info.field_name} cannot be negative. Received: {v}")
        return v

    @model_validator(mode="after")
    def validate_top_n_requires_n_max(self) -> "ProcessingConfig":
        """Reject a Top-N reduction toggle that would silently do nothing.

        ``reduce_to_top_n_peaks=True`` without a positive ``n_max`` is a
        configuration error: the processing pipeline would otherwise accept
        the toggle and leave every spectrum unreduced, silently.
        """
        if self.reduce_to_top_n_peaks and (self.n_max is None or self.n_max <= 0):
            raise ValueError(
                "reduce_to_top_n_peaks=True requires n_max to be set to a "
                "positive value."
            )
        return self

    # Metadata context
    instrument: Optional[str] = None
    mode: Literal["positive", "negative", ""] = ""
    solvents: List[SolventConfig] = Field(default_factory=list)

    # Legacy support (keeping for backward compatibility if needed)
    precursor_mz: float = Field(default=0.0)
    retention_time: float = Field(default=0.0)

    @field_validator("precursor_mz")
    @classmethod
    def validate_precursor_mz(cls, v: float) -> float:
        """
        Validate that the precursor m/z value is non-negative.

        Parameters
        ----------
        v : float
            The precursor m/z value to validate.

        Returns
        -------
        float
            The validated precursor m/z value.

        Raises
        ------
        ValueError
            If ``precursor_mz`` is negative.
        """
        if v < 0:
            raise ValueError(f"precursor_mz must be non-negative. Received: {v}")
        return v


class SimilarityConfig(MassFlowBaseModel):
    """
    Settings for similarity scoring with classical and ML-based algorithms.

    Classical algorithms (always available):
        - ``cosine``: CosineGreedy via matchms
        - ``modified_cosine``: ModifiedCosine via matchms

    Machine-learning algorithms (require ``pip install massflow[ml]``):
        - ``spec2vec``: Spec2Vec embeddings via Gensim
        - ``ms2deepscore``: MS2DeepScore via PyTorch
        - ``consensus``: Consensus scoring combining multiple engines
        - ``cascade``: Cascaded scoring with hierarchical filtering

    Legacy ``tolerance`` is retained for compatibility with existing configs.
    """

    algorithm: Literal[
        "cosine",
        "modified_cosine",
        "spec2vec",
        "ms2deepscore",
        "consensus",
        "cascade",
    ] = Field(
        default="cosine",
        description=(
            "Similarity algorithm: 'cosine', 'modified_cosine' (always available), "
            "'spec2vec', 'ms2deepscore', 'consensus', or 'cascade' "
            "(require massflow[ml])."
        ),
    )

    # Fixed-unit Tolerances
    ms1_tolerance: float = Field(
        default=0.02, description="Precursor mass tolerance in Da"
    )
    resolution_ppm: Optional[float] = Field(
        default=None,
        description="Optional: Precursor mass resolution in ppm. If set, this overrides ms1_tolerance for MS1 filtering.",
    )
    ms2_tolerance: float = Field(
        default=0.02, description="Fragment mass tolerance in Da"
    )

    # Legacy alias documented for compatibility: ``tolerance`` (fragment
    # tolerance in Da) maps onto ``ms2_tolerance``.  Kept as a REAL field so
    # it is never silently ignored; excluded from the normalized config so
    # provenance always shows the effective ``ms2_tolerance``.
    tolerance: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("tolerance"),
        exclude=True,
        description=(
            "DEPRECATED legacy alias for ms2_tolerance (fragment mass tolerance in Da)."
        ),
    )

    @model_validator(mode="after")
    def apply_legacy_tolerance(self) -> "SimilarityConfig":
        """Map the deprecated ``tolerance`` key onto ``ms2_tolerance``.

        Setting both keys with different values is ambiguous and rejected;
        setting only the legacy key applies it as the fragment tolerance
        with a deprecation warning.
        """
        if "tolerance" in self.model_fields_set and self.tolerance is not None:
            if (
                "ms2_tolerance" in self.model_fields_set
                and self.ms2_tolerance != self.tolerance
            ):
                raise ValueError(
                    "Both 'tolerance' (legacy) and 'ms2_tolerance' were set "
                    "with different values; remove the deprecated "
                    "'tolerance' key and use 'ms2_tolerance' only."
                )
            if self.tolerance < 0:
                raise ValueError(
                    f"tolerance cannot be negative. Received: {self.tolerance}"
                )
            if self.ms2_tolerance != self.tolerance:
                logger.warning(
                    "'tolerance' is deprecated; use 'ms2_tolerance'. "
                    "Applying tolerance=%.4f as ms2_tolerance.",
                    self.tolerance,
                )
            self.ms2_tolerance = self.tolerance
        return self

    min_score: float = 0.6
    analog_search: bool = False
    min_matched_peaks: int = 3
    fdr_threshold: float = 0.01
    rt_tolerance: Optional[float] = Field(
        default=None,
        description="Optional: Retention time tolerance in minutes. When set, query-reference pairs with an RT difference exceeding this value are rejected.",
    )

    # --- Consensus engine settings (used when algorithm='consensus') ---
    consensus_weights: dict[str, float] = Field(
        default_factory=lambda: {"cosine": 0.5, "modified_cosine": 0.5},
        description=(
            "Per-engine weights for consensus scoring. Keys are algorithm names "
            "('cosine', 'modified_cosine', 'spec2vec', 'ms2deepscore'). "
            "Weights are normalised internally; only relative proportions matter."
        ),
    )
    consensus_min_engines: int = Field(
        default=1,
        ge=1,
        description="Minimum number of sub-engines that must score a candidate for it to be retained.",
    )

    # --- Cascade engine settings (used when algorithm='cascade') ---
    cascade_lower_bound: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Score threshold applied at each cascade stage to pass candidates forward.",
    )
    cascade_upper_bound: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Final score threshold applied after the last cascade stage.",
    )
    cascade_stages: list[str] = Field(
        default_factory=lambda: ["cosine", "modified_cosine"],
        description="Ordered list of algorithms used as cascade stages, from fastest to slowest.",
    )

    # --- Numba peak pre-filter (used with algorithm='modified_cosine') ---
    enable_numba_prefilter: bool = Field(
        default=True,
        description=(
            "When True, modified_cosine scoring uses the Numba-accelerated "
            "peak/neutral-loss matching prefilter to skip query-reference "
            "pairs that cannot reach min_matched_peaks before exact scoring. "
            "Produces identical results to full scoring and falls back "
            "automatically when numba is not installed."
        ),
    )

    # --- HNSW ANN index settings (used by cascade when hnsw_enabled) ---
    hnsw_enabled: bool = Field(
        default=False,
        description=(
            "When True, cascade searches build a HNSW (Hierarchical Navigable "
            "Small World) index over binned reference spectra and use it for "
            "sub-linear candidate retrieval before exact scoring. Spectral "
            "cosine/modified-cosine are non-metric, so HNSW only generates "
            "candidates; exact scoring is always applied afterwards."
        ),
    )
    hnsw_m: int = Field(
        default=32,
        ge=1,
        description=(
            "HNSW construction parameter M: maximum connections per node per "
            "layer. Higher values densify the graph, improving recall on "
            "non-metric spectral data at the cost of memory."
        ),
    )
    hnsw_ef_construction: int = Field(
        default=400,
        ge=1,
        description=(
            "HNSW construction parameter ef_construction: dynamic candidate "
            "list size during graph build. Higher values make heuristic "
            "pruning gentler, which prevents recall degradation on "
            "non-metric spectral data."
        ),
    )
    hnsw_ef_search: int = Field(
        default=200,
        ge=1,
        description=(
            "HNSW query parameter ef_search: dynamic candidate list size at "
            "query time. Must be >= hnsw_candidates_per_query; higher values "
            "trade latency for recall."
        ),
    )
    hnsw_candidates_per_query: int = Field(
        default=200,
        ge=1,
        description=(
            "Number of HNSW candidates retrieved per query spectrum before "
            "exact scoring in the cascade."
        ),
    )
    hnsw_bin_width: float = Field(
        default=1.0,
        gt=0.0,
        description="m/z bin width (Da) used to vectorize spectra for the HNSW index.",
    )
    hnsw_mz_min: float = Field(
        default=0.0,
        ge=0.0,
        description="Lower m/z bound (inclusive) of the HNSW binning range.",
    )
    hnsw_mz_max: float = Field(
        default=2000.0,
        gt=0.0,
        description="Upper m/z bound (exclusive) of the HNSW binning range.",
    )
    hnsw_random_seed: int = Field(
        default=42,
        description="Random seed for HNSW graph construction (deterministic builds).",
    )

    # --- ML Router / Orchestrator settings (post-v0.1) -----------------------
    enable_routing: bool = Field(
        default=False,
        description=(
            "When True, the orchestrator classifies each query spectrum by "
            "structural difficulty and dispatches it to the appropriate "
            "scoring engine (fast classical engine for 'easy' spectra, "
            "ML consensus engine for 'hard' spectra)."
        ),
    )
    routing_easy_engine: Literal["cosine", "modified_cosine"] = Field(
        default="modified_cosine",
        description="Engine used for 'easy' (high-quality) query spectra when routing is enabled.",
    )
    routing_hard_engine: Literal["consensus", "cascade"] = Field(
        default="consensus",
        description="Engine used for 'hard' (low-quality / chimeric) query spectra when routing is enabled.",
    )
    routing_fallback_engine: Literal["cosine", "modified_cosine"] = Field(
        default="modified_cosine",
        description=(
            "Fallback engine used when the 'hard' ML engine fails or times out. "
            "Must be a classical engine so it is always available."
        ),
    )
    routing_precursor_purity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Precursor purity below which a spectrum is routed to the hard engine.",
    )
    routing_snr_threshold: float = Field(
        default=3.0,
        ge=0.0,
        description="Signal-to-noise ratio below which a spectrum is routed to the hard engine.",
    )
    routing_chimeric_action: Literal["hard", "easy"] = Field(
        default="hard",
        description="How to route chimeric spectra: 'hard' (ML engine) or 'easy' (classical engine).",
    )
    routing_min_difficulty_flags: int = Field(
        default=1,
        ge=0,
        le=5,
        description=(
            "Minimum number of active boolean triage flags required before "
            "a spectrum is routed to the hard engine. Set to 0 to always "
            "route flagged spectra to the hard engine."
        ),
    )
    routing_ml_timeout_seconds: float = Field(
        default=300.0,
        ge=1.0,
        description="Maximum seconds allowed for a single ML engine chunk before falling back.",
    )

    # --- Remote ML engine endpoints (massflow-ml satellite boundary) ---
    ml_endpoints: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Remote ML engine endpoints keyed by algorithm name, e.g. "
            "{'spec2vec': 'http://ml-host:8080/spec2vec'} or "
            "{'ms2deepscore': 'grpc://ml-host:9090'}. When an endpoint is "
            "configured for an algorithm, scoring is routed to the remote "
            "service (REST JSON or the massflow.v1.ml gRPC contract) instead "
            "of requiring a local installation of the heavy dependencies."
        ),
    )
    ml_request_timeout_seconds: float = Field(
        default=10.0,
        ge=0.1,
        description=(
            "Per-request timeout (seconds) for remote ML engine calls. "
            "Timed-out calls count as failures for the circuit breaker."
        ),
    )
    ml_circuit_breaker_threshold: int = Field(
        default=3,
        ge=1,
        description=(
            "Consecutive remote ML failures that open the circuit breaker. "
            "While open, calls fail fast and orchestrators fall back to "
            "classical scoring without paying the network timeout."
        ),
    )
    ml_circuit_breaker_cooldown_seconds: float = Field(
        default=60.0,
        ge=0.0,
        description=(
            "Seconds the circuit breaker stays open before allowing one "
            "trial call (half-open)."
        ),
    )

    # Leaf engines that can appear as consensus members / cascade stages.
    _LEAF_ENGINES: tuple[str, ...] = (
        "cosine",
        "modified_cosine",
        "spec2vec",
        "ms2deepscore",
    )

    @model_validator(mode="after")
    def validate_hnsw_parameters(self) -> "SimilarityConfig":
        """Ensure HNSW construction parameters define a usable graph.

        ``ef_construction`` must be at least ``M`` (hnswlib's heuristic
        pruning is unstable below this); the binning range must have positive
        width.
        """
        if self.hnsw_ef_construction < self.hnsw_m:
            raise ValueError(
                "hnsw_ef_construction must be >= hnsw_m for stable HNSW graph "
                f"construction. Received hnsw_m={self.hnsw_m}, "
                f"hnsw_ef_construction={self.hnsw_ef_construction}."
            )
        if self.hnsw_mz_min >= self.hnsw_mz_max:
            raise ValueError(
                "hnsw_mz_min must be < hnsw_mz_max for a non-empty binning "
                f"range. Received [{self.hnsw_mz_min}, {self.hnsw_mz_max})."
            )
        return self

    @model_validator(mode="after")
    def validate_engine_combinations(self) -> "SimilarityConfig":
        """Reject engine selections that would silently ignore settings.

        * HNSW candidate retrieval only exists inside the cascade engine:
          ``hnsw_enabled=True`` with any other ``algorithm`` is a
          configuration error (the index would be built and never used).
        * ``cascade_stages`` must be a non-empty list of leaf engines.
        * ``consensus_weights`` keys must be leaf engines with positive
          weights.
        """
        if self.hnsw_enabled and self.algorithm != "cascade":
            raise ValueError(
                "hnsw_enabled=True requires algorithm='cascade': HNSW "
                "candidate retrieval is only used by the cascade engine."
            )
        if not self.cascade_stages:
            raise ValueError("cascade_stages must be a non-empty list of engines.")
        for stage in self.cascade_stages:
            if stage not in self._LEAF_ENGINES:
                raise ValueError(
                    f"cascade_stages contains unknown engine {stage!r}; valid "
                    f"leaf engines: {list(self._LEAF_ENGINES)}."
                )
        for name, weight in self.consensus_weights.items():
            if name not in self._LEAF_ENGINES:
                raise ValueError(
                    f"consensus_weights contains unknown engine {name!r}; valid "
                    f"leaf engines: {list(self._LEAF_ENGINES)}."
                )
            if weight <= 0:
                raise ValueError(
                    f"consensus_weights[{name!r}] must be a positive weight; "
                    f"received {weight}."
                )
        return self

    @field_validator("ml_endpoints")
    @classmethod
    def validate_ml_endpoints(
        cls, endpoints: dict[str, str], info: ValidationInfo
    ) -> dict[str, str]:
        """Ensure every ML endpoint uses a supported transport scheme."""
        for algorithm, endpoint in endpoints.items():
            if not isinstance(endpoint, str) or not endpoint.strip():
                raise ValueError(
                    f"ml_endpoints['{algorithm}'] must be a non-empty URL string."
                )
            normalized = endpoint.strip()
            if not normalized.startswith(("http://", "https://", "grpc://")):
                raise ValueError(
                    f"ml_endpoints['{algorithm}']='{endpoint}' has an "
                    f"unsupported scheme; use http://, https://, or grpc://."
                )
        return endpoints

    @field_validator("rt_tolerance")
    @classmethod
    def validate_rt_tolerance(
        cls, v: Optional[float], info: ValidationInfo
    ) -> Optional[float]:
        """Ensure rt_tolerance is not negative."""
        if v is not None and v < 0:
            raise ValueError(f"{info.field_name} cannot be negative. Received: {v}")
        return v

    @field_validator("ms1_tolerance", "ms2_tolerance")
    @classmethod
    def validate_mass_tolerances(cls, v: float, info: ValidationInfo) -> float:
        """Ensure mass tolerances are not negative."""
        if v < 0:
            raise ValueError(f"{info.field_name} cannot be negative. Received: {v}")
        return v

    @field_validator("min_score", "fdr_threshold")
    @classmethod
    def validate_score_ranges(cls, v: float, info: ValidationInfo) -> float:
        """Ensure scores and thresholds are within [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"{info.field_name} must be between 0.0 and 1.0. Received: {v}"
            )
        return v

    @field_validator("min_matched_peaks")
    @classmethod
    def validate_min_matched_peaks(cls, v: int, info: ValidationInfo) -> int:
        """Ensure minimum matched peaks is not negative."""
        if v < 0:
            raise ValueError(f"{info.field_name} cannot be negative. Received: {v}")
        return v


class WorkflowConfig(MassFlowBaseModel):
    """
    High-level workflow feature flags (reserved for future pipeline stages).

    This model currently has no active fields; all orchestration is handled
    directly by :mod:`MassFlow.workflow`. Fields will be added here as new
    pipeline stages (e.g. peak picking, retention-time alignment) are implemented.
    """

    pass


class ExportConfig(MassFlowBaseModel):
    """
    Declared export preferences for result output.

    The annotation workflow writes per-file result reports in the configured
    format. Only 'csv' and 'mztab' are part of the stable v0.1 contract.
    """

    format: Literal["csv", "mztab"] = Field(
        default="csv",
        description="Output format: 'csv' or 'mztab'.",
    )


class MassFlowConfig(MassFlowBaseModel):
    """
    Root configuration object loaded from MassFlow YAML files.

    This model is the contract passed between the CLI, workflow, processing,
    and similarity layers.
    """

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    input: InputConfig
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    modifications: dict[str, dict] = Field(
        default_factory=dict,
        description=(
            "User-defined chemical modifications registered into pyteomics "
            "(name -> {formula, type}) before any spectral processing occurs."
        ),
    )
    config_path: Optional[Path] = Field(
        default=None,
        description=(
            "Absolute path of the YAML configuration file this object was "
            "loaded from (None for programmatically constructed configs). "
            "Set by from_yaml(); written into provenance."
        ),
    )

    # Legacy alias for root output_directory to map to project.output_directory
    @property
    def output_directory(self) -> Path:
        """
        Retrieve the output directory from the project configuration.

        This property serves as an alias to ``self.project.output_directory``,
        providing backward compatibility or convenient access to the output path.

        Returns
        -------
        Path
            The path to the configured output directory.
        """
        return self.project.output_directory

    def normalized_config(self) -> dict[str, Any]:
        """Canonical, JSON-safe representation of the effective configuration.

        This is the normalized configuration representation written into
        provenance (per-file reports and the run-level provenance file): it
        contains the schema version, the absolute source config path (when
        loaded from YAML), the full effective configuration with all paths in
        their resolved (absolute) form, and a SHA-256 digest over the
        canonical JSON so a downstream consumer can verify that a result was
        produced by exactly this configuration.

        Returns
        -------
        dict
            ``{"schema_version": 1, "config_file": ...,
            "effective_config": {...}, "config_digest_sha256": ...}``
        """
        payload: dict[str, Any] = {
            "schema_version": 1,
            "config_file": str(self.config_path) if self.config_path else None,
            "effective_config": json.loads(self.model_dump_json()),
        }
        canonical = json.dumps(payload, sort_keys=True)
        payload["config_digest_sha256"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        return payload

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "MassFlowConfig":
        """
        Load and validate a ``MassFlowConfig`` from a YAML file.

        The file is parsed using a custom YAML loader that tracks line numbers,
        and then validated by Pydantic against the nested configuration models
        defined in this module. Human-readable error messages with exact line
        numbers are provided when validation fails.

        Parameters
        ----------
        path : str or Path
            The file system path to the YAML configuration file.

        Returns
        -------
        MassFlowConfig
            Validated configuration populated from the YAML document.

        Raises
        ------
        FileNotFoundError
            If the specified file path does not exist.
        ValueError
            If the YAML content does not conform to the MassFlow schema.
        yaml.YAMLError
            If the YAML document cannot be parsed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            data = yaml.load(f, Loader=LineNumberLoader)

        if data is None:
            data = {}

        # The line-number loader injects ``__lines__`` bookkeeping into every
        # mapping; strip it before validation (strict models reject unknown
        # keys) but keep it for error reporting.
        raw_data = data
        clean_data = _strip_line_markers(data)

        try:
            config_instance = cls(**clean_data)
        except ValidationError as e:
            error_messages = []
            for err in e.errors():
                loc = err["loc"]
                msg = err["msg"].replace("Value error, ", "")

                current = raw_data
                line_num = "Unknown"
                for i, part in enumerate(loc):
                    if (
                        isinstance(current, dict)
                        and "__lines__" in current
                        and part in current["__lines__"]
                    ):
                        if i == len(loc) - 1:
                            line_num = current["__lines__"][part]
                        current = current.get(part)
                    elif isinstance(current, list) and isinstance(part, int):
                        current = current[part]
                    else:
                        break

                key_path = " -> ".join(str(k) for k in loc)
                if err["type"] == "extra_forbidden":
                    msg = _format_extra_key_error(cls, loc, msg)
                error_messages.append(f"Line {line_num}, Key '{key_path}': {msg}")

            raise ValueError(
                "Configuration validation failed:\n" + "\n".join(error_messages)
            ) from e

        # Record the canonical (absolute) source path for provenance.
        config_instance.config_path = path.resolve()

        # Resolve relative paths against the configuration file's directory
        # (deterministic, independent of the caller's working directory),
        # unless the documented compatibility mode is enabled.
        if not _compat_cwd_paths():
            base_dir = path.resolve().parent
            config_instance.project.output_directory = _resolve_config_path(
                config_instance.project.output_directory, base_dir
            )
            config_instance.input.input_path = _resolve_config_path(
                config_instance.input.input_path, base_dir
            )
            if config_instance.input.library_path is not None:
                config_instance.input.library_path = _resolve_config_path(
                    config_instance.input.library_path, base_dir
                )

        # Register any user-defined chemical modifications into pyteomics
        # before any spectral processing occurs.
        register_custom_modifications(config_instance.modifications)

        return config_instance


def _strip_line_markers(data: Any) -> Any:
    """Recursively remove the ``__lines__`` bookkeeping keys injected by
    :class:`LineNumberLoader`."""
    if isinstance(data, dict):
        return {
            key: _strip_line_markers(value)
            for key, value in data.items()
            if key != "__lines__"
        }
    if isinstance(data, list):
        return [_strip_line_markers(item) for item in data]
    return data


def _compat_cwd_paths() -> bool:
    """Whether the legacy CWD-relative path resolution is enabled.

    Controlled by the ``MASSFLOW_COMPAT_CWD_PATHS`` environment variable
    (set to ``1``/``true``/``yes``).  Documented in
    ``docs/user-guide/configuration.md``.
    """
    return os.environ.get(_COMPAT_CWD_PATHS_ENV, "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _resolve_config_path(p: Path, base_dir: Path) -> Path:
    """Resolve *p* against *base_dir* (the YAML file's directory).

    ``~`` is expanded first; absolute paths are returned unchanged.
    """
    p = p.expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def _resolve_model_type(annotation: Any) -> Optional[type]:
    """Unwrap Optional/Union/Annotated annotations to find a nested model."""
    for candidate in get_args(annotation):
        if isinstance(candidate, type) and hasattr(candidate, "model_fields"):
            return candidate
    return None


def _format_extra_key_error(model_cls: type, loc: tuple, _original_msg: str) -> str:
    """Human-readable message for an unknown configuration key, with a
    spelling suggestion when one is close enough."""
    current_cls: Any = model_cls
    for part in loc[:-1]:
        if isinstance(part, int):
            return f"Unknown configuration key '{loc[-1]}'."
        fields = current_cls.model_fields
        if part not in fields:
            return f"Unknown configuration key '{loc[-1]}'."
        annotation = fields[part].annotation
        if isinstance(annotation, type) and hasattr(annotation, "model_fields"):
            current_cls = annotation
        else:
            nested = _resolve_model_type(annotation)
            if nested is None:
                return f"Unknown configuration key '{loc[-1]}'."
            current_cls = nested

    bad_key = str(loc[-1])
    allowed = sorted(current_cls.model_fields.keys())
    suggestion = difflib.get_close_matches(bad_key, allowed, n=1, cutoff=0.6)
    where = f" under '{' -> '.join(str(k) for k in loc[:-1])}'" if loc[:-1] else ""
    if suggestion:
        return (
            f"Unknown configuration key '{bad_key}'{where}. "
            f"Did you mean '{suggestion[0]}'?"
        )
    return (
        f"Unknown configuration key '{bad_key}'{where}. "
        f"Allowed keys: {', '.join(allowed)}."
    )
