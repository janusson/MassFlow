"""
Configuration schema and validation for Yogimass workflows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Optional at runtime but part of core deps
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore


class ConfigError(ValueError):
    """
    Configuration validation error with dotted-path context.
    """

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def _coerce_path(value: Any, *, path: str) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    raise ConfigError(path, "Expected a string or Path.")


def _coerce_optional_path(value: Any, *, path: str) -> Path | None:
    if value is None:
        return None
    return _coerce_path(value, path=path)


def _coerce_bool(value: Any, *, path: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in {0, 1}:
        return bool(value)
    raise ConfigError(path, "Expected a boolean value.")


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _check_unknown_keys(section: Mapping[str, Any], allowed: set[str], *, prefix: str) -> None:
    for key in section:
        if key not in allowed:
            path = key if prefix in {"", "<root>"} else f"{prefix}.{key}"
            raise ConfigError(path, f"Unknown key '{key}'.")


@dataclass
class InputConfig:
    paths: list[Path] = field(default_factory=list)
    format: str = "mgf"
    recursive: bool = False
    msdial_output: Path | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> InputConfig:
        allowed = {"path", "paths", "format", "recursive", "msdial_output"}
        _check_unknown_keys(data, allowed, prefix="input")

        raw_paths: list[Any] = []
        if "path" in data:
            raw_paths.extend(_listify(data.get("path")))
        if "paths" in data:
            raw_paths.extend(_listify(data.get("paths")))
        paths = [_coerce_path(item, path="input.paths") for item in raw_paths if item is not None]

        fmt = str(data.get("format", "mgf")).lower()
        if fmt not in {"mgf", "msp", "msdial"}:
            raise ConfigError("input.format", f"Unsupported format '{fmt}'.")

        recursive = _coerce_bool(data.get("recursive", False), path="input.recursive")
        msdial_output = _coerce_optional_path(data.get("msdial_output"), path="input.msdial_output")
        return cls(paths=paths, format=fmt, recursive=recursive, msdial_output=msdial_output)


@dataclass
class LibraryConfig:
    path: Path | None = None
    build: bool = True
    input_format: str | None = None
    recursive: bool | None = None
    storage: str | None = None
    overwrite: bool = True
    sources: list[Path] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, outputs: "OutputsConfig") -> LibraryConfig:
        allowed = {"path", "build", "input_format", "recursive", "format", "overwrite", "sources"}
        _check_unknown_keys(data, allowed, prefix="library")

        path_value = data.get("path") or outputs.library
        path = _coerce_optional_path(path_value, path="library.path")

        build_default = path is not None
        build = _coerce_bool(data.get("build", build_default), path="library.build")
        input_format = data.get("input_format")
        if input_format is not None:
            input_format = str(input_format).lower()
            if input_format not in {"mgf", "msp", "msdial"}:
                raise ConfigError("library.input_format", f"Unsupported format '{input_format}'.")

        recursive_value = data.get("recursive")
        recursive = _coerce_bool(recursive_value, path="library.recursive") if recursive_value is not None else None
        storage = data.get("format")
        if storage is not None and storage not in {"json", "sqlite"}:
            raise ConfigError("library.format", "Format must be 'json' or 'sqlite'.")
        overwrite = _coerce_bool(data.get("overwrite", True), path="library.overwrite")
        sources = [
            _coerce_path(item, path="library.sources") for item in _listify(data.get("sources")) if item is not None
        ]
        return cls(
            path=path,
            build=build,
            input_format=input_format,
            recursive=recursive,
            storage=storage,
            overwrite=overwrite,
            sources=sources,
        )


@dataclass
class SimilarityConfig:
    enabled: bool = False
    queries: list[Path] = field(default_factory=list)
    query_format: str | None = None
    recursive: bool | None = None
    top_n: int = 5
    min_score: float = 0.0
    backend: str = "naive"
    vectorizer: str = "spec2vec"
    embedding_model: str | None = None
    output: Path | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SimilarityConfig:
        allowed = {
            "search",
            "queries",
            "query_format",
            "recursive",
            "top_n",
            "min_score",
            "backend",
            "vectorizer",
            "embedding_model",
            "output",
        }
        _check_unknown_keys(data, allowed, prefix="similarity")

        enabled = _coerce_bool(data.get("search", False), path="similarity.search")
        queries = [
            _coerce_path(item, path="similarity.queries") for item in _listify(data.get("queries")) if item is not None
        ]
        if queries and not enabled:
            enabled = True

        query_format = data.get("query_format")
        if query_format is not None:
            query_format = str(query_format).lower()
            if query_format not in {"mgf", "msp", "msdial"}:
                raise ConfigError("similarity.query_format", f"Unsupported format '{query_format}'.")
        if "recursive" in data:
            recursive = _coerce_bool(data.get("recursive"), path="similarity.recursive")
        else:
            recursive = None

        try:
            top_n = int(data.get("top_n", 5))
        except (TypeError, ValueError) as exc:
            raise ConfigError("similarity.top_n", "top_n must be an integer.") from exc
        if top_n <= 0:
            raise ConfigError("similarity.top_n", "top_n must be greater than zero.")

        try:
            min_score = float(data.get("min_score", 0.0))
        except (TypeError, ValueError) as exc:
            raise ConfigError("similarity.min_score", "min_score must be a number.") from exc
        if min_score < 0:
            raise ConfigError("similarity.min_score", "min_score cannot be negative.")

        backend = str(data.get("backend", "naive")).lower()
        if backend not in {"naive", "annoy", "faiss"}:
            raise ConfigError("similarity.backend", f"Unsupported backend '{backend}'.")

        vectorizer = str(data.get("vectorizer", "spec2vec")).lower()
        if vectorizer not in {"spec2vec", "embedding"}:
            raise ConfigError("similarity.vectorizer", f"Unsupported vectorizer '{vectorizer}'.")

        embedding_model = data.get("embedding_model")
        if embedding_model is not None:
            embedding_model = str(embedding_model)
            if vectorizer != "embedding":
                raise ConfigError(
                    "similarity.embedding_model",
                    "embedding_model is only valid when vectorizer='embedding'.",
                )

        output = _coerce_optional_path(data.get("output"), path="similarity.output")

        return cls(
            enabled=enabled,
            queries=queries,
            query_format=query_format,
            recursive=recursive,
            top_n=top_n,
            min_score=min_score,
            backend=backend,
            vectorizer=vectorizer,
            embedding_model=embedding_model,
            output=output,
        )


@dataclass
class NetworkConfig:
    enabled: bool = False
    input: Path | None = None
    library: Path | None = None
    metric: str | None = None
    threshold: float | None = None
    knn: int | None = None
    reference_mz: list[float] | None = None
    directed: bool = False
    output: Path | None = None
    summary: Path | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, outputs: "OutputsConfig") -> NetworkConfig:
        allowed = {
            "enabled",
            "input",
            "library",
            "metric",
            "threshold",
            "knn",
            "reference_mz",
            "directed",
            "output",
            "summary",
        }
        _check_unknown_keys(data, allowed, prefix="network")

        enabled = _coerce_bool(data.get("enabled", False), path="network.enabled")
        output = _coerce_optional_path(data.get("output") or outputs.network, path="network.output")
        summary = _coerce_optional_path(data.get("summary") or outputs.network_summary, path="network.summary")
        if output and not enabled:
            enabled = True

        network_input = _coerce_optional_path(data.get("input"), path="network.input")
        library = _coerce_optional_path(data.get("library") or outputs.library, path="network.library")

        metric = data.get("metric")
        if metric is not None:
            metric = str(metric).lower()
            if metric not in {"cosine", "modified_cosine", "modified-cosine", "spec2vec", "embedding"}:
                raise ConfigError("network.metric", f"Unsupported metric '{metric}'.")

        threshold = data.get("threshold")
        if threshold is not None:
            try:
                threshold = float(threshold)
            except (TypeError, ValueError) as exc:
                raise ConfigError("network.threshold", "threshold must be numeric.") from exc
            if threshold <= 0:
                raise ConfigError("network.threshold", "threshold must be greater than zero.")

        knn = data.get("knn")
        if knn is not None:
            try:
                knn = int(knn)
            except (TypeError, ValueError) as exc:
                raise ConfigError("network.knn", "knn must be an integer.") from exc
            if knn <= 0:
                raise ConfigError("network.knn", "knn must be greater than zero.")

        if threshold is not None and knn is not None:
            raise ConfigError("network.threshold", "Specify either threshold or knn, not both.")

        reference_mz = None
        if data.get("reference_mz") is not None:
            try:
                reference_mz = [float(value) for value in data.get("reference_mz", [])]
            except (TypeError, ValueError) as exc:
                raise ConfigError("network.reference_mz", "reference_mz must be a list of numbers.") from exc

        directed = _coerce_bool(data.get("directed", False), path="network.directed")

        return cls(
            enabled=enabled,
            input=network_input,
            library=library,
            metric=metric,
            threshold=threshold,
            knn=knn,
            reference_mz=reference_mz,
            directed=directed,
            output=output,
            summary=summary,
        )


@dataclass
class CurationConfig:
    enabled: bool = False
    output: Path | None = None
    qc_report: Path | None = None
    min_peaks: int = 3
    min_total_ion_current: float = 0.05
    max_single_peak_fraction: float = 0.9
    precursor_tolerance: float = 0.01
    similarity_threshold: float = 0.95
    require_precursor_mz: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, outputs: "OutputsConfig") -> CurationConfig:
        allowed = {
            "enabled",
            "output",
            "qc_report",
            "min_peaks",
            "min_total_ion_current",
            "max_single_peak_fraction",
            "precursor_tolerance",
            "similarity_threshold",
            "require_precursor_mz",
        }
        _check_unknown_keys(data, allowed, prefix="curation")

        enabled = _coerce_bool(data.get("enabled", False), path="curation.enabled")
        output = _coerce_optional_path(data.get("output") or outputs.curated_library, path="curation.output")
        qc_report = _coerce_optional_path(data.get("qc_report") or outputs.qc_report, path="curation.qc_report")

        try:
            min_peaks = int(data.get("min_peaks", 3))
        except (TypeError, ValueError) as exc:
            raise ConfigError("curation.min_peaks", "min_peaks must be an integer.") from exc
        if min_peaks <= 0:
            raise ConfigError("curation.min_peaks", "min_peaks must be greater than zero.")

        try:
            min_total_ion_current = float(data.get("min_total_ion_current", 0.05))
        except (TypeError, ValueError) as exc:
            raise ConfigError("curation.min_total_ion_current", "Value must be numeric.") from exc
        if min_total_ion_current < 0:
            raise ConfigError("curation.min_total_ion_current", "Value cannot be negative.")

        try:
            max_single_peak_fraction = float(data.get("max_single_peak_fraction", 0.9))
        except (TypeError, ValueError) as exc:
            raise ConfigError("curation.max_single_peak_fraction", "Value must be numeric.") from exc
        if not 0 < max_single_peak_fraction <= 1:
            raise ConfigError("curation.max_single_peak_fraction", "Value must be between 0 and 1.")

        try:
            precursor_tolerance = float(data.get("precursor_tolerance", 0.01))
        except (TypeError, ValueError) as exc:
            raise ConfigError("curation.precursor_tolerance", "Value must be numeric.") from exc
        if precursor_tolerance <= 0:
            raise ConfigError("curation.precursor_tolerance", "Value must be greater than zero.")

        try:
            similarity_threshold = float(data.get("similarity_threshold", 0.95))
        except (TypeError, ValueError) as exc:
            raise ConfigError("curation.similarity_threshold", "Value must be numeric.") from exc
        if not 0 < similarity_threshold <= 1:
            raise ConfigError("curation.similarity_threshold", "Value must be between 0 and 1.")

        require_precursor_mz = _coerce_bool(
            data.get("require_precursor_mz", True),
            path="curation.require_precursor_mz",
        )

        return cls(
            enabled=enabled,
            output=output,
            qc_report=qc_report,
            min_peaks=min_peaks,
            min_total_ion_current=min_total_ion_current,
            max_single_peak_fraction=max_single_peak_fraction,
            precursor_tolerance=precursor_tolerance,
            similarity_threshold=similarity_threshold,
            require_precursor_mz=require_precursor_mz,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_peaks": self.min_peaks,
            "min_total_ion_current": self.min_total_ion_current,
            "max_single_peak_fraction": self.max_single_peak_fraction,
            "precursor_tolerance": self.precursor_tolerance,
            "similarity_threshold": self.similarity_threshold,
            "require_precursor_mz": self.require_precursor_mz,
        }


@dataclass
class OutputsConfig:
    library: Path | None = None
    curated_library: Path | None = None
    qc_report: Path | None = None
    search_results: Path | None = None
    network: Path | None = None
    network_summary: Path | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> OutputsConfig:
        allowed = {
            "library",
            "curated_library",
            "qc_report",
            "search_results",
            "network",
            "network_summary",
        }
        _check_unknown_keys(data, allowed, prefix="outputs")
        return cls(
            library=_coerce_optional_path(data.get("library"), path="outputs.library"),
            curated_library=_coerce_optional_path(data.get("curated_library"), path="outputs.curated_library"),
            qc_report=_coerce_optional_path(data.get("qc_report"), path="outputs.qc_report"),
            search_results=_coerce_optional_path(data.get("search_results"), path="outputs.search_results"),
            network=_coerce_optional_path(data.get("network"), path="outputs.network"),
            network_summary=_coerce_optional_path(data.get("network_summary"), path="outputs.network_summary"),
        )


@dataclass
class WorkflowConfig:
    input: InputConfig
    library: LibraryConfig
    similarity: SimilarityConfig
    network: NetworkConfig
    outputs: OutputsConfig
    curation: CurationConfig

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> WorkflowConfig:
        if not isinstance(data, Mapping):
            raise ConfigError("<root>", "Configuration root must be a mapping/object.")
        allowed_top = {"input", "library", "similarity", "network", "outputs", "curation"}
        _check_unknown_keys(data, allowed_top, prefix="<root>")

        outputs = OutputsConfig.from_mapping(data.get("outputs", {}))
        input_cfg = InputConfig.from_mapping(data.get("input", {}))
        library_cfg = LibraryConfig.from_mapping(data.get("library", {}), outputs=outputs)
        similarity_cfg = SimilarityConfig.from_mapping(data.get("similarity", {}))
        network_cfg = NetworkConfig.from_mapping(data.get("network", {}), outputs=outputs)
        curation_cfg = CurationConfig.from_mapping(data.get("curation", {}), outputs=outputs)

        config = cls(
            input=input_cfg,
            library=library_cfg,
            similarity=similarity_cfg,
            network=network_cfg,
            outputs=outputs,
            curation=curation_cfg,
        )
        config._validate()
        return config

    def _validate(self) -> None:
        if not (self.input.paths or self.library.sources or self.similarity.queries or self.library.path):
            raise ConfigError("input.paths", "Provide at least one input path, library, or query source.")

        if self.library.build and self.library.path is None:
            raise ConfigError("library.path", "Provide a library path when build is enabled.")

        if self.similarity.enabled:
            if self.library.path is None:
                raise ConfigError("similarity.search", "Similarity search requires a library.path.")
            if not (self.similarity.queries or self.input.paths):
                raise ConfigError("similarity.queries", "Provide queries or reuse input.paths for search.")

        if self.network.enabled:
            if self.network.output is None:
                raise ConfigError("network.output", "Building a network requires an output path.")
            if self.network.threshold is None and self.network.knn is None:
                raise ConfigError("network.threshold", "Specify either a threshold or knn for networking.")
            if not (self.network.input or self.library.path):
                raise ConfigError("network.input", "Provide network.input or library.path for network construction.")

        if self.curation.enabled and self.library.path is None:
            raise ConfigError("curation.enabled", "Curation requires a library.path.")

    def with_library_path(self, path: Path) -> "WorkflowConfig":
        """
        Return a shallow copy with the library path populated (used after build).
        """
        self.library.path = path
        if self.network.library is None:
            self.network.library = path
        return self


def _load_config_data(config: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config
    if isinstance(config, (str, Path)):
        config_path = Path(config)
        text = config_path.read_text(encoding="utf-8")
        suffix = config_path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ConfigError("<root>", "PyYAML is required to parse YAML configuration files.")
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return data
    raise ConfigError("<root>", "Config must be a path or mapping.")


def load_config(config: str | Path | Mapping[str, Any]) -> WorkflowConfig:
    """
    Load, parse, and validate a Yogimass workflow configuration.
    """
    data = _load_config_data(config)
    return WorkflowConfig.from_mapping(data)


__all__ = [
    "ConfigError",
    "CurationConfig",
    "InputConfig",
    "LibraryConfig",
    "NetworkConfig",
    "OutputsConfig",
    "SimilarityConfig",
    "WorkflowConfig",
    "load_config",
]
