"""
High-level orchestrator for Yogimass workflows.

Responsibilities:
- Load raw inputs (MGF/MSP/MS-DIAL), build/update spectral libraries.
- Search libraries, build similarity networks, run curation/QC, and write reports.
- Execute config-driven pipelines via ``run_from_config``.

Depends on:
- ``yogimass.similarity`` for processing, libraries, search.
- ``yogimass.networking`` for network construction.
- ``yogimass.curation`` for QC/de-duplication.
- ``yogimass.reporting`` for summaries and report writing.

Avoids:
- Direct CLI parsing (handled in ``yogimass.cli``).
- UI/plotting; keeps to orchestration and composition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from matchms import Spectrum
from matchms.importing import load_from_mgf, load_from_msp

from yogimass import curation
from yogimass.config import ConfigError, ProcessorConfig, WorkflowConfig, load_config
from yogimass.networking import network as network_builder
from yogimass.reporting import (
    SearchMatch,
    summarize_network,
    write_curation_report,
    write_network_summary,
    write_search_results,
)
from yogimass.similarity.library import LocalSpectralLibrary
from yogimass.similarity.metrics import spec2vec_vectorize
from yogimass.similarity.processing import SpectrumProcessor
from yogimass.similarity.search import LibrarySearcher
from yogimass.utils.logging import get_logger

logger = get_logger(__name__)


def load_data(
    inputs: str | Path | Sequence[str | Path],
    *,
    input_format: str = "mgf",
    recursive: bool = False,
    msdial_output_dir: str | Path | None = None,
) -> Any:
    """
    Load spectra or MS-DIAL tables from the given inputs.

    - For ``mgf``/``msp`` formats, returns a list of ``Spectrum`` objects.
    - For ``msdial``, returns a combined dataframe using ``clean_and_combine_msdial``.
    """
    paths = _coerce_paths(inputs)
    fmt = input_format.lower()
    if fmt in {"mgf", "msp"}:
        loader = load_from_mgf if fmt == "mgf" else load_from_msp
        suffixes = {f".{fmt}"}
        files = list(_iter_files(paths, suffixes, recursive=recursive))
        spectra = []
        for file_path in files:
            spectra.extend(list(loader(str(file_path))))
        if not spectra:
            logger.warning("No spectra loaded from %s input(s).", fmt)
        logger.info("Loaded %d spectra from %d %s file(s).", len(spectra), len(files), fmt.upper())
        return spectra
    if fmt == "msdial":
        try:
            from yogimass.io.msdial_clean_combine import clean_and_combine_msdial
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError("MS-DIAL support requires pandas and related utilities.") from exc
        if not paths:
            raise ValueError("At least one input path is required for msdial data.")
        input_dir = paths[0]
        output_dir = Path(msdial_output_dir) if msdial_output_dir is not None else input_dir
        dataframe, _ = clean_and_combine_msdial(input_dir, output_dir)
        logger.info("Loaded MS-DIAL dataframe with %d rows.", len(dataframe))
        return dataframe
    raise ValueError(f"Unsupported input_format: {input_format}")


def build_library(
    sources: Sequence[Spectrum] | str | Path | Sequence[str | Path],
    library_path: str | Path,
    *,
    input_format: str = "mgf",
    recursive: bool = False,
    msdial_output_dir: str | Path | None = None,
    storage: str | None = None,
    processor: SpectrumProcessor | None = None,
    overwrite: bool = True,
    vectorizer: Callable[[Spectrum], Mapping[str, float]] | None = None,
) -> LocalSpectralLibrary:
    """
    Build or update a ``LocalSpectralLibrary`` from spectra.
    """
    spectra = _ensure_spectra(
        sources,
        input_format=input_format,
        recursive=recursive,
        msdial_output_dir=msdial_output_dir,
    )
    library = LocalSpectralLibrary(library_path, storage=storage)
    processor = processor or SpectrumProcessor()
    added = 0
    for spectrum in spectra:
        processed = processor.process(spectrum)
        library.add_spectrum(processed, vectorizer=vectorizer or spec2vec_vectorize, overwrite=overwrite)
        added += 1
    logger.info("Added %d spectra to library at %s (total=%d).", added, library.path, len(library))
    return library


def search_library(
    queries: Sequence[Spectrum] | str | Path | Sequence[str | Path],
    library_path: str | Path,
    *,
    input_format: str = "mgf",
    recursive: bool = False,
    msdial_output_dir: str | Path | None = None,
    top_n: int = 5,
    min_score: float = 0.0,
    backend: str = "naive",
    vectorizer: Callable[[Spectrum], Mapping[str, float]] | None = None,
    processor: SpectrumProcessor | None = None,
) -> list[SearchMatch]:
    """
    Search query spectra against a stored library and return flattened matches.
    """
    spectra = _ensure_spectra(
        queries,
        input_format=input_format,
        recursive=recursive,
        msdial_output_dir=msdial_output_dir,
    )
    library = LocalSpectralLibrary(library_path)
    searcher = LibrarySearcher(
        library,
        processor=processor or SpectrumProcessor(),
        vectorizer=vectorizer or spec2vec_vectorize,
        backend=backend,
    )
    results: list[SearchMatch] = []
    for idx, spectrum in enumerate(spectra):
        query_id = (
            spectrum.get("name")
            or spectrum.get("compound_name")
            or spectrum.get("spectrum_id")
            or f"query-{idx}"
        )
        hits = searcher.search_spectrum(spectrum, top_n=top_n, min_score=min_score)
        for hit in hits:
            results.append(
                SearchMatch(
                    query_id=query_id,
                    hit_id=hit.identifier,
                    score=hit.score,
                    precursor_mz=hit.precursor_mz,
                    metadata=hit.metadata,
                )
            )
    logger.info("Completed search for %d queries against %s.", len(spectra), library_path)
    return results


def curate_library(
    input_library_path: str | Path,
    output_library_path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    qc_report_path: str | Path | None = None,
):
    """
    Run quality filtering and de-duplication on a stored library.
    """
    source_library = LocalSpectralLibrary(input_library_path)
    entries = list(source_library.iter_entries())
    result = curation.curate_entries(entries, config=config)

    output_library = LocalSpectralLibrary(output_library_path, storage=source_library.storage)
    output_library.write_entries(result.curated_entries)

    report_path = _qc_report_path(output_library_path, qc_report_path)
    write_curation_report(result, report_path)
    result.summary["output_library"] = str(output_library.path)
    result.summary["qc_report"] = str(report_path)
    logger.info(
        "Curated library -> %s (kept=%d, dropped=%d, merged_groups=%d)",
        output_library.path,
        result.summary["kept"],
        result.summary["dropped"],
        result.summary["merged_groups"],
    )
    return result


def build_network(
    *,
    input_dir: str | Path | None = None,
    library_path: str | Path | None = None,
    metric: str = "cosine",
    threshold: float | None = None,
    knn: int | None = None,
    processor: SpectrumProcessor | None = None,
    reference_mz: Sequence[float] | None = None,
    undirected: bool = True,
    output_path: str | Path | None = None,
):
    """
    Build a similarity network from either spectra on disk or a stored library.
    """
    if not input_dir and not library_path:
        raise ValueError("Provide either input_dir or library_path for network building.")
    if input_dir and library_path:
        raise ValueError("Choose either input_dir or library_path, not both.")
    if library_path and metric in {"cosine", "modified_cosine", "modified-cosine"}:
        raise ValueError("Library input requires a vector-based metric such as 'spec2vec'.")

    if library_path:
        nodes, edges = network_builder.build_network_from_library(
            library_path,
            metric=metric,
            threshold=threshold,
            knn=knn,
            undirected=undirected,
        )
    else:
        nodes, edges = network_builder.build_network_from_folder(
            input_dir,
            metric=metric,
            threshold=threshold,
            knn=knn,
            processor=processor,
            reference_mz=reference_mz,
            undirected=undirected,
        )
    if output_path:
        network_builder.export_network(nodes, edges, output_path, undirected=undirected)
    logger.info("Built network (%d nodes, %d edges).", len(nodes), len(edges))
    return nodes, edges


def run_from_config(config: str | Path | Mapping[str, Any] | WorkflowConfig) -> None:
    """
    Execute a pipeline based on a YAML/JSON configuration file or a loaded mapping.
    """
    cfg = config if isinstance(config, WorkflowConfig) else load_config(config)
    inputs_cfg = cfg.input
    outputs_cfg = cfg.outputs

    input_paths = inputs_cfg.paths
    input_format = inputs_cfg.format
    recursive = inputs_cfg.recursive
    vectorizer_choice = cfg.similarity.vectorizer
    if cfg.network.metric == "embedding":
        vectorizer_choice = "embedding"
    vectorizer = _vectorizer_for_config(vectorizer_choice, cfg.similarity.embedding_model)
    processor = _processor_from_config(cfg.similarity.processor)

    data = load_data(
        input_paths,
        input_format=input_format,
        recursive=recursive,
        msdial_output_dir=inputs_cfg.msdial_output,
    )
    spectra = data if isinstance(data, list) else []

    library_path = cfg.library.path
    library_sources = cfg.library.sources or input_paths
    library_input_format = cfg.library.input_format or input_format
    library_recursive = cfg.library.recursive if cfg.library.recursive is not None else recursive
    library_built = None
    active_library_path = library_path
    if cfg.library.build:
        if not library_path:
            raise ConfigError("library.path", "Library path is required when building a library.")
        library_built = build_library(
            library_sources if library_sources else spectra,
            library_path,
            input_format=library_input_format,
            recursive=library_recursive,
            msdial_output_dir=inputs_cfg.msdial_output,
            storage=cfg.library.storage,
            overwrite=cfg.library.overwrite,
            vectorizer=vectorizer,
            processor=processor,
        )
        active_library_path = library_built.path
    elif library_path:
        active_library_path = library_path

    if cfg.curation.enabled:
        if not active_library_path:
            raise ConfigError("curation.enabled", "Curation enabled but no library path is available.")
        curated_output = cfg.curation.output or outputs_cfg.curated_library or _derive_curated_path(active_library_path)
        qc_report_path = cfg.curation.qc_report or outputs_cfg.qc_report
        curate_library(
            active_library_path,
            curated_output,
            config=cfg.curation.to_dict(),
            qc_report_path=qc_report_path,
        )
        active_library_path = curated_output

    search_results: list[SearchMatch] = []
    search_enabled = bool(cfg.similarity.enabled and active_library_path)
    if search_enabled:
        query_sources = cfg.similarity.queries or input_paths
        try:
            search_results = search_library(
                query_sources if query_sources is not None else spectra,
                active_library_path,
                input_format=cfg.similarity.query_format or input_format,
                recursive=cfg.similarity.recursive if cfg.similarity.recursive is not None else recursive,
                msdial_output_dir=inputs_cfg.msdial_output,
                top_n=cfg.similarity.top_n,
                min_score=cfg.similarity.min_score,
                backend=cfg.similarity.backend,
                vectorizer=vectorizer,
                processor=processor,
            )
        except (ImportError, ValueError) as exc:
            raise ConfigError("similarity.backend", str(exc)) from exc
        results_output = outputs_cfg.search_results or cfg.similarity.output
        if results_output:
            write_search_results(search_results, results_output)

    network_enabled = cfg.network.enabled
    if network_enabled:
        network_output = cfg.network.output or outputs_cfg.network
        undirected = not cfg.network.directed
        net_input_dir = cfg.network.input
        net_library_path = cfg.network.library or active_library_path
        network_metric = cfg.network.metric or (
            "embedding"
            if vectorizer_choice == "embedding"
            else ("spec2vec" if (net_library_path and not net_input_dir) else "cosine")
        )
        nodes, edges = build_network(
            input_dir=net_input_dir,
            library_path=net_library_path if not net_input_dir else None,
            metric=network_metric,
            threshold=cfg.network.threshold,
            knn=cfg.network.knn,
            processor=processor,
            reference_mz=cfg.network.reference_mz,
            undirected=undirected,
            output_path=network_output,
        )
        summary_output = cfg.network.summary or outputs_cfg.network_summary
        if summary_output:
            summary = summarize_network(nodes, edges)
            write_network_summary(summary, summary_output)

    if library_built:
        logger.info("Library written to %s", library_built.path)
    if search_results and not (outputs_cfg.search_results or cfg.similarity.output):
        logger.info("Generated %d search hits (no output file configured).", len(search_results))


def _coerce_paths(inputs: str | Path | Sequence[str | Path] | None) -> list[Path]:
    if inputs is None:
        return []
    if isinstance(inputs, (str, Path)):
        return [Path(inputs)]
    return [Path(item) for item in inputs]


def _iter_files(paths: Iterable[Path], suffixes: set[str], *, recursive: bool) -> Iterable[Path]:
    for base in paths:
        if base.is_dir():
            pattern = "**/*" if recursive else "*"
            for candidate in base.glob(pattern):
                if candidate.is_file() and candidate.suffix.lower() in suffixes:
                    yield candidate
        elif base.is_file():
            if base.suffix.lower() in suffixes:
                yield base
        else:
            raise FileNotFoundError(f"Input path does not exist: {base}")


def _ensure_spectra(
    sources: Sequence[Spectrum] | str | Path | Sequence[str | Path],
    *,
    input_format: str,
    recursive: bool,
    msdial_output_dir: str | Path | None = None,
) -> list[Spectrum]:
    if isinstance(sources, Sequence) and sources and isinstance(sources[0], Spectrum):  # type: ignore[index]
        return list(sources)  # type: ignore[arg-type]
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a core dependency
        pd = None  # type: ignore
    if pd is not None and isinstance(sources, pd.DataFrame):
        from yogimass.io.msdial_process import msdial_dataframe_to_spectra

        return msdial_dataframe_to_spectra(sources)
    loaded = load_data(
        sources,
        input_format=input_format,
        recursive=recursive,
        msdial_output_dir=msdial_output_dir,
    )
    if not isinstance(loaded, list):
        if pd is not None and isinstance(loaded, pd.DataFrame):
            from yogimass.io.msdial_process import msdial_dataframe_to_spectra

            return msdial_dataframe_to_spectra(loaded)
        raise ValueError(f"Expected spectra list, got {type(loaded)} for format '{input_format}'.")
    return loaded


def _derive_curated_path(library_path: str | Path) -> Path:
    base = Path(library_path)
    suffix = base.suffix or ".json"
    return base.with_name(f"{base.stem}_curated{suffix}")


def _qc_report_path(output_library_path: str | Path, explicit_report: str | Path | None) -> Path:
    if explicit_report is not None:
        return Path(explicit_report)
    base = Path(output_library_path)
    return base.with_name(f"{base.stem}_qc.json")


def _vectorizer_for_config(vectorizer_name: str, embedding_model: str | None):
    if vectorizer_name == "embedding":
        from yogimass.similarity.embeddings import embedding_vectorizer

        model = embedding_model or "spec2vec-lite"
        return lambda spectrum: embedding_vectorizer(spectrum, model_name=model)
    return spec2vec_vectorize


def _processor_from_config(processor_cfg: ProcessorConfig | None) -> SpectrumProcessor:
    cfg = processor_cfg or ProcessorConfig()
    return SpectrumProcessor(
        normalization=cfg.normalization,
        min_relative_intensity=cfg.min_relative_intensity,
        min_absolute_intensity=cfg.min_absolute_intensity,
        max_peaks=cfg.max_peaks,
        mz_dedup_tolerance=cfg.mz_dedup_tolerance,
        float_dtype=cfg.float_dtype,
    )


__all__ = [
    "build_library",
    "build_network",
    "curate_library",
    "load_data",
    "run_from_config",
    "search_library",
]
