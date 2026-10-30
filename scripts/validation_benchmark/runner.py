"""Observe the existing annotation pipeline; do not implement alternative matching."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import platform

import numpy as np
import resource
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

from MassFlow import io, processing, workflow
from MassFlow.config import MassFlowConfig
from MassFlow.similarity import SimilarityEngine
from scripts.validation_benchmark import dataset, metrics
from scripts.benchmark_reporting import attribute_truth_candidates


def _scalar(value: Any) -> Any:
    """Convert production NumPy scalars, preserving strict JSON nonfinite rejection."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported result type: {type(value).__name__}")


def _dump(path: Path, value: Any) -> None:
    """Write strict JSON; missing scientific values must be null, not NaN."""
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, sort_keys=True, default=_scalar)
        + "\n"
    )


def verify_parameters(parameters: dict[str, Any], expected: str) -> None:
    """Reject unsupported or altered frozen algorithm configuration.

    Parameters
    ----------
    parameters : dict
        Frozen benchmark configuration.
    expected : str
        Independently recorded canonical SHA256.

    Returns
    -------
    None
        Raises ValueError on schema drift or tampering.

    Examples
    --------
    >>> # verify_parameters(parameters, recorded_digest)
    """
    if (
        parameters.get("schema_version") != 1
        or dataset.manifest_digest(parameters) != expected
    ):
        raise ValueError("Frozen parameters schema/digest mismatch")


def _git(*arguments: str) -> str:
    """Read local source identity without modifying Git state."""
    return subprocess.check_output(["git", *arguments], text=True).strip()


def _environment() -> dict[str, Any]:
    """Resolve distribution names correctly instead of looking up requirement strings."""
    versions: dict[str, str | None] = {}
    # ``massflow`` is the public/provenance key; the installable distribution is
    # ``massflow-msms`` because the PyPI name ``massflow`` is owned by an
    # unrelated project.
    distributions = {
        "massflow": "massflow-msms",
        "matchms": "matchms",
        "numpy": "numpy",
        "scipy": "scipy",
        "polars": "polars",
        "pyteomics": "pyteomics",
        "pydantic": "pydantic",
        "rdkit": "rdkit",
    }
    for key, name in distributions.items():
        try:
            versions[key] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = None
    source_root = Path(__file__).resolve().parents[2]
    return {
        "application_commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "tracked_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": versions,
        "lock_sha256": hashlib.sha256(
            (source_root / "uv.lock").read_bytes()
        ).hexdigest(),
        "pipeline": "MassFlow.workflow.run_annotation_pipeline",
        "seed": 42,
        "harness_source_sha256": {
            str(path.relative_to(source_root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
    }


def run_arm(
    queries: list[dataset.Record],
    library: list[dataset.Record],
    settings: dict[str, Any],
    output: Path,
    *,
    absent: set[str],
    excluded_accessions: set[str] | None = None,
    processing_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run production ingestion, processing, matching, estimation and native export.

    Parameters
    ----------
    queries : list[Record]
        Blinded queries selected before scoring.
    library : list[Record]
        Reference records with frozen source accessions and declared labels.
    settings : dict
        Exact production SimilarityConfig overrides.
    output : Path
        New empty output directory; existing results are never overwritten.
    absent : set[str]
        Connectivity groups deliberately absent from this library.

    Returns
    -------
    dict
        Machine-readable results, per-query outcomes and actual run metadata.

    Examples
    --------
    >>> # run_arm(queries, references, settings, Path('new-run'), absent=set())
    """
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Refusing to overwrite benchmark run: {output}")
    output.mkdir(parents=True)
    query_spectra = [
        record.spectrum(f"query_{i:04d}", blind=True)
        for i, record in enumerate(queries)
    ]
    references = [record.spectrum(record.accession) for record in library]
    io.save_spectra_to_mgf(query_spectra, output / "query.mgf")
    io.save_spectra_to_msp(references, output / "library.msp")
    config = MassFlowConfig.model_validate(
        {
            "project": {"name": output.name, "output_directory": output / "native"},
            "input": {
                "input_path": output / "query.mgf",
                "library_path": output / "library.msp",
                "format": "mgf",
            },
            "processing": processing_settings
            or {
                "filter_by_mz": False,
                "filter_min_peaks": True,
                "min_peaks": 3,
                "repair_inchi_inchikey_smiles": False,
            },
            "similarity": {
                **settings,
                "collect_retrieval_diagnostics": True,
            },
            "export": {"format": "csv"},
        }
    )
    # Disabling chemical-string repair prevents answer inference from blinded names;
    # real formula/adduct/5-ppm validation still executes on reference records.
    _dump(output / "effective_config.json", config.model_dump(mode="json"))
    captured: list[dict[str, Any]] = []
    seen: dict[str, set[str]] = defaultdict(set)
    reasons: dict[str, list[str]] = defaultdict(list)
    depth = 0
    original_search = SimilarityEngine.search
    original_load = io.load_spectra
    original_process = processing.process_spectra
    original_metadata = processing.metadata_processing
    original_peak = processing.peak_processing
    original_physics = processing.physical_integrity_reason

    def observe_search(engine: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal depth
        depth += 1
        try:
            result = original_search(engine, *args, **kwargs)
            if depth == 1:
                captured.extend(copy.deepcopy(result))
            return result
        finally:
            depth -= 1

    def observe_load(*args: Any, **kwargs: Any) -> Any:
        for spectrum in original_load(*args, **kwargs):
            seen["loaded"].add(str(spectrum.get("id")))
            yield spectrum

    def observe_process(*args: Any, **kwargs: Any) -> Any:
        for spectrum in original_process(*args, **kwargs):
            seen["processed"].add(str(spectrum.get("id")))
            yield spectrum

    def observe_metadata(spectrum: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_metadata(spectrum, *args, **kwargs)
        if result is None and spectrum is not None:
            reasons[str(spectrum.get("id"))].append("metadata_processing")
        return result

    def observe_peak(spectrum: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_peak(spectrum, *args, **kwargs)
        if result is None and spectrum is not None:
            reasons[str(spectrum.get("id"))].append("peak_processing")
        return result

    def observe_physics(spectrum: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_physics(spectrum, *args, **kwargs)
        if result:
            reasons[str(spectrum.get("id"))].append("physical_integrity:" + result)
        return result

    begin = time.perf_counter()
    error = None
    results = []
    try:
        with (
            patch.object(SimilarityEngine, "search", observe_search),
            patch.object(io, "load_spectra", observe_load),
            patch.object(processing, "process_spectra", observe_process),
            patch.object(processing, "metadata_processing", observe_metadata),
            patch.object(processing, "peak_processing", observe_peak),
            patch.object(processing, "physical_integrity_reason", observe_physics),
        ):
            results = workflow.run_annotation_pipeline(config)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    elapsed = time.perf_counter() - begin
    processed = results[0].query_spectra if results else []
    scored_ids = (
        {str(s.get("id")) for s in processed}
        if results and results[0].status != "failed"
        else set()
    )
    reference_map = {r.accession: r for r in library}
    for row in captured:
        reference_id = row["reference_id"]
        base = (
            reference_id[:-6]
            if row.get("is_decoy") and reference_id.endswith("_decoy")
            else reference_id
        )
        if base not in reference_map:
            raise ValueError(f"Untraceable reference ID: {reference_id}")
        row["inchikey"] = reference_map[base].inchikey
        # Q values are attached by the workflow to copies, not the observed search rows.
    exported = {
        r["query_id"]: r.get("q_value") for result in results for r in result.results
    }
    for row in captured:
        row["q_value"] = exported.get(row["query_id"])
    accounting = []
    for i, record in enumerate(queries):
        identifier = f"query_{i:04d}"
        why = list(reasons.get(identifier, []))
        if identifier in scored_ids:
            state = "scored"
        elif identifier in seen["loaded"] and identifier not in seen["processed"]:
            state = "processing_rejected"
            if not why:
                why.append(
                    "minimum_peak_count"
                    if record.mz.size < config.processing.min_peaks
                    else "processing_drop_unclassified"
                )
        elif error or (results and results[0].status == "failed"):
            state = "pipeline_failed"
        elif identifier not in seen["loaded"]:
            state = "io_validation_rejected"
            why.append("io_validation")
        else:
            state = "processing_rejected"
            if not why:
                why.append(
                    "minimum_peak_count"
                    if record.mz.size < 3
                    else "processing_drop_unclassified"
                )
        accounting.append(
            {
                "query_id": identifier,
                "accession": record.accession,
                "inchikey": record.inchikey,
                "connectivity": record.connectivity,
                "eligible": record.accession not in (excluded_accessions or set()),
                "present": record.connectivity not in absent,
                "state": state,
                "reasons": why
                + (
                    ["potential_spectral_leakage"]
                    if record.accession in (excluded_accessions or set())
                    else []
                ),
                "stratum": record.metadata["instrument_type"]
                + " | "
                + record.metadata["collision_energy"],
                "polarity": record.metadata["polarity"],
                "adduct": record.metadata["adduct"],
                "peak_count": len(record.mz),
                "source_sha256": record.source_sha256,
            }
        )
    reference_ledger = []
    for record in library:
        identifier = record.accession
        why = list(reasons.get(identifier, []))
        if identifier in seen["processed"]:
            state = "processed"
        elif identifier not in seen["loaded"]:
            state = "io_validation_rejected"
            why.append("io_validation")
        else:
            state = "processing_rejected"
            if not why:
                why.append(
                    "minimum_peak_count"
                    if record.mz.size < config.processing.min_peaks
                    else "processing_drop_unclassified"
                )
        reference_ledger.append(
            {
                "accession": identifier,
                "inchikey": record.inchikey,
                "source_path": record.source_path,
                "source_sha256": record.source_sha256,
                "source_url": f"https://raw.githubusercontent.com/MassBank/MassBank-data/{dataset.SOURCE_COMMIT}/{record.source_path}",
                "metadata": record.metadata,
                "state": state,
                "reasons": why,
            }
        )
    provenance = _environment()
    provenance.update(
        {
            "wall_seconds": elapsed,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
            "memory_scope": "whole_process_high_water_including_earlier_arms",
            "query_sha256": hashlib.sha256(
                (output / "query.mgf").read_bytes()
            ).hexdigest(),
            "library_sha256": hashlib.sha256(
                (output / "library.msp").read_bytes()
            ).hexdigest(),
            "effective_config": config.model_dump(mode="json"),
        }
    )
    fdr = results[0].fdr_summary if results else None
    calibration: dict[str, Any] = {
        "state": "estimated_uncalibrated"
        if fdr
        and fdr.get("n_decoy_competitions", 0) > 0
        and fdr.get("n_target_competitions", 0) > 0
        else "unavailable",
        "empirical_calibration": "pending",
        "counts": fdr,
        "method": "actual_production_pooled_query_maxima",
        "limitations": [
            "source_standard_labels_not_independently_authenticated",
            "no_independent_deployment_validation",
            "decoy_exchangeability_unvalidated",
            "query_q_not_candidate_specific",
        ],
        "threshold_diagnostics": [],
    }
    for threshold in (0.01, 0.05, 0.1, 1.0):
        accepted = [
            r
            for r in captured
            if r.get("q_value") is not None and r["q_value"] <= threshold
        ]
        calibration["threshold_diagnostics"].append(
            {
                "legacy_q_threshold": threshold,
                "counts": metrics.evaluate(accounting, accepted)["counts"],
                "observed_fdp": metrics.evaluate(accounting, accepted)["observed_fdp"],
                "interpretation": "descriptive_observed_FDP_only_not_error_control",
            }
        )
    reference_rejection_reasons: dict[str, list[str]] = {
        key: value for key, value in reasons.items() if key in reference_map
    }
    payload = {
        "schema_version": 1,
        "metrics": metrics.evaluate(accounting, captured),
        "queries": accounting,
        "calibration": calibration,
        "provenance": provenance,
        "error": error,
        "file_status": results[0].status if results else "failed",
        "fatal_errors": (
            results[0].fatal_errors
            if results
            else [error]
            if error is not None
            else ["No per-file result was returned."]
        ),
        "warnings": results[0].warnings if results else [],
        "candidate_rows": len(captured),
        "retrieval_diagnostics": (
            results[0].retrieval_diagnostics if results else None
        ),
        "reference_records_after_processing": len(
            seen["processed"] & set(reference_map)
        ),
        "reference_accounting": {
            "attempted": len(library),
            "states": dict(Counter(r["state"] for r in reference_ledger)),
            "rejection_reasons": dict(
                Counter(reason for r in reference_ledger for reason in r["reasons"])
            ),
        },
        "filter_observability": {
            "measured": [
                "io_validation",
                "metadata_processing",
                "peak_processing",
                "physical_integrity",
                "minimum_peak_count",
                "qualified_candidate_presence",
            ],
            "not_separately_observed": [
                "all_precursor_candidate_pairs",
                "all_adduct_candidate_pairs",
                "score_only_vs_matched_peak_only_rejections",
            ],
            "limitation": "Search exports qualified hits only. Empty candidate sets cannot distinguish precursor, score and peak-gate losses; no per-gate counts are inferred.",
        },
        "reference_rejection_reasons": reference_rejection_reasons,
    }
    truth_labels = {
        f"query_{query_index:04d}": {
            "inchikey": record.inchikey,
            "connectivity": record.connectivity,
            "source": str(record.metadata.get("instrument_type", "unknown")),
            "peak_count": int(record.mz.size),
            "true_lib_records": [
                reference.accession
                for reference in library
                if reference.inchikey == record.inchikey
            ],
        }
        for query_index, record in enumerate(queries)
    }
    truth_attribution = attribute_truth_candidates(
        truth_labels,
        results[0].results if results else [],
        config=config,
        prepared_library_path=str(output / "native" / "library_library.db"),
        run_diagnostics=(results[0].retrieval_diagnostics if results else None),
        identity_endpoint="inchikey",
    )
    accounting_by_id = {row["query_id"]: row for row in accounting}
    eligible_attributions: Counter[str] = Counter()
    for query_id, query_attribution in truth_attribution.get("by_query", {}).items():
        query_accounting = accounting_by_id[query_id]
        query_attribution["truth_identity_endpoint"] = "full InChIKey"
        query_attribution["benchmark_eligible"] = query_accounting["eligible"]
        query_attribution["truth_library_present_by_connectivity"] = query_accounting[
            "present"
        ]
        query_attribution["truth_reference_preparation_rejections"] = {
            reference_id: reference_rejection_reasons.get(reference_id, [])
            for reference_id in query_attribution.get(
                "truth_reference_ids_missing_after_preparation", []
            )
        }
        if query_accounting["eligible"]:
            eligible_attributions[str(query_attribution["attribution"])] += 1
    truth_attribution["primary_eligible_classification_counts"] = dict(
        sorted(eligible_attributions.items())
    )
    payload["truth_attribution"] = truth_attribution

    for name, values in [
        ("queries", accounting),
        ("candidates", captured),
        ("references", reference_ledger),
    ]:
        (output / f"{name}.jsonl").write_text(
            "".join(
                json.dumps(row, allow_nan=False, sort_keys=True, default=_scalar) + "\n"
                for row in values
            )
        )
    _dump(output / "result.json", payload)
    return payload


def _main() -> None:
    """Freeze a real-data cohort or execute its pinned selection through production."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=False)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--parameters-sha256")
    args = parser.parse_args()
    records, inventory = dataset._load_archive(dataset._fetch(args.cache))
    queries, library, selection = dataset._select(records, 20, 100)
    if args.command == "freeze":
        if args.manifest.exists():
            raise ValueError("Refusing to overwrite frozen manifest")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "dataset_id": "massbank-ae-diagnostic-v1",
            "source": {
                "url": dataset.SOURCE_URL,
                "commit": dataset.SOURCE_COMMIT,
                "sha256": dataset.SOURCE_SHA256,
                "license": "CC BY or CC0 per included record",
                "accessed_on_utc": time.strftime("%Y-%m-%d", time.gmtime()),
            },
            "selection": selection,
            "criteria": {
                "query_source": "Athens_Univ",
                "reference_source": "Eawag",
                "confidence": "standard compound",
                "polarity": "POSITIVE",
                "adduct": "[M+H]+",
                "ms_level": "MS2",
                "split_unit": "connectivity",
                "selection_order": "SHA256(massflow-mvb1-v1|connectivity)",
                "study_state": "diagnostic_not_confirmatory",
                "protocol_amendments": [
                    "Restrict source laboratories prospectively to Athens queries and Eawag references.",
                    "Deposited processed centroid peak lists; raw centroiding and independent standard identity are not universally verifiable.",
                    "Numba prefilter disabled because main has known mixed-frame defect; no thresholds tuned to outcomes.",
                    "Diagnostic data execution is allowed despite unresolved release qualification.",
                ],
            },
            "queries": [
                {
                    "accession": r.accession,
                    "inchikey": r.inchikey,
                    "source_path": r.source_path,
                    "sha256": r.source_sha256,
                    "metadata": r.metadata,
                }
                for r in queries
            ],
            "reference_accessions": [r.accession for r in library],
            "inventory_counts": inventory["counts"],
        }
        _dump(args.manifest, manifest)
        print(dataset.manifest_digest(manifest))
        return
    if not args.parameters or not args.parameters_sha256:
        raise ValueError(
            "Require frozen --parameters and independent --parameters-sha256"
        )
    parameters = json.loads(args.parameters.read_text())
    verify_parameters(parameters, args.parameters_sha256)
    manifest = json.loads(args.manifest.read_text())
    if not args.manifest_sha256:
        raise ValueError("Require independent --manifest-sha256 pin")
    dataset.verify_manifest(manifest, args.manifest_sha256)
    if [r.accession for r in queries] != [q["accession"] for q in manifest["queries"]]:
        raise ValueError("Source selection changed")
    if [r.accession for r in library] != manifest["reference_accessions"]:
        raise ValueError("Library membership changed")
    if args.output.exists():
        raise ValueError("Refusing to overwrite benchmark output")
    args.output.mkdir(parents=True)
    _dump(args.output / "inventory.json", inventory)
    leakage = dataset.audit_leakage(queries, library)
    _dump(args.output / "leakage.json", leakage)
    excluded = {
        pair["query_accession"]
        for pair in leakage["exact_cross_pairs"] + leakage["near_cross_pairs"]
    }
    summaries = {}
    for arm in parameters["arms"]:
        absent = set() if arm == "present" else set(selection["absent_groups"])
        selected_library = [r for r in library if r.connectivity not in absent]
        for algorithm in parameters["algorithms"]:
            name = f"{arm}-{algorithm}"
            summaries[name] = run_arm(
                queries,
                selected_library,
                {**parameters["similarity"], "algorithm": algorithm},
                args.output / name,
                absent=absent,
                excluded_accessions=excluded,
                processing_settings=parameters["processing"],
            )
    _dump(
        args.output / "summary.json",
        {
            "schema_version": 1,
            "manifest_digest": args.manifest_sha256,
            "parameters_digest": args.parameters_sha256,
            "execution_status": "failed"
            if any(v["file_status"] == "failed" for v in summaries.values())
            else "completed",
            "runs": summaries,
        },
    )
    print(
        json.dumps({k: v["metrics"]["counts"] for k, v in summaries.items()}, indent=2)
    )

    if any(v["file_status"] == "failed" for v in summaries.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _main()
