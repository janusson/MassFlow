"""
Workflow orchestration engine for MassFlow.
Executes the processing pipeline based on the provided configuration.
"""

import csv
import logging
from pathlib import Path
from typing import Any, List, Optional

from matchms import Scores, Spectrum
from matchms.importing import load_from_mgf, load_from_msp
from pydantic import ValidationError

from MassFlow.config import MassFlowConfig
from MassFlow.processing import metadata_processing, peak_processing
from MassFlow.similarity import SimilarityCalculator, get_similarity_calculator

logger = logging.getLogger(__name__)


def load_spectra(file_path: Path, file_format: str) -> List[Spectrum]:
    """
    Load spectral data from a file.

    Args:
        file_path: Path to the spectral data file.
        file_format: Format of the file ('mgf', 'msp', 'mzml').

    Returns:
        List of matchms Spectrum objects.

    Raises:
        ValueError: If the format is not supported.
    """
    path_str = str(file_path)
    fmt = file_format.lower()

    if fmt == "mgf":
        spectra = list(load_from_mgf(path_str))
    elif fmt == "msp":
        spectra = list(load_from_msp(path_str))
    else:
        # mzml loading would typically use load_from_mzml if available in matchms
        raise ValueError(f"Unsupported format: {fmt}")

    return [s for s in spectra if s is not None]


def process_spectrum(spectrum: Spectrum, config: MassFlowConfig) -> Optional[Spectrum]:
    """Apply metadata and peak processing to a single spectrum."""
    if config.processing.clean_metadata:
        spectrum = metadata_processing(spectrum)

    if spectrum is None:
        return None

    spectrum = peak_processing(
        spectrum,
        min_intensity=config.processing.min_intensity,
        normalize=config.processing.normalize_intensity,
    )

    return spectrum


def save_results(results: List[dict[str, Any]], output_dir: Path) -> Path:
    """Save the similarity search results to a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.csv"

    headers = [
        "Query_ID",
        "Query_Name",
        "Match_Name",
        "Score",
        "Matches",
        "Smiles",
        "InChIKey",
    ]

    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    return results_file


def run_workflow(config_path: str | Path) -> None:
    """
    Execute the MassFlow pipeline.

    Args:
        config_path: Path to the YAML configuration file.
    """
    try:
        # 1. Configuration Loading
        config = MassFlowConfig.from_yaml(config_path)
        logger.info(f"Loaded configuration from {config_path}")

        # 2. Strategy Initialization
        calculator = get_similarity_calculator(config.similarity)
        # Determine the score name based on the algorithm for dynamic extraction
        score_name = (
            "CosineGreedy_score"
            if config.similarity.algorithm == "cosine"
            else "ModifiedCosine_score"
        )
        matches_name = (
            "CosineGreedy_matches"
            if config.similarity.algorithm == "cosine"
            else "ModifiedCosine_matches"
        )

        # 3. Reference Library Preparation
        reference_spectra: List[Spectrum] = []
        if config.input.reference_library:
            logger.info(f"Loading reference library: {config.input.reference_library}")
            ref_raw = load_spectra(
                config.input.reference_library,
                config.input.reference_library.suffix.lstrip("."),
            )
            reference_spectra = [
                s
                for s in (process_spectrum(spec, config) for spec in ref_raw)
                if s is not None
            ]
            logger.info(f"Processed {len(reference_spectra)} reference spectra.")

        # 4. Query Data Ingestion & Processing
        logger.info(f"Loading query spectra: {config.input.file_path}")
        query_raw = load_spectra(config.input.file_path, config.input.format)

        results = []
        processed_count = 0

        for query_spec in query_raw:
            processed_query = process_spectrum(query_spec, config)
            if processed_query is None:
                continue

            processed_count += 1

            # 5. Similarity Search
            if reference_spectra:
                scores: Scores = calculator.calculate(
                    reference_spectra, [processed_query]
                )

                # Get top hits for this specific query
                best_matches = scores.scores_by_query(processed_query, sort=True)

                if best_matches:
                    top_hit_spectrum, score_data = best_matches[0]
                    current_score = score_data[score_name]

                    if current_score >= config.similarity.min_score:
                        results.append(
                            {
                                "Query_ID": processed_query.get("id", "N/A"),
                                "Query_Name": processed_query.get(
                                    "compound_name",
                                    processed_query.get("name", "Unknown"),
                                ),
                                "Match_Name": top_hit_spectrum.get(
                                    "compound_name",
                                    top_hit_spectrum.get("name", "Unknown"),
                                ),
                                "Score": f"{current_score:.4f}",
                                "Matches": score_data[matches_name],
                                "Smiles": top_hit_spectrum.get("smiles", ""),
                                "InChIKey": top_hit_spectrum.get("inchikey", ""),
                            }
                        )

        # 6. Save Results
        if results:
            out_path = save_results(results, config.output_directory)
            logger.info(
                f"Processed {processed_count} spectra. Results saved to {out_path}"
            )
        else:
            logger.info(
                f"Processed {processed_count} spectra. No matches found above threshold."
            )

    except ValidationError as e:
        logger.critical(f"Configuration validation failed: {e}")
        raise
    except FileNotFoundError as e:
        logger.critical(f"Required file not found: {e}")
        raise
    except Exception as e:
        logger.critical(f"An unexpected error occurred during workflow execution: {e}")
        raise
