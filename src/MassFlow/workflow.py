"""
Workflow orchestration engine for MassFlow.
Executes the processing pipeline based on the provided configuration.
"""

import logging
from pathlib import Path
from typing import List

from matchms import Scores, Spectrum
from pydantic import ValidationError

from MassFlow import io, processing
from MassFlow.config import MassFlowConfig
from MassFlow.similarity import get_similarity_calculator

logger = logging.getLogger(__name__)


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
            ref_raw = io.load_spectra(
                config.input.reference_library,
                config.input.reference_library.suffix.lstrip("."),
            )
            # Use process_spectra for batch processing
            reference_spectra = list(
                processing.process_spectra(ref_raw, config.processing)
            )
            logger.info(f"Processed {len(reference_spectra)} reference spectra.")

        # 4. Query Data Ingestion & Processing
        logger.info(f"Loading query spectra: {config.input.file_path}")
        query_raw = io.load_spectra(config.input.file_path, config.input.format)

        # Use process_spectra to handle the entire collection
        processed_query_spectra = processing.process_spectra(
            query_raw, config.processing
        )

        results = []
        processed_count = 0

        for processed_query in processed_query_spectra:
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
                    current_matches = score_data[matches_name]

                    if (
                        current_score >= config.similarity.min_score
                        and current_matches >= config.similarity.min_matched_peaks
                    ):
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
                                "Matches": current_matches,
                                "Smiles": top_hit_spectrum.get("smiles", ""),
                                "InChIKey": top_hit_spectrum.get("inchikey", ""),
                            }
                        )

        # 6. Save Results
        if results:
            out_path = config.output_directory / "results.csv"
            io.save_match_results(results, out_path)
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
