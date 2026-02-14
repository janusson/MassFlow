"""
Workflow orchestration for MassFlow.
"""

import logging
from pathlib import Path

from MassFlow import io
from MassFlow.similarity import SimilarityEngine

logger = logging.getLogger(__name__)


def run_annotation_pipeline(
    experimental_path: Path,
    reference_path: Path,
    output_directory: Path,
    min_score: float = 0.7,
    tolerance: float = 0.01
) -> None:
    """
    Executes the full annotation pipeline:
    1. Load Experimental Data (using robust sanitization)
    2. Load Reference Library
    3. Perform Similarity Search
    4. Save Results
    """
    
    # 1. Load Experimental Data
    logger.info(f"Loading experimental data from: {experimental_path}")
    exp_fmt = experimental_path.suffix.lstrip(".")
    # Using list() to materialize generator immediately for processing
    query_spectra = list(io.load_spectra(experimental_path, exp_fmt))
    logger.info(f"Loaded {len(query_spectra)} query spectra.")

    # 2. Load Reference Library
    logger.info(f"Loading reference library from: {reference_path}")
    ref_fmt = reference_path.suffix.lstrip(".")
    reference_spectra = list(io.load_spectra(reference_path, ref_fmt))
    logger.info(f"Loaded {len(reference_spectra)} reference spectra.")

    # 3. Similarity Search
    logger.info("Starting similarity search...")
    engine = SimilarityEngine(tolerance=tolerance)
    results = engine.search(
        query_spectra=query_spectra,
        reference_spectra=reference_spectra,
        min_score=min_score,
        top_n=5
    )
    logger.info(f"Found {len(results)} matches.")

    # 4. Save Results
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Determine Output Filename based on input
    output_filename = experimental_path.stem + "_annotated.csv"
    output_path = output_directory / output_filename
    
    io.save_match_results(results, output_path)
    logger.info(f"Pipeline complete. Results saved to {output_path}")
