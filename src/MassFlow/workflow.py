"""
Workflow orchestration engine for MassFlow.
Executes the processing pipeline based on the provided configuration.
"""
import logging
from typing import Iterator
import datetime
from matchms.importing import load_from_mgf, load_from_msp
from matchms import Spectrum

from MassFlow.config import MassFlowConfig
from MassFlow.processing import metadata_processing, peak_processing
from MassFlow import similarity
import csv

logger = logging.getLogger(__name__)

def load_data(config: MassFlowConfig) -> Iterator[Spectrum]:
    """
    Load spectral data based on configuration.

    Args:
        config: The MassFlow configuration object.

    Returns:
        Iterator over Loaded Spectrum objects.

    Raises:
        ValueError: If the format specified in config is not supported.
    """
    path = str(config.input.file_path)
    fmt = config.input.format.lower()
    
    if fmt == "mgf":
        return load_from_mgf(path)
    elif fmt == "msp":
        return load_from_msp(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

def run_workflow(config: MassFlowConfig):
    """
    Execute the MassFlow pipeline.
    
    Args:
        config: The configuration object.
    """
    logger.info("Starting MassFlow workflow...")
    
    logger.info("Starting MassFlow workflow...")
    
    # 2. Preparation: Load Reference Library
    reference_spectra = []
    if config.input.reference_library:
        ref_path = config.input.reference_library
        logger.info(f"Loading reference library from {ref_path}...")
        
        # We use the generic load_data function, repurposing a temp config or just calling imports directly
        # To stay clean, we'll use matchms directly here, but applying our processing
        from matchms.importing import load_from_msp, load_from_mgf
        from MassFlow.processing import process_spectra
            
        if str(ref_path).endswith(".mgf"):
             ref_iter = load_from_mgf(str(ref_path))
        else:
             ref_iter = load_from_msp(str(ref_path))
             
        # Load into memory for search
        reference_spectra = list(process_spectra(ref_iter))
        logger.info(f"Loaded {len(reference_spectra)} reference spectra.")
            
    # Prepare Output CSV
    results_file = config.output_directory / "results.csv"
    if not config.output_directory.exists():
        config.output_directory.mkdir(parents=True, exist_ok=True)
        
    csv_file = open(results_file, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Query_ID", "Query_Name", "Match_Name", "Score", "Matches", "Smiles", "InChIKey"])

    # 3. Ingestion
    spectra = load_data(config)
    
    # 4. Processing
    processed_count = 0
    for spectrum in spectra:
        # Metadata cleaning
        if config.processing.clean_metadata:
            spectrum = metadata_processing(spectrum)
        
        if spectrum is None:
            continue

        # Peak filtering
        spectrum = peak_processing(
            spectrum,
            min_intensity=config.processing.min_intensity,
            # Mapping config fields to processing args
            # Note: config might need more fields to fully match processing capability
            normalize=config.processing.normalize_intensity
        )
        
        if spectrum is None:
            continue
            
        processed_count += 1
        
        # Similarity Search
        if reference_spectra:
             scores = similarity.calculate_cosscores(reference_spectra, [spectrum], tolerance=config.similarity.tolerance)
             # Get top hit
             best_matches = scores.scores_by_query(spectrum, "CosineGreedy_score", sort=True)
             
             if best_matches:
                 top_hit = best_matches[0]
                 match_spectrum = top_hit[0]
                 score_data = top_hit[1]
                 
                 if score_data['CosineGreedy_score'] >= config.similarity.min_score:
                     csv_writer.writerow([
                         spectrum.get("id", "N/A"),
                         spectrum.get("compound_name", spectrum.get("name", "Unknown")),
                         match_spectrum.get("compound_name", match_spectrum.get("name", "Unknown")),
                         f"{score_data['CosineGreedy_score']:.4f}",
                         score_data['CosineGreedy_matches'],
                         match_spectrum.get("smiles", ""),
                         match_spectrum.get("inchikey", "")
                     ])
        
    logger.info(f"Processed {processed_count} spectra. Results saved to {results_file}")
    csv_file.close()

