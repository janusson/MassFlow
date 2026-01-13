import os
import yaml
import sys
from matchms.importing import load_from_mgf
from src.dataloader import LibraryLoader
from src.engine import Searcher
from src.utils import get_logger
from src.filters import apply_filters

logger = get_logger(__name__)

def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config_path = "config/settings.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    logger.info("Initializing Yogimass...")
    config = load_config(config_path)

    # 1. Load Reference Library
    loader = LibraryLoader(config)
    ref_file = config['data'].get('reference_library_file')
    if not ref_file:
         logger.error("No reference library file specified in config.")
         sys.exit(1)

    try:
        library = loader.load_library(ref_file)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    if not library:
        logger.error("Loaded library is empty.")
        sys.exit(1)

    # 2. Load Query Spectrum (Demo: Load first spectrum from query file or just use one from library for testing)
    query_file = config['data'].get('query_file')
    query_spectrum = None

    if query_file:
        query_path = os.path.join(config['data']['raw_library_path'], query_file)
        if os.path.exists(query_path):
             # Just load the first one for demo
             try:
                query_spectra = list(load_from_mgf(query_path))
                if query_spectra:
                    query_spectrum = query_spectra[0]
                    # Apply filters to query as well
                    query_spectrum = apply_filters(query_spectrum, config['filters'])
             except Exception as e:
                 logger.warning(f"Could not load query file: {e}")

    if query_spectrum is None:
        logger.info("No query file found or loaded. Using the first spectrum from reference library as query.")
        if library:
            query_spectrum = library[0]
        else:
            logger.error("No spectra available to use as query.")
            sys.exit(1)

    # 3. Initialize Search Engine
    searcher = Searcher(config)

    # 4. Perform Search
    logger.info(f"Searching for matches for query spectrum (ID: {query_spectrum.metadata.get('spectrum_id', 'N/A')})...")
    matches = searcher.search(query_spectrum, library, n_best=5)

    # 5. Output Results
    print("\n--- Top 5 Matches ---")
    for i, (match, score) in enumerate(matches):
        match_id = match.metadata.get('spectrum_id') or match.metadata.get('title') or "Unknown ID"
        # Score might be a numpy array or tuple depending on matchms version
        # For CosineGreedy, it returns (score, matches)
        try:
             # Try to convert to float directly
             score_val = float(score)
        except (TypeError, ValueError):
             # If that fails, it might be a tuple or numpy void/record
             if isinstance(score, tuple):
                 score_val = float(score[0])
             elif hasattr(score, 'dtype') and score.dtype.names:
                  # It's a structured numpy scalar (e.g. from matchms scores)
                  score_val = float(score[0])
             else:
                  # Fallback
                  score_val = 0.0

        print(f"{i+1}. ID: {match_id} | Score: {score_val:.4f}")

if __name__ == "__main__":
    main()
