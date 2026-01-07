import os
import pickle
from typing import List, Generator
from matchms.importing import load_from_mgf, load_from_msp
from matchms import Spectrum
from src.utils import get_logger
from src.filters import apply_filters

logger = get_logger(__name__)

class LibraryLoader:
    def __init__(self, config: dict):
        self.config = config
        self.raw_path = config['data']['raw_library_path']
        self.cache_path = config['data']['cache_path']
        self.filters = config['filters']

    def load_library(self, filename: str) -> List[Spectrum]:
        """
        Loads a spectral library. Checks cache first, otherwise processes raw file.
        """
        base_name = os.path.splitext(filename)[0]
        cache_file = os.path.join(self.cache_path, f"{base_name}.pickle")
        raw_file = os.path.join(self.raw_path, filename)

        # Check cache
        if os.path.exists(cache_file):
            logger.info(f"Loading cached library from {cache_file}...")
            try:
                with open(cache_file, 'rb') as f:
                    library = pickle.load(f)
                logger.info(f"Loaded {len(library)} spectra from cache.")

                # Apply filters to cached library
                logger.info("Applying filters to cached library...")
                filtered_library = []
                for spectrum in library:
                    processed_spectrum = apply_filters(spectrum, self.filters)
                    if processed_spectrum is not None:
                        filtered_library.append(processed_spectrum)

                logger.info(f"Retained {len(filtered_library)} spectra after filtering.")
                return filtered_library

            except Exception as e:
                logger.error(f"Failed to load cache: {e}. Falling back to raw file.")

        # Load raw file
        if not os.path.exists(raw_file):
            raise FileNotFoundError(f"Raw library file not found: {raw_file}")

        logger.info(f"Loading raw library from {raw_file}...")

        spectra_generator = self._get_loader(raw_file)
        library = []

        for spectrum in spectra_generator:
            if spectrum is not None:
                 library.append(spectrum)

        # Save raw (parsed) library to cache
        # We cache the data BEFORE filtering so that changing filters in YAML
        # takes effect on the next run without re-parsing the large text file.
        self._save_to_cache(library, cache_file)

        logger.info(f"Loaded {len(library)} raw spectra. Applying filters...")

        # Apply filters to the loaded library (whether from cache or raw)
        filtered_library = []
        for spectrum in library:
            processed_spectrum = apply_filters(spectrum, self.filters)
            if processed_spectrum is not None:
                filtered_library.append(processed_spectrum)

        logger.info(f"Retained {len(filtered_library)} spectra after filtering.")

        return filtered_library

    def _get_loader(self, filepath: str) -> Generator[Spectrum, None, None]:
        """
        Returns the appropriate matchms loader based on file extension.
        """
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.mgf':
            return load_from_mgf(filepath)
        elif ext == '.msp':
            return load_from_msp(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _save_to_cache(self, library: List[Spectrum], cache_file: str):
        """
        Saves the processed library to a pickle file.
        """
        logger.info(f"Saving processed library to cache: {cache_file}...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(library, f)
            logger.info("Cache saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
