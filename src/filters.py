from typing import List, Dict, Any, Callable
from matchms import Spectrum
import matchms.filtering as ms_filters
from src.utils import get_logger

logger = get_logger(__name__)

def apply_filters(spectrum: Spectrum, filter_config: List[Dict[str, Any]]) -> Spectrum:
    """
    Applies a list of filters defined in the configuration to a spectrum.
    """
    if spectrum is None:
        return None

    for filter_step in filter_config:
        filter_name = filter_step.get("name")
        params = {k: v for k, v in filter_step.items() if k != "name"}

        filter_func = get_filter_function(filter_name)

        if filter_func:
            # logger.info(f"Applying filter: {filter_name} with params: {params}")
            if params:
                 spectrum = filter_func(spectrum, **params)
            else:
                 spectrum = filter_func(spectrum)

            if spectrum is None:
                # logger.warning(f"Spectrum removed by filter: {filter_name}")
                return None
        else:
            logger.warning(f"Filter function not found: {filter_name}")

    return spectrum

def get_filter_function(name: str) -> Callable:
    """
    Maps a string name to a matchms filtering function.
    """
    # Map of available filters
    # This can be expanded to include custom filters
    filter_map = {
        "default_filters": ms_filters.default_filters,
        "normalize_intensities": ms_filters.normalize_intensities,
        "select_by_intensity": ms_filters.select_by_intensity,
        "select_by_mz": ms_filters.select_by_mz,
        "reduce_to_number_of_peaks": ms_filters.reduce_to_number_of_peaks,
        # "clean_metadata": ms_filters.clean_metadata, # Deprecated or not available
        "add_parent_mass": ms_filters.add_parent_mass,
        # Add more mappings as needed
    }

    return filter_map.get(name)
