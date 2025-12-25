"""
Utility functions for Yogimass.
"""


def get_spectrum_inchi(spectrum):
    """Return InChI string if available, otherwise fall back to InChIKey."""
    return spectrum.get("inchi") or spectrum.get("inchikey")
