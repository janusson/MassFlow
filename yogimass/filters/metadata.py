"""
Metadata and peak processing filters for Yogimass.
"""

from __future__ import annotations

try:  # matchms<0.24 exposed set_ionmode_na_when_missing
    from matchms.filtering import (  # type: ignore
        set_ionmode_na_when_missing as _set_ionmode_na_when_missing,
    )
except ImportError:  # pragma: no cover - depends on installed matchms version

    def _set_ionmode_na_when_missing(spectrum):
        ionmode = spectrum.get("ionmode")
        if ionmode is None or (isinstance(ionmode, str) and not ionmode.strip()):
            spectrum.set("ionmode", "n/a")
        return spectrum


def ensure_name_metadata(spectrum):
    """Populate ``name`` metadata from best available source if missing."""
    if spectrum.get("name"):
        return spectrum
    for key in ("compound_name", "title"):
        value = spectrum.get(key)
        if value:
            metadata_store = getattr(
                getattr(spectrum, "_metadata", None), "_data", None
            )
            if metadata_store is not None:
                dict.__setitem__(metadata_store, "name", value)
            break
    return spectrum


def metadata_processing(spectrum):
    from matchms.filtering import default_filters
    from matchms.filtering import repair_inchi_inchikey_smiles
    from matchms.filtering import harmonize_undefined_inchi
    from matchms.filtering import harmonize_undefined_inchikey
    from matchms.filtering import harmonize_undefined_smiles
    from matchms.filtering import clean_compound_name
    from matchms.filtering import derive_adduct_from_name
    from matchms.filtering import derive_formula_from_name
    from matchms.filtering import derive_ionmode
    from matchms.filtering import make_charge_int

    spectrum = default_filters(spectrum)
    spectrum = repair_inchi_inchikey_smiles(spectrum)
    spectrum = derive_adduct_from_name(spectrum)
    spectrum = derive_formula_from_name(spectrum)
    spectrum = harmonize_undefined_smiles(spectrum)
    spectrum = harmonize_undefined_inchi(spectrum)
    spectrum = harmonize_undefined_inchikey(spectrum)
    spectrum = clean_compound_name(spectrum)
    spectrum = ensure_name_metadata(spectrum)
    spectrum = derive_ionmode(spectrum)
    spectrum = make_charge_int(spectrum)
    spectrum = _set_ionmode_na_when_missing(spectrum)
    return spectrum


def peak_processing(spectrum):
    from matchms.filtering import default_filters
    from matchms.filtering import normalize_intensities
    from matchms.filtering import select_by_intensity
    from matchms.filtering import select_by_relative_intensity
    from matchms.filtering import select_by_mz

    spectrum = default_filters(spectrum)
    spectrum = select_by_intensity(spectrum, intensity_from=0.01)
    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.08)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_mz(spectrum, mz_from=10, mz_to=1000)
    return spectrum
