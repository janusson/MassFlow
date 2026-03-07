"""
Data validation layer for MassFlow using Pydantic.

This module provides Pydantic-based schemas and utility functions to validate
and clean spectral metadata. It defines the ``SpectrumSchema`` to enforce strict
types on fields like precursor m/z and retention time, and includes logic to
coerce or filter out "dirty" data (e.g., non-numeric strings in numeric fields)
before further processing occurs.
"""

import logging
from typing import Any, Optional

from matchms import Spectrum
from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


class SpectrumSchema(BaseModel):
    """
    Pydantic schema for validating and cleaning spectral metadata.
    """

    precursor_mz: float = Field(
        ..., gt=0, description="Precursor m/z, must be positive."
    )
    retention_time: Optional[float] = Field(
        None, description="Retention time in seconds/minutes."
    )
    ccs: Optional[float] = Field(None, description="Collision Cross Section.")
    compound_name: Optional[str] = Field(None, description="Name of the compound.")
    adduct: Optional[str] = Field(None, description="Adduct type (e.g., [M+H]+).")
    charge: Optional[int] = Field(None, description="Charge state.")

    @field_validator("retention_time", "ccs", "precursor_mz", mode="before")
    @classmethod
    def clean_numeric_fields(cls, v: Any) -> Optional[float]:
        """
        Coerce dirty inputs in numeric fields to floats or None.

        This validator acts on fields that are expected to be numeric (e.g.,
        retention time, collision cross section, and precursor m/z). It handles
        common issues found in mass spectral metadata where numeric values are
        polluted with text (like 'CCS:', 'N/A', 'NaN', 'None').

        Parameters
        ----------
        v : Any
            The input value to be cleaned. It can be of any type, though
            typically it is a float, int, or str.

        Returns
        -------
        float or None
            Returns a validated float if the input can be successfully converted.
            Returns None if the input is None, matches known garbage patterns,
            or cannot be converted to a float.
        """
        if v is None:
            return None

        if isinstance(v, (float, int)):
            return float(v)

        if isinstance(v, str):
            v_str = v.strip()
            # Check for known garbage patterns
            if not v_str or any(
                x in v_str.upper() for x in ["CCS", "N/A", "NONE", "NAN"]
            ):
                return None

            try:
                return float(v_str)
            except ValueError:
                return None

        return None

    @field_validator("charge", mode="before")
    @classmethod
    def clean_charge(cls, v: Any) -> Optional[int]:
        """
        Coerce the charge field to an integer or None.

        This validator attempts to safely convert various representations of charge
        states into an integer format. It first tries to cast the value to a float
        (to handle cases like "1.0") before finally casting to an integer.

        Parameters
        ----------
        v : Any
            The input value for charge to be cleaned.

        Returns
        -------
        int or None
            Returns a valid integer representing the charge state.
            Returns None if the input is None or cannot be coerced to an integer.
        """
        if v is None:
            return None
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return None


def validate_and_clean_spectrum(spectrum: Spectrum) -> Optional[Spectrum]:
    """
    Validate and clean a matchms Spectrum object against the SpectrumSchema.

    This function extracts key metadata from a matchms ``Spectrum`` object, handles
    inconsistent key naming (e.g., 'retention_time' vs 'retentiontime'), and
    passes the data through the Pydantic ``SpectrumSchema``. If validation is
    successful, the original spectrum's metadata dictionary is updated with
    the coerced, strictly typed values. Invalid fields that are optional
    are explicitly removed to prevent downstream errors in matchms filters.

    Parameters
    ----------
    spectrum : matchms.Spectrum
        The raw spectrum object to validate and clean.

    Returns
    -------
    matchms.Spectrum or None
        Returns the original spectrum object with its metadata updated in place
        if validation succeeds.
        Returns None if validation fails critically (e.g., missing a required
        field like 'precursor_mz').

    Raises
    ------
    ValidationError
        Caught internally. Logs the validation error details for debugging
        and returns None, effectively dropping the invalid spectrum.
    """
    if spectrum is None:
        return None

    # Prepare raw data dictionary for validation
    # matchms keys can be inconsistent, so we map them
    raw_data = {
        "precursor_mz": spectrum.get("precursor_mz"),
        "retention_time": spectrum.get("retention_time")
        or spectrum.get("retentiontime"),
        "ccs": spectrum.get("ccs") or spectrum.get("CCS"),
        "compound_name": spectrum.get("compound_name") or spectrum.get("name"),
        "adduct": spectrum.get("adduct"),
        "charge": spectrum.get("charge"),
    }

    try:
        # Validate against schema
        clean_model = SpectrumSchema(**raw_data)

        # Update spectrum metadata with cleaned values
        # We explicitly set them to ensure consistency (e.g., float vs str)
        if clean_model.precursor_mz is not None:
            spectrum.set("precursor_mz", clean_model.precursor_mz)

        if clean_model.retention_time is not None:
            spectrum.set("retention_time", clean_model.retention_time)
        else:
            # Explicitly remove if it was invalid/garbage to prevent matchms errors
            spectrum.metadata.pop("retention_time", None)
            spectrum.metadata.pop("retentiontime", None)

        if clean_model.ccs is not None:
            spectrum.set("ccs", clean_model.ccs)
        else:
            spectrum.metadata.pop("ccs", None)
            spectrum.metadata.pop("CCS", None)

        if clean_model.compound_name:
            spectrum.set("compound_name", clean_model.compound_name)

        if clean_model.charge is not None:
            spectrum.set("charge", clean_model.charge)

        return spectrum

    except ValidationError as e:
        # Log basic info about why it failed (e.g. missing precursor)
        scan_id = spectrum.get("scan") or spectrum.get("id") or "unknown_scan"
        logger.debug(f"Skipping spectrum {scan_id}: Validation failed - {e.errors()}")
        return None
