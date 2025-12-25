"""
Post-processing helpers for combined MS-DIAL results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from matchms import Spectrum

from yogimass.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_SORT_COLUMN = "Model ion area"
DEFAULT_NAME_COLUMN = "Name"
DEFAULT_DEDUPLICATE_COLUMNS = ("Experiment",)
DEFAULT_NAME_FILL_VALUE = "Unknown"


def load_summary_csv(summary_path: str | Path) -> pd.DataFrame:
    """
    Load a combined MS-DIAL CSV summary into a dataframe.
    """
    summary_path = Path(summary_path)
    df = pd.read_csv(summary_path)
    logger.info("Loaded combined MS-DIAL summary: %s (%d rows)", summary_path, len(df))
    return df


def process_msdial(
    summary_source: str | Path | pd.DataFrame,
    *,
    sort_column: str = DEFAULT_SORT_COLUMN,
    dropna_column: str | None = DEFAULT_NAME_COLUMN,
    fillna_column: str | None = DEFAULT_NAME_COLUMN,
    fillna_value: str = DEFAULT_NAME_FILL_VALUE,
    deduplicate_by: Sequence[str] | None = DEFAULT_DEDUPLICATE_COLUMNS,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Sort and filter a combined MS-DIAL dataframe.
    - If ``summary_source`` is a path, it will be read as CSV.
    - Rows missing ``dropna_column`` are removed (if provided). Missing values in
      ``fillna_column`` are populated before dropping so important peaks are not lost.
    - The dataframe is sorted by ``sort_column`` (if present).
    - After sorting, duplicate entries for the same experiment (or other columns) are
      removed while keeping the most intense row first.
    """
    if isinstance(summary_source, (str, Path)):
        df = load_summary_csv(summary_source)
    else:
        df = summary_source.copy()

    if fillna_column and fillna_column in df.columns:
        df[fillna_column] = df[fillna_column].fillna(fillna_value)

    if dropna_column and dropna_column in df.columns:
        df = df.dropna(subset=[dropna_column])

    if sort_column and sort_column in df.columns:
        df = df.sort_values(by=sort_column, ascending=ascending)

    if deduplicate_by:
        dedupe_columns = [col for col in deduplicate_by if col in df.columns]
        if dedupe_columns:
            df = df.drop_duplicates(subset=dedupe_columns, keep="first")

    return df.reset_index(drop=True)


def summarize_by_experiment(
    summary_df: pd.DataFrame, value_column: str = DEFAULT_NAME_COLUMN
) -> pd.DataFrame:
    """
    Count the number of non-null ``value_column`` entries per experiment.
    """
    if "Experiment" not in summary_df.columns:
        raise KeyError("summary_df must contain an 'Experiment' column.")
    if value_column not in summary_df.columns:
        raise KeyError(f"summary_df must contain '{value_column}'.")
    counts = (
        summary_df.dropna(subset=[value_column])
        .groupby("Experiment")[value_column]
        .count()
        .rename("count")
        .reset_index()
    )
    return counts


def _parse_msms_string(peaks_string: str) -> tuple[np.ndarray, np.ndarray]:
    peaks: list[tuple[float, float]] = []
    for entry in peaks_string.strip().split():
        if ":" not in entry:
            continue
        try:
            mz, intensity = entry.split(":")
            peaks.append((float(mz), float(intensity)))
        except ValueError:
            continue
    if not peaks:
        return np.array([], dtype="float32"), np.array([], dtype="float32")
    mz_values, intensities = zip(*peaks)
    return np.asarray(mz_values, dtype="float32"), np.asarray(
        intensities, dtype="float32"
    )


def msdial_dataframe_to_spectra(processed_data: pd.DataFrame) -> list[Spectrum]:
    """
    Convert a cleaned MS-DIAL dataframe into ``matchms.Spectrum`` objects.
    Rows missing MS/MS data are skipped.
    """
    spectra: list[Spectrum] = []
    if processed_data.empty:
        return spectra
    for _, row in processed_data.iterrows():
        peaks_string = str(row.get("MS/MS spectrum", "") or "")
        mz_array, intensities = _parse_msms_string(peaks_string)
        if mz_array.size == 0 or intensities.size == 0:
            continue
        alignment_id = row.get("Alignment ID")
        experiment = row.get("Experiment")
        name = row.get("Name")
        try:
            alignment_id_int = int(alignment_id) if alignment_id is not None else None
        except (TypeError, ValueError):
            alignment_id_int = None
        metadata = {
            "name": str(name) if name else f"alignment-{alignment_id_int or 'unknown'}",
            "compound_name": name,
            "alignment_id": alignment_id_int,
            "experiment": experiment,
            "precursor_mz": row.get("Average Mz"),
        }
        spectrum = Spectrum(
            mz=mz_array,
            intensities=intensities,
            metadata={k: v for k, v in metadata.items() if v is not None},
        )
        spectra.append(spectrum)
    if not spectra:
        logger.warning("No valid MS-DIAL MS/MS spectra were parsed from the dataframe.")
    return spectra
