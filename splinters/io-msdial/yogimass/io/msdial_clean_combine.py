"""
Utilities for cleaning and combining MS-DIAL exports.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd

from yogimass.utils.logging import get_logger

logger = get_logger(__name__)


def list_msdial_txt(directory: str | Path) -> List[Path]:
    """
    Return MS-DIAL export files (tab-delimited .txt) detected in ``directory``.
    The first line must contain either "Scan" or "Alignment ID" to be considered valid.
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning("MS-DIAL directory does not exist: %s", directory)
        return []
    results: list[Path] = []
    for pattern in ("*.txt", "*.tsv"):
        for path in sorted(directory.glob(pattern)):
            try:
                first_line = path.open("r", encoding="utf-8-sig").readline()
            except OSError as exc:
                logger.error("Unable to read %s: %s", path, exc)
                continue
            if "Scan" in first_line or "Alignment ID" in first_line:
                results.append(path)
    logger.info("Found %d MS-DIAL exports in %s", len(results), directory)
    return results


def load_msdial_data(tsv_path: str | Path) -> pd.DataFrame:
    """
    Read an MS-DIAL alignment export (tab-delimited) into a dataframe.
    """
    tsv_path = Path(tsv_path)
    df = pd.read_table(tsv_path, sep="\t")
    logger.debug("Loaded %d rows from %s", len(df), tsv_path.name)
    return df


def _clean_name_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Name" not in df.columns:
        return df
    cleaned = df.copy()
    cleaned["Name"] = (
        cleaned["Name"]
        .astype(str)
        .str.lower()
        .str.replace("_", " ", regex=False)
        .str.replace("+", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.replace('"', "", regex=False)
    )
    return cleaned


def _append_experiment_column(df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["Experiment"] = experiment
    return cleaned


def clean_msdial_dataframe(df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    """
    Normalize an MS-DIAL dataframe by cleaning the compound names and tagging the experiment.
    """
    if df.empty:
        return df.copy()
    cleaned = _clean_name_column(df)
    cleaned = _append_experiment_column(cleaned, experiment_name)
    return cleaned


def export_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)


def combine_results(
    dataframes: Sequence[pd.DataFrame],
    output_dir: str | Path,
    *,
    timestamp: str | None = None,
    write_combined: bool = True,
) -> Tuple[pd.DataFrame, Path | None]:
    """
    Combine a sequence of cleaned MS-DIAL dataframes into a single dataframe.
    Optionally writes the result into ``output_dir/combined_results``.
    """
    if not dataframes:
        logger.warning("No MS-DIAL dataframes provided for combining.")
        return pd.DataFrame(), None
    combined = pd.concat(dataframes, ignore_index=True)
    combined_path = None
    if write_combined:
        output_root = Path(output_dir) / "combined_results"
        output_root.mkdir(parents=True, exist_ok=True)
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_path = output_root / f"combined_msdial_results_{timestamp}.csv"
        combined.to_csv(combined_path, index=False)
        logger.info("Saved combined MS-DIAL results to %s", combined_path)
    return combined, combined_path


def extract_msms_peaks(processed_data: pd.DataFrame, alignment_id: int) -> pd.DataFrame:
    """
    Parse the ``MS/MS spectrum`` column for a given alignment ID into an m/z vs intensity table.
    """
    if "Alignment ID" not in processed_data.columns:
        raise KeyError("Processed data must contain 'Alignment ID'.")
    if "MS/MS spectrum" not in processed_data.columns:
        raise KeyError("Processed data must contain 'MS/MS spectrum'.")
    alignment_values = pd.to_numeric(processed_data["Alignment ID"], errors="coerce")
    mask = alignment_values == int(alignment_id)
    subset = processed_data.loc[mask]
    if subset.empty:
        raise ValueError(f"Alignment ID {alignment_id} not found.")
    spectrum_string = subset.iloc[0]["MS/MS spectrum"]
    if not isinstance(spectrum_string, str) or not spectrum_string.strip():
        raise ValueError(f"No MS/MS spectrum for alignment ID {alignment_id}.")
    peaks = []
    for entry in spectrum_string.strip().split():
        if ":" not in entry:
            continue
        mz, intensity = entry.split(":")
        try:
            peaks.append((float(mz), float(intensity)))
        except ValueError:
            continue
    peak_df = pd.DataFrame(peaks, columns=["m/z", "intensity"])
    return peak_df.sort_values("m/z").reset_index(drop=True)


def clean_and_combine_msdial(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    timestamp: str | None = None,
    write_intermediate: bool = True,
    write_combined: bool = True,
) -> Tuple[pd.DataFrame, Path | None]:
    """
    End-to-end helper:
    - locate MS-DIAL exports in ``input_dir``
    - clean each dataframe, annotate with experiment name, and optionally write CSV copies
    - combine the cleaned dataframes and optionally write a single summary CSV
    Returns the combined dataframe and the path to the combined CSV (if written).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    msdial_files = list_msdial_txt(input_dir)
    cleaned_dataframes: List[pd.DataFrame] = []
    for export_path in msdial_files:
        df = load_msdial_data(export_path)
        experiment_name = export_path.stem
        cleaned = clean_msdial_dataframe(df, experiment_name=experiment_name)
        cleaned_dataframes.append(cleaned)
        if write_intermediate:
            export_path_csv = output_dir / f"{experiment_name}.csv"
            export_dataframe(cleaned, export_path_csv)

    combined_df, combined_path = combine_results(
        cleaned_dataframes,
        output_dir,
        timestamp=timestamp,
        write_combined=write_combined,
    )
    return combined_df, combined_path
