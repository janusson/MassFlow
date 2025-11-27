from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matchms", reason="MSDIAL helpers import yogimass.io, which depends on matchms.")

from yogimass.io import msdial


MSDIAL_HEADER = "\t".join([
    "Alignment ID",
    "Average Mz",
    "Name",
    "Model ion area",
    "MS/MS spectrum",
])


def _write_export(path: Path, rows):
    lines = [MSDIAL_HEADER]
    for row in rows:
        lines.append("\t".join(map(str, row)))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_clean_and_combine_msdial(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    rows_a = [
        (1, 123.45, "Compound_A", 1000, "100:10 200:40"),
        (2, 234.56, "Compound_B", 800, "150:30 250:40"),
    ]
    rows_b = [
        (3, 345.67, "Compound_C", 500, "120:5 240:15"),
    ]
    _write_export(input_dir / "experiment_a.txt", rows_a)
    _write_export(input_dir / "experiment_b.txt", rows_b)

    combined_df, combined_path = msdial.clean_and_combine_msdial(
        input_dir,
        output_dir,
        timestamp="teststamp",
    )

    assert len(combined_df) == 3
    assert set(combined_df["Experiment"]) == {"experiment_a", "experiment_b"}
    assert combined_path is not None
    assert combined_path.name == "combined_msdial_results_teststamp.csv"
    assert combined_path.exists()

    per_experiment_csv = output_dir / "experiment_a.csv"
    assert per_experiment_csv.exists()

    peaks_df = msdial.extract_msms_peaks(combined_df, alignment_id=1)
    assert peaks_df.iloc[0]["m/z"] == 100.0
    assert peaks_df.iloc[-1]["intensity"] == 40.0


def test_process_msdial_sorts_and_drops():
    df = pd.DataFrame(
        {
            "Alignment ID": [1, 2, 3],
            "Model ion area": [10, 500, 50],
            "Name": ["compound x", None, "compound y"],
            "Experiment": ["A", "B", "A"],
        }
    )

    processed = msdial.process_msdial(df)

    assert list(processed["Model ion area"]) == [500, 50]
    assert processed["Name"].isna().sum() == 0

    summary = msdial.summarize_by_experiment(processed)
    assert summary.loc[summary["Experiment"] == "A", "count"].item() == 1


def test_dataframe_to_spectra_parses_metadata():
    df = pd.DataFrame(
        {
            "Alignment ID": [5],
            "Average Mz": [111.1],
            "Name": ["compound_x"],
            "Experiment": ["exp1"],
            "MS/MS spectrum": ["50:10 75:5"],
        }
    )

    spectra = msdial.msdial_dataframe_to_spectra(df)

    assert len(spectra) == 1
    spectrum = spectra[0]
    assert spectrum.metadata["alignment_id"] == 5
    assert spectrum.metadata["precursor_mz"] == 111.1
    assert spectrum.metadata["experiment"] == "exp1"
    assert spectrum.peaks.mz.tolist() == [50.0, 75.0]
