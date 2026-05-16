#!/usr/bin/env python3
"""
Smoke test for MassFlow end-to-end numeric precision.
Validates that similarity scores and q_values are stable to 6 decimal places (1E-6).
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
from matchms import Spectrum
from matchms.exporting import save_as_mgf, save_as_msp

from MassFlow.config import (
    InputConfig,
    MassFlowConfig,
    ProcessingConfig,
    SimilarityConfig,
)
from MassFlow.workflow import run_annotation_pipeline


def create_golden_file(golden_path: Path):
    """Write the statically calculated golden results."""
    # Theoretical cosine calculation (intensity_power=1.0, mz_power=0.0):
    # Q1: [100, 50, 20, 10, 5] vs R1: [100, 50, 20, 10, 5] => Exact match => 1.000000
    # Q2: [100, 50, 20, 10, 5] (mz 201 unmatched) vs R1: [100, 50, 20, 10, 5] (mz 200 unmatched)
    # Sum(Q2^2) = 13025, Sum(R1^2) = 13025
    # Matched peaks = 100^2 + 50^2 + 10^2 + 5^2 = 12625
    # Score = 12625 / sqrt(13025 * 13025) = 12625 / 13025 = 0.969289827
    golden_csv = (
        "query_id,reference_id,score,matched_peaks,q_value\n"
        "Q1,R1,1.000000,5,0.500000\n"
        "Q2,R1,0.969290,4,0.500000\n"
    )
    with open(golden_path, "w") as f:
        f.write(golden_csv)


def main():
    from rich.console import Console

    console = Console()

    smoke_dir = Path("smoke_test_run")
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    smoke_dir.mkdir(parents=True)

    try:
        # 1. Setup Data
        query1 = Spectrum(
            mz=np.array([100.0, 150.0, 200.0, 250.0, 300.0]),
            intensities=np.array([100.0, 50.0, 20.0, 10.0, 5.0]),
            metadata={"id": "Q1", "precursor_mz": 100.0, "charge": 1},
        )
        query2 = Spectrum(
            mz=np.array([100.0, 150.0, 201.0, 250.0, 300.0]),
            intensities=np.array([100.0, 50.0, 20.0, 10.0, 5.0]),
            metadata={"id": "Q2", "precursor_mz": 100.0, "charge": 1},
        )

        ref1 = Spectrum(
            mz=np.array([100.0, 150.0, 200.0, 250.0, 300.0]),
            intensities=np.array([100.0, 50.0, 20.0, 10.0, 5.0]),
            metadata={
                "id": "R1",
                "precursor_mz": 100.0,
                "compound_name": "Target1",
                "charge": 1,
            },
        )

        query_file = smoke_dir / "query.mgf"
        library_file = smoke_dir / "library.msp"
        golden_file = smoke_dir / "golden.csv"

        save_as_mgf([query1, query2], str(query_file))
        save_as_msp([ref1], str(library_file))
        create_golden_file(golden_file)

        # 2. Configure Pipeline
        cfg = MassFlowConfig(
            input=InputConfig(
                input_path=query_file,
                library_path=library_file,
            ),
            processing=ProcessingConfig(
                min_peaks=3,
                min_intensity=0.0,
                noise_threshold=0.0,
                normalize_intensity=False,  # We handle our own normalized mock data
            ),
            similarity=SimilarityConfig(
                algorithm="cosine",
                ms1_tolerance=0.1,
                ms2_tolerance=0.5,  # 0.5 allows exact matches but rejects 200 vs 201
                min_score=0.1,
                min_matched_peaks=2,
                fdr_threshold=1.0,  # bypass strict FDR drop for small mock dataset
            ),
        )
        cfg.project.output_directory = smoke_dir / "results"

        # 3. Run Pipeline
        console.print("[bold blue]Running MassFlow Annotation Pipeline...[/bold blue]")
        run_annotation_pipeline(cfg)

        # 4. Compare Outputs
        results_file = smoke_dir / "results" / "query_results.csv"
        if not results_file.exists():
            console.print(
                f"[bold red]FAIL:[/bold red] Expected result file not found at {results_file}"
            )
            sys.exit(1)

        actual_df = pl.read_csv(results_file)
        # Filter out decoys to isolate the target results
        actual_df = actual_df.filter(~pl.col("is_decoy"))

        golden_df = pl.read_csv(golden_file)

        # Merge and compare
        compare_df = golden_df.join(
            actual_df,
            on=["query_id", "reference_id"],
            how="inner",
            suffix="_actual",
        ).rename(
            {
                "score": "score_golden",
                "matched_peaks": "matched_peaks_golden",
                "q_value": "q_value_golden",
            }
        )

        if len(compare_df) != len(golden_df):
            console.print(
                f"[bold red]FAIL:[/bold red] Row count mismatch. Expected {len(golden_df)}, got {len(compare_df)}"
            )
            sys.exit(1)

        tolerance = 1e-6
        passed = True

        for row in compare_df.iter_rows(named=True):
            score_diff = abs(row["score_golden"] - row["score_actual"])
            q_diff = abs(row["q_value_golden"] - row["q_value_actual"])
            peaks_diff = abs(row["matched_peaks_golden"] - row["matched_peaks_actual"])

            if score_diff > tolerance:
                console.print(
                    f"[bold red]FAIL:[/bold red] Score precision exceeded 1E-6 for {row['query_id']}-{row['reference_id']}. "
                    f"Golden: {row['score_golden']:.6f}, Actual: {row['score_actual']:.6f}, Delta: {score_diff:.2E}"
                )
                passed = False

            if q_diff > tolerance:
                console.print(
                    f"[bold red]FAIL:[/bold red] Q-value precision exceeded 1E-6 for {row['query_id']}-{row['reference_id']}. "
                    f"Golden: {row['q_value_golden']:.6f}, Actual: {row['q_value_actual']:.6f}, Delta: {q_diff:.2E}"
                )
                passed = False

            if peaks_diff != 0:
                console.print(
                    f"[bold red]FAIL:[/bold red] Matched peaks mismatch for {row['query_id']}-{row['reference_id']}. "
                    f"Golden: {row['matched_peaks_golden']}, Actual: {row['matched_peaks_actual']}"
                )
                passed = False

        if not passed:
            sys.exit(1)

        console.print(
            "[bold green]SUCCESS:[/bold green] End-to-end numeric precision smoke test passed within 1E-6."
        )

    finally:
        # Cleanup
        if smoke_dir.exists():
            shutil.rmtree(smoke_dir)


if __name__ == "__main__":
    main()
