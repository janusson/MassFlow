import logging
from pathlib import Path

import yaml
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, FloatPrompt

console = Console()
logger = logging.getLogger(__name__)


def run_config_wizard(output_path: Path) -> None:
    """
    Interactively guides the user through creating a MassFlow configuration file.

    The wizard prompts for the most critical settings and saves them to the
    specified YAML file.
    """
    console.print(
        "\n[bold cyan]Welcome to the MassFlow Configuration Wizard![/bold cyan]"
    )
    console.print(
        "This tool will help you create a configuration file for your annotation project.\n"
    )

    # 1. Project Settings
    console.print("[bold yellow]--- Project Settings ---[/bold yellow]")
    project_name = Prompt.ask("Project name", default="MassFlow_Project")
    output_dir = Prompt.ask("Output directory", default="results")

    # 2. Input Settings
    console.print("\n[bold yellow]--- Input Settings ---[/bold yellow]")
    input_path = Prompt.ask("Path to your experimental spectra (file or directory)")
    library_path = Prompt.ask("Path to your reference library (.msp, .db, etc.)")

    console.print("Choose a storage backend:")
    console.print("  - [bold]sqlite[/bold]: Standard, compatible (Default)")
    console.print(
        "  - [bold]hybrid[/bold]: Metadata in SQLite, arrays in Zarr (Faster for large libraries)"
    )
    storage_backend = Prompt.ask(
        "Backend", choices=["sqlite", "hybrid", "zarr"], default="sqlite"
    )

    # 3. Processing Settings
    console.print("\n[bold yellow]--- Processing Settings ---[/bold yellow]")
    min_peaks = IntPrompt.ask("Minimum peaks to keep per spectrum", default=5)
    noise_threshold = FloatPrompt.ask(
        "Noise threshold (intensities below this are dropped)", default=1000.0
    )

    # 4. Similarity Settings
    console.print("\n[bold yellow]--- Similarity Settings ---[/bold yellow]")
    console.print("Choose a similarity algorithm:")
    console.print("  - [bold]cosine[/bold]: Standard cosine similarity")
    console.print("  - [bold]modified_cosine[/bold]: Better for analogues (Default)")
    algorithm = Prompt.ask(
        "Algorithm",
        choices=[
            "cosine",
            "modified_cosine",
            "spec2vec",
            "ms2deepscore",
            "consensus",
            "cascade",
        ],
        default="modified_cosine",
    )

    fdr_threshold = FloatPrompt.ask(
        "FDR threshold (e.g., 0.01 for 1% FDR)", default=0.01
    )
    ms1_tol = FloatPrompt.ask("MS1 precursor tolerance (Da)", default=0.02)

    # 5. Export Settings
    console.print("\n[bold yellow]--- Export Settings ---[/bold yellow]")
    export_format = Prompt.ask("Export format", choices=["csv", "mztab"], default="csv")

    # Construct the config dict
    config_dict = {
        "project": {
            "name": project_name,
            "output_directory": output_dir,
        },
        "input": {
            "input_path": input_path,
            "library_path": library_path,
            "storage_backend": storage_backend,
        },
        "processing": {
            "min_peaks": min_peaks,
            "noise_threshold": noise_threshold,
        },
        "similarity": {
            "algorithm": algorithm,
            "fdr_threshold": fdr_threshold,
            "ms1_tolerance": ms1_tol,
        },
        "export": {
            "format": export_format,
        },
    }

    # Save to YAML
    try:
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        console.print(
            f"\n[bold green]✓ Configuration successfully saved to {output_path}[/bold green]\n"
        )
    except Exception as e:
        console.print(f"\n[bold red]Error saving configuration: {e}[/bold red]")
        raise
