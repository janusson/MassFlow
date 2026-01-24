# MassFlow CLI Design Guide

This document explains the design, architecture, and usage of the `MassFlow` Command Line Interface (CLI).

## Overview

The MassFlow CLI provides a unified entry point for interacting with the MassFlow toolkit. It replaces older standalone scripts (`massflow_buildDB.py`, `massflow_pipeline.py`) with a subcommand-based architecture.

## Architecture

The CLI is built using Python's built-in `argparse` library.

### Entry Point

- **File**: `MassFlow/cli.py`
- **Function**: `main()`
- **Registration**: The CLI is registered as a console script in `pyproject.toml`:

  ```toml
  [project.scripts]
  MassFlow = "MassFlow.cli:main"
  ```

  This allows users to invoke the tool directly using the `MassFlow` command after installation.

### Logging

The CLI implements a custom `ColoredFormatter` to provide visually distinct log messages (colored by severity) when running in a terminal.

- **DEBUG**: Grey
- **INFO**: Green
- **WARNING**: Yellow
- **ERROR**: Red
- **CRITICAL**: Bold Red

## Command Structure

The CLI uses a subcommand structure, meaning calls follow the pattern:
`MassFlow <command> [options]`

### Main Parser

The top-level parser handles version information (`--version`) and dispatches execution to subcommands.

### Subcommands

#### 1. `clean`

Cleans and processes a spectral library.

- **Function**: `run_clean`
- **Arguments**:
  - `--input` (Required): Path to the input library file (`.msp` or `.mgf`).
  - `--output-dir` (Required): Directory where the processed library will be saved.
  - `--format`: Output format. Options: `pickle` (default), `msp`, `mgf`, `json`.
- **Example**:

  ```bash
  MassFlow clean --input data/raw/library.msp --output-dir data/processed --format pickle
  ```

#### 2. `process`

Runs the full MassFlow processing pipeline based on a configuration file.

- **Function**: `run_process`
- **Arguments**:
  - `config`: Path to the YAML configuration file.
- **Example**:

  ```bash
  MassFlow process config.yaml
  ```

#### 3. `plot`

Visualizes a mass spectrum from an MSP file.

- **Function**: `run_plot`
- **Arguments**:
  - `--input` (Required): Path to the input MSP file.
  - `--name`: The name of the specific spectrum to plot.
  - `--more`: Lists the names of all spectra in the file (useful for finding the name to plot).
- **Example**:

  ```bash
  # List available spectra
  MassFlow plot --input data/library.msp --more
  
  # Plot a specific spectrum
  MassFlow plot --input data/library.msp --name "Caffeine"
  ```

#### 4. `search`

*Status: Placeholder / Experimental*
Intended for running similarity searches. Currently prints a placeholder message pointing to `MassFlow.similarity`.

## Internal Logic Flow

1. **Parsing**: `argparse` parses arguments.
2. **Dispatch**: If a subcommand is provided, the `main` function calls the helper function associated with that command (e.g., `run_clean`).
    - This is achieved using `parser.set_defaults(func=run_...)` which binds the function to the `func` attribute of the parsed arguments.
3. **Execution**: The bound function is executed with the parsed arguments.
4. **Exit Code**: The function returns an integer (0 for success, non-zero for error), which `main` passes to `sys.exit()`.

## Extending the CLI

To add a new command:

1. **Define a Handler**: Create a `run_<command>` function in `MassFlow/cli.py` that accepts `args: argparse.Namespace` and returns an `int`.
2. **Register Subparser**: In `main()`, add a new subparser:

    ```python
    new_parser = subparsers.add_parser("newcommand", help="Description")
    new_parser.add_argument("--arg", ...)
    new_parser.set_defaults(func=run_newcommand)
    ```

## Dependencies

- `argparse`: Standard library for CLI parsing.
- `plotnine`: Used by the `plot` command for visualization.
- `pandas`: Used for data handling in plotting.
- `matchms`: Used for loading and handling spectral data.
