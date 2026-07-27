# Command Line Interface (CLI)

The `MassFlow.cli` module uses the `typer` framework to provide the user-facing
terminal commands (`tutorial`, `annotate`, `init`, `convert`, `db`, `serve`, `watch`).
It handles argument parsing and delegates execution to the core workflow.

## Quick Tour

| Command | Description |
|---|---|
| `massflow tutorial` | Generate synthetic tutorial data for evaluating MassFlow locally |
| `massflow annotate` | Run the end-to-end annotation pipeline |
| `massflow init` | Create a starter YAML configuration file |
| `massflow convert` | Convert vendor raw files to mzML via ProteoWizard |
| `massflow db build` | Compile raw spectra into a SQLite/Zarr database |
| `massflow db inspect` | View database statistics |
| `massflow db merge` | Merge multiple databases into one |
| `massflow watch` | Interactive live-reloading annotation mode |
| `massflow serve` | gRPC streaming server for real-time annotation |

For a complete walkthrough using all the core commands on real (synthetic) data, see the [Usage Guide](../user-guide/usage.md).

::: MassFlow.cli
