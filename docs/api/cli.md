# Command Line Interface (CLI)

The `MassFlow.cli` module uses the `typer` framework to provide the user-facing
terminal commands (`tutorial`, `annotate`, `init`, `convert`, `db`, `stream-server`,
`watch`, `tui`). It handles argument parsing and delegates execution to the
core workflow. (`massflow serve` remains as a deprecated alias for
`stream-server` and prints a deprecation notice.)

## Quick Tour

| Command | Description |
|---|---|
| `massflow tutorial` | Generate synthetic tutorial data for evaluating MassFlow locally |
| `massflow annotate` | Run the end-to-end annotation pipeline |
| `massflow init` | Create a starter YAML configuration file |
| `massflow convert` | Convert vendor raw files to mzML via ProteoWizard |
| `massflow db build` | Compile raw spectra into a SQLite/Zarr database (records build lineage: input file, config hash, processing parameters, timestamp) |
| `massflow db inspect` | View database statistics, build history, processing parameters, and target-decoy configuration |
| `massflow db merge` | Merge multiple databases into one (records the input databases in the output's history) |
| `massflow watch` | Interactive live-reloading annotation mode |
| `massflow stream-server` | gRPC streaming server for real-time annotation (`serve` is a deprecated alias) |
| `massflow tui` | Interactive terminal console (find / upload / view / identify) |

For a complete walkthrough using all the core commands on real (synthetic) data, see the [Usage Guide](../user-guide/usage.md).

## Pre-flight validation

Every compute command (`annotate`, `db build`, `db merge`) validates its
inputs before any library store is built or any file is searched: required
files must exist, the reference library must be a loadable open-format/store
input (vendor raw formats and unknown extensions abort with zero output
artifacts), the similarity engine must be constructible with the installed
optional extras (missing `massflow[ml]` aborts with the install fix instead
of a late worker crash), and optional-extras degradations
(`consensus`/`cascade` without `[ml]`, `cascade` + `hnsw_enabled` without
`[hnsw]`) are warned about up-front. Failures print a plain-English `fix:`
hint (see `MassFlow.tui.diagnostics.suggest_fix`), never a raw traceback.
See [the annotation guide](../user-guide/annotation.md) (Pre-flight
validation) for the full contract.

## Terminal console

The interactive console (`massflow tui`) is documented separately in
[the TUI API reference](tui.md).

::: MassFlow.cli
