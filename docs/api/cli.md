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
| `massflow db build` | Compile raw spectra into a SQLite/Zarr database |
| `massflow db inspect` | View database statistics |
| `massflow db merge` | Merge multiple databases into one |
| `massflow watch` | Interactive live-reloading annotation mode |
| `massflow stream-server` | gRPC streaming server for real-time annotation (`serve` is a deprecated alias) |
| `massflow tui` | Interactive terminal console (find / upload / view / identify) |

For a complete walkthrough using all the core commands on real (synthetic) data, see the [Usage Guide](../user-guide/usage.md).

## Terminal console

The interactive console (`massflow tui`) is documented separately in
[the TUI API reference](tui.md).

::: MassFlow.cli
