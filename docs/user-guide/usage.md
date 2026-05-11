# MassFlow CLI Usage Guide

This guide details the core CLI workflows for initializing a project, compiling a reference library, and annotating experimental mass spectra.

## 1. Initialize a Workspace

Generate a default YAML configuration file to define preprocessing, similarity, and export parameters.

```bash
# Generate configuration file in the current directory
uv run massflow init --output massflow_config.yaml

# Overwrite an existing configuration file
uv run massflow init --output massflow_config.yaml --force
```

## 2. Build a Reference Library

Compile raw spectral formats (`.msp`, `.mgf`) into a high-performance SQLite database for rapid, memory-aware annotation workflows.

```bash
# Build a SQLite database from an MSP file
uv run massflow db build \
    --input data/libraries/reference_library.msp \
    --output results/compiled_library.db \
    --config massflow_config.yaml \
    --category "authenticated_standards"
```

## 3. Run an Annotation

Annotate experimental spectra (`.mgf`, `.mzML`) against a compiled SQLite reference library.

**Update `massflow_config.yaml` inputs:**
```yaml
input:
  file_path: "data/experiments/query_file.mgf"
  library_path: "results/compiled_library.db"
  format: "mgf"
```

**Execute the pipeline:**
```bash
uv run massflow annotate --config massflow_config.yaml
```

*Results are exported as defined in the configuration (e.g., `results/query_file_results.csv`).*
