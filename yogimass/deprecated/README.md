# Deprecated Scripts Archive

This folder contains legacy scripts that have been superseded by the modern CLI and workflow system. They are maintained for backward compatibility but **should not be used in new workflows**.

## Scripts

### `yogimass_pipeline.py`

**Replaced by:** `yogimass config run --config <path>`

Legacy wrapper for executing pipelines. Use the unified config-driven CLI instead.

### `yogimass_buildDB.py`

**Replaced by:** `yogimass library build --format msp`

Legacy MSP library builder. Use the modern library build command for new workflows.

### `build_library_from_msp.py`

**Replaced by:** `yogimass library build --format msp`

Legacy MSP library builder helper. Identical functionality to `yogimass_buildDB.py` but with a different name.

### `msdial_fragment_search.py`

**Replaced by:** `yogimass config run --config <file>` (with msdial input)

Legacy MS-DIAL fragment search wrapper. Use config-driven runs for MS-DIAL processing instead.

## Migration Path

For any workflow using these deprecated scripts:

1. Replace `yogimass_pipeline.py` calls with `yogimass config run --config <file>`
2. Replace `yogimass_buildDB.py` or `build_library_from_msp.py` with `yogimass library build --format msp`
3. Replace `msdial_fragment_search.py` with a YAML/JSON config file and `yogimass config run`

Refer to the main README and `examples/` directory for modern workflow examples.
