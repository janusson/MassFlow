# SQLite Library Management

MassFlow v0.1 natively supports persistent, reusable local spectral libraries backed by SQLite.

If you run `massflow annotate` against large, raw text-based formats (like `.msp` or `.mgf`), MassFlow must parse those massive files into Python objects every single time you launch the CLI.

By building a SQLite database, you preprocess the spectra, serialize their mass arrays into optimized binary blobs, and dramatically speed up subsequent annotation runs while reducing memory overhead.

---

## Why Use SQLite Databases?

1.  **Speed:** SQLite blobs bypass the slow string-parsing required for MGF/MSP files.
2.  **Memory:** SQLite allows MassFlow to lazily stream spectra into the scoring engine instead of materializing an entire 10GB `.msp` into RAM at once.
3.  **Organization:** You can merge multiple project-specific libraries together and query them by category.
4.  **Triage Scanning (Core Feature):** During database construction, MassFlow automatically scans spectra for key chemical features (like the Tyrosine immonium ion at 136.076 Da) and flags them in a `triage_flags` bitmask. This allows the workflow to intelligently route structurally significant spectra to advanced machine learning engines downstream without having to re-parse the raw peak arrays.

---

## The Database Workflow

The typical pattern for a local bioinformatics lab is to maintain curated, in-house databases.

### 1. Build a Database
Convert a raw open-format library (like `.msp`) into a MassFlow SQLite `.db` file.

This command streams the input file through the `matchms` filters defined in your configuration file, ensuring the database only stores *clean*, processed spectra.

```shell
uv run massflow db build \
    --input data/libraries/example_library.msp \
    --output results/user_library.db \
    --config massflow_config.yaml \
    --category personal
```

*   `--input`: The raw file to ingest.
*   `--output`: The destination `.db` file.
*   `--config`: The YAML config dictating how the spectra should be cleaned before storage.
*   `--category`: An optional tag (e.g., `personal`, `standards`, `gnps`) to organize the spectra internally.

If the input file is malformed or yields zero valid spectra after processing, the build will explicitly fail and alert you.

### 2. Inspect a Database
Quickly verify the contents and health of a built database without loading the full spectrum objects.

```shell
uv run massflow db inspect results/user_library.db
```

Every SQLite-backed database records its own **build lineage** at creation time: the input file(s) used (with content SHA-256), the effective config hash (`config_digest_sha256`), the processing parameters (including decoy-generation settings), the similarity / target-decoy configuration, the MassFlow version, and an exact UTC build timestamp. `db inspect` acts as the query surface for that history:

**Example Output:**
```text
DATABASE INSPECTION: results/user_library.db
Total Spectra: 15420
Precursor m/z Range: 50.0211 to 1400.9822

Categories:
  - personal: 15420 spectra

Store Metadata
  Backend          sqlite
  Schema Version   1
  Created At (UTC) 2026-09-06T10:00:00+00:00

Library Build History
  #  Built At (UTC)            Category  Spectra  Backend  Source                  Config Digest (sha256)
  1  2026-09-06T10:00:00+00:00 personal  15420    sqlite   data/libraries/lib.msp  5c7725e3d5e77f5b…

Processing Parameters
  min_peaks                   5
  decoy_min_relative_intensity 0.01
  decoy_mz_shift_da           1.0
  ...

Target-Decoy Configuration
  algorithm    cosine
  fdr_threshold 0.05
  ...
```

*   `db build` records one history row per build; rebuilding over changed input or configuration appends a new row, so the table reads as a version history.
*   `db merge` records the exact input databases and per-input spectrum counts (`details.merged_sources`).
*   Databases created before the provenance schema existed (``user_version 0``) are upgraded in place on their next MassFlow open; their spectra are untouched and their history starts at the first recorded build.
*   The same history is queryable programmatically via `SpectralDatabase.get_library_builds()` / `get_latest_library_build()` (all JSON fields parsed), and annotation result sidecars link back to these rows (see `docs/user-guide/results.md`).

### 3. Merge Databases
Combine multiple specialized databases into a single, master search library.

For example, you might maintain one database for authenticated in-house standards, and another for a downloaded GNPS library.

```shell
uv run massflow db merge \
    --inputs results/user_library.db results/standards.db \
    --output results/master_library.db
```

The merged database will retain all the processed spectra and metadata from the inputs.

---

## Using the Database in Annotation

Once your `.db` file is built, simply point your `massflow_config.yaml` to it instead of the raw `.msp` file:

```yaml title="massflow_config.yaml"
input:
  file_path: "data/experiments/experiment.mzML"
  library_path: "results/master_library.db"
  format: "sqlite" # 'db' or 'sqlite'
```

Run `massflow annotate` as normal. The workflow will automatically detect the database and stream the binary spectra directly into the similarity engine.
