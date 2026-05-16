# Workflow: Building Your Master Laboratory Library

Consolidating multiple libraries into a single "Master" SQLite database allows you to standardize your reference data, apply instrument-specific cleaning *once*, and speed up all future annotation runs.

## Phase 1: Define Your "Instrument Quirks" Config

Before building the database, create a `build_config.yaml`. This configuration defines how the raw spectra will be "scrubbed" before they are saved to the database.

**Example `build_config.yaml` for a High-Resolution Orbitrap:**
```yaml
processing:
  # 1. Clean the 'noise' unique to your detector
  noise_threshold: 0.005           # More aggressive (0.5%) cleaning
  min_peak_intensity: 100.0        # Ignore absolute intensity below 100 counts

  # 2. Scientific Integrity (The 'Instrument Quirk' fix)
  # If your instrument consistently drifts, you can define specific
  # metadata normalization here (though MassFlow 1.0 focuses on cleaning).
  normalize_intensity: true        # Scale all libraries to a 0-1000 range for parity
```

---

## Phase 2: Ingest Libraries with "Provenance"

Run the `db build` command for each of your raw libraries (`.msp`, `.mgf`, etc.).

By using the `--category` flag, you can keep track of where each spectrum came from within your master database.

```bash
# Ingest your in-house standards
uv run massflow db build \
    --input libraries/internal_standards.msp \
    --output lib_internal.db \
    --config build_config.yaml \
    --category "in-house-qtof"

# Ingest a public library (like GNPS)
uv run massflow db build \
    --input libraries/ALL_GNPS.msp \
    --output lib_gnps.db \
    --config build_config.yaml \
    --category "public-gnps"
```

---

## Phase 3: The Big Merge

Now, consolidate these pre-cleaned databases into one master file. This process is nearly instantaneous because the spectra are already processed and binary-encoded.

```bash
uv run massflow db merge \
    --inputs lib_internal.db \
    --inputs lib_gnps.db \
    --output master_library.db
```

---

## Phase 4: Verify Your Masterpiece

Inspect the final result to see the breakdown of your consolidated library:

```bash
uv run massflow db inspect master_library.db
```

**Expected Output:**
```text
==================================================
DATABASE INSPECTION: master_library.db
==================================================
Total Spectra: 245,612
Precursor m/z Range: 50.02 to 1800.55

Categories:
  - in-house-qtof: 1,200 spectra
  - public-gnps: 244,412 spectra
==================================================
```

---

## Why this is the "Pro" way:
1. **Consistency:** Because you used the same `build_config.yaml` for all inputs, every spectrum in your master library has been cleaned with the exact same math.
2. **Provenance:** If you see a weird match later, the `category` field tells you exactly which original library it came from.
3. **Daily Use:** You now only need to point your daily `massflow_config.yaml` to `master_library.db`, and MassFlow will search everything at once with zero overhead.
