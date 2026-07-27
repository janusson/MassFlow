# MassFlow CLI Usage Guide

This guide walks you through a complete, self-contained MassFlow workflow — from generating synthetic tutorial data through to annotation results — using nothing but the commands listed below. Every file path in this guide is real; copy and paste each command in order and everything will work.

---

## 0. Generate Tutorial Data

Before running any annotation commands, generate a synthetic dataset that includes a reference library, experimental queries, and a pre-configured YAML config file.

```bash
# Using the dedicated CLI command (recommended):
uv run massflow tutorial

# Or, if you prefer to run the underlying script directly:
uv run python scripts/generate_tutorial_data.py
```

This creates a `tutorial/` directory containing:

| File | Description |
|---|---|
| `tutorial/tutorial_library.msp` | Reference library with 3 steroid standards (Testosterone, Progesterone, Cortisol) |
| `tutorial/tutorial_experimental.mgf` | 4 experimental query spectra (matches, analogues, and noise) |
| `tutorial/tutorial_config.yaml` | Pre-configured analysis parameters (cosine similarity, 0.02 Da tolerance) |

---

## 1. Build a Reference Database

Compile the raw MSP library into a high-performance SQLite database for rapid, memory-aware annotation workflows.

```bash
uv run massflow db build \
    --input tutorial/tutorial_library.msp \
    --output tutorial/results/compiled_library.db \
    --config tutorial/tutorial_config.yaml \
    --category library
```

**Expected output:** `✓ Successfully processed and added 3 spectra to tutorial/results/compiled_library.db.`

---

## 2. Inspect the Database (Optional)

Verify the compiled database contents:

```bash
uv run massflow db inspect tutorial/results/compiled_library.db
```

---

## 3. Initialize a Custom Workspace (Optional)

If you want to generate a fresh configuration file for your own data rather than using the tutorial config:

```bash
# Generate configuration file in the current directory
uv run massflow init --output massflow_config.yaml

# Overwrite an existing configuration file
uv run massflow init --output massflow_config.yaml --force
```

---

## 4. Run an Annotation

Annotate experimental spectra against the compiled SQLite reference library.

**If you're using the tutorial config** (generated in Step 0), first update it to point to the SQLite database instead of the raw MSP file:

```yaml
# In tutorial/tutorial_config.yaml, change:
input:
  library_path: "tutorial/results/compiled_library.db"
```

**Execute the pipeline:**

```bash
uv run massflow annotate --config tutorial/tutorial_config.yaml
```

**Expected output:** `✓ Annotation complete! Results saved to tutorial/results`

Results are exported as defined in the configuration (`tutorial/results/tutorial_experimental_results.csv`).

---

## 5. Check the Results

The CSV results file contains annotation hits with computed scores, matched peaks, and automated `Annotation_Status` tags (`Matched`, `Putative`, or `Unknown`).

```bash
# Quick peek at results:
head -n 5 tutorial/results/tutorial_experimental_results.csv
```

---

## Additional Commands

### Convert Vendor Files

Convert vendor raw files (.raw, .d) to open formats (.mzML) using ProteoWizard msconvert:

```bash
uv run massflow convert --input data/raw/ --output data/experiments/
```

### Watch Mode

Interactive live-reloading that re-runs the pipeline whenever files or config change:

```bash
uv run massflow watch --config tutorial/tutorial_config.yaml
```
