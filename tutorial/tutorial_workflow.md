# Tutorial: High-Performance Workflow with GNPS Libraries

This tutorial walks you through the recommended workflow for using a large public reference library (like `ALL_GNPS.msp`) to annotate your experimental data using MassFlow.

## Prerequisites

Ensure your data is organized as follows:

```text
project_root/
├── libraries/
│   └── ALL_GNPS.msp
└── experiments/
    └── example_spectrum.MSP
```

---

## Step 1: Build a Lightning-Fast SQLite Library

Large `.msp` files (like ALL_GNPS, which can be several gigabytes) are slow to parse repeatedly. We will compress and index it into a MassFlow SQLite database.

**Run the build command:**

```bash
uv run massflow db build --input libraries/ALL_GNPS.msp --output libraries/gnps_reference.db
```

**Why do this?**

- **Speed:** It converts text to binary blobs, making loading nearly instantaneous in future runs.
- **Triage:** MassFlow automatically scans for diagnostic fragments (like immonium ions) and bitmasks them for faster searching.

---

## Step 2: Create your Configuration (`massflow_config.yaml`)

Create a configuration file to define your scientific constraints. We recommend strict 5.0 ppm validation for GNPS data to filter out noisy matches.

```yaml
project:
  name: "GNPS_Test_Drive"
  output_directory: "results/"

input:
  input_path: "experiments/example_spectrum.MSP"
  library_path: "libraries/gnps_reference.db"
  format: "msp"

processing:
  # Standard cleaning for public libraries
  noise_threshold: 0.01          # Remove peaks < 1% of base peak
  max_peaks: 100                 # Keep only top 100 most intense peaks
  normalize_intensity: true      # Scale peaks to 1000 for parity

similarity:
  algorithm: "cosine"
  ms1_tolerance: 0.01            # 10 mDa tolerance for precursor
  ms2_tolerance: 0.02            # 20 mDa tolerance for fragments
  min_score: 0.7                 # Only keep high-confidence hits
  fdr_threshold: 0.05            # 5% False Discovery Rate filter
```

---

## Step 3: Run the Annotation

Now, execute the pipeline. MassFlow will automatically detect that you are using an optimized `.db` library and stream the data efficiently.

```bash
uv run massflow annotate --config massflow_config.yaml
```

---

## Step 4: (Optional) Live Tweak Mode

If you aren't sure about your `min_score` or `noise_threshold`, use the **Watch** mode to see results update in real-time as you save your config:

```bash
uv run massflow watch --config massflow_config.yaml
```

---

## Summary of Preprocessing Applied

1. **Denoising:** The `noise_threshold` removes low-intensity grass that can cause "lucky" but false cosine matches.
2. **Harmonization:** `normalize_intensity` ensures that even if your library and experimental data have different intensity scales (e.g., counts vs. relative %), they are compared on a 0-1000 scale.
3. **Physical Validation:** MassFlow will automatically verify the precursor m/z of the GNPS hits against your experimental spectrum to ensure they are physically possible.
