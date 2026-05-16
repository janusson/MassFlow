# 5-Minute Tutorial: Your First Annotation

!!! abstract "Goal"
    By the end of this 5-minute guide, you'll have MassFlow installed and you will successfully match your mass spectra against a reference library to create a clean, easy-to-read Excel/CSV report.

Welcome! We designed MassFlow specifically for chemists and researchers. There are no confusing menus to click through, and no hidden "black box" math.

---

## 1. Get the Software

First, let's download MassFlow. We use a modern, lightning-fast package manager called `uv` to safely install the software without breaking any existing Python setups on your computer. (Requires Python 3.13+).

Open your terminal and copy-paste these commands:

```shell title="Terminal"
git clone https://github.com/ericjanusson/MassFlow.git
cd MassFlow
uv python pin 3.13
uv sync
```

!!! info "Don't have your own data yet?"
    To run this tutorial, you normally need your own experiment file (like `.mzML`) and a reference library (like `.msp`). If you just want to test if the software works right now, you can run `uv run pytest` to watch MassFlow analyze our built-in mock data!

---

## 2. Create Your Settings File

MassFlow is controlled by a single, readable text file. This means you can share this file with a colleague, and they will get the exact same scientific results you did.

Let's generate your default settings:

```shell title="Terminal"
uv run massflow init
```

!!! success "Output"
    `[INFO] Initialized new MassFlow configuration at: massflow_config.yaml`

---

## 3. Point MassFlow to Your Data

Open that new `massflow_config.yaml` file in any simple text editor (like Notepad, TextEdit, or VS Code).

Don't worry about all the advanced math settings at the bottom yet! For now, just look at the `input` section near the top. Change the paths so they point to where your actual files are saved on your computer:

```yaml title="massflow_config.yaml"
project:
  name: "My_First_Annotation"
  output_directory: "results"

input:
  # Change these to match your actual files!
  file_path: "path/to/your/experiment.mzML"
  library_path: "path/to/your/library.msp"
  format: "mzml"

# ... leave the rest of the file exactly as it is!
```

Save the file and close your editor.

---

## 4. Let it Run!

Now, tell MassFlow to start the analysis using the settings you just saved:

```shell title="Terminal"
uv run massflow annotate --config massflow_config.yaml
```

**What is MassFlow doing right now?**
1. **Cleaning:** It reads both your library and your experiment, automatically filtering out baseline noise and fixing broken metadata.
2. **Matching:** It compares your experimental spectra against the library using the classic `cosine` scoring algorithm.
3. **Checking:** It calculates statistical confidence (False Discovery Rate) to filter out bad or lucky matches.

!!! success "Expected Terminal Output"
    ```text
    [INFO] Loading configuration from massflow_config.yaml
    [INFO] Processing reference library: path/to/your/library.msp
    [INFO] Processing experimental file: path/to/your/experiment.mzML
    [INFO] Running Cosine similarity search...
    [INFO] Results saved to results/experiment_results.csv
    ```

---

## 5. View Your Results

Go to the `results/` folder on your computer. You will find two new files:

1.  `experiment_results.csv`: Your final annotations. You can open this right up in Excel or Pandas!
2.  `experiment_results.report.yaml`: A "receipt" that saves the exact timestamps and settings used to create your CSV. Keep this for your publication records.

**What your CSV looks like:**
```csv title="experiment_results.csv"
query_id,query_precursor_mz,reference_id,reference_name,score,Annotation_Status
query_01,304.15,ref_89,Cocaine,0.95,Matched
query_02,137.05,,,,Unknown
```
*(Tip: Even if MassFlow doesn't find a match for a spectrum, it still lists it as `Unknown`. This proves to you that the spectrum wasn't accidentally skipped!)*

---

## 🎉 You did it!

Now that you've run a basic annotation, check out the user guides to unlock MassFlow's full power:

*   🚀 **Make it 10x faster:** Learn how to convert slow `.msp` libraries into lightning-fast databases using the [`massflow db build`](../user-guide/database.md) command.
*   ⚙️ **Tweak the math:** Explore the powerful noise filters available in the [Configuration Guide](../user-guide/configuration.md).
