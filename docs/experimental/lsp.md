# MassFlow Language Server (Experimental)

!!! warning "Experimental Feature"
    The MassFlow Language Server (LSP) is currently in early beta. It is designed to enhance the developer and researcher experience in modern IDEs like **Zed**, **VS Code**, and **Sublime Text**, but is subject to frequent updates.

The MassFlow Language Server (`MassFlow.server`) brings real-time mass spectrometry insights directly into your text editor. It allows you to see chemical properties, validate SMILES strings, and inspect spectral database matches without leaving your workspace.

---

## Key Features

### 1. Real-time SMILES & Formula Validation
As you type in a YAML config, a CSV results table, or a Markdown report, the server automatically validates chemical identifiers:
*   **SMILES Linting:** Highlights invalid SMILES syntax or valency errors using RDKit.
*   **Formula Checking:** Flags plausible-looking but chemically impossible molecular formulas.

### 2. Information on Hover
Hover your cursor over supported identifiers to see a rich-text tooltip:
*   **SMILES Hover:** Shows the calculated Exact Mass, Formula, and a high-level summary of the molecule.
*   **Scan ID Hover:** If you are working with MassFlow results, hovering over a `Scan_ID` (e.g., `query_42`) will pull the best match name, score, and precursor information from your project metadata.

### 3. Diagnostics
The server publishes standard LSP diagnostics (squiggly underlines), allowing you to catch data entry errors in your library files or configurations before you launch a heavy annotation run.

---

## Setup in Editors

### Zed (Recommended)
MassFlow is designed to work seamlessly with the [Zed editor](https://zed.dev/). You can configure Zed to launch the MassFlow server by adding the following snippet to your `settings.json`:

```json
{
  "lsp": {
    "massflow-lsp": {
      "command": "massflow-server"
    }
  }
}
```

*(Note: Ensure `massflow` is installed in your active Python environment).*

### VS Code
Use the standard "LSP Client" extension and point it to the `massflow-server` executable (or `python -m MassFlow.server`).

---

## Technical Details

The server is built using the `pygls` library and implements the following LSP features:
*   `textDocument/didOpen` / `textDocument/didChange`: Triggers document-wide validation.
*   `textDocument/hover`: Provides the interactive tooltips.

It uses a non-blocking, asynchronous architecture to ensure that your editor remains responsive even when validating long CSV files with thousands of chemical structures.
