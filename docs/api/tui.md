# Terminal Console (TUI)

The `MassFlow.tui` package implements `massflow tui`, a postmodern,
keyboard-first terminal interface for finding, uploading, viewing, and
identifying MS/MS data — no browser, no GUI toolkit, just your terminal
(Ghostty, iTerm2, Kitty, Alacritty, …).

```bash
pip install massflow[tui]
massflow tui --input experiments/run_01.mzML --library libraries/ALL_GNPS.msp
```

## Why a terminal console?

Postmodern interface design for scientific software means *exposing the
machinery instead of hiding it*. The console makes the pipeline's internal
states visible — the quarantine log of rejected spectra, the target/decoy
split behind every q-value, the engine that actually scored a hit (including
fallbacks) — because modern MS/MS analysis (large unknown sets across organic
chemicals, proteins, and peptides, increasingly scored by machine-learning
engines) fails most often for *explainable* reasons, and the interface should
put those reasons on screen.

The visual language is a deliberately artificial "neon lab bench": acid-green
and magenta annotations on a near-black field, monospace glyph plots, and
first-person status lines ("searching… the UI stays responsive"). It is a
layer of honest theatre over the same core pipeline that `massflow annotate`
runs headlessly.

## The four tabs

| Tab | Verb | What it does |
| --- | --- | --- |
| **Browser** | find · upload | Recursive discovery of spectral files (`.mzml`, `.mzxml`, `.mgf`, `.msp`, `.db`, `.zarr`), vendor files flagged with a conversion hint, upload = collision-safe copy into a local workspace, plus a library inspector (backend, spectra count, categories, precursor range). |
| **Viewer** | view | Spectrum-by-spectrum inspection: interactive centroid stick plot (zoom, precursor marker), metadata panel (precursor m/z, RT, charge, adduct, base peak, TIC, spectral entropy). |
| **Identify** | identify | Target-decoy similarity search against the selected library with q-value / p-value calibration; ranked hits with score gauges and a mirror plot (query up, reference down, matched peaks joined). |
| **Diagnostics** | diagnose | Every problem the console has seen, each with a plain-English fix, plus the tail of `massflow_quarantine.log` — every spectrum the validation layer rejected and why. |

## Key bindings

| Key | Action |
| --- | --- |
| `1` `2` `3` `4` | Jump between tabs |
| `?` / `F1` | Help |
| `enter` | Open a directory / load a file in the Browser |
| `u` / `l` | Upload the selected file / load it as a query |
| `+` `-` `z` `p` | Zoom in, zoom out, reset zoom, toggle precursor marker (Viewer) |
| `r` | Run the search (Identify) |
| `g` | Reload the quarantine log (Diagnostics) |
| `ctrl+q` | Quit |

## Error philosophy

The console never dumps raw tracebacks over the interface. Every failure is
captured as a `Problem` with a **stage** (`load-query`, `load-library`,
`search`, …), the exception message, and — when the failure mode is known — a
**hint**: vendor formats point at `massflow convert`, missing ML extras point
at `pip install massflow[ml]`, locked SQLite files tell you to close other
runs, small libraries warn that target-decoy FDR is statistically weak. The
full traceback is preserved in the problem record for when you do need it.

Heavy work (file scans, loads, library censuses, searches) runs in worker
threads, so the UI stays responsive and `ctrl+q` always works.

## Module layout

Every module below the widget layer is a pure function of its inputs (no
terminal, no Textual) so the science-facing behaviour is unit-testable
headlessly:

- `MassFlow.tui.state` — plain dataclasses shared by the layers.
- `MassFlow.tui.spectrum_data` — summaries, max-pool downsampling, mirror
  alignment, formatting (NumPy only, float64 preserved).
- `MassFlow.tui.plot` — pure text renderers: stick plots, mirror plots, score
  gauges, axes.
- `MassFlow.tui.files` — discovery, classification, workspace upload.
- `MassFlow.tui.diagnostics` — problems, hints, quarantine log reader.
- `MassFlow.tui.pipeline` — bridge to `MassFlow.io` / `MassFlow.processing` /
  `MassFlow.similarity` (imported lazily so console startup stays fast).
- `MassFlow.tui.app` — the Textual application.

::: MassFlow.tui.state
::: MassFlow.tui.spectrum_data
::: MassFlow.tui.plot
::: MassFlow.tui.files
::: MassFlow.tui.diagnostics
::: MassFlow.tui.pipeline
