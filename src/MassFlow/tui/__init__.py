"""
Interactive terminal console for MassFlow (``massflow tui``).

A postmodern, keyboard-first console for finding, loading, viewing, and
identifying tandem mass-spectrometry (MS/MS) data without leaving the
terminal. Built on `Textual <https://textual.textualize.io/>`_ so it renders
in any modern terminal emulator (Ghostty, iTerm2, Kitty, Alacritty, ...) with
truecolour, mouse support, and full-width Unicode glyphs.

Layered design
--------------

Every module below the widget layer is a *pure* function of its inputs (no
terminal, no Textual, no I/O side-effects beyond what is documented) so the
science-facing behaviour is unit-testable headlessly:

- :mod:`MassFlow.tui.state` — plain dataclasses shared by the layers.
- :mod:`MassFlow.tui.spectrum_data` — spectrum summaries, downsampling,
  peak mirror-alignment, and formatting helpers (NumPy only, float64).
- :mod:`MassFlow.tui.plot` — text renderers for stick plots, mirror plots,
  score gauges, and axes (return ``list[str]``).
- :mod:`MassFlow.tui.files` — spectral file discovery, classification, and
  workspace "upload" (copy).
- :mod:`MassFlow.tui.diagnostics` — human-first error reports with
  actionable hints, plus the quarantine-log reader.
- :mod:`MassFlow.tui.pipeline` — the bridge to the core annotation modules
  (:mod:`MassFlow.io`, :mod:`MassFlow.processing`,
  :mod:`MassFlow.similarity`); runs synchronously so it can execute inside a
  Textual worker thread.
- :mod:`MassFlow.tui.app` — the Textual application itself (requires the
  ``tui`` extra: ``pip install massflow[tui]``).
"""

from __future__ import annotations

TUI_INSTALL_HINT = (
    "The MassFlow terminal console requires the optional 'textual' package. "
    "Install it with: pip install massflow[tui]"
)

__all__ = [
    "TUI_INSTALL_HINT",
    "state",
    "spectrum_data",
    "plot",
    "files",
    "diagnostics",
    "pipeline",
]
