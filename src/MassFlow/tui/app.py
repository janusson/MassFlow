"""
The MassFlow terminal console (``massflow tui``).

A postmodern, keyboard-first Textual application with four tabs:

- **Browser** — find (recursive spectral-file discovery), upload (copy into a
  local workspace), and load experimental files and libraries.
- **Viewer** — inspect a loaded file spectrum by spectrum, with an
  interactive centroid stick plot (zoom, precursor marker).
- **Identify** — run target-decoy similarity search against a library and
  browse ranked hits with a mirror plot.
- **Diagnostics** — every problem the console has seen, with plain-English
  fixes, plus the core pipeline's quarantine log.

All heavy work (file scans, spectrum loads, library censuses, similarity
searches) runs in Textual worker threads; the UI stays responsive and every
failure is captured as a :class:`MassFlow.tui.diagnostics.Problem` rather
than a traceback vomited over the interface.

Requires the ``tui`` extra: ``pip install massflow[tui]``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, cast

import numpy as np
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Markdown,
    OptionList,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from MassFlow.tui.diagnostics import (
    Problem,
    QuarantineEntry,
    TuiError,
    parse_quarantine_log,
)
from MassFlow.tui.files import FileEntry, copy_into_workspace, human_size
from MassFlow.tui.pipeline import (
    inspect_library,
    load_query_preview,
    run_identification,
)
from MassFlow.tui.plot import (
    render_mirror_plot,
    render_score_gauge,
    render_stick_plot,
)
from MassFlow.tui.spectrum_data import (
    annotation_status,
    format_mz,
    format_retention_time,
)
from MassFlow.tui.state import (
    IdentificationOutcome,
    IdentificationRequest,
    LibraryInfo,
    QueryLoadResult,
    SearchHit,
)

logger = logging.getLogger(__name__)

ALGORITHMS: tuple[str, ...] = (
    "cosine",
    "modified_cosine",
    "consensus",
    "cascade",
    "spec2vec",
    "ms2deepscore",
)

_KIND_GLYPH = {
    "query": "▪",
    "library": "◆",
    "database": "◈",
    "vendor": "⚠",
    "unsupported": "·",
}

_KIND_COLOR = {
    "query": "#2de2e6",
    "library": "#3aff8c",
    "database": "#ff5ad1",
    "vendor": "#ff5470",
    "unsupported": "#55606e",
}

HELP_MARKDOWN = """\
# MassFlow console

## Tabs
| Key | Action |
| --- | --- |
| `1` `2` `3` `4` | Jump to Browser / Viewer / Identify / Diagnostics |
| `?` | This help |

## Browser
| Key | Action |
| --- | --- |
| `↑` `↓` `enter` | Move / open (directories navigate, files load) |
| `u` | Upload the selected file into the workspace |
| `l` | Load the selected file as a query |

## Viewer
| Key | Action |
| --- | --- |
| `↑` `↓` | Previous / next spectrum |
| `+` `-` | Zoom the m/z axis |
| `z` | Reset zoom |
| `p` | Toggle the precursor marker |

## Identify
| Key | Action |
| --- | --- |
| `r` | Run the search |
| `↑` `↓` | Browse hits (mirror plot updates) |

## Diagnostics
| Key | Action |
| --- | --- |
| `g` | Reload the quarantine log |

## Global
| Key | Action |
| --- | --- |
| `ctrl+q` | Quit |
"""


# ---------------------------------------------------------------------------
# Plot widgets
# ---------------------------------------------------------------------------


class SpectrumPlot(Static):
    """A centroid stick plot rendered from pure text glyphs."""

    can_focus = False

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        classes: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._mz: np.ndarray = np.zeros(0, dtype=np.float64)
        self._intensities: np.ndarray = np.zeros(0, dtype=np.float64)
        self._title = ""
        self._precursor_mz: Optional[float] = None
        self._show_precursor = True
        self._zoom = 1.0
        self._center: Optional[float] = None

    def set_peaks(
        self,
        mz: np.ndarray,
        intensities: np.ndarray,
        *,
        title: str = "",
        precursor_mz: Optional[float] = None,
    ) -> None:
        """Set the peaks to draw and reset the zoom window."""
        self._mz = np.asarray(mz, dtype=np.float64)
        self._intensities = np.asarray(intensities, dtype=np.float64)
        self._title = title
        self._precursor_mz = precursor_mz
        self._zoom = 1.0
        self._center = None
        self.refresh()

    def zoom_in(self) -> None:
        self._zoom = max(self._zoom * 0.5, 0.01)
        self.refresh()

    def zoom_out(self) -> None:
        self._zoom = min(self._zoom * 2.0, 8.0)
        self.refresh()

    def zoom_reset(self) -> None:
        self._zoom = 1.0
        self._center = None
        self.refresh()

    def toggle_precursor(self) -> None:
        self._show_precursor = not self._show_precursor
        self.refresh()

    def _window(self) -> tuple[float, float]:
        if self._mz.size == 0:
            return 0.0, 1.0
        full_min = float(np.min(self._mz))
        full_max = float(np.max(self._mz))
        if full_max <= full_min:
            full_max = full_min + 1.0
        span = (full_max - full_min) * self._zoom
        center = (
            self._center if self._center is not None else (full_min + full_max) / 2.0
        )
        return center - span / 2.0, center + span / 2.0

    def render(self) -> str:
        width = max((self.size.width or 80) - 2, 24)
        height = max((self.size.height or 15) - 2, 4)
        x_min, x_max = self._window()
        marker = self._precursor_mz if self._show_precursor else None
        lines = render_stick_plot(
            self._mz,
            self._intensities,
            width=width,
            height=height,
            x_min=x_min,
            x_max=x_max,
            marker_mz=marker,
            title=self._title,
        )
        return "\n".join(lines)


class MirrorPlot(Static):
    """Mirror plot: query peaks up, reference peaks down."""

    can_focus = False

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        id: Optional[str] = None,  # noqa: A002
        classes: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._query_mz: np.ndarray = np.zeros(0, dtype=np.float64)
        self._query_intensities: np.ndarray = np.zeros(0, dtype=np.float64)
        self._reference_mz: np.ndarray = np.zeros(0, dtype=np.float64)
        self._reference_intensities: np.ndarray = np.zeros(0, dtype=np.float64)
        self._tolerance = 0.02
        self._title = ""

    def set_pair(
        self,
        query_mz: np.ndarray,
        query_intensities: np.ndarray,
        reference_mz: np.ndarray,
        reference_intensities: np.ndarray,
        *,
        tolerance: float,
        title: str = "",
    ) -> None:
        self._query_mz = np.asarray(query_mz, dtype=np.float64)
        self._query_intensities = np.asarray(query_intensities, dtype=np.float64)
        self._reference_mz = np.asarray(reference_mz, dtype=np.float64)
        self._reference_intensities = np.asarray(
            reference_intensities, dtype=np.float64
        )
        self._tolerance = tolerance
        self._title = title
        self.refresh()

    def clear(self) -> None:
        self._query_mz = np.zeros(0, dtype=np.float64)
        self._query_intensities = np.zeros(0, dtype=np.float64)
        self._reference_mz = np.zeros(0, dtype=np.float64)
        self._reference_intensities = np.zeros(0, dtype=np.float64)
        self.refresh()

    def render(self) -> str:
        width = max((self.size.width or 80) - 2, 24)
        half_height = max(((self.size.height or 15) - 2) // 2, 3)
        lines = render_mirror_plot(
            self._query_mz,
            self._query_intensities,
            self._reference_mz,
            self._reference_intensities,
            tolerance=self._tolerance,
            width=width,
            half_height=half_height,
            title=self._title,
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Panes
# ---------------------------------------------------------------------------


class PaneBase(Container):
    """Base class for console panes with a typed ``app`` reference.

    Textual types ``app`` as the generic ``App``; the console panes are always
    mounted inside :class:`MassFlowApp`, so this narrows the type so mypy can
    see the session-state helpers (``report_problem``, ``loaded``, ...).
    """

    @property
    def app(self) -> "MassFlowApp":  # type: ignore[override]
        return cast("MassFlowApp", super().app)


class BrowserPane(PaneBase):
    """Find / upload / load pane."""

    def compose(self) -> ComposeResult:
        yield Label("Find", classes="section-label")
        yield Input(
            placeholder="Directory to scan — Enter to scan recursively",
            id="browser-path-input",
        )
        with Horizontal(id="browser-body"):
            with Vertical(id="browser-files"):
                yield Static(
                    "Scan a directory to see spectral files.", id="file-list-empty"
                )
                yield ListView(id="browser-file-list")
            with Vertical(id="browser-side"):
                yield Static("Nothing selected.", id="file-info")
                yield Label("Library", classes="section-label")
                yield Input(
                    placeholder="Library path (.msp / .mgf / .db / .zarr)",
                    id="browser-library-input",
                )
                yield Static("No library selected.", id="library-info")
                with Horizontal():
                    yield Button("Inspect", id="inspect-library-button")
                    yield Button("Use as library", id="use-library-button")

    def on_mount(self) -> None:
        self.query_one("#browser-path-input", Input).value = str(self.app.current_dir)
        self._scan_directory(self.app.current_dir)

    @work(thread=True, exclusive=True, group="scan")
    def _scan_directory(self, directory: Path) -> None:
        from MassFlow.tui.files import discover_spectral_files

        try:
            entries = discover_spectral_files(directory, max_depth=6)
        except OSError as exception:
            self.app.call_from_thread(self._on_scan_failed, exception)
            return
        self.app.call_from_thread(self._on_scan_done, directory, entries)

    def _on_scan_failed(self, exception: OSError) -> None:
        self.app.report_problem(exception, stage="scan")

    def _on_scan_done(self, directory: Path, entries: list[FileEntry]) -> None:
        self.app.current_dir = directory
        self.query_one("#browser-path-input", Input).value = str(directory)
        file_list = self.query_one("#browser-file-list", ListView)
        file_list.clear()
        self._entries = entries
        if not entries:
            self.query_one("#file-list-empty", Static).update(
                f"No spectral files under {directory}"
            )
            self.query_one("#file-list-empty", Static).display = True
            file_list.display = False
            return
        self.query_one("#file-list-empty", Static).display = False
        file_list.display = True
        for entry in entries:
            label = self._entry_label(entry)
            file_list.append(ListItem(Label(label)))
        file_list.index = 0

    def _entry_label(self, entry: FileEntry) -> str:
        glyph = _KIND_GLYPH.get(entry.kind, "·")
        color = _KIND_COLOR.get(entry.kind, "#55606e")
        name = entry.display_name
        if entry.kind == "vendor":
            name += "  (vendor — convert before loading)"
        return f"[{color}]{glyph}[/{color}] {name}"

    @on(Input.Submitted, "#browser-path-input")
    def _on_path_submitted(self, event: Input.Submitted) -> None:
        path = Path(event.value or "").expanduser()
        if path.is_dir():
            self._scan_directory(path)
        elif path.is_file():
            self.app.load_query_file(path)
        else:
            self.app.report_problem(
                FileNotFoundError(f"Path does not exist: {path}"), stage="browser"
            )

    @on(ListView.Selected, "#browser-file-list")
    def _on_file_selected(self, event: ListView.Selected) -> None:
        index = self._selected_index(event.item)
        if index is None or index >= len(self._entries):
            return
        entry = self._entries[index]
        self.query_one("#file-info", Static).update(self._file_info_text(entry))
        if entry.kind == "vendor":
            self.app.report_problem(
                TuiError(
                    f"{entry.path.name} is a vendor format.",
                    stage="load-query",
                    hint=(
                        "Convert it to an open format first: massflow convert "
                        "--input <dir> --output <dir>"
                    ),
                ),
                stage="load-query",
            )
            return
        if entry.kind in {"query", "library", "database"}:
            self.app.load_query_file(entry.path)

    @on(ListView.Highlighted, "#browser-file-list")
    def _on_file_highlighted(self, event: ListView.Highlighted) -> None:
        index = self._selected_index(event.item)
        if index is None or index >= len(self._entries):
            return
        entry = self._entries[index]
        self.query_one("#file-info", Static).update(self._file_info_text(entry))

    def _selected_index(self, item: Optional[ListItem]) -> Optional[int]:
        if item is None:
            return None
        file_list = self.query_one("#browser-file-list", ListView)
        try:
            return file_list.children.index(item)
        except ValueError:
            return None

    def _file_info_text(self, entry: FileEntry) -> str:
        kind_color = _KIND_COLOR.get(entry.kind, "#55606e")
        return (
            f"[bold]{entry.path.name}[/bold]\n"
            f"path: {entry.path}\n"
            f"kind: [{kind_color}]{entry.kind}[/{kind_color}]\n"
            f"size: {human_size(entry.size_bytes)}\n"
            f"format: {entry.format_hint or 'n/a'}"
        )

    @on(Input.Submitted, "#browser-library-input")
    def _on_library_submitted(self, event: Input.Submitted) -> None:
        path = Path(event.value or "").expanduser()
        if not path.exists():
            self.app.report_problem(
                FileNotFoundError(f"Library does not exist: {path}"),
                stage="load-library",
            )
            return
        self.app.set_library_path(path)

    @on(Button.Pressed, "#inspect-library-button")
    def _on_inspect_library(self) -> None:
        path = Path(
            self.query_one("#browser-library-input", Input).value or ""
        ).expanduser()
        if path.exists():
            self.app.inspect_library_async(path)
        else:
            self.app.report_problem(
                FileNotFoundError(f"Library does not exist: {path}"),
                stage="load-library",
            )

    @on(Button.Pressed, "#use-library-button")
    def _on_use_library(self) -> None:
        path = Path(
            self.query_one("#browser-library-input", Input).value or ""
        ).expanduser()
        if path.exists():
            self.app.set_library_path(path)
        else:
            self.app.report_problem(
                FileNotFoundError(f"Library does not exist: {path}"),
                stage="load-library",
            )


class ViewerPane(PaneBase):
    """Spectrum-by-spectrum inspection pane."""

    def compose(self) -> ComposeResult:
        with Horizontal(id="viewer-body"):
            with Vertical(id="viewer-list-column"):
                yield Label("Spectra", classes="section-label")
                yield OptionList(id="viewer-spectrum-list")
            with Vertical(id="viewer-plot-column"):
                yield SpectrumPlot(id="viewer-plot")
                yield Static("Load a file to inspect its spectra.", id="viewer-meta")

    def populate(self, loaded: QueryLoadResult) -> None:
        option_list = self.query_one("#viewer-spectrum-list", OptionList)
        option_list.clear_options()
        labels = []
        for summary in loaded.summaries:
            labels.append(
                f"{summary.spectrum_id[:32]}  ·  "
                f"m/z {format_mz(summary.precursor_mz)}  ·  "
                f"{summary.num_peaks} peaks"
            )
        option_list.add_options(labels)
        if labels:
            option_list.highlighted = 0
            self._show_summary(0)

    def _show_summary(self, index: int) -> None:
        loaded = self.app.loaded
        if loaded is None or not loaded.summaries:
            return
        index = max(0, min(index, len(loaded.summaries) - 1))
        summary = loaded.summaries[index]
        self.app.viewer_index = index
        plot = self.query_one("#viewer-plot", SpectrumPlot)
        plot.set_peaks(
            summary.mz_array,
            summary.intensity_array,
            title=f"{summary.spectrum_id[:48]}",
            precursor_mz=summary.precursor_mz,
        )
        entropy = (
            f"{summary.spectral_entropy:.3f} nats"
            if summary.spectral_entropy is not None
            else "n/a"
        )
        base_peak = (
            f"m/z {format_mz(summary.base_peak_mz)}"
            if summary.base_peak_mz is not None
            else "n/a"
        )
        meta = (
            f"[bold]{summary.spectrum_id}[/bold]\n"
            f"precursor m/z: {format_mz(summary.precursor_mz)}\n"
            f"retention time: {format_retention_time(summary.retention_time_seconds)}\n"
            f"peaks: {summary.num_peaks}   base peak: {base_peak}\n"
            f"charge: {summary.charge if summary.charge is not None else 'n/a'}"
            f"   ionmode: {summary.ionmode or 'n/a'}\n"
            f"adduct: {summary.adduct or 'n/a'}\n"
            f"compound: {summary.compound_name or 'n/a'}\n"
            f"TIC: {summary.total_ion_current:.3g}   entropy: {entropy}\n"
            f"spectrum {index + 1}/{len(loaded.summaries)}"
        )
        self.query_one("#viewer-meta", Static).update(meta)

    @on(OptionList.OptionHighlighted, "#viewer-spectrum-list")
    def _on_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        index = self.query_one("#viewer-spectrum-list", OptionList).highlighted
        if index is not None:
            self._show_summary(int(index))


class IdentifyPane(PaneBase):
    """Similarity search pane."""

    def compose(self) -> ComposeResult:
        yield Label("Identify", classes="section-label")
        with Horizontal(classes="param-row"):
            yield Select(
                [(algorithm, algorithm) for algorithm in ALGORITHMS],
                prompt="engine",
                value="modified_cosine",
                id="identify-algorithm",
                allow_blank=False,
            )
            yield Input(value="0.6", placeholder="min score", id="identify-min-score")
            yield Input(value="10", placeholder="top n", id="identify-top-n")
            yield Input(value="0.05", placeholder="fdr", id="identify-fdr")
            yield Button("Run", variant="primary", id="identify-run")
        yield Static(
            "Load a query file and set a library, then press Run.",
            id="identify-status",
        )
        with Horizontal(id="identify-body"):
            with Vertical(id="identify-results"):
                yield DataTable(
                    id="identify-table", cursor_type="row", zebra_stripes=True
                )
            with Vertical(id="identify-detail"):
                yield Static(
                    "Select a hit to see the mirror plot.", id="identify-hit-detail"
                )
                yield MirrorPlot(id="identify-mirror")

    def on_mount(self) -> None:
        table = self.query_one("#identify-table", DataTable)
        table.add_column("score", key="score", width=7)
        table.add_column("gauge", key="gauge", width=12)
        table.add_column("reference", key="reference")
        table.add_column("matched", key="matched", width=7)
        table.add_column("q-value", key="q", width=8)
        table.add_column("tier", key="tier", width=10)
        table.add_column("Δppm", key="ppm", width=8)

    def populate(self, outcome: IdentificationOutcome) -> None:
        table = self.query_one("#identify-table", DataTable)
        table.clear()
        self._hits: list[SearchHit] = []
        for hit in outcome.hits:
            row_key = str(len(self._hits))
            table.add_row(
                f"{hit.score:.3f}",
                render_score_gauge(hit.score, width=12),
                (hit.reference_name or hit.reference_id)[:40],
                str(hit.matched_peaks),
                f"{hit.q_value:.3f}" if hit.q_value is not None else "n/a",
                hit.annotation_tier or "—",
                (
                    f"{hit.mass_error_ppm:.1f}"
                    if hit.mass_error_ppm is not None
                    else "n/a"
                ),
                key=row_key,
            )
            self._hits.append(hit)

        if outcome.hits:
            table.move_cursor(row=0)
            self._show_hit(outcome.hits[0])

        status = self.query_one("#identify-status", Static)
        warnings = (
            "\n".join(f"⚠ {warning}" for warning in outcome.warnings)
            if outcome.warnings
            else "no warnings"
        )
        status.update(
            f"[bold green]done[/bold green] — {outcome.num_hits} hits across "
            f"{outcome.queries_with_hits}/{outcome.num_queries} queries "
            f"({outcome.duration_seconds:.1f}s, engine: {outcome.engine_used})\n"
            f"{warnings}"
        )

    @staticmethod
    def _fmt(value: Optional[float], spec: str) -> str:
        """Format a float for display, degrading to ``n/a`` for ``None``."""
        return f"{value:{spec}}" if value is not None else "n/a"

    def _show_hit(self, hit: SearchHit) -> None:
        outcome = self.app.last_outcome
        query_peaks = (
            outcome.query_peaks.get(hit.query_id) if outcome is not None else None
        )
        reference_peaks = (
            outcome.hit_reference_peaks.get(hit.reference_id)
            if outcome is not None
            else None
        )

        status = annotation_status(hit.score)
        status_color = {
            "matched": "#3aff8c",
            "putative": "#ffb454",
            "unknown": "#ff5470",
        }[status]
        detail = (
            f"[bold]{hit.reference_name or hit.reference_id}[/bold]\n"
            f"query: {hit.query_id[:40]}\n"
            f"score: {hit.score:.3f}  [{status_color}]{status}[/{status_color}]\n"
            f"matched peaks: {hit.matched_peaks}\n"
            f"q-value: {self._fmt(hit.q_value, '.4f')}   "
            f"p-value: {self._fmt(hit.p_value, '.4g')}\n"
            f"Δmass: {self._fmt(hit.mass_error_ppm, '.1f')} ppm\n"
            f"SMILES: {hit.smiles or 'n/a'}\n"
            f"InChIKey: {hit.inchikey or 'n/a'}\n"
            f"tier: {hit.annotation_tier or '—'}   "
            f"structural: {self._fmt(hit.structural_similarity, '.3f')}"
        )
        self.query_one("#identify-hit-detail", Static).update(detail)

        mirror = self.query_one("#identify-mirror", MirrorPlot)
        if query_peaks is not None and reference_peaks is not None:
            query_mz, query_intensities = query_peaks
            reference_mz, reference_intensities = reference_peaks
            if query_mz.size and reference_mz.size:
                mirror.set_pair(
                    query_mz,
                    query_intensities,
                    reference_mz,
                    reference_intensities,
                    tolerance=self.app._identify_tolerance(),
                    title=(f"query {hit.query_id[:32]}  vs  {hit.reference_id[:32]}"),
                )
            else:
                mirror.clear()
        else:
            mirror.clear()

    @on(DataTable.RowSelected, "#identify-table")
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        index = getattr(event, "cursor_row", -1)
        if 0 <= index < len(self._hits):
            self._show_hit(self._hits[index])

    @on(Button.Pressed, "#identify-run")
    def _on_run(self) -> None:
        self.app.action_run_identification()


class DiagnosticsPane(PaneBase):
    """Problems + quarantine pane."""

    def compose(self) -> ComposeResult:
        yield Label("Problems", classes="section-label")
        yield Static(
            "No problems recorded. The void is calm.", id="diagnostics-problems"
        )
        yield Label(
            "Quarantine (spectra rejected by validation)", classes="section-label"
        )
        yield Static(
            "Press g to reload the quarantine log.", id="diagnostics-quarantine"
        )

    def on_mount(self) -> None:
        self.refresh_content()

    def refresh_content(self) -> None:
        if self.app.problems:
            blocks = [problem.to_text() for problem in self.app.problems[-5:]]
            self.query_one("#diagnostics-problems", Static).update("\n\n".join(blocks))
        else:
            self.query_one("#diagnostics-problems", Static).update(
                "No problems recorded. The void is calm."
            )
        entries: list[QuarantineEntry] = parse_quarantine_log(self.app.quarantine_path)
        if entries:
            lines = [entry.message for entry in entries[-40:]]
            self.query_one("#diagnostics-quarantine", Static).update("\n".join(lines))
        else:
            self.query_one("#diagnostics-quarantine", Static).update(
                f"No quarantine entries in {self.app.quarantine_path}."
            )


class HelpModal(ModalScreen):
    """Modal help overlay."""

    BINDINGS = [
        Binding("escape", "dismiss_modal", "Close"),
        Binding("question_mark", "dismiss_modal", "Close", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Markdown(HELP_MARKDOWN)

    def action_dismiss_modal(self) -> None:
        self.dismiss()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class MassFlowApp(App[None]):
    """The MassFlow terminal console application."""

    TITLE = "MassFlow"
    SUB_TITLE = "console"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("question_mark", "help", "Help", priority=True),
        Binding("f1", "help", "Help", priority=True, show=False),
        Binding("1", "focus_browser", "Browser", show=False),
        Binding("2", "focus_viewer", "Viewer", show=False),
        Binding("3", "focus_identify", "Identify", show=False),
        Binding("4", "focus_diagnostics", "Diagnostics", show=False),
        Binding("u", "upload_selected", "Upload"),
        Binding("l", "load_selected", "Load"),
        Binding("r", "run_identification", "Identify"),
        Binding("g", "reload_quarantine", "Quarantine"),
        Binding("plus", "zoom_in", "Zoom in", show=False),
        Binding("equals_sign", "zoom_in", "Zoom in", show=False),
        Binding("minus", "zoom_out", "Zoom out", show=False),
        Binding("z", "zoom_reset", "Zoom reset", show=False),
        Binding("p", "toggle_precursor", "Precursor", show=False),
    ]

    CSS = """
    Screen {
        background: #0b0f14;
        color: #c8d3dc;
    }

    Header {
        background: #0b0f14;
        color: #3aff8c;
    }

    Footer {
        background: #0b0f14;
    }

    .section-label {
        color: #ff5ad1;
        text-style: bold;
        padding: 0 1;
        margin-top: 1;
    }

    #browser-body, #viewer-body, #identify-body {
        height: 1fr;
    }

    BrowserPane, ViewerPane, IdentifyPane, DiagnosticsPane {
        height: 1fr;
    }

    #browser-files {
        width: 2fr;
        border: round #1c2733;
        padding: 0 1;
    }

    #browser-side {
        width: 1fr;
        border: round #1c2733;
        padding: 0 1;
    }

    #file-info, #library-info, #viewer-meta, #identify-hit-detail {
        height: auto;
        min-height: 8;
        border: tall #1c2733;
        padding: 0 1;
        margin-bottom: 1;
    }

    #viewer-list-column {
        width: 1fr;
        border: round #1c2733;
        padding: 0 1;
    }

    #viewer-plot-column {
        width: 3fr;
        border: round #1c2733;
        padding: 0 1;
    }

    #viewer-plot {
        height: 18;
        color: #2de2e6;
    }

    #identify-results {
        width: 3fr;
        border: round #1c2733;
    }

    #identify-detail {
        width: 2fr;
        border: round #1c2733;
        padding: 0 1;
    }

    #identify-mirror {
        height: 16;
        color: #2de2e6;
    }

    #identify-status, #diagnostics-problems, #diagnostics-quarantine {
        height: auto;
        min-height: 6;
        max-height: 16;
        border: tall #1c2733;
        padding: 0 1;
        margin-bottom: 1;
    }

    .param-row {
        height: 3;
        margin-bottom: 1;
    }

    .param-row Input {
        width: 10;
        margin-right: 1;
    }

    .param-row Select {
        width: 24;
        margin-right: 1;
    }

    DataTable {
        border: none;
    }

    DataTable > .datatable--cursor {
        background: #14202c;
    }

    Button {
        margin-right: 1;
    }

    Markdown {
        padding: 1 2;
    }
    """

    def __init__(
        self,
        *,
        initial_query: Optional[Path] = None,
        initial_library: Optional[Path] = None,
        workspace: Optional[Path] = None,
        quarantine_path: Optional[Path] = None,
        current_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.initial_query = initial_query
        self.initial_library = initial_library
        self.workspace_dir = workspace or (Path.cwd() / "massflow_workspace")
        self.quarantine_path = quarantine_path or Path("massflow_quarantine.log")

        # Mutable session state.
        self.current_dir = current_dir or Path.cwd()
        self.loaded: Optional[QueryLoadResult] = None
        self.library_path: Optional[Path] = None
        self.library_info: Optional[LibraryInfo] = None
        self.last_outcome: Optional[IdentificationOutcome] = None
        self.problems: list[Problem] = []
        self.viewer_index = 0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(id="main-tabs"):
            with TabPane("Browser", id="browser"):
                yield BrowserPane()
            with TabPane("Viewer", id="viewer"):
                yield ViewerPane()
            with TabPane("Identify", id="identify"):
                yield IdentifyPane()
            with TabPane("Diagnostics", id="diagnostics"):
                yield DiagnosticsPane()
        yield Footer()

    def on_mount(self) -> None:
        if self.initial_query is not None:
            self.load_query_file(self.initial_query)
        if self.initial_library is not None:
            self.set_library_path(self.initial_library)

    # --- tab focus helpers --------------------------------------------------

    def _tabs(self) -> TabbedContent:
        return self.query_one("#main-tabs", TabbedContent)

    def _active_pane(self) -> str:
        active = self._tabs().active
        return str(active) if active is not None else "browser"

    def action_focus_browser(self) -> None:
        self._tabs().active = "browser"

    def action_focus_viewer(self) -> None:
        self._tabs().active = "viewer"

    def action_focus_identify(self) -> None:
        self._tabs().active = "identify"

    def action_focus_diagnostics(self) -> None:
        self._tabs().active = "diagnostics"
        self.query_one(DiagnosticsPane).refresh_content()

    # --- viewer zoom actions (gated to the viewer tab) -----------------------

    def action_zoom_in(self) -> None:
        if self._active_pane() == "viewer":
            self.query_one("#viewer-plot", SpectrumPlot).zoom_in()

    def action_zoom_out(self) -> None:
        if self._active_pane() == "viewer":
            self.query_one("#viewer-plot", SpectrumPlot).zoom_out()

    def action_zoom_reset(self) -> None:
        if self._active_pane() == "viewer":
            self.query_one("#viewer-plot", SpectrumPlot).zoom_reset()

    def action_toggle_precursor(self) -> None:
        if self._active_pane() == "viewer":
            self.query_one("#viewer-plot", SpectrumPlot).toggle_precursor()

    # --- cross-pane actions ----------------------------------------------------

    def action_upload_selected(self) -> None:
        if self._active_pane() != "browser":
            return
        self.upload_selected()

    def action_load_selected(self) -> None:
        if self._active_pane() != "browser":
            return
        self.load_selected()

    def action_run_identification(self) -> None:
        self.run_identification_request()

    def action_reload_quarantine(self) -> None:
        if self._active_pane() == "diagnostics":
            self.query_one(DiagnosticsPane).refresh_content()
            self.notify("Quarantine log reloaded.", title="Diagnostics")

    # --- actions ------------------------------------------------------------

    def action_help(self) -> None:
        self.push_screen(HelpModal())

    def upload_selected(self) -> None:
        entry = self._browser_selected_entry()
        if entry is None:
            self.notify("Select a file first.", title="Upload", severity="warning")
            return
        self._upload_worker(entry.path)

    @work(thread=True, exclusive=True, group="upload")
    def _upload_worker(self, source: Path) -> None:
        try:
            destination = copy_into_workspace(source, self.workspace_dir)
        except OSError as exception:
            self.call_from_thread(self.report_problem, exception, "upload")
            return
        self.call_from_thread(self._on_upload_done, source, destination)

    def _on_upload_done(self, source: Path, destination: Path) -> None:
        self.notify(
            f"{source.name} → {destination}", title="Uploaded", severity="information"
        )
        self.current_dir = destination.parent
        self.query_one(BrowserPane)._scan_directory(destination.parent)

    def load_selected(self) -> None:
        entry = self._browser_selected_entry()
        if entry is None:
            self.notify("Select a file first.", title="Load", severity="warning")
            return
        self.load_query_file(entry.path)

    def _browser_selected_entry(self) -> Optional[FileEntry]:
        pane = self.query_one(BrowserPane)
        file_list = pane.query_one("#browser-file-list", ListView)
        index = file_list.index
        if index is None:
            return None
        entries = getattr(pane, "_entries", [])
        if 0 <= index < len(entries):
            return entries[index]
        return None

    def set_library_path(self, path: Path) -> None:
        self.library_path = path
        self.query_one("#browser-library-input", Input).value = str(path)
        self.inspect_library_async(path)

    @work(thread=True, exclusive=True, group="inspect-library")
    def inspect_library_async(self, path: Path) -> None:
        info = inspect_library(path)
        self.call_from_thread(self._on_library_inspected, info)

    def _on_library_inspected(self, info: LibraryInfo) -> None:
        self.library_info = info
        static = self.query_one("#library-info", Static)
        if info.error:
            static.update(f"[bold red]error[/bold red] {info.error}")
            return
        truncated = " (>200k)" if info.truncated else ""
        mz_range = (
            f"{info.precursor_mz_range[0]:.4f}–{info.precursor_mz_range[1]:.4f}"
            if info.precursor_mz_range
            else "n/a"
        )
        categories = (
            ", ".join(f"{name}: {count}" for name, count in info.categories.items())
            or "none"
        )
        static.update(
            f"[bold]{info.path.name}[/bold]\n"
            f"backend: {info.backend}\n"
            f"spectra: {info.total_spectra if info.total_spectra is not None else 'n/a'}{truncated}\n"
            f"precursor m/z: {mz_range}\n"
            f"categories: {categories}"
        )

    def load_query_file(self, path: Path) -> None:
        self._load_query_worker(path)

    @work(thread=True, exclusive=True, group="load-query")
    def _load_query_worker(self, path: Path) -> None:
        try:
            result = load_query_preview(path)
        except TuiError as error:
            self.call_from_thread(self._on_load_failed, error)
            return
        except Exception as exception:  # defensive: never crash the worker
            self.call_from_thread(
                self._on_load_failed,
                TuiError.from_exception(exception, stage="load-query"),
            )
            return
        self.call_from_thread(self._on_query_loaded, result)

    def _on_load_failed(self, error: TuiError) -> None:
        self.report_problem(error, stage=error.stage)

    def _on_query_loaded(self, result: QueryLoadResult) -> None:
        self.loaded = result
        self.viewer_index = 0
        self.query_one(ViewerPane).populate(result)
        self.query_one("#identify-status", Static).update(
            f"Query file ready: {result.path.name} ({len(result.summaries)} spectra)"
        )
        self.query_one("#main-tabs", TabbedContent).active = "viewer"
        if result.quarantined_messages:
            self.notify(
                f"{len(result.quarantined_messages)} spectra quarantined "
                "(see Diagnostics).",
                title="Load",
                severity="warning",
            )
        else:
            self.notify(
                f"Loaded {len(result.summaries)} spectra from {result.path.name}",
                title="Load",
            )

    def run_identification_request(self) -> None:
        loaded = self.loaded
        if loaded is None:
            self.report_problem(
                TuiError(
                    "No query file loaded.",
                    stage="identify",
                    hint="Load an experimental file in the Browser tab first.",
                ),
                stage="identify",
            )
            return
        if self.library_path is None:
            self.report_problem(
                TuiError(
                    "No library selected.",
                    stage="identify",
                    hint="Set a library path (.msp/.mgf/.db/.zarr) in the Browser tab.",
                ),
                stage="identify",
            )
            return

        pane = self.query_one(IdentifyPane)
        algorithm = pane.query_one("#identify-algorithm", Select).value
        if algorithm is None or algorithm == Select.BLANK:
            algorithm = "modified_cosine"
        try:
            min_score = float(pane.query_one("#identify-min-score", Input).value)
            top_n = int(pane.query_one("#identify-top-n", Input).value)
            fdr_threshold = float(pane.query_one("#identify-fdr", Input).value)
        except ValueError as exception:
            self.report_problem(
                TuiError(
                    f"Bad search parameter: {exception}",
                    stage="identify",
                    hint="min score and fdr are decimals (0–1); top n is an integer.",
                ),
                stage="identify",
            )
            return
        if not 0.0 <= min_score <= 1.0:
            self.report_problem(
                TuiError(f"min score {min_score} is outside 0–1.", stage="identify"),
                stage="identify",
            )
            return
        if top_n < 1:
            self.report_problem(
                TuiError(f"top n must be ≥ 1, got {top_n}.", stage="identify"),
                stage="identify",
            )
            return

        request = IdentificationRequest(
            query_path=loaded.path,
            library_path=self.library_path,
            algorithm=str(algorithm),
            min_score=min_score,
            top_n=top_n,
            fdr_threshold=fdr_threshold,
        )
        pane.query_one("#identify-status", Static).update(
            "[bold cyan]searching…[/bold cyan] (the UI stays responsive)"
        )
        self._identify_worker(request)

    def _identify_tolerance(self) -> float:
        # ms2 tolerance used by the mirror plot; defaults match the config.
        return 0.02

    @work(thread=True, exclusive=True, group="identify")
    def _identify_worker(self, request: IdentificationRequest) -> None:
        try:
            outcome = run_identification(request)
        except TuiError as error:
            self.call_from_thread(self._on_identify_failed, error)
            return
        except Exception as exception:  # defensive: never crash the worker
            self.call_from_thread(
                self._on_identify_failed,
                TuiError.from_exception(exception, stage="identify"),
            )
            return
        self.call_from_thread(self._on_identify_done, outcome)

    def _on_identify_failed(self, error: TuiError) -> None:
        self.query_one("#identify-status", Static).update(
            f"[bold red]failed[/bold red] — {error}"
        )
        self.report_problem(error, stage=error.stage)

    def _on_identify_done(self, outcome: IdentificationOutcome) -> None:
        self.last_outcome = outcome
        self.query_one(IdentifyPane).populate(outcome)
        self.notify(
            f"{outcome.num_hits} hits across {outcome.queries_with_hits} queries",
            title="Identify",
        )

    def report_problem(
        self, exception: BaseException, *, stage: str = "unknown"
    ) -> None:
        """Record a problem, notify the user, and refresh the diagnostics tab."""
        problem = (
            exception
            if isinstance(exception, Problem)
            else Problem.from_exception(exception, stage=stage)
        )
        self.problems.append(problem)
        logger.error("%s: %s", problem.stage, problem.detail)
        self.query_one(DiagnosticsPane).refresh_content()
        self.notify(
            problem.detail,
            title=f"error: {problem.stage}",
            severity="error",
            timeout=8,
        )


def run() -> None:
    """Entry point used by the ``massflow tui`` command."""
    MassFlowApp().run()


if __name__ == "__main__":
    run()
