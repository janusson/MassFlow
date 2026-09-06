"""
Tests for MassFlow.tui.app — the Textual console application.

These tests run headlessly via Textual's ``run_test`` pilot. They never touch
the core annotation modules: fake :class:`QueryLoadResult` /
:class:`IdentificationOutcome` payloads are fed straight into the UI handlers,
so the suite stays fast and deterministic.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("textual")

from MassFlow.tui.diagnostics import TuiError
from MassFlow.tui.files import FileEntry
from MassFlow.tui.state import (
    IdentificationOutcome,
    IdentificationRequest,
    QueryLoadResult,
    SearchHit,
    SpectrumSummary,
)
from MassFlow.tui.app import (
    BrowserPane,
    DiagnosticsPane,
    HelpModal,
    MassFlowApp,
    SpectrumPlot,
)

pytestmark = pytest.mark.asyncio


def make_summary(spec_id: str, precursor_mz: float) -> SpectrumSummary:
    mz = np.array([50.0, 100.0, 150.0], dtype=np.float64)
    intensities = np.array([0.5, 1.0, 0.25], dtype=np.float64)
    return SpectrumSummary(
        spectrum_id=spec_id,
        precursor_mz=precursor_mz,
        retention_time_seconds=60.0,
        num_peaks=3,
        charge=1,
        ionmode="positive",
        adduct="[M+H]+",
        compound_name=None,
        base_peak_mz=100.0,
        base_peak_intensity=1.0,
        total_ion_current=1.75,
        spectral_entropy=0.9,
        mz_array=mz,
        intensity_array=intensities,
    )


def make_loaded(tmp_path: Path) -> QueryLoadResult:
    return QueryLoadResult(
        path=tmp_path / "query.mgf",
        format_hint="mgf",
        summaries=[make_summary("s1", 100.0), make_summary("s2", 200.0)],
    )


def make_outcome(tmp_path: Path) -> IdentificationOutcome:
    hit = SearchHit.from_search_result(
        {
            "query_id": "s1",
            "query_precursor_mz": 100.0,
            "reference_id": "ref_1",
            "reference_name": "Caffeine",
            "reference_precursor_mz": 100.01,
            "score": 0.95,
            "matched_peaks": 3,
            "q_value": 0.01,
            "p_value": 0.001,
            "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            "inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N",
            "mass_error_ppm": 2.5,
            "annotation_tier": "level_1",
            "structural_similarity": 0.9,
            "score_breakdown": {"cosine": 0.95},
        }
    )
    query_mz = np.array([50.0, 100.0, 150.0], dtype=np.float64)
    query_intensities = np.array([0.5, 1.0, 0.25], dtype=np.float64)
    return IdentificationOutcome(
        request=IdentificationRequest(
            query_path=tmp_path / "query.mgf",
            library_path=tmp_path / "library.msp",
        ),
        engine_used="cosine",
        hits=[hit],
        num_queries=2,
        num_references=8,
        duration_seconds=0.1,
        fdr_threshold=0.05,
        query_peaks={"s1": (query_mz, query_intensities)},
        hit_reference_peaks={"ref_1": (query_mz, query_intensities)},
    )


class TestAppComposition:
    async def test_composes_tabs(self):
        app = MassFlowApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            for pane_id in ("browser", "viewer", "identify", "diagnostics"):
                assert app.query_one(f"#{pane_id}") is not None

    async def test_tab_switch_bindings(self):
        app = MassFlowApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("2")
            await pilot.pause()
            assert app._active_pane() == "viewer"
            await pilot.press("4")
            await pilot.pause()
            assert app._active_pane() == "diagnostics"
            await pilot.press("1")
            await pilot.pause()
            assert app._active_pane() == "browser"

    async def test_help_modal_opens_and_closes(self):
        app = MassFlowApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("?")
            await pilot.pause()
            assert isinstance(app.screen, HelpModal)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, HelpModal)


class TestViewer:
    async def test_loaded_file_populates_viewer(self, tmp_path):
        app = MassFlowApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_query_loaded(make_loaded(tmp_path))
            await pilot.pause()
            assert app._active_pane() == "viewer"
            option_list = app.query_one("#viewer-spectrum-list")
            assert option_list.option_count == 2
            meta = str(app.query_one("#viewer-meta").render())
            assert "s1" in meta
            assert "spectrum 1/2" in meta

    async def test_spectrum_plot_renders(self, tmp_path):
        app = MassFlowApp()
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app._on_query_loaded(make_loaded(tmp_path))
            await pilot.pause()
            plot = app.query_one("#viewer-plot", SpectrumPlot)
            rendered = str(plot.render())
            assert len(rendered.splitlines()) > 5

    async def test_zoom_actions(self, tmp_path):
        app = MassFlowApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_query_loaded(make_loaded(tmp_path))
            await pilot.pause()
            plot = app.query_one("#viewer-plot", SpectrumPlot)
            app.action_zoom_in()
            await pilot.pause()
            assert plot._zoom < 1.0
            app.action_zoom_reset()
            await pilot.pause()
            assert plot._zoom == 1.0
            app.action_zoom_out()
            await pilot.pause()
            assert plot._zoom > 1.0

    async def test_toggle_precursor_marker(self, tmp_path):
        app = MassFlowApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_query_loaded(make_loaded(tmp_path))
            await pilot.pause()
            plot = app.query_one("#viewer-plot", SpectrumPlot)
            before = plot._show_precursor
            app.action_toggle_precursor()
            await pilot.pause()
            assert plot._show_precursor is not before

    async def test_zoom_ignored_outside_viewer_tab(self):
        app = MassFlowApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            plot = app.query_one("#viewer-plot", SpectrumPlot)
            plot._zoom = 1.0
            app.action_zoom_in()  # active tab is "browser"
            await pilot.pause()
            assert plot._zoom == 1.0


class TestIdentify:
    async def test_populate_renders_hits(self, tmp_path):
        app = MassFlowApp()
        async with app.run_test(size=(140, 44)) as pilot:
            await pilot.pause()
            app._on_identify_done(make_outcome(tmp_path))
            await pilot.pause()
            table = app.query_one("#identify-table")
            assert table.row_count == 1
            detail = str(app.query_one("#identify-hit-detail").render())
            assert "Caffeine" in detail
            assert "0.950" in detail
            mirror = str(app.query_one("#identify-mirror").render())
            assert len(mirror.splitlines()) > 3
            status = str(app.query_one("#identify-status").render())
            assert "1 hits" in status

    async def test_requires_query_and_library(self, tmp_path):
        app = MassFlowApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.run_identification_request()
            await pilot.pause()
            assert len(app.problems) == 1
            assert app.problems[-1].stage == "identify"
            assert app.problems[-1].hint is not None

    async def test_bad_parameter_records_problem(self, tmp_path):
        app = MassFlowApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_query_loaded(make_loaded(tmp_path))
            app.library_path = tmp_path / "library.msp"
            await pilot.pause()
            app.query_one("#identify-min-score").value = "not-a-number"
            app.run_identification_request()
            await pilot.pause()
            assert len(app.problems) == 1
            assert app.problems[-1].stage == "identify"

    async def test_out_of_range_min_score(self, tmp_path):
        app = MassFlowApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app._on_query_loaded(make_loaded(tmp_path))
            app.library_path = tmp_path / "library.msp"
            await pilot.pause()
            app.query_one("#identify-min-score").value = "1.5"
            app.run_identification_request()
            await pilot.pause()
            assert len(app.problems) == 1
            assert "0–1" in app.problems[-1].detail


class TestDiagnostics:
    async def test_report_problem_updates_pane(self, tmp_path):
        app = MassFlowApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.report_problem(
                TuiError("bad vendor", stage="load-query", hint="convert first")
            )
            await pilot.pause()
            assert len(app.problems) == 1
            content = str(app.query_one("#diagnostics-problems").render())
            assert "bad vendor" in content
            assert "convert first" in content

    async def test_quarantine_log_rendered(self, tmp_path):
        log = tmp_path / "quarantine.log"
        log.write_text(
            "2026-01-01 - Quarantined Spectrum | Reason: Missing precursor_mz\n"
        )
        app = MassFlowApp(quarantine_path=log)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one(DiagnosticsPane).refresh_content()
            await pilot.pause()
            content = str(app.query_one("#diagnostics-quarantine").render())
            assert "Missing precursor_mz" in content

    async def test_empty_quarantine_message(self, tmp_path):
        app = MassFlowApp(quarantine_path=tmp_path / "nope.log")
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one(DiagnosticsPane).refresh_content()
            await pilot.pause()
            content = str(app.query_one("#diagnostics-quarantine").render())
            assert "No quarantine entries" in content


class TestBrowser:
    async def test_scan_results_populate_list(self, tmp_path):
        app = MassFlowApp(current_dir=tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            pane = app.query_one(BrowserPane)
            # Wait for the initial background scan of the (empty) temp dir to
            # settle so it cannot race with the entries injected below.
            for _ in range(50):
                await pilot.pause(0.1)
                if getattr(pane, "_entries", None) is not None:
                    break
            entries = [
                FileEntry(
                    path=tmp_path / "run.mzml",
                    kind="query",
                    format_hint="mzml",
                    size_bytes=10,
                ),
                FileEntry(
                    path=tmp_path / "vendor.raw",
                    kind="vendor",
                    format_hint="raw",
                    size_bytes=20,
                ),
            ]
            pane._on_scan_done(tmp_path, entries)
            await pilot.pause()
            file_list = app.query_one("#browser-file-list")
            assert len(file_list.children) == 2
            info = str(app.query_one("#file-info").render())
            assert "run.mzml" in info

    async def test_upload_missing_selection_warns(self):
        app = MassFlowApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.upload_selected()
            await pilot.pause()
            assert app.problems == []  # a soft notification, not a problem
