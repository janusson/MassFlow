# MassFlow Product Contract & Capability Matrix

> **STATUS: AUTHORITATIVE — single source of truth for what MassFlow is, what it ships, and what it must not be mistaken for.**
>
> This document is the result of a full repository reconciliation audit (2026-08-25).
> Where this document conflicts with `README.md`, `ARCHITECTURE.md`, `AGENTS.md`,
> `.github/copilot-instructions.md`, `docs/index.md`, `docs/CHANGELOG.md`, or the
> external research specification, **this document governs** until those files are
> updated to match it.

---

## 1. Authoritative product scope (the contract)

**MassFlow v0.1 is a local-first, config-driven MS/MS annotation engine for
small-molecule / metabolomics-style spectral libraries.** It ingests open-format
spectral files, processes spectra with `matchms`, searches them against local
reference libraries with classical spectral similarity, calibrates confidence
with target-decoy FDR, and exports tabular results with provenance sidecars.

The stable product surface (must not regress; all `@pytest.mark.core`-eligible
paths are expected to pass):

| Surface | Detail |
| --- | --- |
| CLI commands | `massflow init`, `tutorial`, `annotate`, `convert`, `db build`, `db inspect`, `db merge` |
| Configuration | YAML → `MassFlowConfig` (`project`, `input`, `processing`, `similarity`, `export`) |
| Ingestion | `mzML`, `mzXML`, `MGF`, `MSP`, MassFlow SQLite (`.db`/`.sqlite`); vendor raw formats rejected with `UnsupportedVendorFormatError` |
| Processing | `matchms`-based metadata cleaning, RT extraction, intensity/m/z filters, top-N reduction, normalization |
| Similarity | `cosine`, `modified_cosine` (always available) |
| Confidence | Per-query target-decoy competition (TDC): entropy-preserving decoy generation, per-query q-values, empirical p-values as diagnostic (see the [FDR statistical contract](user-guide/scoring_logic.md)) |
| Storage | SQLite libraries (`massflow db build/inspect/merge`); optional hybrid SQLite+Zarr backend; BLOB→Zarr migration scripts |
| Exports | Per-input CSV and mzTab-M result tables, YAML provenance sidecar with per-file status/spectra counts/degradation flags (`<input_stem>_results.<ext>` + `.report.yaml`), explicit `<stem>_failed.report.yaml` for failed files (never an empty CSV), MSP/MGF spectrum export |
| Scientific integrity | 5 ppm precursor validation and theoretical isotopic envelopes in the `MassFlow.models` / `MassFlow.cheminformatics` layer; streaming ingestion gate |
| Python API | `MassFlow.config`, `io`, `processing`, `similarity`, `database`, `storage`, `zarr_store`, `workflow`, `models`, `cheminformatics`, `protocols`, `ml_client` |

**What is NOT in the v0.1 contract:**

- Peptide/proteomics search of any kind (see §6).
- TIIP indexing, O(1) peptide retrieval, BIN/BEST consensus spectra (see §6).
- SIRIUS / MS-GF+ / MaRaCluster / FragPipe interoperability (see §6).
- FBMN export, GraphML molecular networking, `massflow visualize` — documented
  in places but **not implemented** (§4.4). Treat as aspirational until shipped.
- ML engines (`spec2vec`, `ms2deepscore`), meta-engines (`consensus`, `cascade`),
  routing, streaming, TUI, HNSW, and `watch` — all **experimental**, not part of
  the support promise (§4.2).

---

## 2. Capability matrix

Status legend: **✅ Stable** (implemented, tested, part of the v0.1 contract) ·
**🧪 Experimental** (implemented and tested, outside the support promise) ·
**🟡 Partial** (implemented for a subset of the documented surface) ·
**📕 Missing** (documented somewhere in-repo, no code) · **📋 Planned** (roadmap/research only).

### 2.1 Implemented and tested — stable core (✅)

| Capability | Where | Test evidence |
| --- | --- | --- |
| `massflow annotate` config-driven pipeline | `cli.py`, `workflow.py` (`FileExecutionResult` per-file failure model) | `test_workflow.py`, `test_cli.py`, `test_end_to_end_mvp.py`, `test_pipeline_integrity.py`, `test_annotation_coverage.py`, `test_failure_model.py` |
| `massflow init` / `massflow tutorial` | `cli.py`, `scripts/generate_tutorial_data.py` | `test_cli.py` |
| YAML config schema + validation with line numbers | `config.py` | `test_config.py`, `test_core_edge_cases.py` |
| mzML / mzXML / MGF / MSP ingestion; vendor rejection; quarantine log | `io.py` | `test_io.py`, `test_io_validation_layer.py`, `test_validation_scenarios.py` |
| `matchms` processing pipeline | `processing.py` | `test_processing.py` |
| `cosine`, `modified_cosine` scoring | `similarity.py` | `test_mathematical_proof.py`, `test_similarity.py`, `test_ms1_prefilter.py` |
| MS1 prefilter (Da and ppm modes) | `similarity.py` | `test_ms1_prefilter.py` |
| Entropy-preserving decoys + per-query target-decoy FDR + diagnostic p-values | `similarity.py` (`generate_decoys`, `calculate_fdr`, `calibrate_query_level_fdr`), `workflow.py` | `test_decoy_generation.py`, `test_fdr.py`, `test_fdr_statistics.py` (contract tests) |
| SQLite library build / inspect / merge | `database.py`, `cli.py` | `test_database.py`, `test_cli_db.py` |
| CSV, mzTab-M, YAML report, MSP/MGF export | `io.py`, `workflow.py` | `test_io.py`, `test_workflow.py`, `test_cli.py` |
| 5 ppm precursor validation + isotopic envelopes (model layer) | `models.py`, `cheminformatics.py` | `test_precursor_physics.py`, `test_scientific_boundaries.py`, `test_cheminformatics.py`, `test_isotopic_distribution.py`, `test_adduct_validation.py` |

### 2.2 Implemented and tested — experimental (🧪)

| Capability | Where | Notes |
| --- | --- | --- |
| Numba peak/neutral-loss prefilter for `modified_cosine` | `acceleration.py` | Default-on (`enable_numba_prefilter: true`) but numerically identical to full scoring; pure-NumPy fallback. Treat as an optimization, not a science change |
| HNSW two-channel candidate index (`[binned m/z, binned neutral losses]`) | `hnsw.py` | Requires `[hnsw]` extra (`hnswlib`); opt-in via `hnsw_enabled` (default **false**) |
| `cascade` meta-engine (HNSW → prefilter → exact rescoring) | `similarity.py` | Experimental; `[ml]`-free classical stages work, full mode needs extras |
| `consensus` meta-engine | `similarity.py` | Experimental; classical sub-engines always available |
| `spec2vec`, `ms2deepscore` (local) | `similarity.py` | Require `[ml]` extra; guarded imports, graceful fallback |
| Remote ML boundary (REST/gRPC + circuit breaker) | `ml_client.py`, `protocols.py`, `protos/massflow/v1/ml.proto` | Works without heavy deps; satellite reference server in `examples/massflow-ml-satellite/` |
| ML routing of "easy"/"hard" spectra | `similarity.py` (`MLRouter`), `workflow.py` | Opt-in via `enable_routing` (default **false**); triage flags stored by `database.py` |
| Hybrid SQLite+Zarr storage + standalone Zarr backend | `storage.py`, `zarr_store.py`, `database.py` | `input.storage_backend: sqlite|zarr|hybrid` (default **sqlite**); migrations `scripts/migrations/0001_*.py`, `0002_*.py` |
| gRPC streaming server (`stream-server`; `serve` deprecated alias) | `streaming/` | `StreamSpectra` + `GetStatus`; bounded queue with high-water-mark shedding |
| TUI (`massflow tui`) | `tui/` | Requires `[tui]` extra |
| `massflow watch` | `cli.py` | Requires `[watch]` extra |
| `massflow convert` (ProteoWizard `msconvert` wrapper) | `convert.py`, `cli.py` | External binary required; see contradiction C-6 |

### 2.3 Partially implemented (🟡)

| Capability | Reality | Gap |
| --- | --- | --- |
| 5 ppm scientific validation in the **annotate** path | Implemented and tested in `models.py`/`cheminformatics.py`; enforced as an ingestion gate in the **streaming** path | Not enforced on library/query spectra inside the classical `annotate` pipeline (spectra flow through `matchms` only). Docs (`docs/index.md`, ARCHITECTURE §"Scientific Data Integrity") overstate pipeline-wide enforcement |
| `@pytest.mark.core` stable-contract gate | Marker defined in `pyproject.toml` and described in `AGENTS.md` | Only **one** test file (`test_zarr_hybrid.py`) actually applies it; the gate is not exercised as documented |
| Triage bitmask flags | `database.py` computes and stores `triage_flags` JSON on insert; consumed by `MLRouter` thresholds | Only meaningful with `enable_routing: true` (default false); no standalone user surface |

### 2.4 Documented but missing (📕)

| Capability | Claimed in | Reality |
| --- | --- | --- |
| FBMN export (`consensus_spectra.mgf` + CSV pair) | `AGENTS.md` §5.4 (stable output), `docs/index.md` (Stable), `docs/user-guide/results.md`, `.github/copilot-instructions.md` | **No code anywhere in `src/`** (grep for `FBMN`/`consensus_spectra` returns zero matches in source). Export surface is CSV/mzTab-M only |
| GraphML molecular networking + `massflow visualize` + `viz` extra | `docs/user-guide/annotation.md` (§Network Visualization), `docs/index.md` (Experimental), `AGENTS.md` §6.1 | **No `visualize` command in `cli.py`, no graphml code in `src/`, no `viz` extra in `pyproject.toml`** |
| LSP language server | `docs/index.md` (Experimental: "Language Server (LSP)") | Removed; `docs/api/server.md` documents the removal. The stale row in `docs/index.md` is the only live reference |
| `core`-marked stable test suite as the CI gate | `AGENTS.md` §2.5/§6.1 | See §2.3 — marker discipline not applied |

### 2.5 Planned only (📋)

| Item | Status |
| --- | --- |
| Generative spectral augmentation (ML) | `docs/post-v0.1-roadmap.md` — research only |
| Differentiable physics-informed neural networks (PINNs) | `docs/post-v0.1-roadmap.md` — research only |
| Satellite repo split (`massflow-ml`), Zarr storage, gRPC streaming | **Roadmap items already implemented in-tree** — roadmap is stale, not planned |
| TIIP peptide indexing, O(1) peptide retrieval, BIN/BEST consensus, SIRIUS/MS-GF+/MaRaCluster interop, proteomics peptide search | External research specification only — **not implemented, not planned for v0.x** (§6) |

---

## 3. Test ↔ contract mapping

The stable contract is covered by these test files (they must keep passing on every
change; they are the executable form of §1):

- `test_workflow.py`, `test_cli.py`, `test_cli_db.py`, `test_config.py`
- `test_io.py`, `test_io_validation_layer.py`, `test_validation_scenarios.py`
- `test_processing.py`, `test_similarity.py`, `test_mathematical_proof.py`,
  `test_ms1_prefilter.py`, `test_decoy_generation.py`, `test_fdr.py`
- `test_database.py`, `test_precursor_physics.py`, `test_cheminformatics.py`,
  `test_isotopic_distribution.py`, `test_adduct_validation.py`
- `test_end_to_end_mvp.py`, `test_pipeline_integrity.py`, `test_annotation_coverage.py`

Experimental/optional surfaces are covered by: `test_ml_boundary.py`,
`test_ml_guards.py`, `test_acceleration.py`, `test_streaming.py`,
`test_tui_*.py`, `test_zarr_hybrid.py`, `test_zarr_store.py`, `test_migrations.py`,
`test_convert.py`. These may evolve without expanding the v0.1 support promise.

**Known gap:** the `core` pytest marker should be applied to the stable test set
above so `uv run pytest -m core` runs exactly the contract. This is a test-only
change, deliberately deferred (no refactoring in this audit).

---

## 4. Contradiction log (as of 2026-08-25)

Each row: the conflicting claims, where they live, the verified reality, and the
resolution this document mandates. Rows marked **✓ resolved** were acted on by
the complexity-audit pass (2026-08-25); the others remain open until the
referenced files are updated.

| # | Conflict | Sources | Verified reality | Resolution |
| --- | --- | --- | --- | --- |
| C-1 | **Version identity**: docs reference a "v0.2-era Orchestrator API" and a "v1.0 engine lockdown"; package is 0.1.0 | `docs/ARCHITECTURE.md` §Data models; `docs/api/consensus.md`; `docs/api/server.md` vs `pyproject.toml` (`version = "0.1.0"`), `src/MassFlow/__init__.py` (`"0.1.0"`), `docs/CHANGELOG.md` (`[0.1.0] - 2026-05-10`) | v0.2-era modules (`MassFlow.consensus`, `MassFlow.server`) were removed at some point, but **no v1.0 (or 0.2.0) release exists** | **✓ resolved** — ghost-version references rewritten in `docs/ARCHITECTURE.md`, `docs/api/consensus.md`, `docs/api/server.md`, `docs/index.md`, and `docs/user-guide/validation.md`; the removed modules are now described without version qualifiers against the shipped v0.1 baseline. Forward-looking "v0.2 schema bump" mentions in `docs/COMPLEXITY_AUDIT.md` remain as deferred plans (R-1) |
| C-2 | `consensus`/`cascade` labeled **Stable** vs **Experimental** | `README.md` (Similarity engines table) vs `ARCHITECTURE.md`, `AGENTS.md` §6.1, `docs/index.md` | Meta-engines are implemented, tested, but outside the support promise (they degrade to classical when `[ml]` is absent) | **✓ resolved** — README engines table now marks cascade/consensus/spec2vec/ms2deepscore Experimental; the CLI and workflow flag experimental surfaces at run start |
| C-3 | HNSW presented as a **core engineering pillar** vs an **experimental pre-stage**; sample config enables it (`hnsw_enabled: true`) vs default `false` | `README.md` (§Performance architecture) vs `config.py` (default), `ARCHITECTURE.md`, `AGENTS.md` | HNSW requires the `[hnsw]` extra and is only used by the experimental `cascade` engine; default config leaves it off | **✓ resolved** — README pillars reframed as experimental/optional; pipeline-diagram note added; `hnsw_enabled` remains default-false and cascade-only |
| C-4 | **FBMN export claimed stable**; no implementation | `AGENTS.md` §5.4, `docs/index.md`, `docs/user-guide/results.md`, `.github/copilot-instructions.md` | Zero FBMN/`consensus_spectra` code in `src/` | **Missing, not stable.** Do not claim it; implement or de-document (open decision D-3) |
| C-5 | **GraphML networking + `visualize` documented**; no implementation, no `viz` extra | `docs/user-guide/annotation.md`, `docs/index.md`, `AGENTS.md` §6.1 | No `visualize` command, no graphml code, no `viz` extra in `pyproject.toml` | **Missing.** Do not claim it; implement or de-document (open decision D-3) |
| C-6 | **Vendor conversion**: "intentionally does not perform vendor raw conversion internally" / "Do not add internal conversion logic" vs shipped `massflow convert` command (and `docs/index.md` calling it Stable) | `docs/ARCHITECTURE.md` §Unsupported formats; `AGENTS.md` §5.4 vs `cli.py`, `convert.py`, `docs/user-guide/annotation.md` | `convert.py` shells out to the **external** ProteoWizard `msconvert` binary; no in-house conversion code. The annotate pipeline itself still rejects vendor formats | The **annotate path** rejects vendor formats; `convert` is an optional external-tool wrapper. Classified **experimental** (§2.2) until the docs decide otherwise (open decision D-5) |
| C-7 | `massflow watch` labeled **Stable** vs experimental | `docs/index.md` vs `AGENTS.md` §6.1, `pyproject.toml` (`[watch]` extra) | Implemented, requires `watchfiles`, long-running interactive loop, not covered by the v0.1 support promise | **✓ resolved** — `docs/index.md` row now Experimental; CLI help marks it EXPERIMENTAL |
| C-8 | **Duplicate architecture sources of truth**: byte-identical `ARCHITECTURE.md` at repo root and `docs/ARCHITECTURE.md`; different files point at each | `ARCHITECTURE.md` (root), `docs/ARCHITECTURE.md`, `README.md`, `AGENTS.md`, `.github/copilot-instructions.md`, `mkdocs.yml` | Root copy is an exact byte-identical duplicate (diff: empty) of `docs/ARCHITECTURE.md`; mkdocs serves only `docs/ARCHITECTURE.md` | **✓ resolved (D-2 executed)** — root duplicate deleted; `docs.yml` copy step removed; AGENTS/CONTRIBUTING repointed at `docs/ARCHITECTURE.md`; `docs/ARCHITECTURE.md` is a superset of the deleted copy |
| C-9 | `zarr` is both a **core dependency and an optional extra**; README instructs `uv sync --extra zarr` | `pyproject.toml` (`zarr>=3.2.1` in `dependencies` **and** `[zarr]` extra), `README.md` §Installation | Zarr is always installed | **✓ resolved (D-6 executed)** — `[zarr]` extra removed from `pyproject.toml`; README + installation docs updated |
| C-10 | **`core` marker gate documented but unapplied** | `AGENTS.md` §2.5/§6.1, `pyproject.toml` vs test suite | Only `test_zarr_hybrid.py` (1 test) carries `@pytest.mark.core` | **✓ resolved by retirement** — the pytest-config audit removed the marker machinery; the contract gate is the default pytest selection + the explicit `-m scientific` group (see `docs/COMPLEXITY_AUDIT.md` R-3) |
| C-11 | **5 ppm "built-in strict physical integrity checks"** (Stable) vs enforcement surface | `docs/index.md`, `docs/ARCHITECTURE.md` §Scientific Data Integrity vs `models.py`, `workflow.py`, `streaming/engine.py` | Validators are real and tested, but only enforced as a gate in the **streaming** path; the classical annotate path does not construct `SpectrumMetadata` | **✓ docs fixed** — `docs/index.md` row now states the model-layer scope and the streaming-only gate; enforcement decision D-8 remains open |
| C-12 | **Roadmap lists as future work what is already implemented** | `docs/post-v0.1-roadmap.md` (§1 satellite repo, §2 Zarr, §3 gRPC streaming) | All three exist in-tree (`ml_client.py` + `examples/massflow-ml-satellite/`, `zarr_store.py`, `streaming/`) | Roadmap is stale; remaining items are only §4 (generative augmentation, PINNs) |
| C-13 | `docs/index.md` lists **LSP as current experimental** while the module is removed | `docs/index.md` vs `docs/api/server.md` | `MassFlow.server` (LSP) does not exist | **✓ resolved** — `docs/index.md` row now says "Removed" and points at `docs/api/server.md` |
| C-14 | Broken version probe pattern | `src/MassFlow/__init__.py` | `try: __version__ = "0.1.0" except PackageNotFoundError` can never raise; `importlib.metadata` imported but unused | **✓ resolved** — probe removed; plain `__version__ = "0.1.0"` |
| C-15 | README's `storage_backend: "hybrid"` example and hybrid-first narrative vs default `"sqlite"` | `README.md` §Performance architecture/§Quickstart config vs `config.py` (`storage_backend` default `"sqlite"`) | Default DB builds are BLOB SQLite; hybrid is opt-in | **✓ resolved** — README pillar framing now states sqlite is the default and hybrid/Zarr is optional |

---

## 5. External research specification — explicit non-contract status

A technical specification/research document exists **outside this repository**
describing a substantially different architecture:

> TIIP peptide indexing, O(1) peptide retrieval, BIN/BEST consensus generation,
> SIRIUS / MS-GF+ / MaRaCluster interoperability, and proteomics-oriented peptide
> search.

**Verified status (full-text search of this repository, 2026-08-25):**

- `TIIP`, `BIN/BEST`, `MaRaCluster`, `MS-GF`, `SIRIUS`, `FragPipe`, and
  peptide-indexing terms appear **nowhere** in `src/`, `tests/`, `pyproject.toml`,
  configuration files, or documentation.
- No config schema, CLI command, module, test, or data contract relates to
  peptide search.
- The repository's "consensus" vocabulary refers to `ConsensusEngine` weighted
  spectral scoring — **not** BIN/BEST consensus-spectrum generation.

**Mandate for all future agents:**

1. The research specification is **research context only**. It is not a
   requirement, a roadmap item, or a description of the current product.
2. Do **not** implement TIIP, peptide indexing, BIN/BEST, or
   SIRIUS/MS-GF+/MaRaCluster interop as part of MassFlow v0.x work.
3. If work on the research vision begins, it must start with a new product
   contract (a successor document to this one), not by extending the v0.1
   surfaces.
4. Any file that describes MassFlow (README, ARCHITECTURE, AGENTS,
   copilot-instructions, mkdocs docs) must be written from §1 of this document.

---

## 6. Core product vs experimental extensions vs future research (one-glance)

| Tier | Contents | Change discipline |
| --- | --- | --- |
| **Core (v0.1 contract, §1)** | `annotate`/`init`/`tutorial`/`db *`/`convert` CLI; YAML config; mzML/mzXML/MGF/MSP/SQLite ingestion; `matchms` processing; `cosine`/`modified_cosine`; entropy decoys + FDR + empirical p-values; CSV/mzTab-M + YAML exports; model-layer 5 ppm/isotopic validation; SQLite + optional hybrid Zarr storage | Must not regress; full test suite + coverage gate; extra scrutiny on changes |
| **Experimental (§2.2)** | HNSW, cascade, consensus, spec2vec/ms2deepscore, remote ML boundary + satellite, MLRouter, streaming server, TUI, `watch`, Numba prefilter, hybrid/Zarr storage, `convert` | May evolve freely; must not break the core; guards and fallbacks required |
| **Future research** | Generative spectral augmentation, PINNs (in-repo roadmap); TIIP peptide indexing, O(1) peptide retrieval, BIN/BEST consensus, SIRIUS/MS-GF+/MaRaCluster interop, proteomics peptide search (external spec) | Do not implement as v0.x requirements; requires a new contract first |

---

## 7. Open architectural decisions (unresolved)

| ID | Decision | Options | Impact |
| --- | --- | --- | --- |
| D-1 | Version identity | Keep 0.1.0 baseline and purge "v0.2/v1.0" doc language, or release 0.2.0 and rename the contract | Affects every doc and the changelog; contract text assumes 0.1.0 |
| D-2 | Duplicate `ARCHITECTURE.md` at repo root | Delete root copy (mkdocs already serves `docs/ARCHITECTURE.md`) or keep synced | **✓ executed (complexity-audit pass)** — root copy deleted; `docs.yml` copy step removed; references repointed |
| D-3 | FBMN export and GraphML/`visualize` (both documented, neither implemented) | Implement them as experimental, or de-document them | Directly determines whether §2.4 rows become 🧪 or disappear |
| D-4 | `core` marker discipline | Apply `@pytest.mark.core` to the §3 test set and gate CI on it, or drop the marker machinery | Makes the contract machine-enforceable |
| D-5 | `massflow convert` status | Promote to stable (with `msconvert` prerequisite documented) or keep experimental | Affects §2.2/§4 C-6 |
| D-6 | Redundant `[zarr]` extra | Remove the extra (zarr is core) or demote zarr to the extra | **✓ executed (complexity-audit pass)** — extra removed; zarr stays a core dependency |
| D-7 | Research spec custody | Keep the TIIP/BIN/BEST specification external, or add it to the repo under an explicit `research/` (non-product) path | Future agents need a guaranteed-visible "research only" marker |
| D-8 | 5 ppm enforcement in the annotate path | Enforce `SpectrumMetadata` validation on library spectra during `annotate`, or narrow the docs to match the current streaming-only gate | Scientific-integrity claim vs runtime behavior |

---

*Generated by the 2026-08-25 architecture reconciliation audit. No functional
code was changed during this audit; this document is the only deliverable that
declares product scope.*
