# MassFlow Release Readiness — massflow-msms 0.1.0

**Assessment date:** 2026-10-27
**Branch:** `codex/final-release-candidate` (draft PR #134)
**Qualified code/artifact SHA:** `1d0cdeb1a972b269f3187b4a3950a6d23ecf9c44`
**Candidate line:** `release/0.1.0-candidate` at the same SHA
**Base:** `main` at `ffef50bff913fcf400cb3de72595dab50e4f3fd8`
**Assessment:** **GO for publication readiness under the documented exploratory scope**, conditional on owner-only PyPI/environment setup and recorded approval. **NO publication was performed by this work.**

No GitHub PR was merged, no tag was created, no package was uploaded, and no release was created. `main`, the primary checkout, and all pre-existing dirty worktrees were preserved.

---

## 1. Supported release scope

MassFlow 0.1.0 is an **exploratory MS/MS reference-library candidate-retrieval and annotation-evidence toolkit**. It retrieves and ranks library candidates, reports spectral similarity and provenance, and exposes numerical target/decoy diagnostics. It does **not** claim confirmed identities or calibrated error control.

**Qualified core surface**

- `massflow init`, `massflow annotate`, `massflow db build`, `massflow db inspect`, `massflow db merge`.
- mzML, mzXML, MGF, MSP, and MassFlow SQLite inputs. Vendor raw formats are rejected with an actionable error.
- `matchms` metadata/peak processing with float64 m/z and intensities, explicit missingness, and fragment-only m/z windows.
- Classical `cosine` and `modified_cosine` candidate ranking under declared tolerances.
- SQLite library stores with build lineage.
- CSV results plus YAML provenance/report sidecars.
- Explicit confidence states (`estimated_uncalibrated`, `unavailable`, `invalid`); no state implies calibrated FDR.
- Source-record identity and acquisition provenance, separate from chemical identity.

**Not claimed for 0.1.0**

- Independent identification validation or authenticated standard-compound truth.
- Calibrated FDR, q-value error control, or candidate-specific error probabilities.
- Confirmed molecular identity from similarity alone.
- Validated performance across untested instruments, matrices, ion modes, or acquisition protocols.
- Standards-conformant mzTab-M output.
- Supported/qualified ML, HNSW, streaming, TUI, watch, or Numba acceleration.

Independent identification validation and calibrated FDR are **deferred scientific research**. They are prerequisites for stronger identification/FDR claims and for 1.0, not prerequisites for publishing this documented exploratory candidate-retrieval toolkit.

`mztab` remains an **experimental, non-conformant mzTab-shaped table**. It omits `mzTab-version` and states in-file that it is not mzTab-M. CSV is the stable export. The synthetic tutorial generator is a source-checkout-only script; the unsupported `massflow tutorial` CLI command has been removed.

Optional extras (`ml`, `hnsw`, `watch`, `tui`, `chem`) are **experimental/unqualified**. `uv sync --all-extras` is a developer convenience, not a support claim. Release qualification installs only the default/core dependency set.

---

## 2. Integrated work and deferred work

Integration was by merge/cherry-pick onto a clean branch. Source PRs remain open; no GitHub PR was merged.

| Work | Disposition | Evidence |
| --- | --- | --- |
| #116 scientific stabilization | Integrated | Merge `88b3a13`; includes #101/#103–#107, #110, #111/#112 content and processing-semantics revision 4 |
| #117 retrieval diagnostics | Integrated | Stacked commits through `83f4e9e`, merged in `88b3a13` |
| #131 retention-time units | Integrated | Merge `88b3a13` from `5068de3` |
| #133 non-finite raw precursor m/z | Integrated | `46a02b5` |
| #118 NaN MS1-prefilter fix | **Superseded** | Replaced by the complete unified object/array prefilter in `abc331f` |
| #119 certifi security update | Integrated | `df92472` |
| #120 guarded release workflow | Integrated | `8c41221`; extended with lock-pinned qualification and OIDC PyPI publication |
| #121 RDKit install contract | Integrated | `5ce07f6` |
| #114 Pillow security update | Integrated | `97014ba`; Pillow 12.3.0 → 13.0.0 |
| #123 mzTab-M compliance | **Stable claim withdrawn** | `561f9f3`; experimental non-conformant table only |
| Tutorial command removal | Resolved | `31e3ee2`; source-checkout generator retained, installed command removed |
| Core-only qualification + OIDC PyPI | Resolved | `31e3ee2`; provenance fix in `1d0cdeb` |

**Deferred**

- #115 filelock and #113 setuptools: optional-ML/dev-only advisories; core runtime audit is clean.
- #92 advisory preflight: feature work, outside the frozen release scope.
- #112/#111/#110/#107/#106/#105/#104/#103/#101: content already integrated through #116; keep PRs open to avoid double-merging.
- #89/#88/#86/#80 network-aware annotation stack: experimental molecular networking, explicitly out of scope.
- Windows and optional-ML/HNSW qualification: not exercised and not claimed.

---

## 3. Final candidate SHA and branch

- **Qualified code/artifact SHA:** `1d0cdeb1a972b269f3187b4a3950a6d23ecf9c44`
- **PR branch:** `codex/final-release-candidate`
- **Release-candidate branch:** `release/0.1.0-candidate` at the same SHA
- **PR:** https://github.com/janusson/MassFlow/pull/134
- **Qualification run:** https://github.com/janusson/MassFlow/actions/runs/40439344703 — `success`; publish job correctly skipped.
- The final `RELEASE_READINESS.md` update is documentation-only; the PR description records the resulting head SHA. No scoring, preprocessing, retrieval, ranking, confidence, packaging-dependency, or metric code changed after the qualified SHA.

---

## 4. Verification evidence

### 4.1 Guarded non-publishing qualification (exact SHA `1d0cdeb`)

The guarded `release.yml` workflow was dispatched in `qualify` mode on `release/0.1.0-candidate`. It checked out the exact SHA, verified version/lock/tag inputs, installed the **core** dependency set with `uv sync --frozen`, and passed:

- full release suite with coverage: **1,696 passed, 7 skipped, 35 deselected; 83.56% coverage**
- scientific-validation: **388 passed**
- core-contract: **291 passed**
- Ruff format/lint, mypy, and `uv lock --check`
- wheel/sdist build, lock-exported installed-wheel smoke, `uv pip check`, CLI/init/annotation smoke
- provenance, artifact hashing, and artifact upload

Provenance recorded:

- `candidate_sha = 1d0cdeb1a972b269f3187b4a3950a6d23ecf9c44`
- `version = 0.1.0`, `action = qualify`
- `python = 3.13.18`, `platform = Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`
- `lock_sha256 = bf227f487dba380428fbc8b05820ea902953a0ab8aace721fc2c5530b04d1497`
- `pyproject_sha256 = c976926c4be25bead911f3c0f2ab2d8150d6e8648fcaba9d99be50aa080c14c2`

Hosted PR/release-branch CI also passed: CodeQL (actions/python), smoke, lockfile/lint, installed-wheel-contract, Linux tests, and macOS tests.

Local macOS 27.0.1 arm64/Python 3.13.12 gates also passed: 1,696 tests, 83.56% coverage; 388 scientific; 291 core; 13 optional skipped; Ruff, mypy, lock check, and strict MkDocs. One streaming-security test failed transiently on the first local full-suite run and passed on the focused rerun and the next full run; it is not reproducible. The hosted qualification run passed the suite.

### 4.2 Qualified artifacts

Built by the qualification workflow from `1d0cdeb`:

| Artifact | SHA-256 |
| --- | --- |
| `massflow_msms-0.1.0-py3-none-any.whl` | `b376a304d80ae8c7e195d21b0cba5cbe432ef556b3672c64196d055c1c27049a` |
| `massflow_msms-0.1.0.tar.gz` | `6e0635ed88a8d3d779168c5d1113b1afa7eb18b3edfdde3b854609375c24c11f` |
| `SHA256SUMS` | `c32d6837e1fc8e73bbe68bf4cb132ee09e26e4951dc2b43e36c53f260c5c7c92` |

The qualified artifact set contains exactly the wheel, sdist, and `SHA256SUMS`; hidden build housekeeping files are excluded.

### 4.3 Isolated installed-artifact tests

The downloaded qualified wheel and sdist were installed separately into fresh Python 3.13.12 environments outside the checkout using the lock-exported runtime requirements (`--no-deps` for the project artifact). For both:

- `MassFlow.__file__` resolved from the isolated `site-packages`.
- `uv pip check` reported all packages compatible.
- `massflow --help` no longer advertises `tutorial`; `massflow tutorial` exits nonzero with `No such command 'tutorial'`.
- `massflow init`, `massflow db build`, `massflow db inspect`, and `massflow annotate` passed.
- CSV output, `run_provenance.json`, and `massflow_version=0.1.0` passed.
- Experimental mzTab-shaped output was written, contained no `mzTab-version` header, and contained the explicit non-conformance notice.
- `MassFlow.generated` and `MassFlow.streaming.generated.massflow.v1.streaming_pb2` imported.
- Core-only `pip-audit --path` reported **no known vulnerabilities**.

Dependency contract: core includes `matchms`, `grpcio`, and `protobuf>=7.35.0,<8`; RDKit is transitive through `matchms`, not a direct core requirement; no-extra core install imports and `_HAS_RDKIT` is true. The qualified environments used Pillow 13.0.0, protobuf 7.35.1, RDKit 2026.3.1, and matchms 0.33.1.

### 4.4 Frozen benchmark (unchanged definition)

The frozen MassBank diagnostic was run on `97014ba54e16cf2e495e32116888fe532e4aae2b` with the committed manifest and parameters unchanged. The post-benchmark diff to `1d0cdeb` touches only `src/MassFlow/cli.py` (tutorial removal), tests, documentation, and release workflows; it does **not** touch scoring, preprocessing, retrieval, ranking, confidence, or metric code.

| Arm | Exact top-1 | Exact top-5 | Candidate calls | Absent-truth calls | Confidence state | Surviving decoy competitions |
| --- | --- | --- | --- | --- | --- | --- |
| Present cosine | 52/99 | 53/99 | 55 | — | `unavailable` | 0 |
| Present modified cosine | 27/99 | 46/99 | 92 | — | `estimated_uncalibrated` | 1 |
| Mixed cosine | 26/49 | 27/49 | 29 | 2/50 | `unavailable` | 0 |
| Mixed modified cosine | 16/49 | 21/49 | 91 | 46/50 | `estimated_uncalibrated` | 1 |

Positive: the candidate reproduces the frozen #116/#117 counts exactly; cosine remains the stronger candidate ranker. Negative findings are preserved: modified cosine calls 46/50 absent truths in the mixed arm and 75/91 calls are wrong at legacy q ≤ 0.05; confidence is not calibrated (0 or 1 decoy competition); eight minimum-peak drops and one prespecified leakage exclusion remain in accounting; the Numba prefilter is disabled in the frozen parameters, so this cohort does not qualify the mixed-frame pruning fix; deposited labels are not independently authenticated.

Benchmark result hashes remain those recorded for `97014ba` in the PR evidence. This is descriptive cohort evidence, not a deployment error rate or calibrated-FDR certificate.

---

## 5. Remaining real blockers and qualification limits

1. **PyPI ownership/pending publisher (owner action).** `massflow` on PyPI is owned by an unrelated GPL-3.0 project. `massflow-msms` is unregistered. The owner must create a pending publisher for `massflow-msms` (owner `janusson`, repo `MassFlow`, workflow `release.yml`, environment `release`) as described in [`.github/PYPI_TRUSTED_PUBLISHING.md`](.github/PYPI_TRUSTED_PUBLISHING.md). No API token is used.
2. **Release approval model (owner decision).** The `release` environment must have a required reviewer. Option A is an independent reviewer with `prevent_self_review: true` (preferred). Option B is a solo-maintainer auditable approval: record the SHA, version, tag, qualification run URL, artifact hashes, evidence URL, notes, and an explicit approval statement in a GitHub issue/PR comment or committed approval record. Option B is auditable but **not independent**; release notes must say so.
3. **Optional extras are unqualified.** ML/HNSW/TUI/watch/chem are experimental. `[ml]` still resolves a filelock advisory that is outside the qualified core; `setuptools` has a low-severity dev/build advisory. Do not advertise `--all-extras` as supported functionality.
4. **mzTab-M conformance.** The stable claim is withdrawn; no standards-valid exporter is claimed. CSV is the only stable export.
5. **Mandatory cache/store migrations.** Managed caches rebuild automatically under processing-semantics revision 4. Explicit `.db`, `.sqlite`, and `.zarr` stores are not rewritten; rebuild from raw sources into a new destination, inspect record identity/metadata, then switch paths. Do not append corrected records to an old invalid store and call it migration. See `docs/user-guide/processing.md`.
6. **Platform limits.** Hosted Linux and macOS CI pass; local evidence is macOS arm64. Windows and other platforms/architectures are not qualified and are not claimed.
7. **Default-branch alert dashboard.** GitHub showed 5 critical/24 high/13 medium/3 low open pip alerts via API at inspection (push banner showed 6 critical). The candidate upgrades Pillow and certifi; the qualified core install audits clean. Re-check after merge; remaining advisories are expected to be dev/optional-only.

Independent identification validation and calibrated FDR are **not** in this blocker list; they are deferred research and stronger-claim prerequisites (§1).

---

## 6. Proposed package/version metadata

```toml
[project]
name = "massflow-msms"
version = "0.1.0"
requires-python = ">=3.13,<3.14"
license = {text = "MIT"}
```

- **Distribution:** `massflow-msms` (pending publisher not yet created).
- **Import package:** `MassFlow`; **CLI:** `massflow` (unchanged).
- **Version:** `0.1.0`, pre-1.0 exploratory candidate. No 1.0 bump; the project’s 1.0 decision remains NO-GO.
- **Alternative:** `massflow-ms` is also unregistered; changing it is packaging metadata only.
- **Never use:** `massflow` — owned by an unrelated project.

---

## 7. Publication steps (owner-only, after explicit approval)

1. Create the PyPI pending publisher and confirm the GitHub `release` environment/reviewer policy as described in `.github/PYPI_TRUSTED_PUBLISHING.md`.
2. Merge the approved candidate via PR; do not push directly to `main`.
3. Create an annotated tag `v0.1.0` on the merged SHA.
4. Run `.github/workflows/release.yml` in `qualify` mode on the exact SHA. Review the full suite, scientific/core tests, lock-pinned installed smoke, `dist/SHA256SUMS`, and `release-provenance.json`. This does not publish.
5. Run `publish` mode with `confirmation="PUBLISH 0.1.0"`, the exact SHA/tag/version, evidence URL, and migration/rollback notes. The protected `release` environment must approve.
6. The publish job verifies the qualified wheel/sdist against `SHA256SUMS` and provenance, creates the GitHub release, stages only those two files, and uploads them to PyPI with OIDC Trusted Publishing and PEP 740 attestations. It does not rebuild.
7. Verify from a clean environment outside the checkout: `pip install massflow-msms==0.1.0`, `massflow --help`, `massflow init`, and a small CSV annotation. Compare PyPI file hashes with the GitHub release `SHA256SUMS`/provenance.
8. Rollback: yank affected PyPI files (do not delete/reuse the version), delete the GitHub release, fix forward with a new patch version/tag, and requalify.

---

## 8. GO / NO-GO

**GO for publication readiness.** The qualified core candidate has green local and hosted gates, a clean core dependency audit, verified qualified artifacts, a tested non-publishing qualification workflow, and a bounded evidence-backed scope. The only remaining steps are owner-only setup and approval: PyPI pending publisher, `release` environment configuration, and the owner’s choice of independent or auditable solo approval.

**NO publication is performed or authorized by this report.** If the owner requires independent review, complete Option A before publishing; otherwise use Option B and label the release as solo-approved, not independently reviewed. Do not promote this candidate to validated unknown-compound identification or calibrated-FDR use.
