# MassFlow Release Readiness — 0.1.0 candidate

**Assessment date:** 2026-10-26
**Branch:** `codex/final-release-candidate`
**Qualified code/artifact SHA:** `97014ba54e16cf2e495e32116888fe532e4aae2b`
**Integration base:** `main` at `ffef50bff913fcf400cb3de72595dab50e4f3fd8`
**Status:** draft PR / human review candidate. **GO for review; NO-GO for public publication** until the blockers in §5 close.

No GitHub PR was merged, no tag was created, no package was published, and `main` was not modified. This candidate integrates existing work on a clean branch. The primary checkout and its pre-existing changes were left untouched.

---

## 1. Exact supported release scope

MassFlow 0.1.0 is a **reproducible MS/MS reference-library candidate-retrieval and annotation-evidence toolkit**. It produces spectral similarity scores, ranked library candidates, source/acquisition provenance, and query-level numerical target/decoy diagnostics.

**Stable/supported surfaces**

- Configuration-first CLI: `massflow init`, `massflow annotate`, `massflow db build`, `massflow db inspect`, `massflow db merge`.
- Open-format ingestion: mzML, mzXML, MGF, MSP, and MassFlow SQLite stores. Vendor raw formats are rejected with an actionable error.
- `matchms`-based metadata/peak processing with float64 m/z and intensities, missing values preserved as missing/NaN, and explicit fragment-window semantics.
- Classical `cosine` and `modified_cosine` candidate ranking under declared tolerances.
- SQLite library stores with lineage/build provenance.
- CSV result export plus YAML provenance/report sidecars.
- Explicit confidence states: `estimated_uncalibrated`, `unavailable`, or `invalid`. No state implies calibrated error control.
- Source-record identity and acquisition provenance, separate from chemical identity.

**Not claimed**

- Validated unknown-compound identification.
- Calibrated FDR or reliable q-value/error control.
- Confirmed molecular identity from spectral similarity alone.
- Validated performance across untested instruments, matrices, ion modes, or acquisition protocols.
- A validated mzTab-M standards-conformant exchange format.
- Production support for molecular networking, learned scoring, approximate search, streaming, TUI, or Numba acceleration.

`mztab` is retained as an **experimental, non-conformant mzTab-shaped table**. It deliberately omits the `mzTab-version` header and states in-file that it is not mzTab-M. CSV is the stable export.

---

## 2. Integrated PRs and deferred work

Integration was by merge/cherry-pick onto a clean branch; the source PRs remain open. No PR was merged on GitHub.

| Work | Disposition | Candidate evidence |
| --- | --- | --- |
| #116 scientific stabilization | Integrated | Merge `88b3a13`; includes the #101/#103–#107, #110, #111/#112 content and processing-semantics revision 4 |
| #117 retrieval diagnostics | Integrated | Stacked commits through `83f4e9e`, merged in `88b3a13` |
| #131 retention-time units | Integrated | Merge `88b3a13` from `5068de3`; config/export/cache overlaps resolved and retested |
| #133 non-finite raw precursor m/z | Integrated | Cherry-pick `46a02b5`; NaN/±Inf quarantined with a distinct reason |
| #118 NaN MS1-prefilter fix | **Superseded, not cherry-picked** | Replaced by the more complete unified object/array prefilter in `abc331f`, which covers NaN, infinities, non-positive sentinels, ppm/Da boundaries, and object/array equivalence |
| #119 certifi security update | Integrated | Cherry-pick `df92472`; lock-only update to `certifi==2026.10.23` |
| #120 guarded release workflow | Integrated | Cherry-pick `8c41221`; extended in `e3734a8` so qualification installs the wheel against lock-exported runtime requirements |
| #121 RDKit installation contract | Integrated | Cherry-pick `5ce07f6`; README/installation/CI disclose transitive RDKit through `matchms` |
| #114 Pillow security update | Integrated | Cherry-pick `97014ba`; `Pillow 12.3.0 → 13.0.0`, core via `matchms → matplotlib` |
| #123 mzTab-M compliance | **Resolved by withdrawing the stable claim** | `561f9f3` removes the `mzTab-version` assertion, marks `mztab` experimental/non-conformant, and keeps CSV stable. No HUPO-PSI validator conformance is claimed. |

**Deferred with reasons**

| Work | Reason |
| --- | --- |
| #115 filelock 3.29.7 → 4.2.1 | Only reachable through the optional `[ml]` extra (`torch`); the stable core audit is clean. Review before claiming ML-extra support. |
| #113 setuptools 83.0.0 → 84.1.0 | Build/dev-only (low-severity advisory); not in the core runtime install. |
| #92 advisory feasibility preflight | New feature, not release-blocking; outside the frozen scope. |
| #112 release-scope draft, #111 benchmark draft, #110 confidence draft, #107/#106/#105/#104/#103/#101 correctness PRs | Their content is already integrated through #116; keeping the PRs open avoids double-merging. |
| #89/#88/#86/#80 network-aware annotation stack | Experimental molecular networking; explicitly out of the 0.1.0 scope and not required for candidate retrieval. |
| Windows and optional-ML qualification | Not exercised; no support claim is made. |

---

## 3. Final candidate SHA and branch

- **Branch:** `codex/final-release-candidate`
- **Qualified code/artifact SHA:** `97014ba54e16cf2e495e32116888fe532e4aae2b`
- **Base:** `main` `ffef50bff913fcf400cb3de72595dab50e4f3fd8`
- **Commits since base:** the #116/#117/#131 stack, then `46a02b5`, `df92472`, `8c41221`, `5ce07f6`, `561f9f3`, `05b4c03`, `f121a49`, `e3734a8`, `97014ba`.
- This `RELEASE_READINESS.md` is added by a final documentation-only commit. The PR description records the final head SHA. No runtime, packaging metadata, lockfile, test, or workflow behavior changes in that report commit.

---

## 4. Test, benchmark and installation evidence

### 4.1 Software gates (qualified SHA `97014ba`)

Platform: macOS 27.0.1 arm64 (build 26A434), Python 3.13.12, uv 0.13.5 lock tooling.

| Gate | Command | Result |
| --- | --- | --- |
| Full release suite + coverage | `uv run pytest --cov=src/MassFlow --cov-report=xml --cov-fail-under=80 -v` | **1,695 passed, 7 skipped, 35 deselected; 83.39% coverage** |
| Scientific validation | `uv run pytest -m scientific -q` | **388 passed, 1,349 deselected** |
| Optional dependencies | `uv run pytest -m optional -q` | **13 skipped, 1,724 deselected** (HNSW extra unavailable) |
| Lint | `uv run ruff check .` | Passed |
| Format | `uv run ruff format --check .` | 157 files already formatted |
| Types | `uv run mypy .` | Success, 166 source files |
| Lockfile | `uvx --from uv==0.13.5 uv lock --check` | Passed |
| Strict docs | `uv run mkdocs build --strict` | Passed; upstream Material 2.0 advisory and unlisted-page notices are informational |
| Commit hooks | pre-commit on each commit | Passed |

Existing non-fatal SQLite `ResourceWarning`s remain in the suite. Hosted Linux/macOS CI on this exact SHA is **pending**; it runs when the draft PR is opened. Local evidence is macOS arm64 only.

### 4.2 Frozen experimental benchmark (unchanged definition)

The frozen MassBank diagnostic was rerun with the committed manifest and parameters unchanged:

```sh
uv run python -m scripts.validation_benchmark.runner run \
  --cache "$HOME/.cache/massflow-validation" \
  --output "$HOME/.cache/massflow-validation/runs/ae-v1-final-97014ba" \
  --manifest benchmarks/massbank-v1/manifest.json \
  --manifest-sha256 66e27364488feb3e4dc74a452d964d0240d2763e1d1dd40c94ff33c400410b49 \
  --parameters benchmarks/massbank-v1/parameters.json \
  --parameters-sha256 096a6ea137ec010d28c5b657a7225bf57c274cc15fa0e3125166b0b5dd446738
```

Provenance recorded `application_commit=97014ba54e16cf2e495e32116888fe532e4aae2b` and `dirty=false` for all four arms.

| Arm | Exact top-1 | Exact top-5 | Candidate calls | Absent-truth calls | Confidence state | Surviving decoy competitions |
| --- | --- | --- | --- | --- | --- | --- |
| Present cosine | 52/99 | 53/99 | 55 | — | `unavailable` | 0 |
| Present modified cosine | 27/99 | 46/99 | 92 | — | `estimated_uncalibrated` | 1 |
| Mixed cosine | 26/49 | 27/49 | 29 | 2/50 | `unavailable` | 0 |
| Mixed modified cosine | 16/49 | 21/49 | 91 | 46/50 | `estimated_uncalibrated` | 1 |

**Positive findings**

- The candidate reproduces the frozen PR #116/#117 metric dictionaries and per-query counts exactly. No scoring, ranking, candidate-domain, or legacy-q change was introduced by the packaging/security/export changes.
- Cosine remains the better-calibrated candidate ranker on this cohort; 52/99 exact top-1 and 53/99 exact top-5 for present truths.
- CSV/mzTab-shaped exports and provenance round-trip in the installed artifact.

**Negative findings (retained, not hidden)**

- Modified cosine has a high wrong-candidate rate: 46/50 absent truths called in the mixed arm, and 75/91 candidate calls wrong at legacy q ≤ 0.05 (descriptive cohort FDP 82.4%). This is a candidate-ranking limitation, not a deployment error rate.
- Confidence is **not calibrated**: 0 or 1 surviving decoy competition cannot support error control. States are `unavailable` or `estimated_uncalibrated` only.
- Eight queries per arm were rejected by the minimum-peak gate and one query is a prespecified potential-leakage exclusion; these remain in the accounting.
- The frozen parameters disable the Numba prefilter, so this cohort does not qualify the mixed-frame pruning fix. Separate synthetic regressions cover that correction.
- Deposited standard labels are not independently re-authenticated; raw-scan lineage and acquisition independence are not fully verifiable.

Result hashes:

- `present-cosine/result.json` — `ee1c6aacabcb74737f28673df67f574c534054869beada713c2f602007838a98`
- `present-modified_cosine/result.json` — `23011372402555f68b41f6e0f8e47dfb3d16ffbce4af6761cf7d8c69d7a0a819`
- `mixed-cosine/result.json` — `da2c9a6de407c4205cfc8dacc81eb28cc7bc7f0905d5ebe903bec21d86d20874`
- `mixed-modified_cosine/result.json` — `d6b78e30785b7ab220724ad71fbf928e4d9afc8215ef0c462c4c622389edf287`
- `summary.json` — `dda6ba8b727b2ebc421a4aff82d0df44fb3d785a18f4ed156ff813b741b603d3`

Large source/native outputs remain outside Git.

### 4.3 Artifacts

Built from `97014ba` with `uvx --from uv==0.13.5 uv build`:

| Artifact | SHA-256 |
| --- | --- |
| `massflow_msms-0.1.0-py3-none-any.whl` | `318115b2e5a0b69f6016e7b74c83f9f5930b48337c5f4aad53caed1636b48205` |
| `massflow_msms-0.1.0.tar.gz` | `543b91f0d9eb711e15d70e1fc38973f57a7a66edc4d41924c9a85601c8ff4b15` |

Lockfile SHA-256: `bf227f487dba380428fbc8b05820ea902953a0ab8aace721fc2c5530b04d1497`.
`pyproject.toml` SHA-256: `c976926c4be25bead911f3c0f2ab2d8150d6e8648fcaba9d99be50aa080c14c2`.

### 4.4 Isolated installation evidence

Three environments were created **outside the checkout** on Python 3.13.12:

1. `wheel/latest resolution` — installed the wheel with current PyPI resolution.
2. `wheel/lock-constrained` — installed lock-exported runtime requirements, then the wheel with `--no-deps`.
3. `sdist/lock-constrained` — installed lock-exported runtime requirements, then the sdist with `--no-deps`.

For each environment:

- `uv pip check` — all installed packages compatible.
- `MassFlow.__file__` resolved inside the isolated environment's `site-packages` (not the checkout).
- `massflow --help`, `massflow init`, `massflow db build`, `massflow db inspect`, `massflow annotate` — passed.
- CSV output and `run_provenance.json` were written; provenance `massflow_version` and `dependencies.massflow` were `0.1.0`.
- `MassFlow.generated` and `MassFlow.streaming.generated.massflow.v1.streaming_pb2` imported successfully.
- Experimental `mztab` output was written, contained **no** `mzTab-version` header, and contained the explicit `NOT mzTab-M standards-conformant` notice.
- `uvx --python 3.13 pip-audit --path <lock-constrained site-packages>` — **no known vulnerabilities** in the core runtime install. (`massflow-msms` is not yet on PyPI, so the audit skips the project itself, as expected.)

Dependency contract confirmed:

- Core `Requires-Dist` includes `matchms`, `grpcio`, and `protobuf>=7.35.0,<8`.
- RDKit is **not** a direct core requirement, but locked `matchms` requires `rdkit>=2024.3.5` transitively; `rdkit` imports and `_HAS_RDKIT` is true in a no-extra core install.
- The latest-resolution environment used newer transitive versions (e.g. Pillow 13.0.0, protobuf 7.36.2, RDKit 2026.9.1). The lock-constrained environments used the committed lock (protobuf 7.35.1, RDKit 2026.3.1). The release workflow now uses lock-exported requirements for qualification.

**Known installed-command limitation:** `massflow tutorial` is advertised by the CLI but fails from both wheel and sdist because `scripts/generate_tutorial_data.py` is a repository script and is not packaged. It is outside the installed support promise in this candidate. Before publication, either package the generator or hide/remove the installed command; do not claim it as supported.

---

## 5. Remaining genuine release blockers

1. **PyPI distribution ownership.** The name `massflow` on PyPI is owned by an unrelated project (`lyriqdd`, MSI preprocessing framework, GPL-3.0, versions 0.1.0–0.1.2). The candidate uses `massflow-msms`, which returned HTTP 404 (unregistered) on 2026-10-26. The maintainer must register and own the chosen name and configure trusted publishing before any upload. `massflow-ms` is also free but was not chosen.
2. **Hosted CI on the exact SHA.** Local combined verification is complete, but Linux/macOS hosted jobs on `97014ba` have not run yet. Windows remains unqualified. Do not substitute earlier PR CI for this.
3. **Independent release approval.** The `release` GitHub Environment currently has one collaborator, so the required reviewer is the maintainer. Add a second reviewer and set `prevent_self_review: true` before publication.
4. **Independent scientific validation.** Source standard labels are not independently authenticated; acquisition independence is not proven; no independent batches, matrices, instruments, ion modes, or calibrated uncertainty are available. Confidence remains uncalibrated.
5. **mzTab-M conformance.** The stable claim is withdrawn, but no standards-valid mzTab-M exporter exists. CSV is the only stable export. A future conformance effort needs HUPO-PSI validator fixtures and logs.
6. **Legacy persistent stores.** Externally supplied SQLite/Zarr/hybrid stores are not rewritten automatically. Rebuild from raw sources into a new destination and review IDs/metadata before switching paths.
7. **Installed `massflow tutorial`.** As above, it fails outside a source checkout. Either package the generator or remove/hide the command before publication.
8. **Optional extras are not release-qualified.** `[ml]` still resolves `filelock 3.29.7` with a high-severity advisory (only reachable through the optional Torch path); `setuptools 83.0.0` has a low-severity build/dev advisory. The core runtime audit is clean. Review #115/#113 before claiming optional-ML support.
9. **Default-branch alert dashboard.** GitHub reported 5 critical, 24 high, 13 medium, and 3 low open pip alerts at inspection. The candidate upgrades Pillow and certifi, and the core runtime audit is clean; the dashboard will only re-evaluate after merge. Confirm the remaining alert set is dev/optional-only after merge.

---

## 6. Proposed package/version metadata

```toml
[project]
name = "massflow-msms"
version = "0.1.0"
requires-python = ">=3.13,<3.14"
license = {text = "MIT"}
```

- **Distribution name:** `massflow-msms` (owned-name proposal; not yet registered).
- **Import package:** `MassFlow` (unchanged).
- **CLI:** `massflow` (unchanged).
- **Version:** `0.1.0`; this is a pre-1.0 release candidate. No 1.0 version bump. The project’s own 1.0 decision remains **NO-GO**.
- **Alternative name:** `massflow-ms` is also unregistered and may be preferred if the maintainer wants a shorter name. Changing it later is a packaging-metadata change, not an import/CLI rename.
- **Never use:** `massflow` — not owned by this project, and uploading would collide with an unrelated GPL-3.0 project.

---

## 7. PyPI/GitHub publication steps

1. **Review this draft PR.** Resolve review comments and confirm the restricted scope and known limitations. Do not merge until explicitly approved.
2. **Register the distribution name.** Create/claim `massflow-msms` on PyPI and configure Trusted Publishing (OIDC) for `janusson/MassFlow` or an API token in the protected environment. Confirm the owner account. Do not upload under `massflow`.
3. **Harden the release environment.** Add a second reviewer to the `release` environment, enable `prevent_self_review`, and confirm branch/tag protection.
4. **Merge via PR.** Merge the approved candidate to `main`; do not push directly. Record the merged SHA.
5. **Tag.** Create an annotated tag `v0.1.0` on the merged SHA.
6. **Qualify.** Run `.github/workflows/release.yml` in `qualify` mode with the exact SHA, version, tag, evidence URL, and migration/rollback notes. Review the full suite, scientific tests, core-contract tests, artifact hashes, lock-pinned installed-wheel smoke, and `release-provenance.json`.
7. **Publish the GitHub release.** Run `publish` with confirmation `PUBLISH 0.1.0` and the protected environment approval. This creates the GitHub release with wheel, sdist, `SHA256SUMS`, and provenance. It does **not** publish to PyPI.
8. **Publish to PyPI separately.** On the tagged SHA, rebuild with `uv build`, verify the wheel/sdist hashes match the qualified artifacts, then `uv publish --trusted-publishing automatic` (or `twine upload`). Do not upload unverified local rebuilds.
9. **Verify the public install.** In a clean environment outside the checkout: `pip install massflow-msms==0.1.0`, import `MassFlow`, run `massflow --help`, `massflow init`, a small CSV annotation, and confirm provenance. Record the public artifact hashes.
10. **Post-release.** Keep the mzTab-shaped export marked experimental; do not promote calibrated FDR, confirmed identity, or platform/matrix claims without new evidence. Update the release notes with the exact hashes and known limitations.

---

## 8. GO / NO-GO recommendation

**GO for draft PR and human integration review** of this 0.1.0 candidate under the restricted candidate-retrieval scope.

**NO-GO for public PyPI/GitHub publication** at this time. The code gates are green and the core runtime audit is clean, but publication is blocked by:

- unregistered PyPI distribution name and unconfigured trusted publishing;
- hosted CI not yet run on `97014ba`;
- no independent release reviewer;
- unresolved scientific-validation/calibration limits;
- broken installed `massflow tutorial` command (or an explicit decision to remove/hide it);
- optional-ML dependency advisories that are outside the stable scope but still visible.

Publication can proceed after §5 blockers 1–3 and 7–9 are closed, the final merged SHA passes the guarded qualification workflow, and the maintainer explicitly approves the bounded claims above. Do not convert this candidate into a validated unknown-identification or calibrated-FDR product.
