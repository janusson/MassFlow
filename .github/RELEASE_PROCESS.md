# MassFlow release process

This document describes the **manual, qualification-first** release path. It
replaces the former tag-triggered workflow: pushing a version tag no longer
publishes anything.

## Guarantees

The release workflow (`/.github/workflows/release.yml`) is started only by
`workflow_dispatch`. Publication requires all of the following:

1. A deliberate workflow run with `action: publish`.
2. A full 40-character `candidate_sha` that is reachable from `main` or a
   `release/*` branch.
3. A `version` that exactly matches `pyproject.toml`.
4. An existing **annotated** tag `v<version>` that peels to `candidate_sha`.
5. A successful qualification job on that exact SHA.
6. The protected `release` GitHub Environment approval.
7. The exact confirmation phrase `PUBLISH <version>`.
8. Non-empty migration and rollback notes.

A tag alone cannot publish, and a branch push cannot publish. The workflow has
no `push` trigger.

## 1. Prepare and qualify a candidate

1. Finish the code and evidence on `main` or a `release/*` branch. Do not qualify
   a feature-branch commit.
2. Set the release version in `pyproject.toml`.
3. Assemble independent evidence for the exact candidate SHA and version using
   the acceptance checklist below. The evidence must be reachable at an
   `https://` URL or a committed `docs/` path. An evidence commit changes the
   candidate SHA; qualify the final combined candidate.
4. Run the workflow in `qualify` mode without creating or supplying an official
   release tag (UI or CLI):
   ```bash
   gh workflow run release.yml --ref main \
     -f action=qualify \
     -f candidate_sha=<candidate_sha> \
     -f version=1.0.0 \
     -f evidence_url=https://example.invalid/massflow-1.0.0-qualification
   ```
   `qualify` does not publish. It checks out the exact SHA, verifies the input
   agreements, runs the lockfile/lint/type/full-coverage/scientific/core gates,
   builds the wheel and sdist, installs the wheel in a clean environment, runs a
   CLI/init/annotation smoke, records SHA-256 hashes and provenance, and uploads
   the qualified artifacts.

## 2. Review before publication

Review the qualification job output and the uploaded `release-candidate`
artifact. Confirm:

- The candidate SHA and version match the intended release.
- The full suite and scientific tests passed.
- The wheel and sdist hashes in `dist/SHA256SUMS` match the provenance file.
- The installed-package smoke resolved MassFlow from `site-packages` and wrote
  a result CSV.
- The evidence URL is complete and does not rely on an uncommitted local run.

### Issue #109 acceptance evidence

A successful `qualify` job covers only part of issue #109. Before accepting the
candidate, the human reviewer must verify all of the following for the same SHA
and version. Missing or failed evidence blocks acceptance. Preserve uncalibrated
outcomes explicitly and review their implications; do not relabel them as passed.

- Build both wheel and sdist from the candidate and install each independently
  in clean supported OS/Python environments. Run from outside the checkout.
  The workflow currently installs only the wheel on Linux/Python 3.13.
- Exercise installed CLI init/annotate and database build/inspect/merge workflows,
  core-only installation, and optional-dependency availability and absence
  contracts. Source tests and the installed annotation smoke are partial evidence.
- Inspect both distributions for required package data, metadata, declared
  dependencies, and licenses. A successful build alone does not satisfy this.
- Record full coverage, lint, type, and documentation gate results. The workflow
  runs coverage/lint/type gates but has no documentation build gate.
- Run the frozen measured-data benchmark on the combined candidate for issue
  #109 acceptance, including when this PR changes only release governance.
  Preserve failed and uncalibrated outcomes and their interpretation.
- Record distribution SHA-256 values, source-content, lockfile, configuration,
  and benchmark-data hashes, plus commands and environment identities. The
  workflow records candidate SHA, lockfile/pyproject hashes and artifact hashes;
  it does not supply the complete source/configuration/data hash manifest.
- Review candidate-specific migration and rollback instructions. Non-empty text
  is an automated check, not evidence that the procedures are sufficient.
- Preserve test reports, benchmark results, and failure diagnostics durably.
  The workflow currently uploads release artifacts only after success and keeps
  them for 14 days; archive acceptance evidence separately, including failed runs.
- Record explicit human acceptance of the SHA/version and authorization to
  create the official tag. Issue #109 authorizes preparation only.

The `release` environment currently requires a reviewer. **Independent review
is not yet available in this repository because there is only one collaborator.**
Before an official release, a repository administrator should add a second
reviewer and set `prevent_self_review: true` on the environment. Until then, the
environment gate is a deliberate human action but is not independent approval.

## 3. Publish

Only after qualification, the complete acceptance evidence, and the human
prerequisites below are accepted, obtain explicit human authorization to create
and push the official annotated tag at the reviewed SHA:

```bash
git tag -a v1.0.0 <candidate_sha> -m "MassFlow 1.0.0"
git push origin v1.0.0
```

Pushing the tag does not run the release workflow. If qualification fails, fix
the candidate and qualify the new SHA before creating any official tag.
Publication requires separate explicit human authorization. Then run the
workflow again in `publish` mode:

```bash
gh workflow run release.yml --ref main \
  -f action=publish \
  -f candidate_sha=<candidate_sha> \
  -f version=1.0.0 \
  -f tag=v1.0.0 \
  -f confirmation="PUBLISH 1.0.0" \
  -f evidence_url=https://example.invalid/massflow-1.0.0-qualification \
  -f migration_notes="<migration notes>" \
  -f rollback_notes="<rollback procedure>"
```

The publish dispatch re-runs qualification from scratch on the same SHA, waits for
the `release` environment approval, re-verifies every artifact hash and the
annotated tag, and only then creates a GitHub release. It attaches the wheel,
sdist, `SHA256SUMS`, and `release-provenance.json`. It does not publish to PyPI.

These are newly built artifacts, not the binaries from the earlier qualify run.
Before approving the environment gate, review this publish run's qualification
output, provenance and artifact hashes against the accepted candidate evidence.

## Provenance and rollback

`release-provenance.json` records the repository, candidate SHA, version, tag,
workflow run identifiers, evidence URL, migration/rollback notes, Python and
platform, `uv.lock` and `pyproject.toml` hashes, and every artifact SHA-256.
Keep it with the release.

Rollback procedure:

1. Delete the faulty GitHub release. Do not move or reuse the published tag.
2. Restore the previous qualified release as the supported version.
3. Fix forward with a new patch version and a new annotated tag.
4. Re-run qualification and publication through this workflow.

If the defect is a security or data-integrity issue, publish a concise advisory
and state which versions are affected. Do not claim that a successful workflow
proves a vulnerability was unreachable.

## Human prerequisites before first use

- Add an independent reviewer to the `release` environment and enable
  `prevent_self_review`.
- Configure and verify allowed deployment branches for the `release` environment
  (the dispatch workflow ref), and decide whether administrator bypass is allowed.
  Naming the environment in YAML does not establish these protections.
- Confirm branch protection prevents direct unreviewed pushes to `main`.
- Confirm tag protection prevents rewriting release tags.
- Confirm the qualification evidence is committed or hosted durably.
- Confirm the supported OS/Python matrix in the evidence matches the release
  notes. The current candidate has local macOS evidence and hosted Linux/macOS
  CI; Windows is not qualified unless explicitly tested.
