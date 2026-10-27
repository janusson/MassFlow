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
3. Assemble the independent qualification evidence: full-suite result,
   scientific-validation result, packaging/installed-wheel smoke, and the frozen
   benchmark result if the candidate changes scoring, preprocessing, retrieval,
   ranking, confidence, or reported metrics. The evidence must be reachable at
   an `https://` URL or a committed `docs/` path.
4. Create and push an **annotated** tag only when the candidate is ready for
   qualification:
   ```bash
   git tag -a v1.0.0 <candidate_sha> -m "MassFlow 1.0.0 candidate"
   git push origin v1.0.0
   ```
   Pushing the tag does not run the release workflow.
5. Run the workflow in `qualify` mode (UI or CLI):
   ```bash
   gh workflow run release.yml --ref main \
     -f action=qualify \
     -f candidate_sha=<candidate_sha> \
     -f version=1.0.0 \
     -f tag=v1.0.0 \
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
- The installed-package smoke used the lock-exported runtime requirements,
  resolved MassFlow from `site-packages`, imported the shipped streaming
  protobuf resources, passed `uv pip check`, and wrote a result CSV.
- The evidence URL is complete and does not rely on an uncommitted local run.

The `release` environment currently requires a reviewer. **Independent review
is not yet available in this repository because there is only one collaborator.**
Before an official release, a repository administrator should add a second
reviewer and set `prevent_self_review: true` on the environment. Until then, the
environment gate is a deliberate human action but is not independent approval.

## 3. Publish

Only after the qualification and review are accepted, run the workflow again in
`publish` mode:

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

The `publish` job re-runs qualification from scratch on the same SHA, waits for
the `release` environment approval, re-verifies every artifact hash and the
annotated tag, and only then creates a GitHub release. It attaches the wheel,
sdist, `SHA256SUMS`, and `release-provenance.json`. It does not publish to PyPI.

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
- Confirm branch protection prevents direct unreviewed pushes to `main`.
- Confirm tag protection prevents rewriting release tags.
- Confirm the qualification evidence is committed or hosted durably.
- Confirm the supported OS/Python matrix in the evidence matches the release
  notes. The current candidate has local macOS evidence and hosted Linux/macOS
  CI; Windows is not qualified unless explicitly tested.
