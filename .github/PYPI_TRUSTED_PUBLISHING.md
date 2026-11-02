# PyPI Trusted Publishing for massflow-msms

MassFlow publishes the distribution **`massflow-msms`** from the guarded
`.github/workflows/release.yml` workflow. It uses GitHub Actions OIDC Trusted
Publishing: no PyPI API token, password, or repository secret is stored.

Do **not** publish under `massflow`. That name on PyPI is owned by an unrelated
GPL-3.0 project. The import package remains `MassFlow` and the CLI remains
`massflow`; only the distribution name differs.

## One-time PyPI setup (owner)

1. Sign in to PyPI as the owner account for this project.
2. Open **Publishing → Add a new pending publisher**.
3. Enter exactly:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `massflow-msms` |
   | Owner | `janusson` |
   | Repository name | `MassFlow` |
   | Workflow name | `release.yml` |
   | Environment name | `release` |

4. Save the pending publisher. The first successful trusted upload creates the
   project. A pending publisher is sufficient; the project need not exist yet.
5. Do not create an API token for this workflow.

## One-time GitHub setup (owner)

1. Create or confirm the `release` environment under
   **Settings → Environments**.
2. Add a required reviewer. Choose the owner's approval policy:
   - independent reviewer plus `prevent_self_review: true`; or
   - solo-maintainer auditable approval, recorded as described in
     [`RELEASE_PROCESS.md`](RELEASE_PROCESS.md).
3. Confirm branch protection for `main` and tag protection for `v*`.
4. No PyPI secret is added to the environment or repository.

## Publication path

1. Run `release.yml` in `qualify` mode on the exact candidate SHA. This does not
   publish. Review the full-suite/scientific/core results, installed-wheel smoke,
   `dist/SHA256SUMS`, and `release-provenance.json`.
2. Create an annotated tag `v<version>` on that SHA.
3. Run `release.yml` in `publish` mode with the exact candidate SHA, version,
   tag, `confirmation="PUBLISH <version>"`, evidence URL, migration notes, and
   rollback notes.
4. The publish job:
   - downloads the qualified artifact from the qualifying run;
   - verifies `SHA256SUMS` and `release-provenance.json`;
   - stages **only** the verified wheel and sdist;
   - creates the GitHub release;
   - authenticates to PyPI with `id-token: write` and
     `pypa/gh-action-pypi-publish@release/v1`;
   - uploads the same files with PEP 740 attestations.

There is no rebuild in the publish job. A changed hash is a hard failure.

## Verification after publication

```sh
python -m venv /tmp/massflow-pypi-verify
/tmp/massflow-pypi-verify/bin/pip install massflow-msms==<version>
/tmp/massflow-pypi-verify/bin/massflow --help
```

Confirm the downloaded wheel/sdist SHA-256 values match the GitHub release
`SHA256SUMS` and `release-provenance.json`. Compare the PyPI project files page
hashes, not only the local files.

## Rollback

Yank the affected files on PyPI; do not delete or reuse the version. Delete the
GitHub release and keep the yank reason in the release notes. Fix forward with a
new patch version and new annotated tag. Re-run qualification and publication.
