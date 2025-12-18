# Splinters (future repos)

This directory holds code moved out of the core Yogimass package. The intent is to
split each subfolder into its own repository once the core is stable.

## Why this exists

- Keep the core package focused on spectrum processing defaults and similarity/scoring.
- Preserve legacy functionality for later extraction without blocking core work.

## Contents

- `workflows/`: config runner, CLI, pipeline orchestration, examples, docs, scripts
- `similarity-search/`: library storage/search, indexing backends, batch helpers
- `io-msdial/`: file I/O and MS-DIAL integration
- `networking/`: similarity graph construction/export
- `curation/`: QC/curation and reporting
- `deprecated/`: legacy entrypoints
- `utils/`: misc utilities and formula helpers

## How to pick this up later

1. Decide the next repo to extract (one subfolder per repo is the assumption).
2. Move the subfolder into a new repo and restore any packaging/docs that were
   moved here (e.g., `workflows/README.md`, `workflows/docs/`).
3. Update imports and dependencies so they point at the new repo name.
4. Port any tests that live under `workflows/tests/` or add new tests in the new repo.
5. Release the new repo, then delete the corresponding splinter folder here.

## Practical extraction tips

- Prefer `git subtree split` or `git filter-repo` so history is preserved.
- Keep `yogimass` package paths intact until you rename the package.
- After extraction, remove any leftover references from this core repo.

## Notes for future you

- The core repo intentionally removed `yogimass.config` (workflow config), CLI,
  I/O, networking, and curation. If you need them, look in the matching splinter.
- The only config kept in core is `yogimass.config.ProcessorConfig`.
