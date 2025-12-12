# Yogimass library schema

Yogimass stores processed spectra in a lightweight library format suitable for similarity search. Libraries can be persisted as JSON (default) or SQLite; both encodings share the same logical schema.

## Schema version

- `schema_version` identifies the on-disk format; the current version is `1.0`.
- JSON libraries include `"schema_version": "1.0"` at the top level.
- SQLite libraries store the version in a `metadata` table (`key='schema_version'`).
- Older libraries without a version are treated as legacy and accepted by default.

## Logical model

- A library is a collection of **entries**.
- Each entry has:
  - `identifier` (string, unique) — spectrum name or generated ID.
  - `precursor_mz` (float or null) — precursor m/z if available.
  - `metadata` (object) — arbitrary key/value pairs copied from the source spectrum (nulls are dropped; `compound_name` is mirrored to `name` when missing).
  - `vector` (object) — sparse Spec2Vec-style vector mapping token strings to float weights.
- Peak lists are **not** stored; only metadata and vectors are persisted for search.

## JSON encoding

JSON libraries are a mapping with a version and an entry list:

```json
{
  "schema_version": "1.0",
  "entries": [
    {
      "identifier": "spectrum-123",
      "precursor_mz": 321.123,
      "metadata": {"name": "example", "instrument": "QTOF"},
      "vector": {"token_0": 0.12, "token_1": 0.04}
    }
  ]
}
```

## SQLite encoding

SQLite libraries store the same fields in a `spectra` table with columns:

- `identifier` TEXT PRIMARY KEY
- `precursor_mz` REAL
- `metadata` TEXT (JSON string)
- `vector` TEXT (JSON string)

## Invariants and compatibility

- `identifier` is unique and required; updates replace matching identifiers when `overwrite=True`.
- `metadata` keys are free-form; readers should ignore unknown keys.
- `vector` keys/values are floats; ordering is not significant.
- Future versions may add optional fields; older readers should ignore unknown fields rather than failing.
- Compatibility helpers (`read_schema_version`, `assert_schema_compatible`) live in `yogimass.library_io`.

## Export helpers

- `export_library_summary_csv(library_path, output_path)` writes a simple CSV with identifiers, precursor m/z, names, and retention times for downstream tools.
