# Audit evidence and reproducibility

This folder records an audit and program design, not a newly executed chemical
identification/calibration study. [verification.json](verification.json) records
checks performed on the delivered specification. The application baseline is
`ffef50bff913fcf400cb3de72595dab50e4f3fd8`; production code, tests and lockfile are
unchanged in this branch.

## Historical recount

[recount_saved_outputs.py](recount_saved_outputs.py) uses only Python's standard
library. It reads historical files without writing to them or importing MassFlow:

```sh
python3 docs/validation/program/evidence/recount_saved_outputs.py \
  --base /tmp/massflow-benchmark-20261018 > /tmp/massflow-validation-recount.json
```

Compare `runs` and `asset_hashes` with [audit-snapshot.json](audit-snapshot.json).
The original private/local historical files are not redistributed here; if they
are unavailable, the recount cannot be rerun from this folder alone. Each file is
identified by SHA-256 and size. The helper uses the existing label-to-library-ID
mapping, requires no tied best score (and fails explicitly if one appears), and
checks no new chemical truth. Preprocessing rejection counts come from historical
sidecars, not reconstructed execution. Candidate rows with empty scores are
Unknown rows and do not count as calls.

[github-snapshot.json](github-snapshot.json) records the live issue/PR metadata
inspected; it is a dated snapshot, not a continuing status monitor. PR test claims
are not independent evidence that a correction is merged or chemically validated.

## Specification checks

Run from this worktree root:

```sh
uv sync --locked
uv run mkdocs build --strict
uv run ruff check docs/validation/program/evidence/recount_saved_outputs.py
uv run ruff format --check docs/validation/program/evidence/recount_saved_outputs.py
uv run mypy docs/validation/program/evidence/recount_saved_outputs.py
uv run pytest tests/test_mathematical_proof.py tests/test_fdr_statistics.py tests/test_scientific_validation.py -q
uv run pytest --cov=src/MassFlow --cov-report=xml --cov-fail-under=80 -v
```

Validate the JSON schemas/template without adding a project dependency:

```sh
uv run --with jsonschema python - <<'PY'
from pathlib import Path
import json
import jsonschema
root = Path('docs/validation/program')
for filename in ('manifest.schema.json', 'record.schema.json'):
    jsonschema.Draft202012Validator.check_schema(json.loads((root / filename).read_text()))
schema = json.loads((root / 'manifest.schema.json').read_text())
template = json.loads((root / 'manifest.template.json').read_text())
validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
validator.validate(template)
template['status'] = 'frozen'
assert len(list(validator.iter_errors(template))) == 6
print('Two valid schemas; empty draft allowed; empty frozen manifest rejected.')
PY
```

These checks verify document links/build, schema logic, recount arithmetic and
existing application tests. They do not validate acquisition independence, original
data rights, chemical identity or FDR. The empty template is not a frozen dataset.

## Independent review

A separate read-only reviewer inspected the specification and relevant pinned
source. Three substantive design findings were corrected and checked again:

1. Split/select/assign absence by whole connectivity groups, keeping full keys for
   strict scoring; this prevents variant leakage and accidental removal of a
   supposedly present truth.
2. Match intended-use scope to ESI reference-spectrum retrieval and capture
   ionization/inlet/centroiding evidence; no LC/matrix validation inferred from FIA.
3. Limit binomial intervals to justified fixed-rule independent sampling; recompute
   q during grouped resampling, and label pilot resampling as sensitivity evidence.

The reviewer reported no remaining P1 finding in that focused follow-up. This
review is a design check, not empirical validation or release authorization.
