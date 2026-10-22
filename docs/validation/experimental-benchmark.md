# Executable MassBank diagnostic

An isolated benchmark now runs deposited experimental standard spectra through
`MassFlow.workflow.run_annotation_pipeline`. The source, dataset selection and
configuration are frozen. This does not change the historical audit/specification
or complete every experiment in the [scientific validation program](program/README.md).

The executable specification and reproduction command are in
[`benchmarks/massbank-v1/README.md`](https://github.com/janusson/MassFlow/tree/codex/experimental-validation-benchmark/benchmarks/massbank-v1).
It uses positive ESI [M+H]+ records from Athens queries and Eawag references,
retains every query outcome, audits duplicates and reports exact-structure retrieval
separately from confidence calibration. Source labels are depositor assertions of
standard compounds, not independently authenticated identity. Limited contributor,
adduct and polarity coverage precludes a general identification-accuracy claim.

Four arms compare cosine and modified cosine with all truths present, then with
50 connectivity groups removed from the library. Numerical q estimates are
uncalibrated; no nominal FDR is accepted as empirical error control. Known unmerged
correctness fixes remain a condition of interpretation. The measured results and
remaining observability limits are documented in the benchmark directory.
