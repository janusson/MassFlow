# Storage Backends

MassFlow stores reference libraries in SQLite, chunked Zarr, or a hybrid of
both. `MassFlow.storage` defines the abstract `SpectralStore` contract and the
`create_spectral_store` factory; `MassFlow.zarr_store` implements the pure-Zarr
and hybrid (SQLite metadata + Zarr peak arrays) backends.

## Peak fidelity contract

Storing a spectrum never silently changes its scientific data:

- The **flat** layout (default) and the hybrid `ZarrPeakArrayStore` store
  every peak exactly — arbitrary peak counts, no capacity, no reduction.
- The **tensor** layout has a fixed per-spectrum capacity
  (`max_peaks_per_spectrum`). What happens to an over-capacity spectrum is
  governed by the explicit `peak_reduction` policy:
  - `"none"` (default): `add_spectra` raises `PeakCapacityError` and the
    batch is rejected atomically — nothing is written.
  - `"topN"`: the spectrum is explicitly reduced to its N most intense
    peaks (deterministic, stable tie-breaking) and the reduction is recorded
    in the spectrum's stored provenance (`extra_metadata.peak_reduction`),
    visible on every read.
- The active layout, capacity, and reduction policy are persisted in a store
  manifest and validated on every open; reopening a store with mismatched
  parameters raises `ValueError`. Pre-manifest (legacy) stores are inferred
  from their arrays and may only be opened with the strict policy.

::: MassFlow.storage

::: MassFlow.zarr_store
