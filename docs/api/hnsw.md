# HNSW Candidate Indexing

The `MassFlow.hnsw` module wraps hnswlib to provide sub-linear approximate
candidate retrieval for the `CascadeEngine`. Spectral vectors are binned into a
two-channel representation — `[binned exact m/z, binned neutral losses]` — so
the index can find precursor-shifted analogues that an exact-m/z-only index
would miss. Requires the `hnsw` extra.

::: MassFlow.hnsw
