# HNSW Candidate Indexing

The `MassFlow.hnsw` module is an experimental hnswlib-backed approximate
candidate-retrieval stage for `CascadeEngine`. Spectral vectors use a two-channel
representation — `[binned exact m/z, binned neutral losses]` — to support shifted
analogues. Approximate retrieval has no general recall guarantee and requires the
`hnsw` extra.

::: MassFlow.hnsw
