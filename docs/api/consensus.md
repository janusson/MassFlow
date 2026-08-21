# Consensus Scoring

Consensus scoring is implemented by the `ConsensusEngine` class in
`MassFlow.similarity`. It computes scores from multiple independent
sub-engines (cosine, modified_cosine, and — when the `[ml]` extra is
installed — spec2vec and ms2deepscore) and produces a weighted consensus
score, improving annotation confidence by leveraging orthogonal scoring
approaches.

Configure it in YAML with `similarity.algorithm: "consensus"`; per-engine
weights live under `similarity.consensus_weights` and the minimum number
of sub-engines that must agree under `similarity.consensus_min_engines`.
If no sub-engine can be built (ML dependencies missing, remote endpoints
unreachable), scoring falls back to `modified_cosine` — a failed ML call
can never crash the run.

!!! note "Standalone `MassFlow.consensus` module removed"

    The v0.2-era orchestrator module `MassFlow.consensus` (with
    `generate_consensus` / `ConsensusResult`) was removed during the v1.0
    engine lockdown. Consensus *scoring* now lives in
    `MassFlow.similarity.ConsensusEngine`, and the engine-agnostic data
    contracts (`ConsensusInput`, `ConsensusResult`, `ConsensusConfig`)
    live in `MassFlow.models`. See the [Similarity page](similarity.md)
    for the full engine family (consensus, cascade, router).

::: MassFlow.similarity.ConsensusEngine
