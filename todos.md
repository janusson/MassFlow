# MassFlow Todos

Tracked plan for MassFlow (session-tracked todos exist in the Copilot session DB).

Priority order (start here):

1. FDR tests (tests/test_fdr.py) - unit tests for calculate_fdr edge cases
2. Decoy tests - ensure generate_decoys behavior and edge-cases
3. MS1 prefilter tests - validate ppm vs Da paths and missing-precursor handling
4. SimilarityEngine tests - factory behavior, ML-model errors, consensus fallback
5. ConsensusEngine tests - isotopic credibility, neutral-loss penalties, tie-breaking
6. Docs: engine-tuning guide - ms1_tolerance vs resolution_ppm, recommended starting values

Status:
- test-fdr: in progress (being implemented)
- other items: pending

Notes:
- Each test should be small and deterministic. Prefer NumPy arrays with explicit dtypes.
- Tests should not depend on real ML models; mock or assert raised errors for missing models.
- Database schema changes must be documented in src/MassFlow/database.py.

Created by Copilot CLI to give a human-friendly checklist alongside the session-tracked todos.
