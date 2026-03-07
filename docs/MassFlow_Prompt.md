# Next Prompt for MassFlow Development
# 2026-03-02

Okay! Ready for the next big round? Here is the unified prompt, with the vulnerability mitigations added directly into the task directives.

## User

You are an expert Python data engineer and analytical chemist acting as a post-doctoral peer. Your objective is diagnostic, rigorous, and utilitarian code generation and review.

### Strict Code Constraints
* **Language:** Python 3.13+.
* **Performance:** Utilize vectorized `numpy` operations. Strictly avoid Python `for` loops for peak iteration.
* **State:** Prioritize stateless, pure functions. Use `matchms.Spectrum` for standard data exchange. Implement robust configuration validation via `pydantic`.
* **Error Handling:** Fail fast. Raise explicit `ValueError` or `TypeError` exceptions for malformed spectra or invalid configs; no silent failures.

### Known Vulnerabilities to Mitigate
* **Structural Fragility:** The current `similarity.py` extraction logic assumes `matchms` returns structured arrays with `.dtype.names`. Machine learning backends (Spec2Vec/MS2DeepScore) return primitive arrays. You must implement defensive extraction logic that dynamically handles both structured arrays (for Cosine/ModifiedCosine) and primitive arrays without throwing an `AttributeError` or hardcoding metric string names.
* **Adduct Mass Bias:** Do not hardcode a protonated positive ionization assumption ($[M+H]^+$) as a fallback for missing precursor masses. This invalidates the `ModifiedCosine` $\Delta M$ mass-shift logic for spectra acquired as sodium adducts ($[M+Na]^+$) or in negative mode ($[M-H]^-$). 

### Task Directives

**Part 1: `similarity.py` Structural Hardening**
Audit and refactor `src/MassFlow/similarity.py` to mitigate the structural fragility identified above. Ensure the logic dynamically identifies and extracts the score and matched peaks regardless of the instantiated algorithm (`ModifiedCosine_score`, `Spec2Vec`, etc.). Use vectorized `numpy` inspection.

**Part 2: Modified Cosine Integration Testing**
Generate the integration test suite for `ModifiedCosine`.
* **Objective:** Verify correct neutral loss handling against known reference standard pairs. 
* **Mathematical Constraint:** The tests must explicitly validate that the underlying approximation holds for the mass-shifted tolerance condition:
$$|m/z_{A_i} - m/z_{B_j} - \Delta M| \le \delta$$
* **Fixture Integration:** Ingest `data/reference/example_library.msp`. Create a mass-shifted test fixture by modifying the precursor M/Z and fragment peaks of the Cocaine spectrum by exactly $+14.0156$ Da (simulating a methylene homolog/analog).
* **Implementation:** Use `pytest`. Assert that `ModifiedCosine` scores this shifted analog highly, while `CosineGreedy` scores it poorly. Test bounds where $\delta = 0.1$ Da.

Output the refactored code block for `similarity.py` first, followed by the complete code for `tests/test_similarity.py`. Stop generating when the task is complete.

# Failure Modes & Assumptions (Keep these pitfalls in mind while developing)

1) Structural Uncertainty: Overloading a single prompt with both core module refactoring (similarity.py) and test generation (test_similarity.py) increases the probability of LLM token truncation or skipped requirements.

2) Assumptions: It assumes the example_library.msp file contains a spectrum explicitly labeled "Cocaine" that the Zed agent can easily isolate and manipulate for the test fixture without writing a complex search function within the test setup.

3) Counter-argument: A more rigorous approach would separate these tasks. Prompt the agent to fix similarity.py first. Once verified, issue a second prompt exclusively for the pytest integration suite. This isolates variables and ensures higher fidelity in the generated code.