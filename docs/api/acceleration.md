# Acceleration

The `MassFlow.acceleration` module provides a Numba-accelerated peak and
neutral-loss prefilter that skips query–reference pairs below
`min_matched_peaks` before full greedy modified-cosine scoring. Results are identical
to the pure-NumPy fallback used when `numba` is not installed.

Modified cosine can combine direct-fragment and precursor-shifted matches.
The prefilter therefore sums the maximum matching counts from both frames
before applying the threshold. This sum is an upper bound: a peak counted in
both frames is still used only once by the full greedy scorer. The bound assumes
the counter and scorer recognize the same tolerance-compatible peak pairs;
floating-point differences at tolerance boundaries remain a separate limitation. See the
[synthetic recall experiment](../validation/mixed-frame-prefilter.md) for the
regression that established this requirement.

::: MassFlow.acceleration
