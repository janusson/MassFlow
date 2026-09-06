# Acceleration

The `MassFlow.acceleration` module provides a Numba-accelerated peak and
neutral-loss prefilter that skips query–reference pairs below
`min_matched_peaks` before exact modified-cosine scoring. Results are identical
to the pure-NumPy fallback used when `numba` is not installed.

::: MassFlow.acceleration
