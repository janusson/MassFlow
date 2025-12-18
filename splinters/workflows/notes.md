# A couple of lightweight cleanups worth considering

scripts/check.sh currently runs compileall + pytest; if you want it to be the standard preflight, we could add a brief note in USAGE/README and maybe ensure it respects PYTEST_ADDOPTS/extra args (it already forwards args).

.gitignore still references MyPy cache? (If so, we can prune that since MyPy is dropped.)

requirements.txt/dev extras: ensure dev extras mirror what CI uses (pytest/pytest-cov are already there), and possibly add networkx to dev if you want graph exports tested locally without optional import errors.

Docs: link the new scripts/check.sh in CONTRIBUTING or USAGE so contributors find it; and ensure the quickstart + new export helpers are mentioned in README’s intro.

Minor test hygiene: a few tests still use absolute temp paths via tmp_path; we could standardize path creation to avoid stray env reliance, but nothing urgent.

Happy to implement any of these if you’d like.
