Yogimass documentation index
============================

Welcome to Yogimass, a modular toolkit for importing, cleaning, exporting, and comparing tandem mass spectrometry (MS/MS) data.

.. contents::
   :depth: 1
   :local:

What is Yogimass?
-----------------

Yogimass wraps the `matchms <https://matchms.readthedocs.io/>`_ ecosystem into a cohesive pipeline. The ``yogimass.pipeline`` module exposes helpers for:

- listing MGF/MSP libraries from a directory
- applying metadata and peak cleaning filters
- exporting curated spectra back to MGF/MSP/JSON/pickle formats
- computing cosine and modified cosine similarity scores between spectra

Current capabilities
--------------------

- **I/O helpers** (``yogimass.io.mgf``) list available libraries, fetch individual spectra, and save cleaned collections.
- **Filtering utilities** (``yogimass.filters.metadata``) normalize compound metadata and peak lists using the standard matchms filters.
- **Scoring** (``yogimass.scoring.cosine``) calculates cosine/modified cosine similarities and surfaces top matches.
- **Example workflow** (``scripts/example_workflow.py``) demonstrates loading an MGF library, cleaning each spectrum, and exporting the cleaned spectra.
- **Batch cleaning** (``yogimass.pipeline``) provides ``batch_clean_mgf_libraries`` / ``batch_clean_msp_libraries`` and a CLI entry point (``yogimass clean``) to sweep entire directories and export cleaned spectra in multiple formats.

What still needs work
---------------------

- Implement the MSDIAL cleaning/processing stubs in ``yogimass/io/msdial_*.py``.
- Add thorough unit/integration tests that cover the full cleaning and scoring workflow.
- Provide packaging metadata (``pyproject.toml`` or ``setup.py``) for installation.
- Expand the documentation with API references, CLI/Notebook examples, and data requirements.

Quick start
-----------

1. Install dependencies: ``pip install -r requirements.txt``
2. Drop raw MGF/MSP libraries anywhere under ``./data`` (or point the CLI/scripts to another directory). Output directories are created automatically and cleaned files follow the ``<library>_cleaned.<ext>`` naming pattern.
3. Run ``python scripts/example_workflow.py`` to clean and export the first available library into ``./out`` (or call ``clean_mgf_library`` / ``clean_msp_library`` or the batch helpers directly for custom workflows).
4. Prefer the CLI for automation: ``yogimass clean ./data ./out --type mgf --formats mgf json``.
