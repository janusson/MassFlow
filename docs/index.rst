Yogimass
========

Yogimass is a config-first toolkit for importing, cleaning, searching, and networking tandem mass spectrometry (MS/MS) data. It wraps the `matchms <https://matchms.readthedocs.io/>`_ ecosystem with a small set of batteries-included workflows and a CLI.

Config-driven quick start
-------------------------

Create a minimal config (mirrors ``examples/simple_workflow.yaml``):

.. code-block:: yaml

   input:
     path: data/example_library.mgf
   library:
     path: out/example_library.json
   similarity:
     search: true
     queries:
       - data/example_library.mgf
   network:
     enabled: true
     metric: spec2vec
     knn: 5
     output: out/example_network.csv
   outputs:
     search_results: out/example_search.csv
     network_summary: out/example_network_summary.json

Run it via the CLI:

.. code-block:: bash

   python -m yogimass.cli config run --config examples/simple_workflow.yaml

All configs are validated up front by ``yogimass.config.load_config`` and raise a ``ConfigError`` with a dotted path (e.g., ``network.threshold``) and a clear message if something is wrong.

Architecture map
----------------

- ``yogimass.workflow`` — orchestration: ``run_from_config``, ``build_library``, ``search_library``, ``build_network``, ``curate_library``.
- ``yogimass.config`` — typed schema, defaults, and validation for YAML/JSON configs.
- ``yogimass.io`` — I/O helpers for MGF/MSP plus MS-DIAL cleaning/combining utilities.
- ``yogimass.similarity`` — spectrum processing, vectorization, library storage/search, and search backends.
- ``yogimass.networking`` — similarity network construction and export.
- ``yogimass.reporting`` / ``yogimass.curation`` — QC, summaries, and writing reports.

Optional extras: install with ``yogimass[annoy]`` or ``yogimass[faiss]`` to experiment with ANN backends, or ``yogimass[spec2vec]`` when layering in model-backed embeddings.

Using MS-DIAL with Yogimass
---------------------------

MS-DIAL exports are supported as tab-delimited ``.txt``/``.tsv`` alignment tables. Expected columns:

- ``Alignment ID`` (int), ``Average Mz`` (float), ``Name`` (string), ``Model ion area`` (numeric), ``MS/MS spectrum`` (space-delimited ``mz:intensity`` pairs).

Example snippet:

.. code-block:: text

   Alignment ID	Average Mz	Name	Model ion area	MS/MS spectrum
   1	123.45	Citrus_A	1200	100:50 150:25 200:10

Config + CLI:

.. code-block:: yaml

   # examples/msdial_workflow.yaml
   input:
     path: tests/data/msdial_small
     format: msdial
     msdial_output: out/msdial_clean
   library:
     path: out/msdial_library.json
     build: true
     input_format: msdial
   network:
     enabled: true
     metric: spec2vec
     knn: 2
     output: out/msdial_network.csv
   outputs:
     search_results: out/msdial_search.csv
     network_summary: out/msdial_network_summary.json

Run with:

.. code-block:: bash

   python -m yogimass.cli config run --config examples/msdial_workflow.yaml

Outputs map back to MS-DIAL concepts: cleaned per-experiment CSVs and a combined summary live under ``input.msdial_output``; libraries/search results reference MS-DIAL ``Alignment ID``/``Name`` metadata; network exports summarize feature-to-feature similarity.

Legacy utilities
----------------

The ``scripts/example_workflow.py`` script remains for developer tinkering, but the canonical path is the config + CLI flow above. Use ``yogimass config run`` for reproducible pipelines.
