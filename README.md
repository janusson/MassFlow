# Yogimass Core

Yogimass focuses on deterministic MS/MS spectrum processing defaults and similarity scoring.

## Scope

- Spectrum processing defaults (filtering, normalization, m/z dedupe, alignment)
- Similarity metrics (cosine, modified cosine) and Spec2Vec-style vectorization
- Scoring helpers backed by `matchms`

## Install

Install the core package:

```bash
pip install -e .
```

## Usage

```python
from matchms import Spectrum
import numpy as np

from yogimass.config import ProcessorConfig
from yogimass.similarity import SpectrumProcessor, cosine_similarity

spectrum = Spectrum(
    mz=np.asarray([50.0, 75.0], dtype="float32"),
    intensities=np.asarray([100.0, 50.0], dtype="float32"),
    metadata={"name": "example"},
)

config = ProcessorConfig.from_mapping(
    {"normalization": "basepeak", "min_relative_intensity": 0.02}
)
processor = SpectrumProcessor(**config.to_kwargs())
processed = processor.process(spectrum)

score = cosine_similarity(processed, processed)
print(score)
```

## CLI

After installing, you can run workflows from the command line:

```bash
yogimass config run --config examples/simple_workflow.yaml
# or
python -m yogimass config run --config examples/simple_workflow.yaml
```

## Splinters

Legacy workflow/CLI, I/O, networking, and MS-DIAL integration code has been moved
into `splinters/` for future extraction into separate repositories:

- `splinters/workflows`: config runner, CLI, pipeline orchestration, examples/docs
- `splinters/similarity-search`: library storage/search and indexing backends
- `splinters/io-msdial`: file I/O and MS-DIAL integration
- `splinters/networking`: similarity graph construction/export
- `splinters/curation`: QC/curation and reporting
- `splinters/deprecated`: legacy entrypoints
- `splinters/utils`: misc utilities and formula helpers

## Optional Installs

Optional extras (enable splinter features such as ANN search, MS-DIAL helpers, or model-backed embeddings):

```bash
# developer/test deps
pip install -e ".[dev]"

# add annoy (ANN-backed search)
pip install -e ".[annoy]"

# MS-DIAL support (pandas)
pip install -e ".[msdial]"

# install common optional extras
pip install -e ".[all]"

# full install (includes all optional extras)
pip install -e ".[full]"
```

> Note: The `faiss`/`faiss-cpu` packages are platform-specific and were removed from the default extras in order to keep the repository minimal and easy to install. If you need FAISS later, you can install it manually (e.g., `pip install faiss-cpu`) and add a FAISS-backed search backend as needed.
