# Yogimass Core

Yogimass focuses on deterministic MS/MS spectrum processing defaults and similarity scoring.

## Scope

- Spectrum processing defaults (filtering, normalization, m/z dedupe, alignment)
- Similarity metrics (cosine, modified cosine) and Spec2Vec-style vectorization
- Scoring helpers backed by `matchms`

## Install

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
