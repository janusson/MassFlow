import time

from matchms.importing import load_from_msp

from MassFlow import processing
from MassFlow.config import MassFlowConfig
from MassFlow.similarity import SimilarityEngine

lib_path = "../MassFlow_Data/libraries/ALL_GNPS.msp"
config = MassFlowConfig.from_yaml("massflow_config.yaml")

print("1. Benchmarking raw parsing (5000 spectra)...")
start = time.time()
raw_gen = load_from_msp(lib_path, metadata_harmonization=False)
raw_spectra = []
for i, s in enumerate(raw_gen):
    raw_spectra.append(s)
    if i >= 4999:
        break
print(f"Raw parsing: {time.time() - start:.2f} seconds")

print("2. Benchmarking processing (5000 spectra)...")
start = time.time()
processed_spectra = list(
    processing.process_spectra_batch(raw_spectra, config.processing)
)
print(f"Processing: {time.time() - start:.2f} seconds")

print("3. Benchmarking Cosine scoring (1 query vs 5000 refs)...")

engine = SimilarityEngine(config.similarity)
start = time.time()
# dummy query
query = processed_spectra[0]
res = engine.search([query], processed_spectra, include_decoys=False)
print(f"Scoring: {time.time() - start:.2f} seconds")
