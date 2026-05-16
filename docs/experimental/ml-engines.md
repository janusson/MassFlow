# Advanced ML Engines (Experimental)

!!! warning "Experimental Feature"
    Machine Learning similarity engines are not part of the stable MassFlow v1.0 contract.

    They are provided as-is, require external model weights, and are subject to breaking changes in future releases. Use them with caution for production annotations.

MassFlow currently includes experimental support for two deep learning-based spectral similarity engines developed by the matchms ecosystem: **Spec2Vec** and **MS2DeepScore**.

These algorithms attempt to learn complex fragmentation patterns and structural relationships from massive reference libraries, allowing them to find structural analogs even when spectra share very few exact fragment mass matches.

---

## Spec2Vec

Spec2Vec treats mass spectra like text documents, where fragment peaks are "words" and spectra are "sentences." It uses a Word2Vec model to learn embeddings for peaks based on their co-occurrence, allowing it to calculate a similarity score between two spectra based on their learned vector representations.

### Configuration

To use Spec2Vec, you must provide a pre-trained Gensim Word2Vec model file in your YAML configuration.

```yaml
similarity:
  algorithm: "spec2vec"
  model_path: "models/spec2vec_model.model"
  min_score: 0.6
  fdr_threshold: 0.05
```

*   `algorithm`: Must be exactly `"spec2vec"`.
*   `model_path`: The file path to your trained Spec2Vec model weights.
*   `min_score`: The minimum similarity score to retain a match.
*   `fdr_threshold`: The target False Discovery Rate.

---

## MS2DeepScore

MS2DeepScore is a Siamese neural network that predicts the structural similarity (e.g., Tanimoto score) between two molecules based solely on their MS/MS spectra. It is highly effective at identifying structural analogs that classical cosine scoring would miss entirely.

### Configuration

To use MS2DeepScore, you must provide a pre-trained PyTorch model file.

```yaml
similarity:
  algorithm: "ms2deepscore"
  model_path: "models/ms2deepscore_model.pt"
  min_score: 0.8
  fdr_threshold: 0.05
```

*   `algorithm`: Must be exactly `"ms2deepscore"`.
*   `model_path`: The file path to your trained PyTorch `.pt` model.
*   `min_score`: The minimum predicted structural similarity score. MS2DeepScore scores are typically calibrated to align with Tanimoto similarities, so a threshold of `0.8` or higher is common.

---

## Important Constraints

1.  **Dependencies:** You must ensure that your Python environment has `spec2vec` and `ms2deepscore` installed. (They are included in the MassFlow `pyproject.toml` dependencies, but may require manual installation of underlying ML frameworks like PyTorch depending on your OS).
2.  **Model Availability:** MassFlow does not bundle these models. You must train your own or download pre-trained weights from Zenodo or the `matchms` documentation.
3.  **Performance:** ML engines are computationally heavy. If you run them on a large query dataset without a GPU, the annotation pipeline will be significantly slower than the classical `cosine` engine.
