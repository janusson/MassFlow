# Post-v1.0 Development Roadmap

With the release of MassFlow v1.0, the core processing and annotation pipelines are stabilized. Future development will focus on scaling, deployment, and advanced machine learning integrations.

## 1. Satellite Repository Integration (`massflow-ml`)
To maintain the lightweight nature of the core `MassFlow` package, heavy machine learning dependencies (e.g., TensorFlow, PyTorch, Gensim) will be isolated.
*   Establish a dedicated `massflow-ml` repository.
*   Migrate advanced models like `Spec2Vec` and `MS2DeepScore` to this satellite repository.
*   Define a stable API boundary so the core orchestrator can seamlessly route queries to external ML engines when they are installed.

## 2. Advanced Storage Architecture
The current SQLite-based storage is highly reliable but faces performance bottlenecks when scaling to tens of millions of spectra.
*   Transition spectral array storage from SQLite BLOBs to chunked, cloud-native formats like **Zarr** or **N5**.
*   This will enable highly parallelized, distributed similarity searches directly against cloud storage without requiring full library downloads.

## 3. Real-Time Instrumentation & Networking
MassFlow is currently designed as a post-acquisition data analysis tool.
*   Deploy **gRPC-based streaming APIs**.
*   This will allow mass spectrometers to stream spectra directly to MassFlow during acquisition, enabling real-time structural annotation and instrument feedback (e.g., dynamic exclusion or targeted acquisition triggers).

## 4. Algorithmic Expansion
Future updates will explore cutting-edge computational chemistry techniques to expand beyond direct spectral matching.
*   **Generative Spectral Augmentation:** Using generative models to simulate MS2 spectra for novel compounds missing from standard libraries, expanding the search space.
*   **Differentiable Physics-Informed Neural Networks (PINNs):** Integrating known physical fragmentation rules into deep learning architectures to improve the accuracy and explainability of structural elucidation.
