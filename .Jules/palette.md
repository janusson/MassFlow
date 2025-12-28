## 2025-12-28 - [Added TQDM Progress Bar]
**Learning:** CLI tools often lack feedback for long operations. Adding a simple progress bar (tqdm) dramatically improves the user experience by providing visibility into processing status.
**Action:** When working on CLI tools with iteration over large datasets, always consider wrapping the main loop with tqdm.
