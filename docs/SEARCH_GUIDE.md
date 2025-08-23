# 🔍 Search & Recommendation Guide ([Back to README](../README.md))

## 1. Overview

This guide covers the Search & Recommendation capability: TF‑IDF-based text retrieval with an inverted index, plus CTR-based re-ranking workflow in the UI (CTR training is done in the "Data Collection & Training" tab).

## 2. Quick Start

1) After starting the system, go to "🔍 Online Retrieval & Ranking".
- Enter a query (e.g., Artificial Intelligence, Machine Learning, Deep Learning).
- Choose ranking mode: TF‑IDF or CTR (CTR requires a trained model from the "📊 Data Collection & Training" tab).
- Click "🔬 Execute Search".

2) Inspect results and record interactions.
- The table shows Document ID, TF‑IDF score, extra info (CTR score or document length), and summary.
- Click a row to view the full text; a click sample will be recorded for CTR training.

3) Train the CTR model (optional).
- Go to "📊 Data Collection & Training" to review samples and stats.
- Click "Train CTR Model"; then return to the search tab to switch to CTR ranking for comparison.

Note: If present, preloaded documents live at `data/preloaded_documents.json` and are auto‑loaded at startup. The UI does not support manual add/remove of documents.

## 3. Technical Notes

- Inverted index & retrieval: `src/search_engine/index_tab/offline_index.py` and `InvertedIndexService`.
- Online search entry: `src/search_engine/search_tab/search_tab.py` calling `IndexService.search/retrieve/rank`.
- CTR data & training: `src/search_engine/data_service.py` and `src/search_engine/training_tab/`.

## 4. FAQ

- No results: Ensure the index has preloaded documents; use queries aligned with the dataset topics.
- CTR score is empty: Collect samples and train a model in the training tab, then switch to CTR ranking.
- Port conflict: Use `quick_start.sh` or change the port and restart.

