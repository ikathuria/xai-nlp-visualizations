# XAI NLP Visualizations

## Project Overview
This repository accompanies a survey paper on **data visualization techniques for Explainable AI (XAI) in Natural Language Processing (NLP)**. The survey reviews state-of-the-art visualization methods used to interpret transformer-based NLP models, including attention heatmaps, embedding projections, and token-level importance scoring.

In addition to the survey, this repository provides **mini-demo implementations** that illustrate how these visualization techniques can be applied to real NLP examples.

---

## Survey Paper
- **PDF:** [`paper/survey_xai_nlp.pdf`](paper/survey_xai_nlp.pdf)
- **Summary:**  
  The paper covers:
  - Token-level explanations using attention and SHAP values  
  - Embedding projection (UMAP) for semantic visualization  
  - Trends, challenges, and future directions in XAI visualizations for NLP

---

## Demo Implementations

### 1. Token-Level Importance Heatmap
- **Notebook:** [`notebooks/token_heatmap.ipynb`](notebooks/token_heatmap.ipynb)  
- **Description:** Visualizes the contribution of individual tokens in a sentence to the model’s prediction using attention weights.  
- **Figure example:** `figures/sentence1_heatmap.png`

### 2. Embedding Projection (UMAP)
- **Notebook:** [`notebooks/embedding_umap.ipynb`](notebooks/embedding_umap.ipynb)  
- **Description:** Projects last-layer token embeddings into 2D space using UMAP, with token importance color-coded.  
- **Figure example:** `figures/sentence1_umap.png`

---
