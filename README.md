
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.123.0-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.57.1-FFD21E?logo=huggingface&logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-2.3.3-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.3.4-013243?logo=numpy&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-1.4.1-FFF000?logo=duckdb&logoColor=black)
![Scikit-Learn](https://img.shields.io/badge/sklearn-1.7.2-F7931E?logo=scikit-learn&logoColor=white)

# SCANNN

A compact repo for generating text embeddings (SentenceTransformers), saving them to HDF5, building high-performance ScaNN nearest-neighbor indexes, running similarity queries, and benchmarking recall vs. brute-force.

## Contents
- Notebooks
  - REAL_MAIN_ScaNN_Embedding_Search_Index.ipynb — end-to-end pipeline: embed, normalize, save, build ScaNN, evaluate, query.
  - ScaNN_Embedding_Search_Index.ipynb — alternate/demo notebook with interactive progress widgets and examples.
- Data
  - agnews_embeddings.h5 — precomputed normalized embeddings (HDF5).
- Environment
  - scann-env/ — optional Python virtual environment included in the workspace.
- Utilities / Scripts
  - runwsl.bat — helper script for Windows Subsystem for Linux usage.
- Metadata
  - .gitignore

## Key features
- Generate embeddings with SentenceTransformers (default model: `all-MiniLM-L6-v2`).
- L2-normalize embeddings (required for dot-product / angular similarity with ScaNN).
- Save/load embeddings in HDF5 with metadata (dimension, description).
- Build ScaNN index (tree + asymmetric hashing + reorder) for fast approximate nearest neighbors.
- Optionally compute exact neighbors with ScaNN brute-force for recall benchmarking.
- Demo queries and interactive outputs for inspecting results.

## Quick start (recommended)
1. Create or activate a Python venv (or use the provided scann-env):
   - python3 -m venv .venv && source .venv/bin/activate
2. Install required packages:
```bash
# language: bash
pip install scann sentence-transformers datasets h5py numpy
```
3. Open either notebook in Colab / Jupyter:
   - [REAL_MAIN_ScaNN_Embedding_Search_Index.ipynb](REAL_MAIN_ScaNN_Embedding_Search_Index.ipynb) — full workflow
   - [ScaNN_Embedding_Search_Index.ipynb](ScaNN_Embedding_Search_Index.ipynb) — demo / widgets

## Typical workflow summary
1. Load text dataset (e.g., Hugging Face `ag_news`) or your own dataset.
2. Generate embeddings with SentenceTransformers:
   - call `generate_and_normalize(data)` to obtain L2-normalized embeddings.
3. Save embeddings to HDF5 (e.g., `agnews_embeddings.h5`) with attributes for dimension/description.
4. Build a ScaNN index with recommended parameters (example uses a tree builder, AH quantization, and a reorder step).
5. Query the index and optionally compute recall vs. ScaNN brute-force searcher.

Example builder snippet (see notebook for full code and parameters):
```python
# language: python
builder = scann.scann_ops_pybind.builder(
    normalized_dataset_embeddings,
    K_NEIGHBORS,
    "dot_product"
)

tree_configured = builder.tree(
    num_leaves=num_leaves,
    num_leaves_to_search=num_leaves_to_search,
    training_sample_size=training_sample_size
)

ah_configured = tree_configured.score_ah(
    8,  # dimensions per subvector
    anisotropic_quantization_threshold=0.2
)

reorder_configured = ah_configured.reorder(REORDER_NEIGHBORS)
searcher = reorder_configured.build()
```

## How to load precomputed embeddings
```python
# language: python
import h5py
with h5py.File('agnews_embeddings.h5','r') as f:
    emb = f['agnews'][:]
    print(emb.shape)
    print(f['agnews'].attrs.get('description'))
```

## Notable variables / functions (see notebooks)
- [`generate_and_normalize`](REAL_MAIN_ScaNN_Embedding_Search_Index.ipynb) — embeds and L2-normalizes inputs.
- [`compute_recall`](REAL_MAIN_ScaNN_Embedding_Search_Index.ipynb) — utility to compute recall@k vs brute-force.
- [`run_query`](ScaNN_Embedding_Search_Index.ipynb) — helper to run sample queries and print results.
- `K_NEIGHBORS`, `REORDER_NEIGHBORS` — index/query parameter constants used when building the ScaNN index.
- ScaNN builder usage via `scann.scann_ops_pybind.builder` (see notebooks).

## Troubleshooting / notes
- Installing scann in some environments may require specific wheels or OS packages. If you encounter "externally-managed-environment" pip errors, use a venv or conda env.
- GPU is used by SentenceTransformers only if a CUDA-enabled PyTorch is available; otherwise CPU fallback is used.
- The included `scann-env/` virtual environment can be used as reference; creating a fresh venv is recommended.

## License & credits
- This repo uses SentenceTransformers and ScaNN (Google). Check respective licenses.
- The notebooks include examples using the Hugging Face `ag_news` dataset for benchmarking.

## Contact / next steps
- Open the notebooks to run the pipeline end-to-end:
  - [REAL_MAIN_ScaNN_Embedding_Search_Index.ipynb](REAL_MAIN_ScaNN_Embedding_Search_Index.ipynb)
  - [ScaNN_Embedding_Search_Index.ipynb](ScaNN_Embedding_Search_Index.ipynb)

### Top contributors:

<a href="https://github.com/nackerboss/SCANNN/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nackerboss/SCANNN" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



