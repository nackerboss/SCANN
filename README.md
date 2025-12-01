
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.123.0-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.57.1-FFD21E?logo=huggingface&logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-2.3.3-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.3.4-013243?logo=numpy&logoColor=white)

# SCANNN

SCANNN is a compact, reproducible workspace for:

- generating sentence embeddings (SentenceTransformers),
- saving embeddings to HDF5,
- building ScaNN approximate nearest-neighbor indexes, and
- benchmarking retrieval quality vs exact search.

Repository layout
- `REAL_MAIN_ScaNN_Embedding_Search_Index.ipynb` — primary end-to-end notebook (embed → save → build → evaluate → query).
- `ScaNN_Embedding_Search_Index.ipynb` — demo notebook with interactive widgets and shorter examples.
- `agnews_embeddings.h5` — example precomputed, normalized embeddings (HDF5).
- `scann-env/` — optional reference Python virtual environment included in the workspace.
- `frontend/` — optional React + Vite demo for interactive search.

Quick start

1. Create and activate a virtual environment

   - Windows (cmd.exe):
     ```cmd
     python -m venv .venv
     .\.venv\Scripts\activate
     ```

   - macOS / Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run Jupyter and open the notebooks

```bash
jupyter lab
```

Open `REAL_MAIN_ScaNN_Embedding_Search_Index.ipynb` for the full pipeline or `ScaNN_Embedding_Search_Index.ipynb` for the demo.

Typical workflow
- Load a text dataset (e.g., HF `ag_news`) or your own corpus.
- Generate embeddings with a SentenceTransformers model (e.g., `all-MiniLM-L6-v2`).
- L2-normalize embeddings (recommended for dot-product queries with ScaNN).
- Save embeddings to HDF5 with descriptive attributes.
- Build a ScaNN index (tree + AH quantization + reorder) and run approximate queries.
- Optionally compute exact nearest neighbors to measure recall@k.

Quick example — load embeddings

```python
import h5py
with h5py.File('agnews_embeddings.h5', 'r') as f:
    embeddings = f['agnews'][:]
    print('shape', embeddings.shape)
    print('description:', f['agnews'].attrs.get('description'))
```

Troubleshooting & tips
- Use a virtual environment to avoid permission/packaging issues.
- If you need GPU inference with SentenceTransformers, install a CUDA-enabled PyTorch build.
- ScaNN installation may require a platform-specific wheel; consult ScaNN docs if `pip install scann` fails.

Next steps (optional)
- Add a `requirements.txt` (already added) or `pyproject.toml` for reproducible installs.
- Add a small runnable script that reproduces the notebook pipeline end-to-end.

License & credits
- Uses SentenceTransformers, ScaNN, and Hugging Face datasets — see the corresponding projects for license details.

Repository
- https://github.com/nackerboss/SCANNN
- `REAL_MAIN_ScaNN_Embedding_Search_Index.ipynb` — primary end-to-end notebook (embed → save → build → evaluate → query).

- `ScaNN_Embedding_Search_Index.ipynb` — demo notebook with interactive widgets and shorter examples.

- `agnews_embeddings.h5` — example precomputed, normalized embeddings (HDF5).

# Contributors

<a href="https://github.com/nackerboss/SCANNN/graphs/contributors">
  <img src="https://camo.githubusercontent.com/f2aa1fd987013e0c261f0eda02150c9a8e62c89c00059d41825d323a61f4060e/68747470733a2f2f636f6e747269622e726f636b732f696d6167653f7265706f3d6e61636b6572626f73732f5343414e4e4e" data-canonical-src="https://contrib.rocks/image?repo=nackerboss/SCANNN" style="max-width: 100%;">
</a>
