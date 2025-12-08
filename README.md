
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.123.0-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.57.1-FFD21E?logo=huggingface&logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-2.3.3-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.3.4-013243?logo=numpy&logoColor=white)

# SCANN

SCANN is a compact, reproducible workspace for:

- generating sentence embeddings (SentenceTransformers),
- saving embeddings to HDF5,
- building ScaNN approximate nearest-neighbor indexes, and
- benchmarking retrieval quality vs exact search.

Repository layout
- `ScaNN_Embedding_Search_Index.ipynb` — simple demo notebook.
- `ScaNN_Embedding_Search_Index_-_Demo_and_Benchmarking.ipynb` — full demo notebook with benchmarking tools.
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

Open `ScaNN_Embedding_Search_Index_-_Demo_and_Benchmarking.ipynb` for the full pipeline or `ScaNN_Embedding_Search_Index.ipynb` for the demo.

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

filename = 'glove-50-angular.hdf5'
dataset_train = 'train'
dataset_test = 'test'
dataset_truth = 'neighbors'

try:
    with h5py.File(filename, 'r') as f:
        loaded_train_embeddings_raw = f[dataset_train][:]
        loaded_train_embeddings = loaded_train_embeddings_raw / np.linalg.norm(loaded_train_embeddings_raw, axis=1, keepdims=True)
        loaded_test_embeddings_raw = f[dataset_test][:]
        loaded_test_embeddings = loaded_test_embeddings_raw / np.linalg.norm(loaded_test_embeddings_raw, axis=1, keepdims=True)
        loaded_truth_embedding_raw = f[dataset_truth][:]
        loaded_truth_embedding = loaded_truth_embedding_raw / np.linalg.norm(loaded_truth_embedding_raw, axis=1, keepdims=True)

        print("\nEmbeddings loaded successfully.")
        print("Train shape:", loaded_train_embeddings.shape)
        print("Test shape:", loaded_test_embeddings.shape)
        print("Truth shape:", loaded_truth_embedding.shape)

except Exception as e:
    print(f"An error occurred during loading: {e}")
```

Troubleshooting & tips
- Use a virtual environment to avoid permission/packaging issues.
- If you need GPU inference with SentenceTransformers, install a CUDA-enabled PyTorch build.
- ScaNN installation may require a platform-specific wheel; consult ScaNN docs if `pip install scann` fails.


License & credits
- Uses SentenceTransformers, ScaNN, and Hugging Face datasets.

Repository
- https://github.com/nackerboss/SCANN
- `ScaNN_Embedding_Search_Index.ipynb` — quick demo notebook.

- `ScaNN_Embedding_Search_Index_-_Demo_and_Benchmarking.ipynb` — Full demo notebook; contains arbitrary queries, and benchmark tools.

- `agnews_embeddings.h5` — example precomputed, normalized embeddings (HDF5).

# Contributors

<a href="https://github.com/nackerboss/SCANN/graphs/contributors">
  <img src="https://camo.githubusercontent.com/f2aa1fd987013e0c261f0eda02150c9a8e62c89c00059d41825d323a61f4060e/68747470733a2f2f636f6e747269622e726f636b732f696d6167653f7265706f3d6e61636b6572626f73732f5343414e4e4e" data-canonical-src="https://contrib.rocks/image?repo=nackerboss/SCANNN" style="max-width: 100%;">
</a>
