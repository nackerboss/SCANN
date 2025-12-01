# -*- coding: utf-8 -*-
"""ScaNN Server for Local Use - Assignment Submission"""

# --- 1. INSTALL DEPENDENCIES ---
# Recommended installation command:
# pip install fastapi uvicorn scann sentence-transformers datasets nest_asyncio 'h5py>=3.0.0'

import os
import time
import numpy as np
import uvicorn
# import nest_asyncio # Not needed for standard local execution
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import scann
from datasets import load_dataset
# from pyngrok import ngrok # REMOVED
import torch
from contextlib import asynccontextmanager
import h5py

# Apply nest_asyncio immediately to allow nested event loops in Colab
# nest_asyncio.apply() # REMOVED for local execution

# --- 2. CONFIGURATION ---
DATASET_SIZE = 120000  # @param {type:"integer"}
MODEL_NAME = 'all-MiniLM-L6-v2'
K_NEIGHBORS = 10

# Global State
embedding_model = None
searcher = None
dataset_texts = []
normalized_dataset_embeddings = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, searcher, dataset_texts, normalized_dataset_embeddings
    print("--- SERVER STARTUP ---")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"1. Init Model ({device})...")
    embedding_model = SentenceTransformer(MODEL_NAME, device=device)

    print(f"2. Load Data ({DATASET_SIZE} items from ag_news)...")
    try:
        # Load the dataset
        dataset = load_dataset('ag_news', split=f'train[:{DATASET_SIZE}]')
        dataset_texts = dataset['text']

        filename = 'agnews_embeddings.h5'
        dataset_name = 'agnews'

        # Load Embeddings (Assumes 'agnews_embeddings.h5' exists locally)
        # NOTE: If this file does not exist, you'll need to run the embedding
        # generation part (currently commented out) once to create it.
        try:
            with h5py.File(filename, 'r') as f:
                normalized_dataset_embeddings = f[dataset_name][:]
                print("\nEmbeddings loaded successfully.")
                print("Shape:", normalized_dataset_embeddings.shape)
                print("Metadata description:", f[dataset_name].attrs['description'])

        except Exception as e:
            print(f"An error occurred during loading HDF5 file: {e}")
            # You might want to halt or generate the embeddings here if needed
            normalized_dataset_embeddings = None # Ensure searcher isn't built if load fails

        if normalized_dataset_embeddings is not None:
            print("4. Build ScaNN Index (Partitioning -> Scoring -> Reordering)...")
            # Theory: Partitioning (Tree)
            num_leaves = int(np.sqrt(DATASET_SIZE))

            builder = scann.scann_ops_pybind.builder(
                normalized_dataset_embeddings, K_NEIGHBORS, "dot_product"
            ).tree(
                num_leaves=num_leaves,
                num_leaves_to_search=max(int(num_leaves*0.1), 10),
                training_sample_size=min(int(DATASET_SIZE*0.1), 10000)
            ).score_ah(
                2, anisotropic_quantization_threshold=0.2 # Theory: Anisotropic Hashing
            ).reorder(
                K_NEIGHBORS * 5 # Theory: Reordering
            )

            searcher = builder.build()
            print("--- READY ---")
        else:
            print("--- NOT READY: EMBEDDINGS FAILED TO LOAD ---")

    except Exception as e:
        print(f"Error during startup: {e}")

    yield
    print("--- SERVER SHUTDOWN ---")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchQuery(BaseModel):
    query_text: str
    k_neighbors: int = 5

class BenchmarkRequest(BaseModel):
    num_queries: int = 100

@app.get("/health")
def health_check():
    return {"status": "ok", "ready": searcher is not None}

@app.post("/search")
def search(query: SearchQuery):
    if not searcher: raise HTTPException(503, "Not ready. Model might still be loading or embeddings are missing.")
    start = time.time()

    # 1. Embed Query
    q_emb = embedding_model.encode([query.query_text])[0]
    q_emb = q_emb / np.linalg.norm(q_emb)

    # 2. Search
    # Ensure we don't ask for more neighbors than ScaNN was configured for (K_NEIGHBORS)
    final_k = min(query.k_neighbors, K_NEIGHBORS)
    indices, distances = searcher.search(q_emb, final_num_neighbors=final_k)

    elapsed = (time.time()-start)*1000

    # Cast idx to int to avoid numpy.uint32 TypeError
    results = [{"rank": i+1, "text": dataset_texts[int(idx)], "similarity": float(d), "dataset_index": int(idx)}
               for i, (idx, d) in enumerate(zip(indices, distances))]

    return {"query": query.query_text, "results": results, "search_time_ms": elapsed}

@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    if not searcher: raise HTTPException(503, "Not ready. Embeddings are missing.")
    print(f"--- STARTING BENCHMARK ({req.num_queries} queries) ---")

    try:
        # Load test set
        test_texts = load_dataset('ag_news', split=f'test[:{req.num_queries}]')['text']
        test_emb = embedding_model.encode(test_texts, convert_to_tensor=False)
        test_emb = test_emb / np.linalg.norm(test_emb, axis=1, keepdims=True)

        # 1. Brute Force (Ground Truth)
        print("Running Brute Force...")
        bf_start = time.time()
        # Create a raw brute force searcher for comparison
        bf_searcher = scann.scann_ops_pybind.builder(normalized_dataset_embeddings, K_NEIGHBORS, "dot_product").score_brute_force().build()
        bf_idx, _ = bf_searcher.search_batched(test_emb)
        bf_time = time.time() - bf_start

        # 2. ScaNN (Approximate)
        print("Running ScaNN...")
        scann_start = time.time()
        scann_idx, _ = searcher.search_batched(test_emb)
        scann_time = time.time() - scann_start

        # 3. Calculate Recall
        recall_sum = 0
        k = 5
        for i in range(len(test_texts)):
            # Intersection of indices
            recall_sum += len(set(bf_idx[i][:k]).intersection(set(scann_idx[i][:k]))) / k

        avg_recall = recall_sum/len(test_texts)

        print(f"Benchmark Complete. Recall: {avg_recall:.4f}")

        return {
            "dataset_size": DATASET_SIZE,
            "num_queries": req.num_queries,
            "results": [
                {
                    "method": "Brute Force",
                    "time_seconds": bf_time,
                    "avg_ms_per_query": (bf_time / req.num_queries) * 1000,
                    "recall": 1.0
                },
                {
                    "method": "ScaNN",
                    "time_seconds": scann_time,
                    "avg_ms_per_query": (scann_time / req.num_queries) * 1000,
                    "recall": avg_recall
                }
            ]
        }
    except Exception as e:
        print(f"Benchmark Error: {e}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    # Removed ngrok setup.
    # This runs the FastAPI server locally on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)