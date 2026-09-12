# Vector Database from Scratch

## What we are building
A pure-Python vector database and nearest-neighbor search engine. It will eventually feature both an exact brute-force search index and a custom approximate nearest neighbor (ANN) index based on HNSW logic, complete with measurements and benchmark plots for speed-vs-accuracy trade-offs.

## Why no existing libraries?
The project strictly prohibits standard vector database solutions (Pinecone, Chroma) and search algorithms (FAISS, `sklearn.neighbors`). Relying on off-the-shelf imports obscures the mathematical and algorithmic foundation of modern AI retrieval. Building it entirely with NumPy ensures a deep, foundational understanding of high-dimensional spatial indexing, metric spaces, and performance trade-offs. 

## What this stage does
This is the foundational stage. It programmatically generates a deterministic, semantically rich dataset of 50,000 texts, utilizes a lightweight local embedding model to map these texts into vector space, normalizes them, and persists the raw arrays and metadata to disk.

## Setup Instructions
1. **Create the virtual environment**: `python -m venv venv`
2. **Activate the environment**: 
   - Linux/Mac: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
3. **Install dependencies**: `pip install -r requirements.txt`

## Execution
* **Generate Dataset and Embeddings:** `python scripts/prepare_data.py`
* **Validate the Generated Data:** `python scripts/validate_data.py`

*Generated files are stored locally in the `data/processed/` directory.*

