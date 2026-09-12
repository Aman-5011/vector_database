import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

class EmbeddingPipeline:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 256):
        """
        Initializes the pipeline with a local embedding model.
        all-MiniLM-L6-v2 is used for fast, high-quality 384-dimensional embeddings.
        """
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Converts text into normalized dense vectors.
        Normalization forces all vectors to have a length of 1, 
        meaning dot product directly equals cosine similarity.
        """
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True 
        )
        return embeddings

    def save_data(self, vectors: np.ndarray, metadata: List[Dict], vectors_path: str, meta_path: str):
        """
        Saves vectors to a fast NumPy binary format (.npy) and metadata to JSONL.
        """
        # Save vectors
        np.save(vectors_path, vectors)
        
        # Save metadata
        with open(meta_path, 'w', encoding='utf-8') as f:
            for record in metadata:
                f.write(json.dumps(record) + '\n')
                
        print(f"Successfully saved {len(vectors)} vectors to {vectors_path}")
        print(f"Successfully saved {len(metadata)} metadata records to {meta_path}")

    @staticmethod
    def load_data(vectors_path: str, meta_path: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Utility to reload data for future indexing/search scripts.
        """
        vectors = np.load(vectors_path)
        metadata = []
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                metadata.append(json.loads(line))
        return vectors, metadata