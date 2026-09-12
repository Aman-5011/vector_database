import numpy as np
from typing import List, Tuple

class ExactIndex:
    def __init__(self, dim: int):
        """
        Initializes an exact nearest-neighbor index.
        """
        self.dim = dim
        self.vectors = np.empty((0, dim), dtype=np.float32)
        self.ids = []
        self.id_to_idx = {}
        
        # We use a boolean mask to handle deletions efficiently without 
        # constantly reallocating the entire numpy array in memory.
        self.is_active = np.empty((0,), dtype=bool)

    def add(self, vectors: np.ndarray, ids: List[str]):
        """
        Bulk adds vectors to the index.
        """
        if len(vectors) != len(ids):
            raise ValueError("Vectors and IDs length mismatch.")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {vectors.shape[1]}.")
        
        for uid in ids:
            if uid in self.id_to_idx:
                raise ValueError(f"Duplicate ID found: {uid}. IDs must be unique.")
                
        start_idx = len(self.ids)
        for i, uid in enumerate(ids):
            self.ids.append(uid)
            self.id_to_idx[uid] = start_idx + i
            
        self.vectors = np.vstack([self.vectors, vectors])
        self.is_active = np.concatenate([self.is_active, np.ones(len(ids), dtype=bool)])

    def insert(self, uid: str, vector: np.ndarray):
        """
        Inserts a single vector into the index.
        """
        if uid in self.id_to_idx:
            raise ValueError(f"ID {uid} already exists.")
        if vector.shape != (self.dim,):
            raise ValueError(f"Expected vector shape ({self.dim},), got {vector.shape}.")
            
        self.ids.append(uid)
        self.id_to_idx[uid] = len(self.ids) - 1
        self.vectors = np.vstack([self.vectors, vector.reshape(1, -1)])
        self.is_active = np.append(self.is_active, True)

    def delete(self, uid: str):
        """
        Marks a vector as deleted so it never appears in future search results.
        """
        if uid not in self.id_to_idx:
            raise KeyError(f"ID {uid} not found.")
        
        idx = self.id_to_idx.pop(uid)
        self.is_active[idx] = False

    def search(self, query: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Performs an exhaustive exact nearest-neighbor search.
        Uses Cosine Similarity (Dot product on normalized vectors).
        Returns a list of (id, similarity_score) sorted descending by score.
        """
        if query.shape != (self.dim,):
            raise ValueError(f"Query must have shape ({self.dim},), got {query.shape}.")
        if k <= 0:
            raise ValueError("k must be greater than 0.")
            
        active_count = np.sum(self.is_active)
        if active_count == 0:
            return []
            
        actual_k = min(k, active_count)
        
        # 1. Exhaustive Distance Calculation: Matrix-vector multiplication
        # Because vectors are normalized, dot product == cosine similarity.
        scores = np.dot(self.vectors, query)
        
        # 2. Mask out deleted records by setting their score to negative infinity
        scores[~self.is_active] = -np.inf
        
        # 3. Find top-k efficiently using argpartition
        # argpartition puts the top actual_k elements at the end, but unordered
        partitioned_indices = np.argpartition(scores, -actual_k)[-actual_k:]
        
        # 4. Extract scores of the top-k and sort them descending
        top_scores = scores[partitioned_indices]
        sorted_relative_indices = np.argsort(top_scores)[::-1]
        
        # 5. Map back to original indices
        final_indices = partitioned_indices[sorted_relative_indices]
        
        results = []
        for idx in final_indices:
            results.append((self.ids[idx], float(scores[idx])))
            
        return results