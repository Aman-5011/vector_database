import math
import heapq
import random
import numpy as np
from typing import List, Tuple, Dict, Set

class HNSWIndex:
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 100, ef_search: int = 50, seed: int = 42):
        """
        Initializes an HNSW-style approximate nearest-neighbor index.
        """
        self.dim = dim
        self.M = M
        self.M0 = 2 * M  # Max connections for layer 0
        self.ef_construction = max(ef_construction, M)
        self.ef_search = ef_search
        
        self.level_mult = 1.0 / math.log(self.M)
        self.rng = random.Random(seed)
        
        # Core data structures
        self.vectors: List[np.ndarray] = []
        self.ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
        self.deleted: Set[int] = set()
        
        # Graph: graph[node_idx][layer] = [neighbor_1, neighbor_2, ...]
        self.graph: List[List[List[int]]] = []
        
        self.max_level = -1
        self.entry_point = None

    def _random_level(self) -> int:
        """Determines the maximum layer for a new node."""
        return int(math.floor(-math.log(self.rng.uniform(1e-10, 1.0)) * self.level_mult))

    def _similarity(self, query: np.ndarray, idx: int) -> float:
        """Calculates cosine similarity (dot product on normalized vectors)."""
        return float(np.dot(query, self.vectors[idx]))

    def _similarity_nodes(self, idx1: int, idx2: int) -> float:
        """Calculates cosine similarity between two existing nodes."""
        return float(np.dot(self.vectors[idx1], self.vectors[idx2]))

    def _greedy_search(self, query: np.ndarray, entry_point: int, level: int) -> int:
        """Finds the single closest node at a given layer."""
        curr_obj = entry_point
        curr_sim = self._similarity(query, curr_obj)
        
        changed = True
        while changed:
            changed = False
            for neighbor in self.graph[curr_obj][level]:
                neighbor_sim = self._similarity(query, neighbor)
                if neighbor_sim > curr_sim:
                    curr_sim = neighbor_sim
                    curr_obj = neighbor
                    changed = True
        return curr_obj

    def _search_layer(self, query: np.ndarray, entry_points: List[int], ef: int, level: int) -> List[Tuple[float, int]]:
        """
        Explores the graph layer to find the 'ef' closest candidates.
        Maintains a candidate queue (C) and a top-results queue (W).
        """
        visited = set(entry_points)
        
        # C is a max-heap (implemented via negative similarities)
        C = []
        # W is a min-heap to maintain the top `ef` closest nodes
        W = []
        
        for ep in entry_points:
            sim = self._similarity(query, ep)
            heapq.heappush(C, (-sim, ep))
            heapq.heappush(W, (sim, ep))
            
        while C:
            neg_c_sim, c = heapq.heappop(C)
            c_sim = -neg_c_sim
            
            furthest_sim_in_W = W[0][0]
            
            if c_sim < furthest_sim_in_W:
                break
                
            for neighbor in self.graph[c][level]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    neighbor_sim = self._similarity(query, neighbor)
                    
                    furthest_sim_in_W = W[0][0]
                    if neighbor_sim > furthest_sim_in_W or len(W) < ef:
                        heapq.heappush(C, (-neighbor_sim, neighbor))
                        heapq.heappush(W, (neighbor_sim, neighbor))
                        if len(W) > ef:
                            heapq.heappop(W)  # Drop the worst
                            
        return W

    def _select_neighbors(self, base_node_idx: int, candidates: List[Tuple[float, int]], M: int) -> List[int]:
        """
        Principled HNSW neighbor-selection heuristic.
        Enforces spatial diversity by checking if a candidate is closer to the 
        base node than it is to any already-selected neighbor.
        """
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = []
        
        for sim_to_base, c_idx in candidates:
            if len(selected) >= M:
                break
                
            is_diverse = True
            for s_idx in selected:
                sim_c_s = self._similarity_nodes(c_idx, s_idx)
                if sim_c_s > sim_to_base:
                    is_diverse = False
                    break
                    
            if is_diverse:
                selected.append(c_idx)
                
        # Fallback to ensure graph connectivity if we pruned too heavily
        if len(selected) < M:
            for sim_to_base, c_idx in candidates:
                if len(selected) >= M:
                    break
                if c_idx not in selected:
                    selected.append(c_idx)
                    
        return selected

    def insert(self, uid: str, vector: np.ndarray):
        """Inserts a single vector into the HNSW graph."""
        if uid in self.id_to_idx:
            raise ValueError(f"ID {uid} already exists.")
        if vector.shape != (self.dim,):
            raise ValueError(f"Expected shape ({self.dim},), got {vector.shape}")

        idx = len(self.vectors)
        self.vectors.append(vector)
        self.ids.append(uid)
        self.id_to_idx[uid] = idx
        
        l = self._random_level()
        self.graph.append([[] for _ in range(l + 1)])
        
        if self.entry_point is None:
            self.entry_point = idx
            self.max_level = l
            return
            
        curr_obj = self.entry_point
        
        for lc in range(self.max_level, l, -1):
            curr_obj = self._greedy_search(vector, curr_obj, lc)
            
        entry_points = [curr_obj]
        
        for lc in range(min(self.max_level, l), -1, -1):
            candidates = self._search_layer(vector, entry_points, self.ef_construction, lc)
            
            neighbors = self._select_neighbors(idx, candidates, self.M if lc > 0 else self.M0)
            
            for neighbor in neighbors:
                self.graph[idx][lc].append(neighbor)
                self.graph[neighbor][lc].append(idx)
                
                M_max = self.M0 if lc == 0 else self.M
                if len(self.graph[neighbor][lc]) > M_max:
                    conn_sims = [(self._similarity_nodes(neighbor, n), n) for n in self.graph[neighbor][lc]]
                    self.graph[neighbor][lc] = self._select_neighbors(neighbor, conn_sims, M_max)
            
            entry_points = [idx for sim, idx in candidates]
            
        if l > self.max_level:
            self.max_level = l
            self.entry_point = idx

    def add(self, vectors: np.ndarray, ids: List[str]):
        """Helper to bulk insert vectors sequentially."""
        if len(vectors) != len(ids):
            raise ValueError("Vectors and IDs mismatch.")
        for uid, vec in zip(ids, vectors):
            self.insert(uid, vec)

    def delete(self, uid: str):
        """
        Lazy deletion: marks node as deleted without physically severing graph edges.
        """
        if uid not in self.id_to_idx:
            raise KeyError(f"ID {uid} not found.")
        idx = self.id_to_idx[uid]
        self.deleted.add(idx)
        del self.id_to_idx[uid]

    def search(self, query: np.ndarray, k: int = 5, ef_search: int = None) -> List[Tuple[str, float]]:
        """
        Approximate search navigating the HNSW graph.
        """
        if query.shape != (self.dim,):
            raise ValueError("Dimension mismatch.")
        if k <= 0:
            raise ValueError("k must be > 0.")
            
        if self.entry_point is None:
            return []
            
        ef = ef_search if ef_search is not None else self.ef_search
        ef = max(ef, k)
        
        curr_obj = self.entry_point
        
        for lc in range(self.max_level, 0, -1):
            curr_obj = self._greedy_search(query, curr_obj, lc)
            
        candidates = self._search_layer(query, [curr_obj], ef, 0)
        
        results = []
        for sim, idx in candidates:
            if idx not in self.deleted:
                results.append((sim, idx))
                
        results.sort(key=lambda x: x[0], reverse=True)
        return [(self.ids[idx], float(sim)) for sim, idx in results[:k]]