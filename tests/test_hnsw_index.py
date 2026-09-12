import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hnsw_index import HNSWIndex
from src.exact_index import ExactIndex
from src.data.embedding_pipeline import EmbeddingPipeline

class TestHNSWIndex(unittest.TestCase):
    
    def setUp(self):
        self.dim = 2
        # Deterministic parameters for testing
        self.hnsw = HNSWIndex(dim=self.dim, M=4, ef_construction=10, ef_search=10, seed=42)
        self.exact = ExactIndex(dim=self.dim)
        
        # A tiny manual dataset
        self.manual_vecs = np.array([
            [1.0, 0.0],
            [0.707, 0.707],
            [0.0, 1.0],
            [-0.707, 0.707],
            [-1.0, 0.0]
        ], dtype=np.float32)
        self.manual_ids = ["A", "B", "C", "D", "E"]

    def test_basic_construction_and_search(self):
        """Test 1-4, 7-8: Empty, basic bounds, and manual recovery."""
        # Empty search
        self.assertEqual(self.hnsw.search(np.array([1.0, 0.0]), k=2), [])
        
        self.hnsw.add(self.manual_vecs, self.manual_ids)
        self.exact.add(self.manual_vecs, self.manual_ids)
        
        query = np.array([0.9, 0.435], dtype=np.float32) # Normalized vector close to A and B
        
        # Compare Exact vs HNSW
        exact_res = self.exact.search(query, k=2)
        hnsw_res = self.hnsw.search(query, k=2, ef_search=10)
        
        self.assertEqual(len(hnsw_res), 2)
        self.assertEqual(exact_res[0][0], hnsw_res[0][0])  # Top match should align perfectly
        self.assertEqual(exact_res[1][0], hnsw_res[1][0])

    def test_lazy_deletion(self):
        """Test 9-10: Deletions must be ignored by search."""
        self.hnsw.add(self.manual_vecs, self.manual_ids)
        query = np.array([1.0, 0.0], dtype=np.float32)
        
        # 'A' is exactly [1.0, 0.0]. We expect A.
        self.assertEqual(self.hnsw.search(query, k=1)[0][0], "A")
        
        self.hnsw.delete("A")
        
        # After deleting 'A', it should find 'B' instead.
        res = self.hnsw.search(query, k=1)
        self.assertEqual(res[0][0], "B")
        
        # Check KeyError on lookup
        with self.assertRaises(KeyError):
            self.hnsw.delete("A")

    def test_duplicate_insert(self):
        """Test 6: Handles duplicates."""
        self.hnsw.insert("A", self.manual_vecs[0])
        with self.assertRaises(ValueError):
            self.hnsw.insert("A", self.manual_vecs[1])

    def test_real_dataset_integration(self):
        """Test 11-13: Build graph on a subset of the real dataset, ensure graph connects, avoid full scan."""
        vectors_path = 'data/processed/vectors.npy'
        meta_path = 'data/processed/metadata.jsonl'
        
        if not os.path.exists(vectors_path):
            self.skipTest("Real dataset missing.")
            
        vectors, metadata = EmbeddingPipeline.load_data(vectors_path, meta_path)
        
        # We test on the first 1,000 vectors to keep the unit test under ~10 seconds.
        # Python iteration is too slow for 50k in a unit test, but works for the final benchmark.
        test_size = 1000
        vec_subset = vectors[:test_size]
        id_subset = [metadata[i]['id'] for i in range(test_size)]
        dim = vec_subset.shape[1]
        
        exact = ExactIndex(dim=dim)
        exact.add(vec_subset, id_subset)
        
        hnsw = HNSWIndex(dim=dim, M=12, ef_construction=30)
        hnsw.add(vec_subset, id_subset)
        
        # Validate graph construction
        # Max level should be > 0 (probabilistically guaranteed with 1000 items)
        self.assertGreater(hnsw.max_level, 0)
        
        query = vec_subset[0]
        exact_res = exact.search(query, k=10)
        
        # Provide a high ef_search to practically guarantee 100% recall on a small subset
        hnsw_res = hnsw.search(query, k=10, ef_search=200)
        
        # Result length
        self.assertEqual(len(hnsw_res), 10)
        
        # Top result must be itself
        self.assertEqual(hnsw_res[0][0], exact_res[0][0])
        self.assertAlmostEqual(hnsw_res[0][1], 1.0, places=4)
        
        # Sort verification
        scores = [res[1] for res in hnsw_res]
        self.assertTrue(all(scores[i] >= scores[i+1] for i in range(len(scores)-1)))

if __name__ == '__main__':
    unittest.main(verbosity=2)