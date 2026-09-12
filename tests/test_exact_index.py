import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.exact_index import ExactIndex
from src.data.embedding_pipeline import EmbeddingPipeline

class TestExactIndex(unittest.TestCase):
    
    def setUp(self):
        self.dim = 2
        self.index = ExactIndex(dim=self.dim)
        
        # Normalized manual dataset
        # v1: [1, 0]
        # v2: [0, 1]
        # v3: [-1, 0]
        self.manual_vecs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0]
        ], dtype=np.float32)
        self.manual_ids = ["v1", "v2", "v3"]

    def test_manual_dataset_exact_math(self):
        """Test 1: Core correctness on a tiny mathematical dataset."""
        self.index.add(self.manual_vecs, self.manual_ids)
        
        query = np.array([1.0, 0.0], dtype=np.float32)
        results = self.index.search(query, k=3)
        
        # Cosine similarity expectations:
        # dot([1,0], [1,0]) = 1.0
        # dot([1,0], [0,1]) = 0.0
        # dot([1,0], [-1,0]) = -1.0
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], "v1")
        self.assertAlmostEqual(results[0][1], 1.0)
        
        self.assertEqual(results[1][0], "v2")
        self.assertAlmostEqual(results[1][1], 0.0)
        
        self.assertEqual(results[2][0], "v3")
        self.assertAlmostEqual(results[2][1], -1.0)

    def test_k_behavior_and_empty_index(self):
        """Test 2, 3, 4, 5: Edge cases on search bounds."""
        # Empty index
        self.assertEqual(self.index.search(np.array([1.0, 0.0]), k=5), [])
        
        self.index.add(self.manual_vecs, self.manual_ids)
        
        # Invalid k
        with self.assertRaises(ValueError):
            self.index.search(np.array([1.0, 0.0]), k=0)
            
        # k larger than active vectors
        results = self.index.search(np.array([1.0, 0.0]), k=10)
        self.assertEqual(len(results), 3)
        
        # Query dimension mismatch
        with self.assertRaises(ValueError):
            self.index.search(np.array([1.0, 0.0, 0.0]), k=2)

    def test_insert_and_delete(self):
        """Test 6, 7, 8: Insert, delete, and duplicate ID checks."""
        # Calculate a mathematically perfect unit vector
        val = np.float32(1.0 / np.sqrt(2.0))
        test_vector = np.array([val, val], dtype=np.float32)
        
        # Insert
        self.index.insert("v4", test_vector)
        res1 = self.index.search(test_vector, k=1)
        self.assertEqual(res1[0][0], "v4")
        self.assertAlmostEqual(res1[0][1], 1.0, places=5)
        
        # Duplicate ID
        with self.assertRaises(ValueError):
            self.index.insert("v4", np.array([1.0, 0.0]))
            
        # Delete
        self.index.delete("v4")
        res2 = self.index.search(test_vector, k=1)
        self.assertEqual(len(res2), 0)  # Empty because v4 was the only vector

    def test_real_dataset_integration(self):
        """Test 9: Larger correctness test on real 50k dataset."""
        vectors_path = 'data/processed/vectors.npy'
        meta_path = 'data/processed/metadata.jsonl'
        
        if not os.path.exists(vectors_path):
            self.skipTest("Real dataset not generated yet. Skipping integration test.")
            
        vectors, metadata = EmbeddingPipeline.load_data(vectors_path, meta_path)
        ids = [record['id'] for record in metadata]
        
        # Initialize index with correct dimension (should be 384)
        dim = vectors.shape[1]
        real_index = ExactIndex(dim=dim)
        real_index.add(vectors, ids)
        
        # Use the first vector as our query
        query = vectors[0]
        query_id = ids[0]
        
        results = real_index.search(query, k=10)
        
        # The top result MUST be the vector itself with score ~1.0
        self.assertEqual(len(results), 10)
        self.assertEqual(results[0][0], query_id)
        self.assertAlmostEqual(results[0][1], 1.0, places=4)
        
        # Scores must be perfectly ordered descending
        scores = [res[1] for res in results]
        self.assertTrue(all(scores[i] >= scores[i+1] for i in range(len(scores)-1)))
        
        # Test deletion over real dataset
        real_index.delete(query_id)
        results_after_delete = real_index.search(query, k=10)
        returned_ids = [res[0] for res in results_after_delete]
        
        self.assertNotIn(query_id, returned_ids)

if __name__ == '__main__':
    unittest.main(verbosity=2)