import unittest
from src.evaluation import calculate_recall_at_k, calculate_qps

class TestEvaluationMetrics(unittest.TestCase):

    def test_recall_perfect_match(self):
        """Test 2: Recall = 1.0 when exact and approx are identical."""
        exact = ["A", "B", "C", "D", "E"]
        approx = ["A", "B", "C", "D", "E"]
        recall = calculate_recall_at_k(exact, approx, k=5)
        self.assertEqual(recall, 1.0)

    def test_recall_no_overlap(self):
        """Test 3: Recall = 0.0 when there is no overlap."""
        exact = ["A", "B", "C"]
        approx = ["X", "Y", "Z"]
        recall = calculate_recall_at_k(exact, approx, k=3)
        self.assertEqual(recall, 0.0)

    def test_recall_partial_overlap(self):
        """Test 1 & 4: Manual math calculation for partial overlaps."""
        exact = ["A", "B", "C", "D", "E"]
        # Approx misses D and E, replaces with X and Y. Order doesn't strictly matter for set intersection.
        approx = ["A", "X", "B", "Y", "C"] 
        
        # Intersection is A, B, C (3 items) out of K=5
        recall = calculate_recall_at_k(exact, approx, k=5)
        self.assertAlmostEqual(recall, 0.6)

    def test_recall_respects_k_limit(self):
        """Even if arrays are long, only the top K should be evaluated."""
        exact = ["A", "B", "C", "D", "E"]
        approx = ["A", "B", "X", "Y", "Z"]
        
        # If K=2, exact=A,B and approx=A,B -> Recall 1.0
        recall_2 = calculate_recall_at_k(exact, approx, k=2)
        self.assertEqual(recall_2, 1.0)
        
        # If K=5, exact=A,B,C,D,E and approx=A,B,X,Y,Z -> Recall 0.4 (2/5)
        recall_5 = calculate_recall_at_k(exact, approx, k=5)
        self.assertAlmostEqual(recall_5, 0.4)

    def test_qps_calculation(self):
        """Test 5: QPS math."""
        # 500 queries in 2.5 seconds = 200 QPS
        qps = calculate_qps(num_queries=500, total_time_seconds=2.5)
        self.assertEqual(qps, 200.0)
        
        # Division by zero safety
        self.assertEqual(calculate_qps(500, 0.0), 0.0)

if __name__ == '__main__':
    unittest.main(verbosity=2)