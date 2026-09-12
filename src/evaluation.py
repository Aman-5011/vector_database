from typing import List

def calculate_recall_at_k(exact_ids: List[str], approx_ids: List[str], k: int) -> float:
    """
    Calculates the Recall@K.
    It measures how many of the true top-K IDs were found in the approximate top-K IDs.
    """
    if k <= 0:
        return 0.0
        
    exact_set = set(exact_ids[:k])
    approx_set = set(approx_ids[:k])
    
    overlap = len(exact_set.intersection(approx_set))
    return float(overlap) / float(k)

def calculate_qps(num_queries: int, total_time_seconds: float) -> float:
    """
    Calculates Queries Per Second.
    """
    if total_time_seconds <= 0.0:
        return 0.0
    return float(num_queries) / total_time_seconds