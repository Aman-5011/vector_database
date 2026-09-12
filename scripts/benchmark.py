import os
import sys
import time
import csv
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.exact_index import ExactIndex
from src.hnsw_index import HNSWIndex
from src.data.embedding_pipeline import EmbeddingPipeline
from src.evaluation import calculate_recall_at_k, calculate_qps

def generate_queries(vectors: np.ndarray, num_queries: int = 500, seed: int = 42) -> np.ndarray:
    """
    Deterministically samples vectors from the dataset and adds a tiny bit of noise
    to simulate real-world semantic queries that are close but not mathematically identical
    to the stored data.
    """
    rng = np.random.default_rng(seed)
    
    # Randomly select base vectors
    indices = rng.choice(len(vectors), size=num_queries, replace=False)
    base_queries = vectors[indices].copy()
    
    # Add slight Gaussian noise
    noise = rng.normal(0, 0.02, base_queries.shape).astype(np.float32)
    queries = base_queries + noise
    
    # Re-normalize to preserve Cosine Similarity property (L2 norm = 1)
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries / norms
    
    return queries

def main():
    vectors_path = 'data/processed/vectors.npy'
    meta_path = 'data/processed/metadata.jsonl'
    results_dir = 'data/benchmarks'
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading dataset...")
    vectors, metadata = EmbeddingPipeline.load_data(vectors_path, meta_path)
    ids = [record['id'] for record in metadata]
    dim = vectors.shape[1]
    
    print("Generating 500 deterministic query vectors...")
    queries = generate_queries(vectors, num_queries=500)
    k = 10
    
    # ==========================================
    # 1. BUILD INDEXES
    # ==========================================
    print("\nBuilding Exact Index...")
    exact_idx = ExactIndex(dim=dim)
    exact_idx.add(vectors, ids)
    
    print("Building HNSW Index (This will take a few minutes in pure Python)...")
    # Using M=16, ef_construction=100 as standard defaults
    hnsw_idx = HNSWIndex(dim=dim, M=16, ef_construction=100, seed=42)
    build_start = time.perf_counter()
    hnsw_idx.add(vectors, ids)
    build_time = time.perf_counter() - build_start
    print(f"HNSW built in {build_time:.2f} seconds.")

    # ==========================================
    # 2. GROUND TRUTH (EXACT SEARCH)
    # ==========================================
    print("\nRunning Ground Truth (Exact) Search...")
    ground_truth_ids = []
    
    exact_start = time.perf_counter()
    for q in queries:
        res = exact_idx.search(q, k=k)
        ground_truth_ids.append([item[0] for item in res])
    exact_time = time.perf_counter() - exact_start
    
    exact_qps = calculate_qps(len(queries), exact_time)
    exact_latency = (exact_time / len(queries)) * 1000
    
    print(f"Exact Search: {exact_qps:.2f} QPS | {exact_latency:.2f} ms/query")

    # ==========================================
    # 3. HNSW EVALUATION
    # ==========================================
    ef_search_values = [10, 20, 50, 100, 200]
    results_data = []

    print("\n" + "-"*60)
    print(f"{'ef_search':<12} {'Recall@10':<12} {'Avg Latency(ms)':<18} {'QPS'}")
    print("-" * 60)

    for ef in ef_search_values:
        approx_ids = []
        
        # Timing ONLY the search loop
        hnsw_start = time.perf_counter()
        for q in queries:
            res = hnsw_idx.search(q, k=k, ef_search=ef)
            approx_ids.append([item[0] for item in res])
        hnsw_time = time.perf_counter() - hnsw_start
        
        # Calculate Metrics
        recalls = [
            calculate_recall_at_k(ground_truth_ids[i], approx_ids[i], k) 
            for i in range(len(queries))
        ]
        
        mean_recall = np.mean(recalls)
        qps = calculate_qps(len(queries), hnsw_time)
        avg_latency = (hnsw_time / len(queries)) * 1000
        
        print(f"{ef:<12} {mean_recall*100:>6.2f}%       {avg_latency:>7.2f}            {qps:>7.2f}")
        
        results_data.append({
            'ef_search': ef,
            'k': k,
            'num_queries': len(queries),
            'recall_at_k': mean_recall,
            'avg_latency_ms': avg_latency,
            'qps': qps,
            'total_time_seconds': hnsw_time
        })
    print("-" * 60)

    # ==========================================
    # 4. SAVE AND PLOT
    # ==========================================
    csv_path = os.path.join(results_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_data[0].keys())
        writer.writeheader()
        writer.writerows(results_data)
    print(f"\nSaved CSV results to {csv_path}")
    
    # Plotting Recall vs QPS
    recalls_plot = [row['recall_at_k'] * 100 for row in results_data]
    qps_plot = [row['qps'] for row in results_data]
    
    plt.figure(figsize=(8, 6))
    plt.plot(recalls_plot, qps_plot, marker='o', linestyle='-', color='b')
    
    # Annotate points with their ef_search values
    for i, ef in enumerate(ef_search_values):
        plt.annotate(f"ef={ef}", (recalls_plot[i], qps_plot[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center')

    plt.title('Speed vs Accuracy Tradeoff (HNSW)')
    plt.xlabel('Recall@10 (%)')
    plt.ylabel('Queries Per Second (QPS)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(results_dir, 'recall_vs_qps.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()