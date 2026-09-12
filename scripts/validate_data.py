import os
import sys
import numpy as np
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.embedding_pipeline import EmbeddingPipeline

def validate():
    vectors_path = 'data/processed/vectors.npy'
    meta_path = 'data/processed/metadata.jsonl'
    
    if not os.path.exists(vectors_path) or not os.path.exists(meta_path):
        print(f"Error: Processed files not found at {vectors_path} and {meta_path}.")
        print("Please run 'python scripts/prepare_data.py' first.")
        sys.exit(1)

    print("Loading data for validation...")
    vectors, metadata = EmbeddingPipeline.load_data(vectors_path, meta_path)
    
    total_records = len(vectors)
    unique_texts = len(set(record['text'] for record in metadata))
    category_counts = Counter(record['category'] for record in metadata)
    has_nans = bool(np.isnan(vectors).any())
    has_infs = bool(np.isinf(vectors).any())

    # Formatted Diagnostic Report
    print("\n" + "="*50)
    print("           DATA QUALITY & INTEGRITY REPORT        ")
    print("="*50)
    print(f"Total Records:             {total_records:,}")
    print(f"Unique Text Strings:       {unique_texts:,}")
    print(f"Embedding Matrix Shape:    {vectors.shape}")
    print(f"Contains NaNs:             {has_nans}")
    print(f"Contains Infs:             {has_infs}")
    print("-" * 50)
    print("Category Breakdown:")
    for category, count in category_counts.items():
        print(f"  - {category:<20}: {count:,} records")
    print("="*50 + "\n")
    
    # Validation Assertions
    assert total_records >= 50000, f"Expected >= 50,000 records, found {total_records}"
    assert total_records == len(metadata), "Vectors count does not match metadata records count"
    assert unique_texts == total_records, f"Found {total_records - unique_texts} duplicate text strings!"
    assert vectors.ndim == 2, f"Expected a 2D matrix, got shape {vectors.shape}"
    assert not has_nans, "Embedding matrix contains NaN values"
    assert not has_infs, "Embedding matrix contains Infinite values"
    
    # Verify Unique UUIDs
    ids = [record['id'] for record in metadata]
    assert len(ids) == len(set(ids)), "Found duplicate IDs in metadata"
    
    # Verify L2 Normalization (Cosine Similarity invariant)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Vectors are not normalized to unit length"
    
    print("All validation checks passed successfully.")

if __name__ == "__main__":
    validate()