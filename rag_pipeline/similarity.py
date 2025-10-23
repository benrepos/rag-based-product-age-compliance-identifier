"""
Similarity calculation functions for RAG retrieval
Uses cosine similarity (standard for embeddings comparison)
"""

import numpy as np

def cosine_similarity_batch(query_emb: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between a query and multiple embeddings efficiently
    This is the main function used by the retriever
    
    Args:
        query_emb: Query embedding vector (1D array)
        embeddings: Matrix of embeddings (2D array: num_embeddings x dimensions)
        
    Returns:
        np.ndarray: Array of similarity scores (higher = more similar)
    """
    # Normalize query
    query_norm = query_emb / np.linalg.norm(query_emb)
    
    # Normalize all embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Compute dot product (cosine similarity for normalized vectors)
    similarities = np.dot(embeddings_norm, query_norm)
    
    return similarities


def get_top_k_indices(similarities: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Get indices of top-k highest similarity scores
    
    Args:
        similarities: Array of similarity scores
        k: Number of top results to return
        
    Returns:
        np.ndarray: Indices of top-k results (sorted by score, descending)
    """
    # Handle edge case where k >= total number of items
    if k >= len(similarities):
        return np.argsort(similarities)[::-1]
    
    # Use argpartition for efficiency (O(n) instead of O(n log n))
    top_k_idx = np.argpartition(similarities, -k)[-k:]
    
    # Sort them by score (descending)
    top_k_idx = top_k_idx[np.argsort(similarities[top_k_idx])[::-1]]
    
    return top_k_idx

