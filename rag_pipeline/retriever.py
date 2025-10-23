"""
Retrieval module for RAG pipeline
Loads embeddings and retrieves relevant law chunks based on similarity
"""

import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict
import re
from rank_bm25 import BM25Okapi
from openai import OpenAI

from .similarity import cosine_similarity_batch, get_top_k_indices
 


class LawRetriever:
    """Retrieves relevant law chunks based on query similarity"""
    
    def __init__(self, 
                 embeddings_file: str = "embeddings/embeddings_store.pkl",
                 model: str = "text-embedding-3-small",
                 dimensions: int = 1536):
        """
        Initialize the law retriever
        
        Args:
            embeddings_file: Path to pickled embeddings store
            model: OpenAI embedding model name
            dimensions: Embedding dimensions
        """
        self.embeddings_file = Path(embeddings_file)
        self.model = model
        self.dimensions = dimensions
        
        # Initialize OpenAI client
        self.client = self._create_openai_client()
        
        # Load embeddings store
        self.embeddings_store = self._load_embeddings()
        self.embeddings = self.embeddings_store['embeddings']
        self.chunks = self.embeddings_store['chunks']
        self.metadata = self.embeddings_store['metadata']
        
        print(f"✓ Loaded {len(self.chunks)} law chunks")
        print(f"  Model: {self.metadata['model']}")
        print(f"  Dimensions: {self.metadata['dimensions']}")
        
        # Prepare BM25 corpus for sparse retrieval
        self._bm25 = self._build_bm25_index()
    
    def _tokenize(self, text: str) -> List[str]:
        return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1]
    
    def _build_bm25_index(self) -> BM25Okapi:
        corpus = [self._tokenize(c.get('chunk_text', '')) for c in self.chunks]
        return BM25Okapi(corpus)
    
    def _create_openai_client(self) -> OpenAI:
        """Create OpenAI client with API key from environment"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it using: export OPENAI_API_KEY='your-api-key'"
            )
        return OpenAI(api_key=api_key)
    
    def _load_embeddings(self) -> Dict:
        """
        Load embeddings store from pickle file
        
        Returns:
            dict: Embeddings store with embeddings, chunks, and metadata
        """
        if not self.embeddings_file.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {self.embeddings_file}\n"
                f"Please run the pipeline first: python run_pipeline.py"
            )
        
        with open(self.embeddings_file, 'rb') as f:
            embeddings_store = pickle.load(f)
        
        return embeddings_store
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query string
        
        Args:
            query: Query text (e.g., product description)
            
        Returns:
            np.ndarray: Query embedding vector
        """
        response = self.client.embeddings.create(
            input=query,
            model=self.model,
            dimensions=self.dimensions
        )
        
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return embedding
    
    def retrieve_relevant_chunks(self, 
                                query: str, 
                                k: int = 5,
                                return_scores: bool = True) -> List[Dict]:
        """
        Retrieve top-k most relevant law chunks for a query
        This is the main method you'll use for retrieval
        
        Args:
            query: Query text (e.g., product description)
            k: Number of chunks to retrieve
            return_scores: Whether to include similarity scores in results
            
        Returns:
            List of dictionaries containing chunk data and similarity scores
        """
        # Embed the query
        query_emb = self.embed_query(query)
        
        # Calculate cosine similarities
        similarities = cosine_similarity_batch(query_emb, self.embeddings)
        
        # Sparse BM25 scores (normalized)
        tokens = self._tokenize(query)
        bm25_scores = np.array(self._bm25.get_scores(tokens), dtype=np.float32)
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        # Boilerplate penalty (headings and generator footers)
        boilerplate = []
        for c in self.chunks:
            txt = (c.get('chunk_text', '') or '').lower()
            is_boiler = (
                ("document generated" in txt) or
                (re.match(r"^\s*part\s+\d+", txt) is not None) or
                (len(txt) < 40)
            )
            boilerplate.append(0.15 if is_boiler else 0.0)
        boilerplate = np.array(boilerplate, dtype=np.float32)

        # No pre-filtering by sales terms or product category

        # Lightweight lexical re-ranking: boost chunks with keyword overlap
        query_terms = {t for t in ''.join([c if c.isalnum() else ' ' for c in query.lower()]).split() if len(t) > 2}
        if query_terms:
            # compute overlap score in [0,1]
            overlap_scores = []
            for chunk in self.chunks:
                text_lc = chunk.get('chunk_text', '').lower()
                # fast contains check by counting term hits
                hits = sum(1 for t in query_terms if t in text_lc)
                overlap_scores.append(hits / max(1, len(query_terms)))
            overlap_scores = np.array(overlap_scores, dtype=np.float32)
            # combine: hybrid dense+sparse with lexical boost
            combined = 0.60 * similarities + 0.30 * bm25_scores + 0.10 * overlap_scores
        else:
            combined = 0.70 * similarities + 0.30 * bm25_scores

        # Penalise bladed-product evidence for clearly non-bladed tool queries
        blade_terms = {"blade", "bladed", "knife", "knives", "razor", "scissors", "sword", "machete"}
        non_bladed_terms = {"hammer", "spanner", "wrench", "pliers", "screwdriver", "mallet"}
        query_has_blade = any(t in query_terms for t in blade_terms) if query_terms else False
        query_has_non_bladed = any(t in query_terms for t in non_bladed_terms) if query_terms else False
        if not query_has_blade:
            # detect phrases that indicate bladed-specific sections
            flags = []
            for chunk in self.chunks:
                txt = chunk.get('chunk_text', '').lower()
                has_bladed_phrase = ("bladed product" in txt) or ("bladed article" in txt)
                flags.append(1.0 if has_bladed_phrase else 0.0)
            flags = np.array(flags, dtype=np.float32)
            # base penalty
            penalty = 0.15 * flags
            # stronger penalty if query explicitly mentions a non-bladed tool
            if query_has_non_bladed:
                penalty += 0.10 * flags
            combined = combined - penalty
        
        # Apply boilerplate penalty
        combined = combined - boilerplate

        # Get top-k indices on combined scores
        top_k_idx = get_top_k_indices(combined, k)
        
        # Retrieve chunks with scores
        results = []
        for idx in top_k_idx:
            chunk = self.chunks[idx].copy()
            
            if return_scores:
                chunk['similarity_score'] = float(similarities[idx])
            
            results.append(chunk)
        
        return results
