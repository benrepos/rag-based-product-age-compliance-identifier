"""
Embeddings Module for UK Law Document Chunks
Uses OpenAI's text-embedding-3-small with batch processing for efficiency
"""

import json
import pickle
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class EmbeddingsGenerator:
    """Generates and manages embeddings for legal document chunks"""
    
    def __init__(self, 
                 embeddings_dir=".",
                 model="text-embedding-3-small",
                 dimensions=1536,
                 batch_size=100,
                 max_workers=5):
        """
        Initialize the embeddings generator
        
        Args:
            embeddings_dir: Directory containing chunks and for saving embeddings
            model: OpenAI embedding model name
            dimensions: Embedding dimensions (1536 for text-embedding-3-small)
            batch_size: Number of texts to embed in one API call
            max_workers: Number of parallel workers for batch processing
        """
        self.embeddings_dir = Path(__file__).parent if embeddings_dir == "." else Path(embeddings_dir)
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        self.chunks_file = self.embeddings_dir / "law_chunks.json"
        self.embeddings_file = self.embeddings_dir / "embeddings_store.pkl"
        
        # Initialize OpenAI client
        self.client = self._create_openai_client()
    
    def _create_openai_client(self):
        """Create OpenAI client with API key from environment"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it using: export OPENAI_API_KEY='your-api-key'"
            )
        return OpenAI(api_key=api_key)
    
    def load_chunks(self):
        """
        Load chunks from JSON file
        
        Returns:
            list: List of chunk dictionaries
        """
        if not self.chunks_file.exists():
            raise FileNotFoundError(
                f"Chunks file not found: {self.chunks_file}\n"
                f"Please run chunk_documents.py first."
            )
        
        print(f"📖 Loading chunks from: {self.chunks_file}")
        
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        print(f"✓ Loaded {len(chunks)} chunk(s)")
        return chunks
    
    def get_batched_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            list: List of embedding vectors
        """
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model,
                dimensions=self.dimensions
            )
            
            embeddings = [r.embedding for r in response.data]
            return embeddings
        
        except Exception as e:
            print(f"✗ Error getting embeddings: {str(e)}")
            raise
    
    def process_batch(self, batch_data):
        """
        Process a single batch of chunks
        
        Args:
            batch_data: Tuple of (batch_idx, batch_chunks)
            
        Returns:
            tuple: (batch_idx, embeddings, success)
        """
        batch_idx, batch_chunks = batch_data
        
        try:
            texts = [chunk['chunk_text'] for chunk in batch_chunks]
            embeddings = self.get_batched_embeddings(texts)
            return (batch_idx, embeddings, True)
        
        except Exception as e:
            print(f"✗ Batch {batch_idx} failed: {str(e)}")
            return (batch_idx, None, False)
    
    def generate_embeddings_parallel(self, chunks: List[Dict]) -> np.ndarray:
        """
        Generate embeddings for all chunks using parallel batch processing
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            numpy array: Matrix of embeddings (num_chunks x dimensions)
        """
        print(f"\n{'='*70}")
        print(f"🚀 GENERATING EMBEDDINGS")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"Dimensions: {self.dimensions}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Parallel workers: {self.max_workers}")
        
        # Create batches
        batches = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batches.append((i // self.batch_size, batch))
        
        print(f"Total batches: {len(batches)}")
        print(f"\nProcessing batches in parallel...")
        
        # Process batches in parallel
        all_embeddings = [None] * len(chunks)
        completed_batches = 0
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch jobs
            future_to_batch = {
                executor.submit(self.process_batch, batch_data): batch_data[0]
                for batch_data in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx, embeddings, success = future.result()
                
                if success and embeddings:
                    # Store embeddings in correct position
                    start_idx = batch_idx * self.batch_size
                    for i, emb in enumerate(embeddings):
                        all_embeddings[start_idx + i] = emb
                    
                    completed_batches += 1
                    progress = (completed_batches / len(batches)) * 100
                    elapsed = time.time() - start_time
                    
                    print(f"  ✓ Batch {batch_idx + 1}/{len(batches)} ({progress:.1f}%) - {elapsed:.1f}s elapsed")
                else:
                    print(f"  ✗ Batch {batch_idx + 1} failed")
        
        # Check if all embeddings were generated
        if None in all_embeddings:
            raise RuntimeError("Some embeddings failed to generate")
        
        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        
        total_time = time.time() - start_time
        print(f"\n✓ All embeddings generated successfully!")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average: {total_time/len(chunks):.3f}s per chunk")
        print(f"  Shape: {embeddings_array.shape}")
        
        return embeddings_array
    
    def save_embeddings(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Save embeddings and metadata to pickle file
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Numpy array of embeddings
        """
        print(f"\n💾 Saving embeddings...")
        
        # Create storage structure
        embeddings_store = {
            'embeddings': embeddings,
            'chunks': chunks,
            'metadata': {
                'model': self.model,
                'dimensions': self.dimensions,
                'num_chunks': len(chunks),
                'embedding_shape': embeddings.shape
            }
        }
        
        # Save to pickle
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(embeddings_store, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size = self.embeddings_file.stat().st_size / 1024
        print(f"✓ Saved to: {self.embeddings_file}")
        print(f"  Size: {file_size:.1f} KB")
    
    def load_embeddings(self):
        """
        Load embeddings from pickle file
        
        Returns:
            dict: Embeddings store with embeddings, chunks, and metadata
        """
        if not self.embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")
        
        print(f"📖 Loading embeddings from: {self.embeddings_file}")
        
        with open(self.embeddings_file, 'rb') as f:
            embeddings_store = pickle.load(f)
        
        print(f"✓ Loaded embeddings")
        print(f"  Model: {embeddings_store['metadata']['model']}")
        print(f"  Chunks: {embeddings_store['metadata']['num_chunks']}")
        print(f"  Shape: {embeddings_store['metadata']['embedding_shape']}")
        
        return embeddings_store
    
    def run_full_pipeline(self):
        """
        Run the complete embeddings generation pipeline
        
        Returns:
            dict: Embeddings store with embeddings, chunks, and metadata
        """
        print(f"\n{'='*70}")
        print(f"🎯 STARTING EMBEDDINGS PIPELINE")
        print(f"{'='*70}\n")
        
        # Load chunks
        chunks = self.load_chunks()
        
        # Generate embeddings
        embeddings = self.generate_embeddings_parallel(chunks)
        
        # Save embeddings
        self.save_embeddings(chunks, embeddings)
        
        print(f"\n{'='*70}")
        print(f"✅ PIPELINE COMPLETE!")
        print(f"{'='*70}\n")
        print(f"Output files:")
        print(f"  • {self.chunks_file} ({len(chunks)} chunks)")
        print(f"  • {self.embeddings_file} ({embeddings.shape[0]} embeddings)")
        
        # Return the embeddings store
        return {
            'embeddings': embeddings,
            'chunks': chunks,
            'metadata': {
                'model': self.model,
                'dimensions': self.dimensions,
                'num_chunks': len(chunks),
                'embedding_shape': embeddings.shape
            }
        }


def main():
    """Main function to run embeddings generation"""
    generator = EmbeddingsGenerator()
    generator.run_full_pipeline()


if __name__ == "__main__":
    main()

