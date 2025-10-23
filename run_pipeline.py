#!/usr/bin/env python3
"""
Full Pipeline Runner: PDF → Text → Chunks → Embeddings
Run this script whenever you add new PDFs to recreate the entire dataset
"""

import os
import sys
from pathlib import Path
from config import OPENAI_API_KEY  # loads .env if needed

def check_api_key():
    """Check if OpenAI API key is set"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not found in environment variables")
        print("Set it with: export OPENAI_API_KEY='your-api-key-here'\n")
        return False
    return True

def run_pdf_extraction():
    """Step 1: Extract text from PDFs"""
    print("\n" + "="*70)
    print("STEP 1: Extracting text from PDFs")
    print("="*70)
    
    from data_ingestion.pdf_reader import LegalDocumentReader
    reader = LegalDocumentReader()
    reader.process_all_pdfs()

def run_chunking():
    """Step 2: Chunk documents"""
    print("\n" + "="*70)
    print("STEP 2: Chunking documents")
    print("="*70)
    
    from data_ingestion.chunk_documents import LegalDocumentChunker
    chunker = LegalDocumentChunker()
    chunker.process_all_documents()

def run_embeddings():
    """Step 3: Generate embeddings"""
    print("\n" + "="*70)
    print("STEP 3: Generating embeddings")
    print("="*70)
    
    from embeddings.embedder import EmbeddingsGenerator
    generator = EmbeddingsGenerator()
    generator.run_full_pipeline()

def main():
    """Run the complete pipeline"""
    print("\n" + "="*70)
    print("🚀 FULL PIPELINE: PDF → TEXT → CHUNKS → EMBEDDINGS")
    print("="*70)
    
    # Check prerequisites
    if not check_api_key():
        sys.exit(1)
    
    try:
        # Run pipeline
        run_pdf_extraction()
        run_chunking()
        run_embeddings()
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE!")
        print("="*70)
        print("\nOutput files:")
        print("  • data_ingestion/raw_laws_txt/*.txt")
        print("  • embeddings/law_chunks.json")
        print("  • embeddings/embeddings_store.pkl")
        print()
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

