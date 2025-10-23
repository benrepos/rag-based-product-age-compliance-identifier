# Embeddings Module - Usage Guide

## Overview
This module generates embeddings for legal document chunks using OpenAI's `text-embedding-3-small` model with batch processing and parallelization for efficiency.

## Setup

### 1. Install Dependencies
```bash
pip install -r ../requirements.txt
```

### 2. Set OpenAI API Key
```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or add it to your shell profile (~/.zshrc or ~/.bashrc):
```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

## Usage

### Generate Embeddings
Once the API key is set, run the embeddings generation:

```bash
cd embeddings
python embedder.py
```

This will:
1. Load chunks from `law_chunks.json`
2. Generate embeddings in parallel batches (100 chunks per batch, 5 workers)
3. Save results to `embeddings_store.pkl`

### Expected Output
```
📖 Loading chunks from: law_chunks.json
✓ Loaded 104 chunk(s)

🚀 GENERATING EMBEDDINGS
Model: text-embedding-3-small
Dimensions: 1536
Total chunks: 104
Batch size: 100
Parallel workers: 5
Total batches: 2

Processing batches in parallel...
  ✓ Batch 1/2 (50.0%) - 1.5s elapsed
  ✓ Batch 2/2 (100.0%) - 2.1s elapsed

✓ All embeddings generated successfully!
  Total time: 2.1s
  Average: 0.020s per chunk
  Shape: (104, 1536)

💾 Saving embeddings...
✓ Saved to: embeddings_store.pkl
  Size: 632.4 KB

✅ PIPELINE COMPLETE!
```

## Configuration

You can customize the embeddings generation by modifying the `EmbeddingsGenerator` parameters:

```python
generator = EmbeddingsGenerator(
    embeddings_dir=".",           # Where to save embeddings
    model="text-embedding-3-small",  # OpenAI model
    dimensions=1536,              # Embedding dimensions
    batch_size=100,              # Chunks per API call (max 2048)
    max_workers=5                # Parallel workers
)
```

## Output Files

### `law_chunks.json`
JSON file containing all document chunks with metadata:
```json
{
  "chunk_id": "Licensing Act 2003 (alcohol)_1_0",
  "law_name": "Licensing Act 2003 (alcohol)",
  "section": "1",
  "chunk_index": 0,
  "chunk_text": "...",
  "token_count": 650,
  "url": "https://www.legislation.gov.uk/..."
}
```

### `embeddings_store.pkl`
Pickle file containing:
- `embeddings`: NumPy array of shape (num_chunks, dimensions)
- `chunks`: List of chunk dictionaries
- `metadata`: Model info and statistics

## Loading Embeddings

To load and use the embeddings in your code:

```python
import pickle
import numpy as np

# Load embeddings
with open('embeddings_store.pkl', 'rb') as f:
    store = pickle.load(f)

embeddings = store['embeddings']  # NumPy array
chunks = store['chunks']          # List of dicts
metadata = store['metadata']      # Dict with info

print(f"Loaded {len(chunks)} chunks")
print(f"Embedding shape: {embeddings.shape}")
```

## Cost Estimation

OpenAI `text-embedding-3-small` pricing (as of 2024):
- $0.020 per 1M tokens

For 104 chunks (~500-800 tokens each):
- Estimated tokens: ~65,000
- Estimated cost: ~$0.0013 (negligible)

## Performance

- **Batch processing**: Process up to 100 chunks per API call
- **Parallel execution**: 5 concurrent workers
- **Expected speed**: ~50 chunks/second

## Troubleshooting

### Error: "OPENAI_API_KEY not found"
Make sure you've set your API key in environment variables.

### Rate Limiting
If you encounter rate limits, reduce `max_workers` or `batch_size`:
```python
generator = EmbeddingsGenerator(max_workers=3, batch_size=50)
```

### Memory Issues
For very large document sets, process in smaller batches:
```python
generator = EmbeddingsGenerator(batch_size=50)
```

