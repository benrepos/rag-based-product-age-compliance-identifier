# RAG-Based Product Age Compliance Identifier

A RAG (Retrieval-Augmented Generation) system for identifying age restrictions on UK products based on legal documents.

## 🎯 Project Overview

This system uses UK legislation and optionally business policy documents to determine age restrictions for products (alcohol, tobacco/vapes, knives, fireworks, etc.). It includes:
- Smart ingestion with OCR fallback for difficult PDFs (leaflets/tables)
- Content-aware chunking (Acts vs guidance leaflets)
- Hybrid retrieval (dense cosine + BM25 sparse + light lexical)
- LLM prior (open‑book UK knowledge) to guide retrieval + LLM reasoning with Pydantic‑validated output and evidence debugging

## 📁 Project Structure

```
rag-based-product-age-compliance-identifier/
├── data_ingestion/          # Stage 1: PDF reading and chunking
│   ├── pdf_reader.py        # Text extraction with pdfplumber + OCR fallback + normalization
│   ├── chunk_documents.py   # Doc-type-aware chunking (Acts vs guidance)
│   ├── raw_law_docs/        # Input: PDF files
│   └── raw_laws_txt/        # Output: Extracted text files
│
├── embeddings/              # Stage 2: Embeddings generation
│   ├── embedder.py          # Batched OpenAI embeddings
│   ├── law_chunks.json      # Chunked documents with metadata
│   ├── embeddings_store.pkl # Pickled embeddings
│   └── README.md            # Embeddings usage guide
│
├── api/
│   └── app.py               # FastAPI with /classify and /classify_debug
│
├── requirements.txt         # Python dependencies
├── run_embeddings.py        # Convenience script
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11 recommended
- OpenAI API key in `.env` (OPENAI_API_KEY=...)
- System tools (for OCR fallback, optional but recommended):
  - macOS: `brew install tesseract ocrmypdf`

### Installation

1. **Clone the repository**
   ```bash
   cd "/Users/ben.drury/Documents/Github repos/rag-based-product-age-compliance-identifier"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set OpenAI API key**
   - Put it in `.env` at repo root: `OPENAI_API_KEY=sk-...`
   - API loads `.env` automatically.

## 📜 Usage

### One-button pipeline

Run full ingestion → chunking → embeddings:

```bash
python run_pipeline.py
```

What happens:
- PDFs in `data_ingestion/raw_law_docs/` are extracted with pdfplumber; if the text is sparse or broken, OCR (ocrmypdf) is attempted automatically; text is normalized (fixes smashed words/bullets) and cached.
- All `.txt` in `data_ingestion/raw_laws_txt/` (including your policy files) are included.
- Doc-type detection:
  - Acts/Regs: paragraph+section chunking (target 500–800 tokens)
  - Guidance/leaflets/tables: sentence/bullet chunking (target 80–300 tokens)
- Embeddings are regenerated into `embeddings/embeddings_store.pkl`.

Add new PDFs:
- Drop the PDF into `data_ingestion/raw_law_docs/` and rerun `python run_pipeline.py`.
- Extraction skips files whose TXT is newer than the PDF.

Add your own policy docs:
- Create a small `.txt` in `data_ingestion/raw_laws_txt/` with clear rule, synonyms, and provenance.
- Example: `Policy - Compressed gases (butane, CO2) age restriction.txt` (already included).
- Rerun `python run_pipeline.py`.

Debug evidence:
- Start API: `uvicorn api.app:app --reload --port 8000`
- Use `POST /classify_debug` with `{ product_description, k }` to see the exact chunks, scores, and snippets sent to the LLM.

### Run the API (FastAPI)

Start the service:
```bash
uvicorn api.app:app --reload --port 8000
# or explicitly use venv python
.venv/bin/python -m uvicorn api.app:app --reload --port 8000
```

Environment:
- Put `OPENAI_API_KEY=...` in `.env` at repo root (auto‑loaded).
- Optional header auth: set `API_KEY=...` in `.env` and pass `x-api-key: ...` on requests.

Endpoints:
- `POST /classify`
  - Body: `{ "product_description": "Bottle of red wine 12% ABV", "k": 5 }`
  - Returns: Pydantic `ClassificationResponse` (restriction_level, confidence, reason, evidence + metadata)
- `POST /classify_debug`
  - Same request as `/classify`, response additionally includes:
    - `prior` (model’s advisory guess: category, expected restriction, hints)
    - `evidence` array with top‑k chunks and similarity scores used for the LLM.
- `GET /health` and `GET /` for health checks

Typical workflow:
1) When sources change (new PDFs or policy `.txt`): run `python run_pipeline.py` to refresh chunks and embeddings.
2) Start the API and query `/classify` in production; use `/classify_debug` during tuning to inspect evidence.

Examples:
```bash
curl -s -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"product_description":"Disposable butane lighter refill 300ml","k":15}' | jq

curl -s -X POST http://localhost:8000/classify_debug \
  -H 'Content-Type: application/json' \
  -d '{"product_description":"CO2 gas cylinder 600g","k":25}' | jq
```

Notes:
- Increase `k` (e.g., 15–30) for more robust retrieval on rare terms.
- If you see boilerplate headings in evidence, the hybrid retriever will still return policy/statute chunks; use `/classify_debug` to validate.

### LLM Prior (advisory) and caching

- The API runs a light “prior” step first (open‑book UK knowledge) using `gpt-4o-mini`:
  - Output: `category_guess`, `expected_restriction`, `rationale_short`, `law_hints`, `query_expansion_terms`.
  - Retrieval query is expanded with these hints/terms, improving recall (e.g., alcohol → s.146 hints).
  - The final classification remains evidence‑only; the prior is advisory and visible in `/classify_debug`.
- Prior calls are cached in‑process with LRU (512 entries). The cache resets when the process restarts.

## ✅ Status

Repo is ready-to-run: ingestion (with OCR), chunking, embeddings, hybrid retrieval, and API with evidence debug.

## 🏛️ Legal Documents

### Currently Processed:
1. **Licensing Act 2003** (alcohol) - 184 pages
2. **Offensive Weapons Act 2019** - 75 pages

### Planned:
- Tobacco & Related Products Regulations 2016
- Criminal Justice Act 1988 (knives)
- Fireworks Regulations 2004

## 🛠️ Technical Details

### Document Chunking
- Acts/Regs: paragraphs + numbered sections; 500–800 tokens
- Guidance/leaflets: sentences/bullets; 80–300 tokens; tail chunks preserved so short lines aren’t lost
- Tokenizer: cl100k_base (OpenAI-compatible)

### Embeddings & Retrieval
- Embeddings: OpenAI `text-embedding-3-small` (1536 dims)
- Hybrid retrieval: 0.60 cosine + 0.30 BM25 + 0.10 lexical overlap; boilerplate penalty for generic headings
- Evidence debug: `POST /classify_debug` shows top‑k with scores/snippets

### Deployment (Cloud Run)
- The container listens on `0.0.0.0:$PORT` (Cloud Run injects `PORT`).
- Command: `uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8080}`
- Set `OPENAI_API_KEY` (required) and optionally `API_KEY` for header auth.

### Chunk Metadata Structure
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

## 📦 Dependencies

```
pypdf==4.0.1
pdfplumber==0.11.4
pillow==10.3.0
tiktoken==0.5.2
openai==1.54.0
httpx==0.27.2
numpy==1.26.3
rank-bm25==0.2.2
fastapi==0.109.0
uvicorn==0.27.0
```

System (optional but recommended):
- tesseract, ocrmypdf (for OCR fallback)

## 🤝 Contributing

This is a personal project for UK product age compliance identification. Suggestions and improvements are welcome!

## 📄 License

This project processes UK legal documents which are Crown Copyright but available under the Open Government Licence v3.0.

## 🔗 Resources

- [UK Legislation](https://www.legislation.gov.uk/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [tiktoken](https://github.com/openai/tiktoken)

## 📝 Notes

- Pipeline auto-OCRs PDFs when layout extraction is poor and normalizes text.
- You can add business policy files as `.txt`; they will be retrieved and clearly cited as policy (not statute).
- Use `/classify_debug` to inspect evidence if a product seems misclassified, then refine sources.

## 🚧 Possible Improvements

- Retrieval quality
  - Add per-source weighting (statute > guidance > policy) with tie‑breaks.
  - Learnable re-ranking (e.g., SVM/LightGBM on features: cosine, BM25, rarity, chunk length, boilerplate score).
  - FAISS/AnnLite vector index for faster search on larger corpora.
- Ingestion/normalization
  - Table extraction where structure matters (pdfplumber tables) and layout‑aware heuristics per document template.
  - Automatic de‑duplication and boilerplate masking across acts (“Document Generated…”, “PART X”).
- Product understanding
  - Config‑driven ontology and synonyms (JSON) with deterministic category mapping prior to retrieval.
  - Optional LLM pass to canonicalize messy product strings (brand/size -> normalized features).
- LLM reasoning
  - Add a strict JSON schema with allowed enums and citation checks (require section names in evidence).
  - Multi‑prompt strategy: statute‑first; fallback to guidance/policy.
- Governance & ops
  - Policy document versioning, validity windows, and audit logs (who added/changed rules).
  - Rate limiting and request logging in the API; `/metrics` for observability.
  - CI: lint (pylint), format (black/isort), tests (pytest) for pipeline and retrieval.

