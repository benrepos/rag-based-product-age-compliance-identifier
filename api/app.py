"""
FastAPI application for product age restriction classification
Orchestrates the full RAG pipeline: Retrieval → Classification
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rag_pipeline import LawRetriever
from llm import AgeRestrictionClassifier, ClassificationResponse
from llm.prior import get_prior_cached
from config import OPENAI_API_KEY, API_KEY  # ensures .env is loaded via config module

# Load environment variables from .env if present
load_dotenv()


# Request model
class ProductRequest(BaseModel):
    """Request body for product classification"""
    product_description: str = Field(
        min_length=5,
        description="Description of the product to classify"
    )
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant law chunks to retrieve (1-20)"
    )


# Initialize FastAPI app
app = FastAPI(
    title="UK Product Age Restriction API",
    description="RAG-based system for determining age restrictions on UK products",
    version="1.0.0"
)

# Initialize components (loaded once at startup)
retriever = None
classifier = None


@app.on_event("startup")
async def startup_event():
    """Initialize components without failing the server if assets are missing."""
    global retriever, classifier
    print("\n🚀 Initializing API...")
    # Initialize classifier (does not require embeddings)
    try:
        classifier = AgeRestrictionClassifier(model="gpt-5-nano", temperature=1)
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ Classifier init failed: {exc}")
        classifier = None
    # Try to load retriever; if embeddings unavailable, defer to first request
    try:
        retriever = LawRetriever(embeddings_file="embeddings/embeddings_store.pkl")
        print("✅ Embeddings loaded")
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ Retriever init deferred: {exc}")
        retriever = None
    print("✅ Startup complete\n")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "UK Product Age Restriction API",
        "version": "1.0.0"
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify_product(
    request: ProductRequest,
    x_api_key: str = Header(None),
) -> ClassificationResponse:
    # Simple header-based API key check
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    """
    Classify a product's age restriction based on UK law
    
    Pipeline:
    1. Retrieve relevant law chunks using similarity search
    2. Use LLM to classify based on retrieved evidence
    3. Return structured classification with metadata
    """
    try:
        # Lazy-load retriever on first request if needed
        global retriever
        if retriever is None:
            retriever = LawRetriever(embeddings_file="embeddings/embeddings_store.pkl")

        # Stage A: Prior (advisory)
        prior = get_prior_cached(request.product_description)
        expansion_terms = " ".join(prior.get("law_hints", []) + prior.get("query_expansion_terms", []))
        expanded_query = f"{request.product_description} {expansion_terms}".strip()

        # Step 1: Retrieve relevant law chunks using expanded query
        relevant_chunks = retriever.retrieve_relevant_chunks(
            query=expanded_query,
            k=request.k
        )
        
        # Step 2: Classify using LLM with retrieved evidence
        result = classifier.classify_with_metadata(
            product_description=request.product_description,
            relevant_chunks=relevant_chunks
        )
        
        return result
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Embeddings not available: {e}. Rebuild image with embeddings or run pipeline first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "retriever_loaded": retriever is not None,
        "classifier_loaded": classifier is not None,
        "num_law_chunks": len(retriever.chunks) if retriever else 0
    }


@app.post("/classify_debug")
async def classify_product_debug(
    request: ProductRequest,
    x_api_key: str = Header(None),
):
    # Simple header-based API key check
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    """
    Debug endpoint: returns the retrieved evidence (top-k chunks with scores)
    alongside the standard classification result. This does not change the
    production response model of /classify.
    """
    try:
        # Lazy-load retriever on first request if needed
        global retriever
        if retriever is None:
            retriever = LawRetriever(embeddings_file="embeddings/embeddings_store.pkl")

        # Stage A: Prior (advisory) for debug visibility
        prior = get_prior_cached(request.product_description)
        expansion_terms = " ".join(prior.get("law_hints", []) + prior.get("query_expansion_terms", []))
        expanded_query = f"{request.product_description} {expansion_terms}".strip()

        # Step 1: Retrieve relevant law chunks with expanded query
        relevant_chunks = retriever.retrieve_relevant_chunks(
            query=expanded_query,
            k=request.k
        )

        # Prepare a compact evidence view
        evidence_view = []
        for c in relevant_chunks:
            snippet = (c.get('chunk_text', '') or '')[:400]
            evidence_view.append({
                'chunk_id': c.get('chunk_id'),
                'law_name': c.get('law_name'),
                'section': c.get('section'),
                'similarity_score': c.get('similarity_score'),
                'token_count': c.get('token_count'),
                'snippet': snippet
            })

        # Step 2: Classify using LLM
        result = classifier.classify_with_metadata(
            product_description=request.product_description,
            relevant_chunks=relevant_chunks
        )

        return {
            'prior': prior,
            'classification': result,
            'evidence': evidence_view
        }

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Embeddings not available: {e}. Rebuild image with embeddings or run pipeline first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification (debug) failed: {str(e)}"
        )
