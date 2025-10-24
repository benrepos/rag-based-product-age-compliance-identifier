"""
LLM prior: open-book UK knowledge used to guide retrieval.

This module asks the model to infer likely category and restriction for a
product description, and to return hints/terms to expand retrieval.
The prior is advisory only; final classification must still rely on evidence.
"""

from typing import Dict, List
import os

from openai import OpenAI
from functools import lru_cache


def _create_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables. "
            "Please set it using: export OPENAI_API_KEY='your-api-key'"
        )
    return OpenAI(api_key=api_key)


def get_prior(product_description: str,
              model: str = "gpt-4o-mini",
              temperature: float = 0.2) -> Dict:
    """
    Produce a lightweight prior for category and expected restriction.

    Returns dict with keys:
      - category_guess: str
      - expected_restriction: one of None|16+|18+|Licensed
      - rationale_short: str
      - law_hints: List[str]
      - query_expansion_terms: List[str]
    """
    client = _create_openai_client()
    system = (
        "You are a UK retail compliance expert. Given ONLY a product description, "
        "infer the likely regulated category and expected restriction under UK rules. "
        "Return JSON with: category_guess, expected_restriction (None|16+|18+|Licensed), "
        "rationale_short (one sentence), law_hints (e.g., 'Licensing Act 2003 s.146'), "
        "query_expansion_terms (synonyms/keywords: e.g., 'alcohol, sale to under 18')."
    )
    user = f"Product: {product_description}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        response_format={"type": "json_object"}
    )

    content = resp.choices[0].message.content or "{}"
    # Be defensive about JSON parsing; the API returns JSON per response_format
    import json  # local import to avoid global dependency at import time
    data = json.loads(content)

    # Normalize fields
    result = {
        "category_guess": data.get("category_guess", "unknown"),
        "expected_restriction": data.get("expected_restriction", "None"),
        "rationale_short": data.get("rationale_short", ""),
        "law_hints": data.get("law_hints", []) or [],
        "query_expansion_terms": data.get("query_expansion_terms", []) or [],
    }
    # Ensure types
    if not isinstance(result["law_hints"], list):
        result["law_hints"] = [str(result["law_hints"])]
    if not isinstance(result["query_expansion_terms"], list):
        result["query_expansion_terms"] = [str(result["query_expansion_terms"])]
    return result


@lru_cache(maxsize=512)
def get_prior_cached(product_description: str,
                     model: str = "gpt-4o-mini",
                     temperature: float = 0.2) -> Dict:
    """Cached wrapper around get_prior to avoid repeat LLM calls for the same input."""
    return get_prior(product_description, model=model, temperature=temperature)


