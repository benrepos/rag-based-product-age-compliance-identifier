"""
Prompt templates for LLM-based age restriction classification
"""

from typing import List, Dict


def build_classification_prompt(product_description: str, chunks: List[Dict]) -> str:
    """
    Build the classification prompt with product description and retrieved law chunks
    
    Args:
        product_description: Description of the product to classify
        chunks: List of relevant law chunks with metadata
        
    Returns:
        str: Formatted prompt for the LLM
    """
    # Format evidence from chunks
    evidence_sections = []
    for i, chunk in enumerate(chunks, 1):
        law_name = chunk.get('law_name', 'Unknown Law')
        section = chunk.get('section', 'Unknown')
        chunk_text = chunk.get('chunk_text', '')
        score = chunk.get('similarity_score', 0)
        
        # Truncate very long chunks
        if len(chunk_text) > 800:
            chunk_text = chunk_text[:800] + "..."
        
        evidence_sections.append(
            f"[Evidence {i}] {law_name} - Section {section} (relevance: {score:.2f})\n{chunk_text}\n"
        )
    
    evidence_text = "\n".join(evidence_sections)
    
    prompt = f"""You are a UK product compliance assistant. Determine if the product is age restricted by CROSS-CHECKING the product description against the legal evidence provided.

Product Description:
{product_description}

Legal Evidence:
{evidence_text}

Instructions:
1. Analyse the product description against the legal evidence; do not ignore product specifics (e.g., whether it actually has a blade).
2. Determine if any age restriction applies; if the evidence is not directly applicable to the product, answer "None".
3. Only cite laws that are actually relevant AND explicitly cover the product type:
   - "Bladed product" requires a blade capable of cutting (e.g., knives, razors)
   - Pointed tools (e.g., screwdrivers) and impact tools (e.g., hammers, mallets, wrenches) are NOT bladed products
   - "Bladed article/product" restrictions for remote sale/delivery do not themselves create a blanket age prohibition for all items; rely on provisions that explicitly prohibit sale to under 18 (e.g., knives, knife blades, razor blades, axes under CJA 1988 s.141A) unless the evidence clearly extends the scope
4. Before deciding 18+, complete this checklist using the evidence text (quote short phrases where possible):
   - Does the statute explicitly list the item (knife, knife blade, razor blade, axe)? If yes, cite section
   - If not listed: does the statutory definition clearly fit this product? Explain why/why not
   - Is the section about remote sale/delivery logistics rather than a sale-to-under-18 offence?
5. Be confident if the evidence is clear, less confident if ambiguous; do not over-generalise.

Respond in JSON format with exactly these fields:
{{
  "restriction_level": "None" | "16+" | "18+" | "Licensed",
  "confidence": 0.0-1.0,
  "reason": "Brief explanation of your reasoning",
  "evidence": ["List of law names used in your decision"]
}}"""
    
    return prompt


SYSTEM_PROMPT = """You are a UK legal compliance expert specializing in age-restricted products. 
Your role is to analyze product descriptions against UK legislation to determine age restrictions.
You must base your answers strictly on the provided legal evidence.
Always respond with valid JSON only, no additional text."""

