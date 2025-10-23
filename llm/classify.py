"""
LLM-based classification module for age restriction determination
Uses Pydantic for structured output validation
"""

import os
from typing import Dict, List
from openai import OpenAI

from .prompts import build_classification_prompt, SYSTEM_PROMPT
from .models import ClassificationResult, ClassificationResponse, ClassificationMetadata


class AgeRestrictionClassifier:
    """Uses LLM to classify products based on retrieved law chunks"""
    
    def __init__(self, model: str = "gpt-5-nano", temperature: float = 0.1):
        """
        Initialize the classifier
        
        Args:
            model: OpenAI model to use (gpt-5-nano recommended)
            temperature: Temperature for generation (lower = more deterministic)
        """
        self.model = model
        self.temperature = temperature
        
        # Initialize OpenAI client
        self.client = self._create_openai_client()
        
        print(f"✓ Initialized classifier with {model}")
    
    def _create_openai_client(self) -> OpenAI:
        """Create OpenAI client with API key from environment"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it using: export OPENAI_API_KEY='your-api-key'"
            )
        return OpenAI(api_key=api_key)
    
    def classify(self, product_description: str, relevant_chunks: List[Dict]) -> ClassificationResult:
        """
        Classify a product's age restriction based on retrieved law chunks
        
        Args:
            product_description: Description of the product
            relevant_chunks: List of relevant law chunks from retriever
            
        Returns:
            ClassificationResult: Pydantic model with classification results
        """
        # Build the prompt
        user_prompt = build_classification_prompt(product_description, relevant_chunks)
        
        # Call the LLM with structured output
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            response_format=ClassificationResult
        )
        
        # Extract the parsed Pydantic model
        result = response.choices[0].message.parsed
        
        return result
    
    def classify_with_metadata(self, 
                              product_description: str, 
                              relevant_chunks: List[Dict]) -> ClassificationResponse:
        """
        Classify a product and include additional metadata
        
        Args:
            product_description: Description of the product
            relevant_chunks: List of relevant law chunks from retriever
            
        Returns:
            ClassificationResponse: Pydantic model with classification and metadata
        """
        # Get the classification
        classification = self.classify(product_description, relevant_chunks)
        
        # Build metadata
        metadata = ClassificationMetadata(
            num_chunks_used=len(relevant_chunks),
            top_similarity_score=relevant_chunks[0].get('similarity_score', 0) if relevant_chunks else 0,
            model_used=self.model,
            laws_retrieved=list(set(chunk.get('law_name', '') for chunk in relevant_chunks))
        )
        
        # Create the full response
        response = ClassificationResponse(
            product_description=product_description,
            classification=classification,
            metadata=metadata
        )
        
        return response

