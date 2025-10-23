"""
Pydantic models for age restriction classification outputs
"""

from typing import List, Literal
from pydantic import BaseModel, Field, field_validator


class ClassificationResult(BaseModel):
    """Age restriction classification result"""
    
    restriction_level: Literal["None", "16+", "18+", "Licensed"] = Field(
        description="The age restriction level for the product"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the classification (0.0 to 1.0)"
    )
    reason: str = Field(
        min_length=10,
        description="Brief explanation of the reasoning behind the classification"
    )
    evidence: List[str] = Field(
        description="List of law names used in the decision"
    )
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence is between 0 and 1"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class ClassificationMetadata(BaseModel):
    """Metadata about the classification process"""
    # Allow fields starting with `model_` such as `model_used`
    model_config = {"protected_namespaces": ()}
    
    num_chunks_used: int = Field(
        description="Number of law chunks retrieved and used"
    )
    top_similarity_score: float = Field(
        description="Highest similarity score from retrieval"
    )
    model_used: str = Field(
        description="LLM model used for classification"
    )
    laws_retrieved: List[str] = Field(
        description="List of all laws retrieved from the database"
    )


class ClassificationResponse(BaseModel):
    """Complete classification response with metadata"""
    
    product_description: str = Field(
        description="The product description that was classified"
    )
    classification: ClassificationResult = Field(
        description="The classification result"
    )
    metadata: ClassificationMetadata = Field(
        description="Metadata about the classification process"
    )

