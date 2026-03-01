"""
Pydantic response models for RentShield AI Analysis Engine API.

All response models include detailed field descriptions and examples
for automatic OpenAPI documentation generation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class IssueCategoryEnum(str, Enum):
    """Valid issue categories for classification."""
    SAFETY = "Safety"
    MAINTENANCE = "Maintenance"
    HARASSMENT = "Harassment"
    DISCRIMINATION = "Discrimination"


class SeverityEnum(str, Enum):
    """Severity levels for issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceQualityEnum(str, Enum):
    """Evidence quality ratings."""
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    STRONG = "strong"


class ConfidenceLevelEnum(str, Enum):
    """Confidence levels for recommendations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RecommendedOutcomeEnum(str, Enum):
    """Possible recommended outcomes for disputes."""
    FAVOR_TENANT = "Favor Tenant"
    FAVOR_LANDLORD = "Favor Landlord"
    MEDIATION_REQUIRED = "Mediation Required"
    INSUFFICIENT_EVIDENCE = "Insufficient Evidence"


class TrustLevelEnum(str, Enum):
    """Trust levels for evidence validation."""
    HIGH_TRUST = "HIGH TRUST"
    MEDIUM_TRUST = "MEDIUM TRUST"
    LOW_TRUST = "LOW TRUST"
    UNTRUSTED = "UNTRUSTED"


# ============================================================================
# Issue Classification Models
# ============================================================================

class IssueClassification(BaseModel):
    """
    Result of AI-powered issue classification.
    
    Example:
        >>> classification = IssueClassification(
        ...     primary_category="Maintenance",
        ...     confidence=85,
        ...     severity="medium",
        ...     reasoning="Issue involves plumbing repairs"
        ... )
    """
    
    primary_category: IssueCategoryEnum = Field(
        ...,
        description="Primary issue category",
        json_schema_extra={"example": "Maintenance"},
    )
    confidence: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence score for classification (0-100)",
        json_schema_extra={"example": 85},
    )
    secondary_category: Optional[IssueCategoryEnum] = Field(
        default=None,
        description="Secondary category if applicable",
        json_schema_extra={"example": "Safety"},
    )
    severity: SeverityEnum = Field(
        ...,
        description="Severity level of the issue",
        json_schema_extra={"example": "medium"},
    )
    reasoning: str = Field(
        ...,
        description="Explanation for the classification",
        json_schema_extra={"example": "Issue involves water damage which is a maintenance concern with potential safety implications."},
    )
    urgency_flag: bool = Field(
        default=False,
        description="Whether issue requires immediate attention",
        json_schema_extra={"example": False},
    )
    keywords_detected: list[str] = Field(
        default_factory=list,
        description="Key terms identified in the description",
        json_schema_extra={"example": ["water leak", "mold", "ceiling"]},
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request ID for tracing",
    )


# ============================================================================
# Evidence Validation Models
# ============================================================================

class EXIFData(BaseModel):
    """
    Extracted EXIF metadata from an image.
    
    Example:
        >>> exif = EXIFData(
        ...     datetime_original="2024:01:15 14:30:00",
        ...     device_make="Apple",
        ...     device_model="iPhone 13"
        ... )
    """
    
    datetime_original: Optional[str] = Field(
        default=None,
        description="Original capture timestamp",
        json_schema_extra={"example": "2024:01:15 14:30:00"},
    )
    device_make: Optional[str] = Field(
        default=None,
        description="Camera/device manufacturer",
        json_schema_extra={"example": "Apple"},
    )
    device_model: Optional[str] = Field(
        default=None,
        description="Camera/device model",
        json_schema_extra={"example": "iPhone 13 Pro"},
    )
    gps_latitude: Optional[float] = Field(
        default=None,
        description="GPS latitude coordinate",
        json_schema_extra={"example": 37.7749},
    )
    gps_longitude: Optional[float] = Field(
        default=None,
        description="GPS longitude coordinate",
        json_schema_extra={"example": -122.4194},
    )
    software: Optional[str] = Field(
        default=None,
        description="Software used to create/edit image",
        json_schema_extra={"example": "16.2"},
    )
    file_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of the file",
        json_schema_extra={"example": "a1b2c3d4e5f6..."},
    )
    dimensions: Optional[str] = Field(
        default=None,
        description="Image dimensions (WxH)",
        json_schema_extra={"example": "4032x3024"},
    )
    file_size: Optional[int] = Field(
        default=None,
        description="File size in bytes",
        json_schema_extra={"example": 2457600},
    )


class TamperAnalysis(BaseModel):
    """
    Analysis of potential image tampering.
    
    Example:
        >>> tamper = TamperAnalysis(
        ...     tamper_probability=0.15,
        ...     indicators=["No editing software detected"],
        ...     conclusion="Low risk of tampering"
        ... )
    """
    
    tamper_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of tampering (0.0-1.0)",
        json_schema_extra={"example": 0.15},
    )
    indicators: list[str] = Field(
        default_factory=list,
        description="Detected tampering indicators",
        json_schema_extra={"example": ["Consistent EXIF data", "No editing software watermarks"]},
    )
    conclusion: str = Field(
        ...,
        description="Summary conclusion about image authenticity",
        json_schema_extra={"example": "Low risk of tampering detected"},
    )


class AlignmentAnalysis(BaseModel):
    """
    Analysis of how well evidence aligns with claims.
    
    Example:
        >>> alignment = AlignmentAnalysis(
        ...     alignment_score=82,
        ...     reasoning="Timestamp matches reported incident date"
        ... )
    """
    
    alignment_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Score indicating alignment between evidence and claims (0-100)",
        json_schema_extra={"example": 82},
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Identified concerns or inconsistencies",
        json_schema_extra={"example": ["Photo taken 3 days after reported incident"]},
    )
    reasoning: str = Field(
        ...,
        description="Detailed explanation of alignment assessment",
        json_schema_extra={"example": "The image timestamp aligns with the reported incident date and visible damage is consistent with the described water leak."},
    )


class EvidenceValidation(BaseModel):
    """
    Complete evidence validation result.
    
    Example:
        >>> validation = EvidenceValidation(
        ...     authenticity_score=75,
        ...     trust_level="MEDIUM TRUST",
        ...     exif_data=EXIFData(),
        ...     tamper_analysis=TamperAnalysis(...),
        ...     alignment_analysis=AlignmentAnalysis(...)
        ... )
    """
    
    authenticity_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall authenticity score (0-100)",
        json_schema_extra={"example": 75},
    )
    trust_level: TrustLevelEnum = Field(
        ...,
        description="Trust level classification",
        json_schema_extra={"example": "MEDIUM TRUST"},
    )
    exif_data: EXIFData = Field(
        ...,
        description="Extracted EXIF metadata",
    )
    tamper_analysis: TamperAnalysis = Field(
        ...,
        description="Tampering detection results",
    )
    alignment_analysis: Optional[AlignmentAnalysis] = Field(
        default=None,
        description="Evidence-claim alignment analysis (if claim provided)",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request ID for tracing",
    )


# ============================================================================
# DAO Recommendation Models
# ============================================================================

class PositionAnalysis(BaseModel):
    """Base model for party position analysis."""
    
    key_arguments: list[str] = Field(
        default_factory=list,
        description="Main arguments presented",
    )
    evidence_strength: int = Field(
        ...,
        ge=0,
        le=100,
        description="Strength of supporting evidence (0-100)",
    )
    credibility_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall credibility score (0-100)",
    )
    supporting_factors: list[str] = Field(
        default_factory=list,
        description="Factors supporting this position",
    )


class TenantPosition(PositionAnalysis):
    """Analysis of tenant's position in the dispute."""
    pass


class LandlordPosition(PositionAnalysis):
    """Analysis of landlord's position in the dispute."""
    pass


class EvidenceEvaluation(BaseModel):
    """Evaluation of evidence quality from both parties."""
    
    tenant_evidence_quality: EvidenceQualityEnum = Field(
        ...,
        description="Quality rating of tenant's evidence",
        json_schema_extra={"example": "good"},
    )
    landlord_evidence_quality: EvidenceQualityEnum = Field(
        ...,
        description="Quality rating of landlord's evidence",
        json_schema_extra={"example": "fair"},
    )
    metadata_authenticity: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall authenticity score of all metadata (0-100)",
        json_schema_extra={"example": 78},
    )
    key_discrepancies: list[str] = Field(
        default_factory=list,
        description="Notable discrepancies found",
    )
    critical_gaps: list[str] = Field(
        default_factory=list,
        description="Critical missing evidence or information",
    )


class DAORecommendationDetails(BaseModel):
    """Detailed DAO voting recommendation."""
    
    tenant_favor_confidence: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence in ruling for tenant (0-100)",
        json_schema_extra={"example": 72},
    )
    landlord_favor_confidence: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence in ruling for landlord (0-100)",
        json_schema_extra={"example": 15},
    )
    neutral_confidence: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence in neutral/mediation outcome (0-100)",
        json_schema_extra={"example": 13},
    )
    recommended_outcome: RecommendedOutcomeEnum = Field(
        ...,
        description="Recommended resolution",
        json_schema_extra={"example": "Favor Tenant"},
    )
    confidence_level: ConfidenceLevelEnum = Field(
        ...,
        description="Overall confidence in recommendation",
        json_schema_extra={"example": "high"},
    )
    reasoning: str = Field(
        ...,
        description="Detailed reasoning for recommendation",
    )
    key_considerations: list[str] = Field(
        default_factory=list,
        description="Key factors considered in decision",
    )
    suggested_resolution: str = Field(
        ...,
        description="Specific suggested resolution action",
    )


class RedFlags(BaseModel):
    """Red flags detected for all parties."""
    
    tenant_concerns: list[str] = Field(
        default_factory=list,
        description="Concerns about tenant's case",
    )
    landlord_concerns: list[str] = Field(
        default_factory=list,
        description="Concerns about landlord's response",
    )
    evidence_concerns: list[str] = Field(
        default_factory=list,
        description="Concerns about evidence quality/authenticity",
    )


class DAORecommendation(BaseModel):
    """
    Complete DAO recommendation for a dispute case.
    
    This is the comprehensive output from case analysis.
    """
    
    case_summary: str = Field(
        ...,
        description="2-3 sentence neutral overview of the case",
    )
    tenant_position: TenantPosition = Field(
        ...,
        description="Analysis of tenant's position",
    )
    landlord_position: LandlordPosition = Field(
        ...,
        description="Analysis of landlord's position",
    )
    evidence_evaluation: EvidenceEvaluation = Field(
        ...,
        description="Evaluation of all evidence",
    )
    dao_recommendation: DAORecommendationDetails = Field(
        ...,
        description="DAO voting recommendation",
    )
    red_flags: RedFlags = Field(
        ...,
        description="Detected red flags",
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="Recommended next steps",
    )
    estimated_resolution_timeline: str = Field(
        ...,
        description="Estimated time to resolution",
        json_schema_extra={"example": "7-14 days"},
    )
    vision_analyses: Optional[list["ImageEvidenceAnalysis"]] = Field(
        default=None,
        description="LLaVA vision analysis results for evidence images (if enabled)",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request ID for tracing",
    )


# ============================================================================
# Fraud Analysis Models
# ============================================================================

class FraudAnalysis(BaseModel):
    """
    Fraud pattern detection results.
    
    Example:
        >>> fraud = FraudAnalysis(
        ...     fraud_risk_score=25,
        ...     indicators=[],
        ...     conclusion="Low fraud risk"
        ... )
    """
    
    fraud_risk_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall fraud risk score (0-100)",
        json_schema_extra={"example": 25},
    )
    indicators: list[str] = Field(
        default_factory=list,
        description="Detected fraud indicators",
    )
    duplicate_evidence: bool = Field(
        default=False,
        description="Whether duplicate evidence was detected",
    )
    timeline_inconsistencies: bool = Field(
        default=False,
        description="Whether timeline inconsistencies were found",
    )
    manipulation_detected: bool = Field(
        default=False,
        description="Whether emotional manipulation was detected",
    )
    conclusion: str = Field(
        ...,
        description="Summary conclusion about fraud risk",
    )


# ============================================================================
# Health Check & Error Models
# ============================================================================

class HealthCheck(BaseModel):
    """
    Health check response for the service.
    
    Example:
        >>> health = HealthCheck(
        ...     status="healthy",
        ...     llm_connected=True,
        ...     version="1.0.0"
        ... )
    """
    
    status: str = Field(
        ...,
        description="Service status",
        json_schema_extra={"example": "healthy"},
    )
    llm_connected: bool = Field(
        ...,
        description="Whether Ollama LLM is reachable",
        json_schema_extra={"example": True},
    )
    version: str = Field(
        ...,
        description="API version",
        json_schema_extra={"example": "1.0.0"},
    )
    model_available: Optional[str] = Field(
        default=None,
        description="Available LLM model name",
        json_schema_extra={"example": "mistral"},
    )


class ErrorResponse(BaseModel):
    """
    Standard error response format.
    
    Example:
        >>> error = ErrorResponse(
        ...     error="validation_error",
        ...     message="Invalid input data",
        ...     request_id="abc123"
        ... )
    """
    
    error: str = Field(
        ...,
        description="Error type identifier",
        json_schema_extra={"example": "validation_error"},
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
        json_schema_extra={"example": "Invalid input: description is required"},
    )
    details: Optional[dict] = Field(
        default=None,
        description="Additional error details",
    )
    request_id: str = Field(
        ...,
        description="Request ID for tracing",
        json_schema_extra={"example": "req-abc123-def456"},
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp",
    )


# ============================================================================
# Batch Response Models
# ============================================================================

class BatchClassificationResponse(BaseModel):
    """Response for batch classification endpoint."""
    
    results: list[IssueClassification] = Field(
        ...,
        description="Classification results for each issue",
    )
    total_processed: int = Field(
        ...,
        description="Total number of issues processed",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request ID for tracing",
    )


# ============================================================================
# Vision Analysis Models (LLaVA Pipeline)
# ============================================================================

class CleanlinessLevelEnum(str, Enum):
    """Cleanliness levels for image analysis."""
    CLEAN = "clean"
    AVERAGE = "average"
    DIRTY = "dirty"
    UNSANITARY = "unsanitary"


class LocationTypeEnum(str, Enum):
    """Location type classification."""
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    UNCLEAR = "unclear"


class VisionAnalysis(BaseModel):
    """
    LLaVA image analysis result.
    
    Contains structured scene description focusing on housing-related
    damage, safety hazards, and maintenance issues.
    
    Example:
        >>> analysis = VisionAnalysis(
        ...     scene_summary="Kitchen ceiling shows water damage with visible mold growth",
        ...     detected_objects=["ceiling", "water stain", "mold"],
        ...     damage_detected=["water damage", "mold growth"],
        ...     safety_hazards=["potential mold spores"],
        ...     cleanliness_level="dirty",
        ...     indoor_outdoor="indoor",
        ...     confidence=85
        ... )
    """
    
    scene_summary: str = Field(
        ...,
        description="2-sentence description of the scene",
        json_schema_extra={"example": "Kitchen ceiling showing significant water damage with visible mold growth. Dark staining indicates prolonged water exposure."},
    )
    detected_objects: list[str] = Field(
        default_factory=list,
        description="Objects detected in the image",
        json_schema_extra={"example": ["ceiling", "water stain", "mold", "light fixture"]},
    )
    damage_detected: list[str] = Field(
        default_factory=list,
        description="Types of damage identified",
        json_schema_extra={"example": ["water damage", "mold growth", "paint peeling"]},
    )
    safety_hazards: list[str] = Field(
        default_factory=list,
        description="Safety hazards identified",
        json_schema_extra={"example": ["mold spores", "electrical hazard near water"]},
    )
    cleanliness_level: CleanlinessLevelEnum = Field(
        ...,
        description="Overall cleanliness assessment",
        json_schema_extra={"example": "dirty"},
    )
    indoor_outdoor: LocationTypeEnum = Field(
        ...,
        description="Location type classification",
        json_schema_extra={"example": "indoor"},
    )
    confidence: int = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence score for the analysis (0-100)",
        json_schema_extra={"example": 85},
    )


class MultimodalVerdict(BaseModel):
    """
    Mistral multimodal reasoning result.
    
    Compares tenant claim, image analysis, and EXIF metadata
    to determine evidence consistency and fraud risk.
    
    Example:
        >>> verdict = MultimodalVerdict(
        ...     image_supports_claim=True,
        ...     consistency_score=82,
        ...     evidence_strength=75,
        ...     detected_inconsistencies=[],
        ...     possible_fraud_signals=[],
        ...     reasoning="Image shows water damage consistent with tenant's claim"
        ... )
    """
    
    image_supports_claim: bool = Field(
        ...,
        description="Whether the image supports the tenant's claim",
        json_schema_extra={"example": True},
    )
    consistency_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Score indicating consistency between claim and evidence (0-100)",
        json_schema_extra={"example": 82},
    )
    evidence_strength: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall strength of the evidence (0-100)",
        json_schema_extra={"example": 75},
    )
    detected_inconsistencies: list[str] = Field(
        default_factory=list,
        description="Inconsistencies found between claim and evidence",
        json_schema_extra={"example": ["Photo timestamp is 3 days after reported incident"]},
    )
    possible_fraud_signals: list[str] = Field(
        default_factory=list,
        description="Potential fraud indicators detected",
        json_schema_extra={"example": []},
    )
    reasoning: str = Field(
        ...,
        description="Short explanation of the verdict",
        json_schema_extra={"example": "The image clearly shows water damage on the ceiling consistent with the tenant's description of a leak."},
    )


class ImageEvidenceAnalysis(BaseModel):
    """
    Complete multimodal evidence analysis result.
    
    Combines EXIF metadata analysis, LLaVA vision analysis,
    and Mistral multimodal reasoning for comprehensive evidence verification.
    
    Example:
        >>> analysis = ImageEvidenceAnalysis(
        ...     exif_analysis=EXIFData(...),
        ...     vision_analysis=VisionAnalysis(...),
        ...     multimodal_verdict=MultimodalVerdict(...),
        ...     final_evidence_score=78,
        ...     trust_level=TrustLevelEnum.MEDIUM_TRUST
        ... )
    """
    
    exif_analysis: EXIFData = Field(
        ...,
        description="Extracted EXIF metadata from the image",
    )
    vision_analysis: VisionAnalysis = Field(
        ...,
        description="LLaVA image understanding results",
    )
    multimodal_verdict: MultimodalVerdict = Field(
        ...,
        description="Mistral multimodal reasoning verdict",
    )
    final_evidence_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Final weighted evidence score (0-100)",
        json_schema_extra={"example": 78},
    )
    trust_level: TrustLevelEnum = Field(
        ...,
        description="Trust level classification based on final score",
        json_schema_extra={"example": "MEDIUM TRUST"},
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Unique request ID for tracing",
    )


# Rebuild models to resolve forward references
DAORecommendation.model_rebuild()
