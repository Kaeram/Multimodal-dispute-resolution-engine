"""
Pydantic models for request/response validation.

This module exports all request and response models used by the API.
"""

from app.models.requests import (
    BatchClassificationRequest,
    CaseAnalysisRequest,
    EvidenceItem,
    ImageEvidenceRequest,
    IssueClassificationRequest,
    PropertyHistory,
)
from app.models.responses import (
    AlignmentAnalysis,
    CleanlinessLevelEnum,
    DAORecommendation,
    ErrorResponse,
    EvidenceEvaluation,
    EvidenceValidation,
    EXIFData,
    FraudAnalysis,
    HealthCheck,
    ImageEvidenceAnalysis,
    IssueClassification,
    LandlordPosition,
    LocationTypeEnum,
    MultimodalVerdict,
    PositionAnalysis,
    RedFlags,
    TamperAnalysis,
    TenantPosition,
    VisionAnalysis,
)

__all__ = [
    # Requests
    "IssueClassificationRequest",
    "CaseAnalysisRequest",
    "BatchClassificationRequest",
    "EvidenceItem",
    "PropertyHistory",
    "ImageEvidenceRequest",
    # Responses
    "IssueClassification",
    "EvidenceValidation",
    "EXIFData",
    "TamperAnalysis",
    "AlignmentAnalysis",
    "DAORecommendation",
    "FraudAnalysis",
    "HealthCheck",
    "ErrorResponse",
    "TenantPosition",
    "LandlordPosition",
    "PositionAnalysis",
    "EvidenceEvaluation",
    "RedFlags",
    # Vision Analysis
    "VisionAnalysis",
    "MultimodalVerdict",
    "ImageEvidenceAnalysis",
    "CleanlinessLevelEnum",
    "LocationTypeEnum",
]
