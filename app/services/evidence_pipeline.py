"""
Evidence Pipeline for RentShield AI Analysis Engine.

Provides multimodal evidence verification combining:
- VisionService (LLaVA) for image understanding
- EvidenceValidator for EXIF metadata analysis
- OllamaLLMService (Mistral) for multimodal reasoning
"""

import json
import os
from datetime import datetime
from typing import Any, Optional

from app.config import get_settings
from app.models.responses import (
    CleanlinessLevelEnum,
    EXIFData,
    ImageEvidenceAnalysis,
    LocationTypeEnum,
    MultimodalVerdict,
    TrustLevelEnum,
    VisionAnalysis,
)
from app.services.evidence_validator import EvidenceValidator
from app.services.llm_service import OllamaLLMService
from app.services.vision_service import VisionService
from app.utils.exceptions import AnalysisError, InvalidImageError
from app.utils.logger import get_logger

logger = get_logger(__name__)


# Multimodal reasoning prompt for Mistral
MULTIMODAL_PROMPT_TEMPLATE = """You are an evidence verification AI for housing disputes.

Analyze the following evidence and determine if the image supports the tenant's claim.

TENANT CLAIM:
{claim_text}

IMAGE ANALYSIS (from vision AI):
{llava_json}

EXIF METADATA:
{exif_json}

{date_context}

Based on this information, provide your verdict. Consider:
1. Does the image content match what the tenant claims?
2. Are there any inconsistencies between the claim and the image?
3. Does the EXIF metadata (timestamp, location) support or contradict the claim?
4. Are there any signs of potential fraud or manipulation?

Return STRICT JSON with no additional text:

{{
  "image_supports_claim": true/false,
  "consistency_score": 0-100,
  "evidence_strength": 0-100,
  "detected_inconsistencies": [],
  "possible_fraud_signals": [],
  "reasoning": "short explanation"
}}"""


class EvidencePipeline:
    """
    Multimodal evidence verification pipeline.
    
    Orchestrates the complete evidence analysis workflow:
    1. Download image from URL
    2. Extract EXIF metadata
    3. Analyze image with LLaVA
    4. Perform multimodal reasoning with Mistral
    5. Calculate final evidence score
    
    Attributes:
        vision_service: LLaVA-based image understanding service.
        evidence_validator: EXIF metadata extractor and validator.
        llm_service: Mistral LLM for multimodal reasoning.
        
    Example:
        >>> pipeline = EvidencePipeline()
        >>> result = pipeline.analyze_evidence(
        ...     image_url="https://storage.example.com/image.jpg",
        ...     claim_text="Water damage in kitchen ceiling"
        ... )
        >>> print(result.final_evidence_score)
        78
    """
    
    def __init__(
        self,
        vision_service: Optional[VisionService] = None,
        evidence_validator: Optional[EvidenceValidator] = None,
        llm_service: Optional[OllamaLLMService] = None,
    ) -> None:
        """
        Initialize the evidence pipeline.
        
        Args:
            vision_service: Optional custom VisionService instance.
            evidence_validator: Optional custom EvidenceValidator instance.
            llm_service: Optional custom OllamaLLMService instance.
        """
        self.vision_service = vision_service or VisionService()
        self.evidence_validator = evidence_validator or EvidenceValidator()
        self.llm_service = llm_service or OllamaLLMService()
        
        logger.info("EvidencePipeline initialized")
    
    def analyze_evidence(
        self,
        image_url: str,
        claim_text: str,
        incident_date: Optional[str] = None,
    ) -> ImageEvidenceAnalysis:
        """
        Perform complete multimodal evidence analysis.
        
        Pipeline steps:
        1. Download image from URL
        2. Extract EXIF metadata
        3. Analyze image with LLaVA
        4. Perform multimodal reasoning with Mistral
        5. Calculate final weighted score
        
        Args:
            image_url: URL of the evidence image.
            claim_text: Tenant's claim or complaint text.
            incident_date: Optional ISO datetime of the incident.
            
        Returns:
            ImageEvidenceAnalysis: Complete analysis result.
            
        Raises:
            InvalidImageError: If image download or validation fails.
            AnalysisError: If analysis pipeline fails.
            
        Example:
            >>> pipeline = EvidencePipeline()
            >>> result = pipeline.analyze_evidence(
            ...     image_url="https://example.com/damage.jpg",
            ...     claim_text="Water leak causing mold growth",
            ...     incident_date="2024-01-15T00:00:00Z"
            ... )
        """
        temp_path = None
        
        try:
            logger.info(
                "Starting evidence analysis pipeline",
                url=image_url[:100],
                claim_length=len(claim_text),
            )
            
            # Step 1: Download image
            logger.info("Step 1: Downloading image")
            temp_path = self.vision_service.download_image(image_url)
            
            # Step 2: Extract EXIF metadata
            logger.info("Step 2: Extracting EXIF metadata")
            exif_data = self._extract_exif_safe(temp_path)
            exif_authenticity = self.evidence_validator.calculate_authenticity_score(exif_data)
            
            # Step 3: Analyze image with LLaVA
            logger.info("Step 3: Analyzing image with LLaVA")
            llava_result = self.vision_service.analyze_image(temp_path)
            vision_analysis = self._build_vision_analysis(llava_result)
            
            # Step 4: Multimodal reasoning with Mistral
            logger.info("Step 4: Performing multimodal reasoning with Mistral")
            multimodal_verdict = self._perform_multimodal_reasoning(
                claim_text=claim_text,
                llava_result=llava_result,
                exif_data=exif_data,
                incident_date=incident_date,
            )
            
            # Step 5: Calculate final score
            final_score = self._calculate_final_score(
                exif_authenticity=exif_authenticity,
                llava_confidence=vision_analysis.confidence,
                consistency_score=multimodal_verdict.consistency_score,
            )
            
            trust_level = self.evidence_validator.get_trust_level(final_score)
            
            logger.info(
                "Evidence analysis completed",
                final_score=final_score,
                trust_level=trust_level.value,
            )
            
            return ImageEvidenceAnalysis(
                exif_analysis=exif_data,
                vision_analysis=vision_analysis,
                multimodal_verdict=multimodal_verdict,
                final_evidence_score=final_score,
                trust_level=trust_level,
            )
            
        except InvalidImageError:
            raise
        except Exception as e:
            logger.error("Evidence analysis pipeline failed", error=str(e))
            raise AnalysisError(f"Evidence analysis failed: {str(e)}")
        finally:
            # Cleanup temp file
            if temp_path:
                self.vision_service.cleanup_temp_file(temp_path)
    
    def _extract_exif_safe(self, image_path: str) -> EXIFData:
        """
        Extract EXIF data with error handling.
        
        Returns empty EXIFData if extraction fails.
        """
        try:
            return self.evidence_validator.extract_exif(image_path)
        except Exception as e:
            logger.warning("EXIF extraction failed", error=str(e))
            return EXIFData()
    
    def _build_vision_analysis(self, llava_result: dict[str, Any]) -> VisionAnalysis:
        """
        Build VisionAnalysis model from LLaVA response.
        
        Handles missing or invalid values with sensible defaults.
        """
        # Parse cleanliness level
        cleanliness_raw = llava_result.get("cleanliness_level", "average").lower()
        try:
            cleanliness = CleanlinessLevelEnum(cleanliness_raw)
        except ValueError:
            cleanliness = CleanlinessLevelEnum.AVERAGE
        
        # Parse indoor/outdoor
        location_raw = llava_result.get("indoor_outdoor", "unclear").lower()
        try:
            location = LocationTypeEnum(location_raw)
        except ValueError:
            location = LocationTypeEnum.UNCLEAR
        
        # Parse confidence
        confidence = llava_result.get("confidence", 50)
        if not isinstance(confidence, int):
            try:
                confidence = int(confidence)
            except (ValueError, TypeError):
                confidence = 50
        confidence = max(0, min(100, confidence))
        
        return VisionAnalysis(
            scene_summary=llava_result.get("scene_summary", "Unable to analyze image"),
            detected_objects=llava_result.get("detected_objects", []),
            damage_detected=llava_result.get("damage_detected", []),
            safety_hazards=llava_result.get("safety_hazards", []),
            cleanliness_level=cleanliness,
            indoor_outdoor=location,
            confidence=confidence,
        )
    
    def _perform_multimodal_reasoning(
        self,
        claim_text: str,
        llava_result: dict[str, Any],
        exif_data: EXIFData,
        incident_date: Optional[str] = None,
    ) -> MultimodalVerdict:
        """
        Perform multimodal reasoning using Mistral.
        
        Compares the tenant claim, LLaVA analysis, and EXIF metadata
        to determine evidence consistency and fraud risk.
        """
        # Build date context if provided
        date_context = ""
        if incident_date:
            date_context = f"REPORTED INCIDENT DATE: {incident_date}"
        
        # Prepare EXIF JSON (only include non-null fields)
        exif_dict = {k: v for k, v in exif_data.model_dump().items() if v is not None}
        
        # Build prompt
        prompt = MULTIMODAL_PROMPT_TEMPLATE.format(
            claim_text=claim_text,
            llava_json=json.dumps(llava_result, indent=2),
            exif_json=json.dumps(exif_dict, indent=2),
            date_context=date_context,
        )
        
        try:
            result = self.llm_service.query(
                prompt=prompt,
                expect_json=True,
                timeout=60,
            )
            
            return self._build_multimodal_verdict(result)
            
        except Exception as e:
            logger.warning("Multimodal reasoning failed, using defaults", error=str(e))
            return MultimodalVerdict(
                image_supports_claim=False,
                consistency_score=50,
                evidence_strength=50,
                detected_inconsistencies=["Unable to perform full analysis"],
                possible_fraud_signals=[],
                reasoning=f"Analysis incomplete: {str(e)[:100]}",
            )
    
    def _build_multimodal_verdict(self, llm_result: dict[str, Any]) -> MultimodalVerdict:
        """
        Build MultimodalVerdict model from Mistral response.
        
        Handles missing or invalid values with sensible defaults.
        """
        # Parse boolean
        supports_claim = llm_result.get("image_supports_claim", False)
        if not isinstance(supports_claim, bool):
            supports_claim = str(supports_claim).lower() in ("true", "yes", "1")
        
        # Parse scores with bounds checking
        def parse_score(value: Any, default: int = 50) -> int:
            if not isinstance(value, (int, float)):
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    return default
            return max(0, min(100, int(value)))
        
        consistency = parse_score(llm_result.get("consistency_score", 50))
        strength = parse_score(llm_result.get("evidence_strength", 50))
        
        return MultimodalVerdict(
            image_supports_claim=supports_claim,
            consistency_score=consistency,
            evidence_strength=strength,
            detected_inconsistencies=llm_result.get("detected_inconsistencies", []),
            possible_fraud_signals=llm_result.get("possible_fraud_signals", []),
            reasoning=llm_result.get("reasoning", "No reasoning provided"),
        )
    
    def _calculate_final_score(
        self,
        exif_authenticity: int,
        llava_confidence: int,
        consistency_score: int,
    ) -> int:
        """
        Calculate final weighted evidence score.
        
        Formula:
            final_score = 0.3 * exif_authenticity + 0.3 * llava_confidence + 0.4 * consistency_score
            
        Args:
            exif_authenticity: EXIF-based authenticity score (0-100).
            llava_confidence: LLaVA confidence score (0-100).
            consistency_score: Multimodal consistency score (0-100).
            
        Returns:
            int: Final weighted score (0-100).
        """
        score = (
            0.3 * exif_authenticity +
            0.3 * llava_confidence +
            0.4 * consistency_score
        )
        
        return max(0, min(100, int(round(score))))
