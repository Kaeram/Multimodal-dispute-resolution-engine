"""
Case Analyzer for RentShield AI Analysis Engine.

Provides AI-powered issue classification, dispute analysis, and
DAO recommendation generation for housing disputes.
"""

from datetime import datetime
from typing import Any, Optional

from app.models.requests import CaseAnalysisRequest
from app.models.responses import (
    ConfidenceLevelEnum,
    DAORecommendation,
    DAORecommendationDetails,
    EvidenceEvaluation,
    EvidenceQualityEnum,
    FraudAnalysis,
    ImageEvidenceAnalysis,
    IssueCategoryEnum,
    IssueClassification,
    LandlordPosition,
    RecommendedOutcomeEnum,
    RedFlags,
    SeverityEnum,
    TenantPosition,
)
from app.services.llm_service import OllamaLLMService
from app.utils.exceptions import AnalysisError
from app.utils.logger import get_logger

logger = get_logger(__name__)


# Keywords for issue classification
CATEGORY_KEYWORDS = {
    "Safety": [
        "fire", "smoke", "carbon monoxide", "gas leak", "electrical",
        "mold", "asbestos", "lead", "pest", "rodent", "infestation",
        "structural", "collapse", "unsafe", "hazard", "emergency",
        "broken lock", "security", "violence", "threatening",
    ],
    "Maintenance": [
        "repair", "broken", "leak", "water damage", "plumbing",
        "heating", "hvac", "air conditioning", "appliance", "window",
        "door", "roof", "floor", "ceiling", "wall", "paint",
        "elevator", "garbage", "trash", "cleaning", "maintenance",
    ],
    "Harassment": [
        "harassment", "threaten", "intimidate", "retaliation",
        "entry without notice", "illegal entry", "privacy",
        "stalking", "verbal abuse", "hostile", "aggressive",
    ],
    "Discrimination": [
        "discrimination", "race", "gender", "disability", "religion",
        "national origin", "familial status", "color", "sex",
        "refused", "denied", "different treatment", "fair housing",
    ],
}

# Severity indicators
CRITICAL_KEYWORDS = [
    "emergency", "fire", "gas leak", "no heat", "no water",
    "flooding", "structural damage", "violence", "threatened",
    "immediate danger", "uninhabitable", "health hazard",
]

HIGH_KEYWORDS = [
    "mold", "electrical issue", "broken lock", "pest infestation",
    "no hot water", "sewage", "major leak", "roof damage",
]


class DisputeCaseAnalyzer:
    """
    Analyzes housing disputes and generates DAO recommendations.
    
    Provides comprehensive case analysis including:
    - Issue classification by category and severity
    - Full dispute analysis with position assessments
    - DAO voting recommendations
    - Fraud pattern detection
    
    Example:
        >>> analyzer = DisputeCaseAnalyzer(llm_service)
        >>> classification = analyzer.classify_issue(
        ...     "Water leak causing mold growth",
        ...     evidence_count=3
        ... )
        >>> print(classification.primary_category)
        'Maintenance'
    """
    
    def __init__(
        self,
        llm_service: OllamaLLMService,
        evidence_pipeline: Optional[Any] = None,
    ) -> None:
        """
        Initialize the case analyzer.
        
        Args:
            llm_service: OllamaLLMService instance for LLM queries.
            evidence_pipeline: Optional EvidencePipeline for vision analysis.
        """
        self.llm = llm_service
        self._evidence_pipeline = evidence_pipeline
    
    def classify_issue(
        self,
        description: str,
        evidence_count: int = 0,
    ) -> IssueClassification:
        """
        Classify an issue by category, severity, and urgency.
        
        Args:
            description: Detailed description of the issue.
            evidence_count: Number of evidence items provided.
            
        Returns:
            IssueClassification: Classification results.
            
        Raises:
            AnalysisError: If classification fails.
            
        Example:
            >>> analyzer = DisputeCaseAnalyzer(llm_service)
            >>> result = analyzer.classify_issue(
            ...     "Heating has been broken for 2 weeks in winter",
            ...     evidence_count=2
            ... )
            >>> print(result.primary_category)
            'Maintenance'
            >>> print(result.severity)
            'high'
        """
        # First, do keyword-based pre-analysis for context
        detected_keywords = self._detect_keywords(description)
        preliminary_category = self._determine_preliminary_category(description)
        preliminary_severity = self._determine_preliminary_severity(description)
        
        prompt = f"""You are an expert housing dispute analyst. Classify the following tenant issue.

ISSUE DESCRIPTION:
{description}

EVIDENCE PROVIDED: {evidence_count} items

Analyze and classify this issue. Consider:
1. What is the PRIMARY category? Choose ONE: Safety, Maintenance, Harassment, or Discrimination
2. Is there a SECONDARY category that also applies? (optional)
3. What is the severity level? (low, medium, high, critical)
4. Does this require URGENT attention?

Respond with JSON in this exact format:
{{
    "primary_category": "Safety|Maintenance|Harassment|Discrimination",
    "confidence": <0-100>,
    "secondary_category": null or "Safety|Maintenance|Harassment|Discrimination",
    "severity": "low|medium|high|critical",
    "reasoning": "<2-3 sentence explanation>",
    "urgency_flag": true|false,
    "keywords_detected": [<list of key terms identified>]
}}"""

        try:
            result = self.llm.query(prompt, expect_json=True)
            
            # Handle parse errors by using keyword-based classification
            if result.get("parse_error"):
                logger.warning(
                    "LLM classification parsing failed, using keyword-based fallback"
                )
                return IssueClassification(
                    primary_category=IssueCategoryEnum(preliminary_category),
                    confidence=60,
                    secondary_category=None,
                    severity=SeverityEnum(preliminary_severity),
                    reasoning="Classification based on keyword analysis due to LLM parsing issues.",
                    urgency_flag=preliminary_severity in ["high", "critical"],
                    keywords_detected=detected_keywords[:10],
                )
            
            # Parse and validate LLM response
            primary_cat = result.get("primary_category", preliminary_category)
            
            # Validate category
            try:
                primary_category = IssueCategoryEnum(primary_cat)
            except ValueError:
                primary_category = IssueCategoryEnum(preliminary_category)
            
            # Handle secondary category
            secondary_cat = result.get("secondary_category")
            secondary_category = None
            if secondary_cat:
                try:
                    secondary_category = IssueCategoryEnum(secondary_cat)
                except ValueError:
                    pass
            
            # Validate severity
            severity_str = result.get("severity", preliminary_severity)
            try:
                severity = SeverityEnum(severity_str.lower())
            except ValueError:
                severity = SeverityEnum(preliminary_severity)
            
            classification = IssueClassification(
                primary_category=primary_category,
                confidence=int(result.get("confidence", 70)),
                secondary_category=secondary_category,
                severity=severity,
                reasoning=result.get("reasoning", "No detailed reasoning provided."),
                urgency_flag=bool(result.get("urgency_flag", False)),
                keywords_detected=result.get("keywords_detected", detected_keywords[:10]),
            )
            
            logger.info(
                "Issue classified",
                category=classification.primary_category.value,
                severity=classification.severity.value,
                confidence=classification.confidence,
                urgency=classification.urgency_flag,
            )
            
            return classification
            
        except Exception as e:
            logger.error("Issue classification failed", error=str(e))
            raise AnalysisError(
                message="Failed to classify issue",
                details={"error": str(e)},
            )
    
    def _detect_keywords(self, text: str) -> list[str]:
        """Detect category-related keywords in text."""
        text_lower = text.lower()
        detected = []
        
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower and keyword not in detected:
                    detected.append(keyword)
        
        return detected
    
    def _determine_preliminary_category(self, text: str) -> str:
        """Determine category based on keyword matching."""
        text_lower = text.lower()
        scores: dict[str, int] = {cat: 0 for cat in CATEGORY_KEYWORDS}
        
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[category] += 1
        
        if max(scores.values()) == 0:
            return "Maintenance"  # Default
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _determine_preliminary_severity(self, text: str) -> str:
        """Determine severity based on keyword matching."""
        text_lower = text.lower()
        
        for keyword in CRITICAL_KEYWORDS:
            if keyword in text_lower:
                return "critical"
        
        for keyword in HIGH_KEYWORDS:
            if keyword in text_lower:
                return "high"
        
        return "medium"
    
    def analyze_dispute(
        self,
        case_data: CaseAnalysisRequest,
    ) -> DAORecommendation:
        """
        Perform comprehensive dispute analysis for DAO recommendation.
        
        Args:
            case_data: Complete case data including complaints and evidence.
            
        Returns:
            DAORecommendation: Comprehensive analysis and recommendation.
            
        Raises:
            AnalysisError: If analysis fails.
            
        Example:
            >>> analyzer = DisputeCaseAnalyzer(llm_service)
            >>> recommendation = analyzer.analyze_dispute(case_data)
            >>> print(recommendation.dao_recommendation.recommended_outcome)
            'Favor Tenant'
        """
        # Build comprehensive case context
        case_context = self._build_case_context(case_data)
        
        # Run vision analysis if enabled
        vision_analyses: list[ImageEvidenceAnalysis] = []
        if case_data.enable_vision_analysis:
            vision_analyses = self._run_vision_analyses(case_data)
        
        prompt = f"""You are an impartial housing dispute arbitrator analyzing a case for DAO (Decentralized Autonomous Organization) voting.

{case_context}

Analyze this case thoroughly and provide a comprehensive recommendation. Be fair and impartial.

Respond with JSON in this exact format:
{{
    "case_summary": "<2-3 sentence neutral overview of the dispute>",
    "tenant_position": {{
        "key_arguments": [<list of main points>],
        "evidence_strength": <0-100>,
        "credibility_score": <0-100>,
        "supporting_factors": [<factors that strengthen tenant's case>]
    }},
    "landlord_position": {{
        "key_arguments": [<list of main points>],
        "evidence_strength": <0-100>,
        "credibility_score": <0-100>,
        "supporting_factors": [<factors that strengthen landlord's case>]
    }},
    "evidence_evaluation": {{
        "tenant_evidence_quality": "poor|fair|good|strong",
        "landlord_evidence_quality": "poor|fair|good|strong",
        "metadata_authenticity": <0-100>,
        "key_discrepancies": [<list of discrepancies>],
        "critical_gaps": [<missing evidence or information>]
    }},
    "dao_recommendation": {{
        "tenant_favor_confidence": <0-100>,
        "landlord_favor_confidence": <0-100>,
        "neutral_confidence": <0-100>,
        "recommended_outcome": "Favor Tenant|Favor Landlord|Mediation Required|Insufficient Evidence",
        "confidence_level": "high|medium|low",
        "reasoning": "<detailed explanation>",
        "key_considerations": [<list of key factors>],
        "suggested_resolution": "<specific action recommended>"
    }},
    "red_flags": {{
        "tenant_concerns": [<any concerns about tenant's case>],
        "landlord_concerns": [<any concerns about landlord's case>],
        "evidence_concerns": [<any concerns about evidence>]
    }},
    "next_steps": [<recommended next steps>],
    "estimated_resolution_timeline": "<e.g., '7-14 days'>"
}}"""

        try:
            result = self.llm.query(prompt, expect_json=True, timeout=180)
            
            if result.get("parse_error"):
                raise AnalysisError(
                    message="Failed to parse LLM response for case analysis",
                    case_id=case_data.issue_id,
                )
            
            # Parse tenant position
            tenant_pos = result.get("tenant_position", {})
            tenant_position = TenantPosition(
                key_arguments=tenant_pos.get("key_arguments", []),
                evidence_strength=int(tenant_pos.get("evidence_strength", 50)),
                credibility_score=int(tenant_pos.get("credibility_score", 50)),
                supporting_factors=tenant_pos.get("supporting_factors", []),
            )
            
            # Parse landlord position
            landlord_pos = result.get("landlord_position", {})
            landlord_position = LandlordPosition(
                key_arguments=landlord_pos.get("key_arguments", []),
                evidence_strength=int(landlord_pos.get("evidence_strength", 50)),
                credibility_score=int(landlord_pos.get("credibility_score", 50)),
                supporting_factors=landlord_pos.get("supporting_factors", []),
            )
            
            # Parse evidence evaluation
            evidence_eval = result.get("evidence_evaluation", {})
            evidence_evaluation = EvidenceEvaluation(
                tenant_evidence_quality=self._parse_quality(
                    evidence_eval.get("tenant_evidence_quality", "fair")
                ),
                landlord_evidence_quality=self._parse_quality(
                    evidence_eval.get("landlord_evidence_quality", "fair")
                ),
                metadata_authenticity=int(evidence_eval.get("metadata_authenticity", 50)),
                key_discrepancies=evidence_eval.get("key_discrepancies", []),
                critical_gaps=evidence_eval.get("critical_gaps", []),
            )
            
            # Parse DAO recommendation
            dao_rec = result.get("dao_recommendation", {})
            dao_recommendation = DAORecommendationDetails(
                tenant_favor_confidence=int(dao_rec.get("tenant_favor_confidence", 50)),
                landlord_favor_confidence=int(dao_rec.get("landlord_favor_confidence", 50)),
                neutral_confidence=int(dao_rec.get("neutral_confidence", 0)),
                recommended_outcome=self._parse_outcome(
                    dao_rec.get("recommended_outcome", "Mediation Required")
                ),
                confidence_level=self._parse_confidence_level(
                    dao_rec.get("confidence_level", "medium")
                ),
                reasoning=dao_rec.get("reasoning", "No detailed reasoning provided."),
                key_considerations=dao_rec.get("key_considerations", []),
                suggested_resolution=dao_rec.get("suggested_resolution", "Requires further review."),
            )
            
            # Parse red flags
            red_flags_data = result.get("red_flags", {})
            red_flags = RedFlags(
                tenant_concerns=red_flags_data.get("tenant_concerns", []),
                landlord_concerns=red_flags_data.get("landlord_concerns", []),
                evidence_concerns=red_flags_data.get("evidence_concerns", []),
            )
            
            recommendation = DAORecommendation(
                case_summary=result.get("case_summary", "Case summary unavailable."),
                tenant_position=tenant_position,
                landlord_position=landlord_position,
                evidence_evaluation=evidence_evaluation,
                dao_recommendation=dao_recommendation,
                red_flags=red_flags,
                next_steps=result.get("next_steps", []),
                estimated_resolution_timeline=result.get("estimated_resolution_timeline", "14-30 days"),
                vision_analyses=vision_analyses if vision_analyses else None,
            )
            
            logger.info(
                "Dispute analysis complete",
                case_id=case_data.issue_id,
                recommended_outcome=recommendation.dao_recommendation.recommended_outcome.value,
                confidence=recommendation.dao_recommendation.confidence_level.value,
            )
            
            return recommendation
            
        except AnalysisError:
            raise
        except Exception as e:
            logger.error(
                "Dispute analysis failed",
                case_id=case_data.issue_id,
                error=str(e),
            )
            raise AnalysisError(
                message="Failed to analyze dispute case",
                case_id=case_data.issue_id,
                details={"error": str(e)},
            )
    
    def _build_case_context(self, case_data: CaseAnalysisRequest) -> str:
        """Build comprehensive case context for LLM prompt."""
        context_parts = [
            f"CASE ID: {case_data.issue_id}",
            f"INCIDENT DATE: {case_data.incident_date.isoformat()}",
            "",
            "TENANT'S COMPLAINT:",
            case_data.tenant_complaint,
            "",
        ]
        
        if case_data.landlord_response:
            context_parts.extend([
                "LANDLORD'S RESPONSE:",
                case_data.landlord_response,
                "",
            ])
        else:
            context_parts.extend([
                "LANDLORD'S RESPONSE:",
                "(No response provided)",
                "",
            ])
        
        # Evidence summary
        context_parts.append("EVIDENCE SUMMARY:")
        context_parts.append(f"- Tenant evidence items: {len(case_data.tenant_evidence)}")
        context_parts.append(f"- Landlord evidence items: {len(case_data.landlord_evidence)}")
        
        # Property history if available
        if case_data.property_history:
            context_parts.extend([
                "",
                "PROPERTY HISTORY:",
                f"- Previous complaints: {case_data.property_history.previous_complaints}",
                f"- Resolution rate: {case_data.property_history.resolution_rate * 100:.0f}%",
            ])
        
        return "\n".join(context_parts)
    
    def _parse_quality(self, value: str) -> EvidenceQualityEnum:
        """Parse evidence quality string to enum."""
        try:
            return EvidenceQualityEnum(value.lower())
        except ValueError:
            return EvidenceQualityEnum.FAIR
    
    def _parse_outcome(self, value: str) -> RecommendedOutcomeEnum:
        """Parse recommended outcome string to enum."""
        mapping = {
            "favor tenant": RecommendedOutcomeEnum.FAVOR_TENANT,
            "favor landlord": RecommendedOutcomeEnum.FAVOR_LANDLORD,
            "mediation required": RecommendedOutcomeEnum.MEDIATION_REQUIRED,
            "insufficient evidence": RecommendedOutcomeEnum.INSUFFICIENT_EVIDENCE,
        }
        return mapping.get(value.lower(), RecommendedOutcomeEnum.MEDIATION_REQUIRED)
    
    def _parse_confidence_level(self, value: str) -> ConfidenceLevelEnum:
        """Parse confidence level string to enum."""
        try:
            return ConfidenceLevelEnum(value.lower())
        except ValueError:
            return ConfidenceLevelEnum.MEDIUM
    
    def _run_vision_analyses(
        self,
        case_data: CaseAnalysisRequest,
    ) -> list[ImageEvidenceAnalysis]:
        """
        Run vision analysis on all evidence images.
        
        Uses EvidencePipeline to analyze each tenant evidence image
        with LLaVA vision model.
        """
        results: list[ImageEvidenceAnalysis] = []
        
        if not self._evidence_pipeline:
            # Lazy import to avoid circular dependencies
            from app.services.evidence_pipeline import EvidencePipeline
            self._evidence_pipeline = EvidencePipeline()
        
        # Process tenant evidence images
        for evidence in case_data.tenant_evidence:
            try:
                logger.info(
                    "Running vision analysis on evidence",
                    url=evidence.file_url[:80],
                )
                
                analysis = self._evidence_pipeline.analyze_evidence(
                    image_url=evidence.file_url,
                    claim_text=case_data.tenant_complaint,
                    incident_date=case_data.incident_date.isoformat() if case_data.incident_date else None,
                )
                results.append(analysis)
                
            except Exception as e:
                logger.warning(
                    "Vision analysis failed for evidence",
                    url=evidence.file_url[:80],
                    error=str(e),
                )
                # Continue with other evidence items
                continue
        
        logger.info(
            "Vision analyses complete",
            total_evidence=len(case_data.tenant_evidence),
            successful=len(results),
        )
        
        return results
    
    def detect_fraud_patterns(
        self,
        case_data: CaseAnalysisRequest,
    ) -> FraudAnalysis:
        """
        Detect potential fraud patterns in a dispute case.
        
        Checks for:
        - Duplicate image submissions
        - Inconsistent timelines
        - Excessive emotional language
        - Evidence-claim mismatches
        
        Args:
            case_data: Complete case data.
            
        Returns:
            FraudAnalysis: Fraud pattern detection results.
            
        Example:
            >>> analyzer = DisputeCaseAnalyzer(llm_service)
            >>> fraud = analyzer.detect_fraud_patterns(case_data)
            >>> print(f"Fraud risk: {fraud.fraud_risk_score}")
        """
        indicators: list[str] = []
        fraud_score = 0
        duplicate_detected = False
        timeline_issues = False
        manipulation_detected = False
        
        # Check for duplicate evidence (by URL)
        evidence_urls = [e.file_url for e in case_data.tenant_evidence]
        if len(evidence_urls) != len(set(evidence_urls)):
            indicators.append("Duplicate evidence files detected")
            fraud_score += 30
            duplicate_detected = True
        
        # Check timeline consistency
        for evidence in case_data.tenant_evidence:
            if evidence.uploaded_at < case_data.incident_date:
                indicators.append(
                    f"Evidence uploaded before incident date: {evidence.file_url}"
                )
                fraud_score += 25
                timeline_issues = True
                break
        
        # Use LLM to detect manipulation language
        prompt = f"""Analyze this tenant complaint for signs of emotional manipulation or exaggeration.

COMPLAINT TEXT:
{case_data.tenant_complaint}

Look for:
1. Excessive emotional language designed to manipulate
2. Dramatic exaggerations
3. Inconsistent or contradictory statements
4. Unrealistic claims

Respond with JSON:
{{
    "manipulation_score": <0-100, higher = more manipulation detected>,
    "concerning_phrases": [<list of concerning phrases or claims>],
    "analysis": "<brief analysis>"
}}"""

        try:
            result = self.llm.query(prompt, expect_json=True, timeout=60)
            
            if not result.get("parse_error"):
                manipulation_score = int(result.get("manipulation_score", 0))
                if manipulation_score > 50:
                    indicators.extend(result.get("concerning_phrases", [])[:3])
                    fraud_score += min(manipulation_score // 2, 30)
                    manipulation_detected = True
                    
        except Exception as e:
            logger.warning("Manipulation detection failed", error=str(e))
        
        # Cap score at 100
        fraud_score = min(fraud_score, 100)
        
        # Generate conclusion
        if fraud_score < 20:
            conclusion = "Low fraud risk detected"
        elif fraud_score < 50:
            conclusion = "Moderate fraud risk - recommend additional verification"
        else:
            conclusion = "High fraud risk - thorough investigation required"
        
        if not indicators:
            indicators.append("No fraud indicators detected")
        
        logger.info(
            "Fraud analysis complete",
            case_id=case_data.issue_id,
            fraud_score=fraud_score,
            indicators_count=len(indicators),
        )
        
        return FraudAnalysis(
            fraud_risk_score=fraud_score,
            indicators=indicators,
            duplicate_evidence=duplicate_detected,
            timeline_inconsistencies=timeline_issues,
            manipulation_detected=manipulation_detected,
            conclusion=conclusion,
        )
