"""
Evidence Validator for RentShield AI Analysis Engine.

Provides image authenticity verification using EXIF metadata analysis,
tampering detection, and LLM-powered evidence-claim alignment assessment.
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from app.config import get_settings
from app.models.responses import (
    AlignmentAnalysis,
    EvidenceValidation,
    EXIFData,
    TamperAnalysis,
    TrustLevelEnum,
)
from app.utils.exceptions import ExifExtractionError, InvalidImageError
from app.utils.logger import get_logger

logger = get_logger(__name__)


# Known editing software indicators
EDITING_SOFTWARE = [
    "photoshop",
    "gimp",
    "lightroom",
    "capture one",
    "affinity",
    "pixelmator",
    "snapseed",
    "vsco",
    "afterlight",
]


class EvidenceValidator:
    """
    Validates image authenticity using EXIF metadata and AI analysis.
    
    Provides comprehensive evidence validation including:
    - EXIF metadata extraction
    - Authenticity scoring
    - Tampering detection
    - Evidence-claim alignment analysis
    
    Example:
        >>> validator = EvidenceValidator()
        >>> exif = validator.extract_exif("path/to/image.jpg")
        >>> score = validator.calculate_authenticity_score(exif)
        >>> print(f"Authenticity score: {score}")
    """
    
    def __init__(self) -> None:
        """Initialize the evidence validator."""
        self.settings = get_settings()
    
    def validate_image_file(self, image_path: str) -> None:
        """
        Validate that a file is a valid image.
        
        Args:
            image_path: Path to the image file.
            
        Raises:
            InvalidImageError: If file is not a valid image.
        """
        path = Path(image_path)
        
        # Check file exists
        if not path.exists():
            raise InvalidImageError(
                message="Image file does not exist",
                filename=str(path),
            )
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > self.settings.MAX_FILE_SIZE:
            raise InvalidImageError(
                message=f"Image file too large (max {self.settings.MAX_FILE_SIZE // (1024*1024)}MB)",
                filename=str(path),
                details={"file_size": file_size},
            )
        
        # Check extension
        if path.suffix.lower() not in self.settings.ALLOWED_EXTENSIONS:
            raise InvalidImageError(
                message=f"Invalid file type. Allowed: {', '.join(self.settings.ALLOWED_EXTENSIONS)}",
                filename=str(path),
                details={"extension": path.suffix},
            )
        
        # Verify it's a valid image
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            raise InvalidImageError(
                message="File is not a valid image",
                filename=str(path),
                details={"error": str(e)},
            )
    
    def extract_exif(self, image_path: str) -> EXIFData:
        """
        Extract EXIF metadata from an image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            EXIFData: Extracted EXIF metadata.
            
        Raises:
            ExifExtractionError: If EXIF extraction fails.
            
        Example:
            >>> validator = EvidenceValidator()
            >>> exif = validator.extract_exif("photo.jpg")
            >>> print(exif.datetime_original)
            '2024:01:15 14:30:00'
        """
        path = Path(image_path)
        
        try:
            self.validate_image_file(image_path)
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(path)
            file_size = path.stat().st_size
            
            with Image.open(path) as img:
                # Get image dimensions
                width, height = img.size
                dimensions = f"{width}x{height}"
                
                # Extract EXIF data using public API (works with all image formats)
                exif_data = img.getexif() or None
                
                # Convert Exif object to dict if not empty
                if exif_data is not None and len(exif_data) == 0:
                    exif_data = None
                
                if exif_data is None:
                    logger.info(
                        "No EXIF data found in image",
                        filename=str(path),
                    )
                    return EXIFData(
                        file_hash=file_hash,
                        dimensions=dimensions,
                        file_size=file_size,
                    )
                
                # Parse EXIF tags
                parsed_exif: dict[str, Any] = {}
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    parsed_exif[tag] = value
                
                # Extract GPS info
                gps_lat, gps_lon = self._extract_gps_info(parsed_exif)
                
                exif_result = EXIFData(
                    datetime_original=self._safe_string(parsed_exif.get("DateTimeOriginal")),
                    device_make=self._safe_string(parsed_exif.get("Make")),
                    device_model=self._safe_string(parsed_exif.get("Model")),
                    gps_latitude=gps_lat,
                    gps_longitude=gps_lon,
                    software=self._safe_string(parsed_exif.get("Software")),
                    file_hash=file_hash,
                    dimensions=dimensions,
                    file_size=file_size,
                )
                
                logger.info(
                    "EXIF extraction successful",
                    filename=str(path),
                    has_datetime=exif_result.datetime_original is not None,
                    has_device=exif_result.device_make is not None,
                    has_gps=exif_result.gps_latitude is not None,
                )
                
                return exif_result
                
        except InvalidImageError:
            raise
        except Exception as e:
            logger.error(
                "EXIF extraction failed",
                filename=str(path),
                error=str(e),
            )
            raise ExifExtractionError(
                message="Failed to extract EXIF metadata",
                filename=str(path),
                details={"error": str(e)},
            )
    
    def _calculate_file_hash(self, path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _safe_string(self, value: Any) -> Optional[str]:
        """Safely convert EXIF value to string."""
        if value is None:
            return None
        try:
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="ignore").strip()
            return str(value).strip() or None
        except Exception:
            return None
    
    def _extract_gps_info(
        self,
        parsed_exif: dict[str, Any],
    ) -> tuple[Optional[float], Optional[float]]:
        """Extract GPS coordinates from EXIF data."""
        gps_info = parsed_exif.get("GPSInfo")
        
        if not gps_info or not isinstance(gps_info, dict):
            return None, None
        
        try:
            # Parse GPS tags
            gps_data: dict[str, Any] = {}
            for key, value in gps_info.items():
                tag = GPSTAGS.get(key, key)
                gps_data[tag] = value
            
            # Extract latitude
            lat = gps_data.get("GPSLatitude")
            lat_ref = gps_data.get("GPSLatitudeRef", "N")
            
            # Extract longitude
            lon = gps_data.get("GPSLongitude")
            lon_ref = gps_data.get("GPSLongitudeRef", "E")
            
            if lat and lon:
                latitude = self._convert_to_degrees(lat)
                longitude = self._convert_to_degrees(lon)
                
                if lat_ref == "S":
                    latitude = -latitude
                if lon_ref == "W":
                    longitude = -longitude
                
                return latitude, longitude
                
        except Exception as e:
            logger.debug(f"Could not parse GPS info: {e}")
        
        return None, None
    
    def _convert_to_degrees(self, value: tuple) -> float:
        """Convert GPS coordinates to decimal degrees."""
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    
    def calculate_authenticity_score(self, exif_data: EXIFData) -> int:
        """
        Calculate authenticity score based on EXIF data.
        
        Scoring algorithm:
        - Has any EXIF data: +40 points
        - Has datetime_original: +30 points
        - Has device_make/model: +20 points
        - Has GPS data: +10 points
        
        Args:
            exif_data: Extracted EXIF metadata.
            
        Returns:
            int: Authenticity score (0-100).
            
        Example:
            >>> validator = EvidenceValidator()
            >>> exif = EXIFData(datetime_original="2024:01:15 14:30:00")
            >>> score = validator.calculate_authenticity_score(exif)
            70  # 40 (has EXIF) + 30 (has datetime)
        """
        score = 0
        
        # Check for any meaningful EXIF data
        has_exif = any([
            exif_data.datetime_original,
            exif_data.device_make,
            exif_data.device_model,
            exif_data.gps_latitude,
            exif_data.software,
        ])
        
        if has_exif:
            score += 40
        
        # Datetime original is most important
        if exif_data.datetime_original:
            score += 30
        
        # Device information adds credibility
        if exif_data.device_make or exif_data.device_model:
            score += 20
        
        # GPS data is strong indicator
        if exif_data.gps_latitude is not None and exif_data.gps_longitude is not None:
            score += 10
        
        logger.debug(
            "Authenticity score calculated",
            score=score,
            has_exif=has_exif,
            has_datetime=exif_data.datetime_original is not None,
            has_device=exif_data.device_make is not None,
            has_gps=exif_data.gps_latitude is not None,
        )
        
        return min(score, 100)
    
    def get_trust_level(self, authenticity_score: int) -> TrustLevelEnum:
        """
        Determine trust level from authenticity score.
        
        Args:
            authenticity_score: Score from 0-100.
            
        Returns:
            TrustLevelEnum: Trust level classification.
        """
        if authenticity_score >= 80:
            return TrustLevelEnum.HIGH_TRUST
        elif authenticity_score >= 50:
            return TrustLevelEnum.MEDIUM_TRUST
        elif authenticity_score >= 20:
            return TrustLevelEnum.LOW_TRUST
        else:
            return TrustLevelEnum.UNTRUSTED
    
    def detect_tampering(self, image_path: str) -> TamperAnalysis:
        """
        Detect potential image tampering.
        
        Checks for:
        - Inconsistent metadata timestamps
        - Software watermarks (Photoshop, GIMP)
        - Missing or stripped EXIF
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            TamperAnalysis: Tampering detection results.
            
        Example:
            >>> validator = EvidenceValidator()
            >>> tamper = validator.detect_tampering("edited_photo.jpg")
            >>> print(f"Tamper probability: {tamper.tamper_probability}")
        """
        indicators: list[str] = []
        tamper_probability = 0.0
        
        try:
            exif_data = self.extract_exif(image_path)
            
            # Check for stripped EXIF
            has_meaningful_exif = any([
                exif_data.datetime_original,
                exif_data.device_make,
                exif_data.gps_latitude,
            ])
            
            if not has_meaningful_exif:
                indicators.append("Missing or stripped EXIF metadata")
                tamper_probability += 0.3
            
            # Check for editing software
            if exif_data.software:
                software_lower = exif_data.software.lower()
                for editor in EDITING_SOFTWARE:
                    if editor in software_lower:
                        indicators.append(f"Editing software detected: {exif_data.software}")
                        tamper_probability += 0.4
                        break
            
            # Check for suspicious patterns
            if exif_data.datetime_original:
                # Check if datetime is in the future
                try:
                    dt_str = exif_data.datetime_original.replace(":", "-", 2)
                    dt = datetime.fromisoformat(dt_str.replace(" ", "T"))
                    if dt > datetime.now():
                        indicators.append("Image timestamp is in the future")
                        tamper_probability += 0.5
                except ValueError:
                    pass
            
            # Cap probability at 1.0
            tamper_probability = min(tamper_probability, 1.0)
            
            # Generate conclusion
            if tamper_probability < 0.2:
                conclusion = "Low risk of tampering detected"
            elif tamper_probability < 0.5:
                conclusion = "Moderate tampering indicators found"
            else:
                conclusion = "High probability of image manipulation"
            
            if not indicators:
                indicators.append("No tampering indicators detected")
            
            logger.info(
                "Tampering analysis complete",
                probability=tamper_probability,
                indicators_count=len(indicators),
            )
            
            return TamperAnalysis(
                tamper_probability=round(tamper_probability, 2),
                indicators=indicators,
                conclusion=conclusion,
            )
            
        except Exception as e:
            logger.error("Tampering detection failed", error=str(e))
            return TamperAnalysis(
                tamper_probability=0.5,
                indicators=["Could not complete tampering analysis"],
                conclusion="Tampering analysis inconclusive due to processing error",
            )
    
    def analyze_evidence_alignment(
        self,
        llm_service: Any,
        exif_data: EXIFData,
        claim_text: str,
        incident_date: Optional[datetime] = None,
    ) -> AlignmentAnalysis:
        """
        Use LLM to assess how well evidence aligns with claims.
        
        Args:
            llm_service: OllamaLLMService instance for LLM queries.
            exif_data: EXIF metadata from the evidence image.
            claim_text: The claim or complaint text to verify against.
            incident_date: Optional reported incident date.
            
        Returns:
            AlignmentAnalysis: Assessment of evidence-claim alignment.
            
        Example:
            >>> validator = EvidenceValidator()
            >>> llm = OllamaLLMService()
            >>> alignment = validator.analyze_evidence_alignment(
            ...     llm, exif_data, "Water leak started Jan 10"
            ... )
        """
        # Build context about the evidence
        evidence_context = []
        
        if exif_data.datetime_original:
            evidence_context.append(f"Photo taken: {exif_data.datetime_original}")
        else:
            evidence_context.append("Photo timestamp: Not available (metadata stripped or missing)")
        
        if exif_data.device_make or exif_data.device_model:
            device = f"{exif_data.device_make or ''} {exif_data.device_model or ''}".strip()
            evidence_context.append(f"Camera device: {device}")
        
        if exif_data.gps_latitude is not None:
            evidence_context.append(
                f"GPS coordinates: {exif_data.gps_latitude:.4f}, {exif_data.gps_longitude:.4f}"
            )
        
        if incident_date:
            evidence_context.append(f"Reported incident date: {incident_date.isoformat()}")
        
        evidence_summary = "\n".join(evidence_context)
        
        prompt = f"""Analyze the alignment between photographic evidence metadata and a tenant's claim.

EVIDENCE METADATA:
{evidence_summary}

TENANT'S CLAIM:
{claim_text}

Analyze whether the evidence appears to support the claim. Consider:
1. Does the photo timestamp align with the reported incident date (if available)?
2. Are there any logical inconsistencies between the claim and the evidence metadata?
3. Is the lack of metadata a concern?

Respond with JSON in this exact format:
{{
    "alignment_score": <0-100, how well evidence supports the claim>,
    "concerns": [<list of specific concerns or inconsistencies found>],
    "reasoning": "<detailed explanation of your assessment>"
}}"""

        try:
            result = llm_service.query(prompt, expect_json=True)
            
            # Handle parse errors
            if result.get("parse_error"):
                return AlignmentAnalysis(
                    alignment_score=50,
                    concerns=["LLM response parsing failed"],
                    reasoning="Could not fully analyze alignment due to LLM response format issues.",
                )
            
            return AlignmentAnalysis(
                alignment_score=int(result.get("alignment_score", 50)),
                concerns=result.get("concerns", []),
                reasoning=result.get("reasoning", "No detailed reasoning provided."),
            )
            
        except Exception as e:
            logger.error("Evidence alignment analysis failed", error=str(e))
            return AlignmentAnalysis(
                alignment_score=50,
                concerns=["Alignment analysis could not be completed"],
                reasoning=f"Analysis failed due to error: {str(e)}",
            )
    
    def validate_evidence(
        self,
        image_path: str,
        claim_text: Optional[str] = None,
        incident_date: Optional[datetime] = None,
        llm_service: Optional[Any] = None,
    ) -> EvidenceValidation:
        """
        Perform complete evidence validation.
        
        This is the main entry point for evidence validation, combining
        EXIF extraction, authenticity scoring, tampering detection, and
        optionally evidence-claim alignment analysis.
        
        Args:
            image_path: Path to the evidence image.
            claim_text: Optional claim to verify evidence against.
            incident_date: Optional reported incident date.
            llm_service: Optional LLM service for alignment analysis.
            
        Returns:
            EvidenceValidation: Complete validation results.
            
        Example:
            >>> validator = EvidenceValidator()
            >>> result = validator.validate_evidence(
            ...     "leak_photo.jpg",
            ...     claim_text="Water leak in kitchen ceiling",
            ...     llm_service=OllamaLLMService()
            ... )
            >>> print(f"Trust level: {result.trust_level}")
        """
        # Extract EXIF
        exif_data = self.extract_exif(image_path)
        
        # Calculate authenticity score
        authenticity_score = self.calculate_authenticity_score(exif_data)
        
        # Get trust level
        trust_level = self.get_trust_level(authenticity_score)
        
        # Detect tampering
        tamper_analysis = self.detect_tampering(image_path)
        
        # Adjust authenticity score based on tampering
        if tamper_analysis.tamper_probability > 0.5:
            authenticity_score = max(0, authenticity_score - 30)
            trust_level = self.get_trust_level(authenticity_score)
        
        # Alignment analysis (if claim provided and LLM available)
        alignment_analysis = None
        if claim_text and llm_service:
            alignment_analysis = self.analyze_evidence_alignment(
                llm_service,
                exif_data,
                claim_text,
                incident_date,
            )
        
        logger.info(
            "Evidence validation complete",
            authenticity_score=authenticity_score,
            trust_level=trust_level.value,
            tamper_probability=tamper_analysis.tamper_probability,
            has_alignment=alignment_analysis is not None,
        )
        
        return EvidenceValidation(
            authenticity_score=authenticity_score,
            trust_level=trust_level,
            exif_data=exif_data,
            tamper_analysis=tamper_analysis,
            alignment_analysis=alignment_analysis,
        )
