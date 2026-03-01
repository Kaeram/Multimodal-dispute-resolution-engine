"""
Tests for Vision Service and Evidence Pipeline.

Tests LLaVA-based image understanding, multimodal evidence verification,
and the complete evidence analysis pipeline.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.models.responses import (
    CleanlinessLevelEnum,
    EXIFData,
    ImageEvidenceAnalysis,
    LocationTypeEnum,
    MultimodalVerdict,
    TrustLevelEnum,
    VisionAnalysis,
)
from app.services.evidence_pipeline import EvidencePipeline
from app.services.vision_service import VisionService
from app.utils.exceptions import InvalidImageError


class TestVisionService:
    """Tests for VisionService class."""
    
    @pytest.fixture
    def service(self) -> VisionService:
        """Create VisionService instance for tests."""
        return VisionService()
    
    @pytest.fixture
    def sample_image_path(self, tmp_path: Path) -> str:
        """Create a sample test image."""
        img = Image.new("RGB", (100, 100), color="red")
        img_path = tmp_path / "test_image.jpg"
        img.save(str(img_path), "JPEG")
        return str(img_path)
    
    @patch("app.services.vision_service.requests.get")
    def test_health_check_success(
        self,
        mock_get: MagicMock,
        service: VisionService,
    ) -> None:
        """Test health check succeeds when LLaVA is available."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [
                {"name": "llava:latest"},
                {"name": "mistral:latest"},
            ]
        }
        
        result = service.health_check()
        
        assert result is True
        mock_get.assert_called_once()
    
    @patch("app.services.vision_service.requests.get")
    def test_health_check_llava_not_available(
        self,
        mock_get: MagicMock,
        service: VisionService,
    ) -> None:
        """Test health check fails when LLaVA is not installed."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "models": [
                {"name": "mistral:latest"},
            ]
        }
        
        result = service.health_check()
        
        assert result is False
    
    @patch("app.services.vision_service.requests.get")
    def test_health_check_connection_error(
        self,
        mock_get: MagicMock,
        service: VisionService,
    ) -> None:
        """Test health check handles connection errors."""
        import requests
        mock_get.side_effect = requests.RequestException("Connection failed")
        
        result = service.health_check()
        
        assert result is False
    
    @patch("app.services.vision_service.requests.get")
    def test_download_image_success(
        self,
        mock_get: MagicMock,
        service: VisionService,
    ) -> None:
        """Test successful image download."""
        import io
        
        # Create mock image content in memory
        img = Image.new("RGB", (50, 50), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, "JPEG")
        image_content = buffer.getvalue()
        
        mock_response = MagicMock()
        mock_response.headers = {"content-length": str(len(image_content))}
        mock_response.iter_content.return_value = [image_content]
        mock_get.return_value = mock_response
        
        result = service.download_image("https://example.com/test.jpg")
        
        assert os.path.exists(result)
        assert result.endswith(".jpg")
        
        # Cleanup
        try:
            os.unlink(result)
        except OSError:
            pass  # Ignore cleanup errors on Windows
    
    def test_download_image_invalid_extension(
        self,
        service: VisionService,
    ) -> None:
        """Test download fails for invalid file extension."""
        with pytest.raises(InvalidImageError) as exc_info:
            service.download_image("https://example.com/file.txt")
        
        assert "Invalid file type" in exc_info.value.message
    
    @patch("app.services.vision_service.requests.post")
    def test_analyze_image_success(
        self,
        mock_post: MagicMock,
        service: VisionService,
        sample_image_path: str,
    ) -> None:
        """Test successful image analysis with LLaVA."""
        mock_response = {
            "response": json.dumps({
                "scene_summary": "Kitchen ceiling with water damage and mold.",
                "detected_objects": ["ceiling", "water stain"],
                "damage_detected": ["water damage", "mold"],
                "safety_hazards": ["mold spores"],
                "cleanliness_level": "dirty",
                "indoor_outdoor": "indoor",
                "confidence": 85,
            })
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        result = service.analyze_image(sample_image_path)
        
        assert result["scene_summary"] == "Kitchen ceiling with water damage and mold."
        assert "water damage" in result["damage_detected"]
        assert result["confidence"] == 85
    
    def test_analyze_image_file_not_found(
        self,
        service: VisionService,
    ) -> None:
        """Test analysis fails for non-existent file."""
        with pytest.raises(InvalidImageError) as exc_info:
            service.analyze_image("/nonexistent/path.jpg")
        
        assert "not found" in exc_info.value.message
    
    def test_parse_json_response_direct(
        self,
        service: VisionService,
    ) -> None:
        """Test JSON parsing from direct JSON string."""
        json_str = '{"scene_summary": "test", "confidence": 80}'
        
        result = service._parse_json_response(json_str)
        
        assert result["scene_summary"] == "test"
        assert result["confidence"] == 80
    
    def test_parse_json_response_markdown_block(
        self,
        service: VisionService,
    ) -> None:
        """Test JSON parsing from markdown code block."""
        json_str = '''Here is the analysis:
```json
{"scene_summary": "test from markdown", "confidence": 75}
```
'''
        
        result = service._parse_json_response(json_str)
        
        assert result["scene_summary"] == "test from markdown"
        assert result["confidence"] == 75
    
    def test_parse_json_response_fallback(
        self,
        service: VisionService,
    ) -> None:
        """Test JSON parsing fallback for invalid input."""
        result = service._parse_json_response("This is not valid JSON at all")
        
        assert "Unable to analyze" in result["scene_summary"] or len(result["scene_summary"]) > 0
        assert result["confidence"] == 30


class TestEvidencePipeline:
    """Tests for EvidencePipeline class."""
    
    @pytest.fixture
    def mock_vision_service(self) -> MagicMock:
        """Create mock VisionService."""
        mock = MagicMock(spec=VisionService)
        mock.download_image.return_value = "/tmp/test_image.jpg"
        mock.analyze_image.return_value = {
            "scene_summary": "Water damaged ceiling with visible mold growth.",
            "detected_objects": ["ceiling", "water stain", "mold"],
            "damage_detected": ["water damage", "mold growth"],
            "safety_hazards": ["mold exposure"],
            "cleanliness_level": "dirty",
            "indoor_outdoor": "indoor",
            "confidence": 82,
        }
        return mock
    
    @pytest.fixture
    def mock_evidence_validator(self) -> MagicMock:
        """Create mock EvidenceValidator."""
        mock = MagicMock()
        mock.extract_exif.return_value = EXIFData(
            datetime_original="2024:01:15 14:30:00",
            device_make="Apple",
            device_model="iPhone 13",
        )
        mock.calculate_authenticity_score.return_value = 70
        mock.get_trust_level.return_value = TrustLevelEnum.MEDIUM_TRUST
        return mock
    
    @pytest.fixture
    def mock_llm_service(self) -> MagicMock:
        """Create mock OllamaLLMService."""
        mock = MagicMock()
        mock.query.return_value = {
            "image_supports_claim": True,
            "consistency_score": 85,
            "evidence_strength": 78,
            "detected_inconsistencies": [],
            "possible_fraud_signals": [],
            "reasoning": "Image clearly shows water damage consistent with claim.",
        }
        return mock
    
    def test_analyze_evidence_success(
        self,
        mock_vision_service: MagicMock,
        mock_evidence_validator: MagicMock,
        mock_llm_service: MagicMock,
    ) -> None:
        """Test successful evidence analysis pipeline."""
        pipeline = EvidencePipeline(
            vision_service=mock_vision_service,
            evidence_validator=mock_evidence_validator,
            llm_service=mock_llm_service,
        )
        
        result = pipeline.analyze_evidence(
            image_url="https://example.com/damage.jpg",
            claim_text="Water damage in kitchen ceiling from leak",
            incident_date="2024-01-15T00:00:00Z",
        )
        
        assert isinstance(result, ImageEvidenceAnalysis)
        assert result.final_evidence_score >= 0
        assert result.final_evidence_score <= 100
        assert result.trust_level == TrustLevelEnum.MEDIUM_TRUST
        
        # Verify pipeline steps were called
        mock_vision_service.download_image.assert_called_once()
        mock_vision_service.analyze_image.assert_called_once()
        mock_evidence_validator.extract_exif.assert_called_once()
        mock_llm_service.query.assert_called_once()
        mock_vision_service.cleanup_temp_file.assert_called_once()
    
    def test_analyze_evidence_calculates_correct_score(
        self,
        mock_vision_service: MagicMock,
        mock_evidence_validator: MagicMock,
        mock_llm_service: MagicMock,
    ) -> None:
        """Test final score calculation matches formula."""
        # Set known values
        mock_evidence_validator.calculate_authenticity_score.return_value = 70  # 0.3 * 70 = 21
        mock_vision_service.analyze_image.return_value["confidence"] = 80  # 0.3 * 80 = 24
        mock_llm_service.query.return_value["consistency_score"] = 90  # 0.4 * 90 = 36
        # Expected: 21 + 24 + 36 = 81
        
        pipeline = EvidencePipeline(
            vision_service=mock_vision_service,
            evidence_validator=mock_evidence_validator,
            llm_service=mock_llm_service,
        )
        
        result = pipeline.analyze_evidence(
            image_url="https://example.com/test.jpg",
            claim_text="Test claim for scoring",
        )
        
        assert result.final_evidence_score == 81
    
    def test_build_vision_analysis_handles_invalid_values(
        self,
        mock_vision_service: MagicMock,
        mock_evidence_validator: MagicMock,
        mock_llm_service: MagicMock,
    ) -> None:
        """Test vision analysis builder handles invalid enum values."""
        pipeline = EvidencePipeline(
            vision_service=mock_vision_service,
            evidence_validator=mock_evidence_validator,
            llm_service=mock_llm_service,
        )
        
        llava_result = {
            "scene_summary": "Test scene",
            "cleanliness_level": "invalid_level",
            "indoor_outdoor": "invalid_location",
            "confidence": "not_a_number",
        }
        
        result = pipeline._build_vision_analysis(llava_result)
        
        assert result.cleanliness_level == CleanlinessLevelEnum.AVERAGE
        assert result.indoor_outdoor == LocationTypeEnum.UNCLEAR
        assert result.confidence == 50
    
    def test_build_multimodal_verdict_handles_invalid_values(
        self,
        mock_vision_service: MagicMock,
        mock_evidence_validator: MagicMock,
        mock_llm_service: MagicMock,
    ) -> None:
        """Test multimodal verdict builder handles invalid values."""
        pipeline = EvidencePipeline(
            vision_service=mock_vision_service,
            evidence_validator=mock_evidence_validator,
            llm_service=mock_llm_service,
        )
        
        llm_result = {
            "image_supports_claim": "yes",  # String instead of bool
            "consistency_score": 150,  # Out of bounds
            "evidence_strength": -10,  # Negative
        }
        
        result = pipeline._build_multimodal_verdict(llm_result)
        
        assert result.image_supports_claim is True
        assert result.consistency_score == 100  # Clamped to max
        assert result.evidence_strength == 0  # Clamped to min


class TestVisionAnalysisModel:
    """Tests for VisionAnalysis Pydantic model."""
    
    def test_valid_vision_analysis(self) -> None:
        """Test creating valid VisionAnalysis model."""
        analysis = VisionAnalysis(
            scene_summary="Test scene with damage visible",
            detected_objects=["wall", "crack"],
            damage_detected=["structural crack"],
            safety_hazards=["unstable structure"],
            cleanliness_level=CleanlinessLevelEnum.AVERAGE,
            indoor_outdoor=LocationTypeEnum.INDOOR,
            confidence=75,
        )
        
        assert analysis.scene_summary == "Test scene with damage visible"
        assert len(analysis.detected_objects) == 2
        assert analysis.confidence == 75
    
    def test_confidence_bounds(self) -> None:
        """Test confidence score bounds validation."""
        with pytest.raises(ValueError):
            VisionAnalysis(
                scene_summary="Test",
                cleanliness_level=CleanlinessLevelEnum.CLEAN,
                indoor_outdoor=LocationTypeEnum.INDOOR,
                confidence=101,  # Invalid: > 100
            )


class TestMultimodalVerdictModel:
    """Tests for MultimodalVerdict Pydantic model."""
    
    def test_valid_multimodal_verdict(self) -> None:
        """Test creating valid MultimodalVerdict model."""
        verdict = MultimodalVerdict(
            image_supports_claim=True,
            consistency_score=85,
            evidence_strength=80,
            detected_inconsistencies=[],
            possible_fraud_signals=[],
            reasoning="Evidence strongly supports the claim.",
        )
        
        assert verdict.image_supports_claim is True
        assert verdict.consistency_score == 85
        assert len(verdict.detected_inconsistencies) == 0


class TestImageEvidenceAnalysisModel:
    """Tests for ImageEvidenceAnalysis Pydantic model."""
    
    def test_valid_image_evidence_analysis(self) -> None:
        """Test creating valid ImageEvidenceAnalysis model."""
        analysis = ImageEvidenceAnalysis(
            exif_analysis=EXIFData(
                datetime_original="2024:01:15 14:30:00",
                device_make="Apple",
            ),
            vision_analysis=VisionAnalysis(
                scene_summary="Damaged ceiling visible",
                cleanliness_level=CleanlinessLevelEnum.DIRTY,
                indoor_outdoor=LocationTypeEnum.INDOOR,
                confidence=80,
            ),
            multimodal_verdict=MultimodalVerdict(
                image_supports_claim=True,
                consistency_score=85,
                evidence_strength=78,
                reasoning="Evidence supports claim",
            ),
            final_evidence_score=78,
            trust_level=TrustLevelEnum.MEDIUM_TRUST,
        )
        
        assert analysis.final_evidence_score == 78
        assert analysis.trust_level == TrustLevelEnum.MEDIUM_TRUST
        assert analysis.vision_analysis.confidence == 80
