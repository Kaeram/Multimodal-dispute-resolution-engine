"""
Tests for Evidence Validator.

Tests EXIF extraction, authenticity scoring, and tampering detection
using test images.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.models.responses import TrustLevelEnum
from app.services.evidence_validator import EvidenceValidator
from app.utils.exceptions import InvalidImageError


class TestEvidenceValidator:
    """Tests for EvidenceValidator class."""
    
    @pytest.fixture
    def validator(self) -> EvidenceValidator:
        """Create validator instance for tests."""
        return EvidenceValidator()
    
    @pytest.fixture
    def sample_image_path(self, tmp_path: Path) -> str:
        """Create a sample test image."""
        img = Image.new("RGB", (100, 100), color="red")
        img_path = tmp_path / "test_image.jpg"
        img.save(str(img_path), "JPEG")
        return str(img_path)
    
    @pytest.fixture
    def sample_png_path(self, tmp_path: Path) -> str:
        """Create a sample PNG test image."""
        img = Image.new("RGB", (100, 100), color="blue")
        img_path = tmp_path / "test_image.png"
        img.save(str(img_path), "PNG")
        return str(img_path)
    
    def test_validate_image_file_success(
        self,
        validator: EvidenceValidator,
        sample_image_path: str,
    ) -> None:
        """Test image validation passes for valid image."""
        # Should not raise
        validator.validate_image_file(sample_image_path)
    
    def test_validate_image_file_not_found(
        self,
        validator: EvidenceValidator,
    ) -> None:
        """Test image validation fails for non-existent file."""
        with pytest.raises(InvalidImageError) as exc_info:
            validator.validate_image_file("/nonexistent/path.jpg")
        
        assert "does not exist" in exc_info.value.message
    
    def test_validate_image_file_wrong_extension(
        self,
        validator: EvidenceValidator,
        tmp_path: Path,
    ) -> None:
        """Test image validation fails for wrong extension."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Not an image")
        
        with pytest.raises(InvalidImageError) as exc_info:
            validator.validate_image_file(str(txt_file))
        
        assert "Invalid file type" in exc_info.value.message
    
    def test_extract_exif_basic(
        self,
        validator: EvidenceValidator,
        sample_image_path: str,
    ) -> None:
        """Test EXIF extraction returns basic metadata."""
        exif_data = validator.extract_exif(sample_image_path)
        
        # Basic images have file info even without EXIF
        assert exif_data.file_hash is not None
        assert exif_data.dimensions == "100x100"
        assert exif_data.file_size > 0
    
    def test_extract_exif_png(
        self,
        validator: EvidenceValidator,
        sample_png_path: str,
    ) -> None:
        """Test EXIF extraction works for PNG files."""
        exif_data = validator.extract_exif(sample_png_path)
        
        assert exif_data.file_hash is not None
        assert exif_data.dimensions == "100x100"
    
    def test_calculate_authenticity_score_no_exif(
        self,
        validator: EvidenceValidator,
    ) -> None:
        """Test authenticity score is 0 for no EXIF data."""
        from app.models.responses import EXIFData
        
        exif_data = EXIFData()
        score = validator.calculate_authenticity_score(exif_data)
        
        assert score == 0
    
    def test_calculate_authenticity_score_with_datetime(
        self,
        validator: EvidenceValidator,
    ) -> None:
        """Test authenticity score includes datetime bonus."""
        from app.models.responses import EXIFData
        
        exif_data = EXIFData(datetime_original="2024:01:15 14:30:00")
        score = validator.calculate_authenticity_score(exif_data)
        
        # 40 (has exif) + 30 (datetime) = 70
        assert score == 70
    
    def test_calculate_authenticity_score_full(
        self,
        validator: EvidenceValidator,
    ) -> None:
        """Test authenticity score with all metadata."""
        from app.models.responses import EXIFData
        
        exif_data = EXIFData(
            datetime_original="2024:01:15 14:30:00",
            device_make="Apple",
            device_model="iPhone 13",
            gps_latitude=37.7749,
            gps_longitude=-122.4194,
        )
        score = validator.calculate_authenticity_score(exif_data)
        
        # 40 + 30 + 20 + 10 = 100
        assert score == 100
    
    def test_get_trust_level_high(
        self,
        validator: EvidenceValidator,
    ) -> None:
        """Test trust level classification for high score."""
        assert validator.get_trust_level(85) == TrustLevelEnum.HIGH_TRUST
        assert validator.get_trust_level(100) == TrustLevelEnum.HIGH_TRUST
    
    def test_get_trust_level_medium(
        self,
        validator: EvidenceValidator,
    ) -> None:
        """Test trust level classification for medium score."""
        assert validator.get_trust_level(50) == TrustLevelEnum.MEDIUM_TRUST
        assert validator.get_trust_level(79) == TrustLevelEnum.MEDIUM_TRUST
    
    def test_get_trust_level_low(
        self,
        validator: EvidenceValidator,
    ) -> None:
        """Test trust level classification for low score."""
        assert validator.get_trust_level(20) == TrustLevelEnum.LOW_TRUST
        assert validator.get_trust_level(49) == TrustLevelEnum.LOW_TRUST
    
    def test_get_trust_level_untrusted(
        self,
        validator: EvidenceValidator,
    ) -> None:
        """Test trust level classification for untrusted score."""
        assert validator.get_trust_level(0) == TrustLevelEnum.UNTRUSTED
        assert validator.get_trust_level(19) == TrustLevelEnum.UNTRUSTED
    
    def test_detect_tampering_no_exif(
        self,
        validator: EvidenceValidator,
        sample_image_path: str,
    ) -> None:
        """Test tampering detection flags missing EXIF."""
        result = validator.detect_tampering(sample_image_path)
        
        # Simple test images have no EXIF, should be flagged
        assert result.tamper_probability > 0
        assert any("EXIF" in ind for ind in result.indicators)
    
    def test_validate_evidence_complete(
        self,
        validator: EvidenceValidator,
        sample_image_path: str,
    ) -> None:
        """Test complete evidence validation."""
        result = validator.validate_evidence(sample_image_path)
        
        assert result.authenticity_score >= 0
        assert result.trust_level is not None
        assert result.exif_data is not None
        assert result.tamper_analysis is not None
    
    def test_validate_evidence_with_claim(
        self,
        validator: EvidenceValidator,
        sample_image_path: str,
    ) -> None:
        """Test evidence validation with claim (without LLM)."""
        result = validator.validate_evidence(
            sample_image_path,
            claim_text="Water damage in kitchen",
        )
        
        # Without LLM, alignment should not be analyzed
        assert result.alignment_analysis is None
    
    @patch.object(EvidenceValidator, "analyze_evidence_alignment")
    def test_validate_evidence_with_llm(
        self,
        mock_alignment: MagicMock,
        validator: EvidenceValidator,
        sample_image_path: str,
    ) -> None:
        """Test evidence validation with LLM alignment analysis."""
        from app.models.responses import AlignmentAnalysis
        
        mock_alignment.return_value = AlignmentAnalysis(
            alignment_score=75,
            concerns=["Minor timing discrepancy"],
            reasoning="Evidence appears to support the claim.",
        )
        
        mock_llm = MagicMock()
        result = validator.validate_evidence(
            sample_image_path,
            claim_text="Water damage in kitchen",
            llm_service=mock_llm,
        )
        
        assert result.alignment_analysis is not None
        assert result.alignment_analysis.alignment_score == 75


class TestFileHash:
    """Tests for file hash calculation."""
    
    def test_hash_consistency(self, tmp_path: Path) -> None:
        """Test that same file produces same hash."""
        validator = EvidenceValidator()
        
        # Create test file
        img = Image.new("RGB", (50, 50), color="green")
        img_path = tmp_path / "hash_test.jpg"
        img.save(str(img_path), "JPEG")
        
        # Extract twice
        exif1 = validator.extract_exif(str(img_path))
        exif2 = validator.extract_exif(str(img_path))
        
        assert exif1.file_hash == exif2.file_hash
    
    def test_hash_uniqueness(self, tmp_path: Path) -> None:
        """Test that different files produce different hashes."""
        validator = EvidenceValidator()
        
        # Create two different images
        img1 = Image.new("RGB", (50, 50), color="red")
        img1_path = tmp_path / "img1.jpg"
        img1.save(str(img1_path), "JPEG")
        
        img2 = Image.new("RGB", (50, 50), color="blue")
        img2_path = tmp_path / "img2.jpg"
        img2.save(str(img2_path), "JPEG")
        
        exif1 = validator.extract_exif(str(img1_path))
        exif2 = validator.extract_exif(str(img2_path))
        
        assert exif1.file_hash != exif2.file_hash
