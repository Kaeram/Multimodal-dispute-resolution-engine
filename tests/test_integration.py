"""
Integration tests for RentShield AI Analysis Engine API.

Tests all API endpoints using FastAPI's TestClient.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create test client for API tests."""
    return TestClient(app)


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a sample test image."""
    img = Image.new("RGB", (100, 100), color="red")
    img_path = tmp_path / "test_image.jpg"
    img.save(str(img_path), "JPEG")
    return img_path


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_health_check(self, client: TestClient) -> None:
        """Test health endpoint returns expected structure."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "llm_connected" in data
        assert "version" in data
    
    def test_health_check_has_request_id(self, client: TestClient) -> None:
        """Test health endpoint includes request ID header."""
        response = client.get("/health")
        
        assert "X-Request-ID" in response.headers


class TestClassifyIssueEndpoint:
    """Tests for the issue classification endpoint."""
    
    @patch("app.api.endpoints.get_case_analyzer")
    def test_classify_issue_success(
        self,
        mock_analyzer: MagicMock,
        client: TestClient,
    ) -> None:
        """Test successful issue classification."""
        from app.models.responses import IssueCategoryEnum, IssueClassification, SeverityEnum
        
        mock_instance = MagicMock()
        mock_instance.classify_issue.return_value = IssueClassification(
            primary_category=IssueCategoryEnum.MAINTENANCE,
            confidence=85,
            severity=SeverityEnum.MEDIUM,
            reasoning="Issue involves water damage.",
            urgency_flag=False,
            keywords_detected=["water", "leak"],
        )
        mock_analyzer.return_value = mock_instance
        
        response = client.post(
            "/api/v1/classify-issue",
            json={
                "description": "Water has been leaking from the ceiling for 2 weeks",
                "evidence_count": 3,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["primary_category"] == "Maintenance"
        assert data["confidence"] == 85
    
    def test_classify_issue_validation_error(
        self,
        client: TestClient,
    ) -> None:
        """Test validation error for short description."""
        response = client.post(
            "/api/v1/classify-issue",
            json={
                "description": "Short",  # Too short
                "evidence_count": 0,
            },
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_classify_issue_missing_description(
        self,
        client: TestClient,
    ) -> None:
        """Test validation error for missing description."""
        response = client.post(
            "/api/v1/classify-issue",
            json={
                "evidence_count": 0,
            },
        )
        
        assert response.status_code == 400


class TestValidateEvidenceEndpoint:
    """Tests for the evidence validation endpoint."""
    
    @patch("app.api.endpoints.get_evidence_validator")
    def test_validate_evidence_success(
        self,
        mock_validator: MagicMock,
        client: TestClient,
        sample_image: Path,
    ) -> None:
        """Test successful evidence validation."""
        from app.models.responses import (
            EvidenceValidation,
            EXIFData,
            TamperAnalysis,
            TrustLevelEnum,
        )
        
        mock_instance = MagicMock()
        mock_instance.validate_evidence.return_value = EvidenceValidation(
            authenticity_score=75,
            trust_level=TrustLevelEnum.MEDIUM_TRUST,
            exif_data=EXIFData(dimensions="100x100"),
            tamper_analysis=TamperAnalysis(
                tamper_probability=0.1,
                indicators=["No issues found"],
                conclusion="Low risk",
            ),
        )
        mock_validator.return_value = mock_instance
        
        with open(sample_image, "rb") as f:
            response = client.post(
                "/api/v1/validate-evidence",
                files={"image": ("test.jpg", f, "image/jpeg")},
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["authenticity_score"] == 75
        assert data["trust_level"] == "MEDIUM TRUST"
    
    def test_validate_evidence_invalid_file_type(
        self,
        client: TestClient,
        tmp_path: Path,
    ) -> None:
        """Test validation error for wrong file type."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Not an image")
        
        with open(txt_file, "rb") as f:
            response = client.post(
                "/api/v1/validate-evidence",
                files={"image": ("test.txt", f, "text/plain")},
            )
        
        assert response.status_code == 400
        data = response.json()
        # Error is nested in detail for HTTPException responses
        error_info = data.get("detail", data)
        assert "invalid_file_type" in error_info.get("error", "")


class TestAnalyzeCaseEndpoint:
    """Tests for the case analysis endpoint."""
    
    @patch("app.api.endpoints.get_case_analyzer")
    def test_analyze_case_success(
        self,
        mock_analyzer: MagicMock,
        client: TestClient,
    ) -> None:
        """Test successful case analysis."""
        from app.models.responses import (
            ConfidenceLevelEnum,
            DAORecommendation,
            DAORecommendationDetails,
            EvidenceEvaluation,
            EvidenceQualityEnum,
            LandlordPosition,
            RecommendedOutcomeEnum,
            RedFlags,
            TenantPosition,
        )
        
        mock_instance = MagicMock()
        mock_instance.analyze_dispute.return_value = DAORecommendation(
            case_summary="Test case summary",
            tenant_position=TenantPosition(
                key_arguments=["Strong evidence"],
                evidence_strength=80,
                credibility_score=75,
                supporting_factors=["Documentation provided"],
            ),
            landlord_position=LandlordPosition(
                key_arguments=["Repairs scheduled"],
                evidence_strength=50,
                credibility_score=60,
                supporting_factors=["Appointment records"],
            ),
            evidence_evaluation=EvidenceEvaluation(
                tenant_evidence_quality=EvidenceQualityEnum.GOOD,
                landlord_evidence_quality=EvidenceQualityEnum.FAIR,
                metadata_authenticity=70,
                key_discrepancies=[],
                critical_gaps=[],
            ),
            dao_recommendation=DAORecommendationDetails(
                tenant_favor_confidence=70,
                landlord_favor_confidence=20,
                neutral_confidence=10,
                recommended_outcome=RecommendedOutcomeEnum.FAVOR_TENANT,
                confidence_level=ConfidenceLevelEnum.MEDIUM,
                reasoning="Tenant has stronger evidence.",
                key_considerations=["Evidence quality"],
                suggested_resolution="Complete repairs within 7 days.",
            ),
            red_flags=RedFlags(),
            next_steps=["Schedule repair"],
            estimated_resolution_timeline="7-14 days",
        )
        mock_analyzer.return_value = mock_instance
        
        response = client.post(
            "/api/v1/analyze-case",
            json={
                "issue_id": "case-123",
                "tenant_complaint": "Heating has been broken for 3 weeks causing health issues.",
                "landlord_response": "We scheduled repairs but tenant was unavailable.",
                "incident_date": "2024-01-10T00:00:00Z",
                "tenant_evidence": [],
                "landlord_evidence": [],
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["dao_recommendation"]["recommended_outcome"] == "Favor Tenant"
    
    def test_analyze_case_missing_complaint(
        self,
        client: TestClient,
    ) -> None:
        """Test validation error for missing complaint."""
        response = client.post(
            "/api/v1/analyze-case",
            json={
                "issue_id": "case-123",
                "incident_date": "2024-01-10T00:00:00Z",
            },
        )
        
        assert response.status_code == 400


class TestBatchClassifyEndpoint:
    """Tests for the batch classification endpoint."""
    
    @patch("app.api.endpoints.get_case_analyzer")
    def test_batch_classify_success(
        self,
        mock_analyzer: MagicMock,
        client: TestClient,
    ) -> None:
        """Test successful batch classification."""
        from app.models.responses import IssueCategoryEnum, IssueClassification, SeverityEnum
        
        mock_instance = MagicMock()
        mock_instance.classify_issue.return_value = IssueClassification(
            primary_category=IssueCategoryEnum.MAINTENANCE,
            confidence=75,
            severity=SeverityEnum.MEDIUM,
            reasoning="Standard maintenance issue.",
            urgency_flag=False,
            keywords_detected=[],
        )
        mock_analyzer.return_value = mock_instance
        
        response = client.post(
            "/api/v1/batch-classify",
            json={
                "issues": [
                    {"description": "Water leak in bathroom ceiling", "evidence_count": 2},
                    {"description": "Noisy neighbors disturbing sleep", "evidence_count": 0},
                ],
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 2
        assert len(data["results"]) == 2
    
    def test_batch_classify_empty_list(
        self,
        client: TestClient,
    ) -> None:
        """Test validation error for empty issues list."""
        response = client.post(
            "/api/v1/batch-classify",
            json={"issues": []},
        )
        
        assert response.status_code == 400


class TestRootEndpoint:
    """Tests for the root endpoint."""
    
    def test_root_returns_info(self, client: TestClient) -> None:
        """Test root endpoint returns service info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "docs" in data
