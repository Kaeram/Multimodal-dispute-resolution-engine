"""
Tests for LLM Service.

Uses mocked Ollama responses to test LLM service functionality
without requiring a running Ollama instance.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from app.services.llm_service import OllamaLLMService
from app.utils.exceptions import LLMConnectionError, LLMTimeoutError


class TestOllamaLLMService:
    """Tests for OllamaLLMService class."""
    
    def test_init_default_settings(self) -> None:
        """Test service initializes with default settings."""
        service = OllamaLLMService()
        
        assert service.base_url == "http://localhost:11434"
        assert service.model == "mistral"
        assert service.timeout > 0
    
    def test_init_custom_settings(self) -> None:
        """Test service initializes with custom settings."""
        service = OllamaLLMService(
            base_url="http://custom:11434",
            model="llama2",
            timeout=60,
        )
        
        assert service.base_url == "http://custom:11434"
        assert service.model == "llama2"
        assert service.timeout == 60
    
    @patch("app.services.llm_service.requests.get")
    def test_test_connection_success(self, mock_get: MagicMock) -> None:
        """Test connection check returns True when Ollama is available."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "mistral:latest"}]
        }
        mock_get.return_value = mock_response
        
        service = OllamaLLMService()
        result = service.test_connection()
        
        assert result is True
    
    @patch("app.services.llm_service.requests.get")
    def test_test_connection_failure(self, mock_get: MagicMock) -> None:
        """Test connection check returns False when Ollama is unavailable."""
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        service = OllamaLLMService()
        result = service.test_connection()
        
        assert result is False
    
    @patch("app.services.llm_service.requests.get")
    def test_test_connection_model_not_found(self, mock_get: MagicMock) -> None:
        """Test connection check returns False when model is not available."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [{"name": "llama2:latest"}]  # Different model
        }
        mock_get.return_value = mock_response
        
        service = OllamaLLMService()
        result = service.test_connection()
        
        assert result is False
    
    @patch("app.services.llm_service.requests.post")
    def test_query_success(self, mock_post: MagicMock) -> None:
        """Test query returns parsed JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"category": "Maintenance", "confidence": 85}'
        }
        mock_post.return_value = mock_response
        
        service = OllamaLLMService()
        result = service.query("Test prompt", expect_json=True)
        
        assert result["category"] == "Maintenance"
        assert result["confidence"] == 85
    
    @patch("app.services.llm_service.requests.post")
    def test_query_json_in_markdown(self, mock_post: MagicMock) -> None:
        """Test query extracts JSON from markdown code blocks."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '```json\n{"category": "Safety"}\n```'
        }
        mock_post.return_value = mock_response
        
        service = OllamaLLMService()
        result = service.query("Test prompt", expect_json=True)
        
        assert result["category"] == "Safety"
    
    @patch("app.services.llm_service.requests.post")
    def test_query_timeout(self, mock_post: MagicMock) -> None:
        """Test query raises LLMTimeoutError on timeout."""
        mock_post.side_effect = requests.exceptions.Timeout()
        
        service = OllamaLLMService()
        service.max_retries = 1  # Reduce retries for test
        
        with pytest.raises(LLMTimeoutError):
            service.query("Test prompt")
    
    @patch("app.services.llm_service.requests.post")
    def test_query_connection_error(self, mock_post: MagicMock) -> None:
        """Test query raises LLMConnectionError on connection failure."""
        mock_post.side_effect = requests.exceptions.ConnectionError()
        
        service = OllamaLLMService()
        service.max_retries = 1
        
        with pytest.raises(LLMConnectionError):
            service.query("Test prompt")
    
    @patch("app.services.llm_service.requests.post")
    def test_query_raw_text(self, mock_post: MagicMock) -> None:
        """Test query returns raw text when expect_json=False."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "This is a plain text response."
        }
        mock_post.return_value = mock_response
        
        service = OllamaLLMService()
        result = service.query("Test prompt", expect_json=False)
        
        assert result["response"] == "This is a plain text response."
    
    @patch("app.services.llm_service.requests.post")
    def test_batch_query(self, mock_post: MagicMock) -> None:
        """Test batch query processes multiple prompts."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"result": "success"}'
        }
        mock_post.return_value = mock_response
        
        service = OllamaLLMService()
        results = service.batch_query(["Prompt 1", "Prompt 2"])
        
        assert len(results) == 2
        assert all(r["result"] == "success" for r in results)
    
    def test_parse_json_response_direct(self) -> None:
        """Test JSON parsing with direct JSON."""
        service = OllamaLLMService()
        result = service._parse_json_response('{"key": "value"}')
        
        assert result["key"] == "value"
    
    def test_parse_json_response_with_markdown(self) -> None:
        """Test JSON parsing with markdown code blocks."""
        service = OllamaLLMService()
        result = service._parse_json_response('```json\n{"key": "value"}\n```')
        
        assert result["key"] == "value"
    
    def test_parse_json_response_invalid(self) -> None:
        """Test JSON parsing with invalid JSON."""
        service = OllamaLLMService()
        result = service._parse_json_response("This is not JSON")
        
        assert result["parse_error"] is True
        assert "raw_response" in result
