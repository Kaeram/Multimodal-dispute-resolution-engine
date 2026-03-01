"""
Ollama LLM Service for RentShield AI Analysis Engine.

Provides integration with local Ollama instance running Mistral model.
Includes retry logic, JSON parsing, and connection health checks.
"""

import json
import re
import time
from typing import Any, Optional

import requests

from app.config import get_settings
from app.utils.exceptions import LLMConnectionError, LLMTimeoutError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaLLMService:
    """
    Service for communicating with local Ollama LLM instance.
    
    This service handles all LLM interactions including connection testing,
    query execution with retry logic, and JSON response parsing.
    
    Attributes:
        base_url: Ollama API base URL.
        model: Name of the model to use.
        timeout: Request timeout in seconds.
        
    Example:
        >>> service = OllamaLLMService()
        >>> if service.test_connection():
        ...     result = service.query("Classify this issue...")
        ...     print(result)
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """
        Initialize the Ollama LLM service.
        
        Args:
            base_url: Ollama API URL. Defaults to settings value.
            model: Model name. Defaults to settings value.
            timeout: Request timeout. Defaults to settings value.
        """
        settings = get_settings()
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT
        self.max_tokens = settings.OLLAMA_MAX_TOKENS
        self.temperature = settings.OLLAMA_TEMPERATURE
        self.max_retries = settings.MAX_RETRIES
        self.retry_delay = settings.RETRY_DELAY
    
    def test_connection(self) -> bool:
        """
        Test connection to Ollama and verify model availability.
        
        Returns:
            bool: True if Ollama is running and model is available.
            
        Example:
            >>> service = OllamaLLMService()
            >>> service.test_connection()
            True
        """
        try:
            # Check if Ollama is running
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )
            
            if response.status_code != 200:
                logger.warning(
                    "Ollama health check failed",
                    status_code=response.status_code,
                )
                return False
            
            # Check if model is available
            models_data = response.json()
            available_models = [
                m.get("name", "").split(":")[0]
                for m in models_data.get("models", [])
            ]
            
            if self.model not in available_models:
                logger.warning(
                    "Model not found in Ollama",
                    model=self.model,
                    available_models=available_models,
                )
                return False
            
            logger.info(
                "Ollama connection successful",
                model=self.model,
                base_url=self.base_url,
            )
            return True
            
        except requests.exceptions.ConnectionError:
            logger.error(
                "Cannot connect to Ollama",
                base_url=self.base_url,
            )
            return False
        except requests.exceptions.Timeout:
            logger.error("Ollama connection timed out")
            return False
        except Exception as e:
            logger.error(
                "Unexpected error testing Ollama connection",
                error=str(e),
            )
            return False
    
    def get_available_model(self) -> Optional[str]:
        """
        Get the available model name from Ollama.
        
        Returns:
            Optional[str]: Model name if available, None otherwise.
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )
            if response.status_code == 200:
                models_data = response.json()
                models = models_data.get("models", [])
                if models:
                    # Return first model that matches our configured model
                    for m in models:
                        name = m.get("name", "").split(":")[0]
                        if name == self.model:
                            return self.model
                    # Fall back to first available model
                    return models[0].get("name", "").split(":")[0]
            return None
        except Exception:
            return None
    
    def query(
        self,
        prompt: str,
        expect_json: bool = True,
        timeout: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Send a prompt to the LLM and get a response.
        
        Args:
            prompt: The prompt to send to the model.
            expect_json: Whether to parse response as JSON.
            timeout: Optional custom timeout for this request.
            system_prompt: Optional system prompt for context.
            
        Returns:
            dict: Parsed JSON response or {"response": raw_text}.
            
        Raises:
            LLMConnectionError: If connection to Ollama fails.
            LLMTimeoutError: If request times out.
            
        Example:
            >>> service = OllamaLLMService()
            >>> result = service.query(
            ...     "Classify this issue: broken window",
            ...     expect_json=True
            ... )
            >>> print(result.get("category"))
        """
        request_timeout = timeout or self.timeout
        last_error: Optional[Exception] = None
        
        # Build the full prompt
        if expect_json:
            json_instruction = (
                "\n\nIMPORTANT: You must respond ONLY with valid JSON. "
                "Do not include any text before or after the JSON object. "
                "Do not use markdown code blocks."
            )
            full_prompt = prompt + json_instruction
        else:
            full_prompt = prompt
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                logger.debug(
                    "Sending query to Ollama",
                    model=self.model,
                    attempt=attempt + 1,
                    prompt_length=len(prompt),
                )
                
                # Build request payload
                payload: dict[str, Any] = {
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "num_predict": self.max_tokens,
                        "temperature": self.temperature,
                    },
                }
                
                if system_prompt:
                    payload["system"] = system_prompt
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=request_timeout,
                )
                
                elapsed = time.time() - start_time
                
                if response.status_code != 200:
                    logger.warning(
                        "Ollama returned error status",
                        status_code=response.status_code,
                        response_text=response.text[:500],
                    )
                    raise LLMConnectionError(
                        message=f"Ollama returned status {response.status_code}",
                        details={"status_code": response.status_code},
                    )
                
                result = response.json()
                response_text = result.get("response", "")
                
                logger.info(
                    "Ollama query completed",
                    elapsed_seconds=round(elapsed, 2),
                    response_length=len(response_text),
                )
                
                if expect_json:
                    return self._parse_json_response(response_text)
                else:
                    return {"response": response_text}
                    
            except requests.exceptions.Timeout:
                last_error = LLMTimeoutError(
                    timeout_seconds=request_timeout,
                )
                logger.warning(
                    "Ollama query timed out",
                    attempt=attempt + 1,
                    timeout=request_timeout,
                )
                
            except requests.exceptions.ConnectionError as e:
                last_error = LLMConnectionError(
                    message="Failed to connect to Ollama",
                    details={"error": str(e)},
                )
                logger.warning(
                    "Connection error to Ollama",
                    attempt=attempt + 1,
                    error=str(e),
                )
                
            except LLMConnectionError:
                raise
                
            except Exception as e:
                last_error = LLMConnectionError(
                    message=f"Unexpected error: {str(e)}",
                    details={"error": str(e)},
                )
                logger.warning(
                    "Unexpected error in Ollama query",
                    attempt=attempt + 1,
                    error=str(e),
                )
            
            # Wait before retry with exponential backoff
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                logger.debug(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        
        # All retries failed
        if isinstance(last_error, LLMTimeoutError):
            raise last_error
        elif isinstance(last_error, LLMConnectionError):
            raise last_error
        else:
            raise LLMConnectionError(
                message="Failed to complete LLM query after all retries",
                details={"attempts": self.max_retries},
            )
    
    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        """
        Parse JSON from LLM response, handling common formatting issues.
        
        Args:
            response_text: Raw text response from LLM.
            
        Returns:
            dict: Parsed JSON object.
        """
        text = response_text.strip()
        
        # Try direct JSON parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code blocks
        json_patterns = [
            r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
            r"```\s*([\s\S]*?)\s*```",       # ``` ... ```
            r"\{[\s\S]*\}",                   # Raw JSON object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # If pattern is the raw JSON match, use directly
                    if pattern == r"\{[\s\S]*\}":
                        return json.loads(match)
                    else:
                        return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        
        # If all parsing fails, return structured error response
        logger.warning(
            "Failed to parse JSON from LLM response",
            response_preview=text[:200],
        )
        return {
            "parse_error": True,
            "raw_response": text,
            "warning": "Could not parse JSON from LLM response",
        }
    
    def batch_query(
        self,
        prompts: list[str],
        expect_json: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Process multiple prompts sequentially.
        
        Args:
            prompts: List of prompts to process.
            expect_json: Whether to parse responses as JSON.
            
        Returns:
            list: List of response dictionaries.
            
        Example:
            >>> service = OllamaLLMService()
            >>> results = service.batch_query([
            ...     "Classify: broken window",
            ...     "Classify: water leak"
            ... ])
        """
        results = []
        
        logger.info(
            "Starting batch query",
            count=len(prompts),
        )
        
        for i, prompt in enumerate(prompts):
            try:
                result = self.query(prompt, expect_json=expect_json)
                results.append(result)
                
                logger.debug(
                    "Batch query progress",
                    completed=i + 1,
                    total=len(prompts),
                )
                
            except Exception as e:
                logger.error(
                    "Batch query item failed",
                    index=i,
                    error=str(e),
                )
                results.append({
                    "error": True,
                    "message": str(e),
                    "index": i,
                })
        
        logger.info(
            "Batch query completed",
            total=len(prompts),
            successful=sum(1 for r in results if not r.get("error")),
        )
        
        return results
