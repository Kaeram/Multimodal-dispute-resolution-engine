"""
Vision Service for RentShield AI Analysis Engine.

Provides LLaVA-based image understanding for housing inspection evidence.
Downloads images, analyzes them using the LLaVA multimodal model via Ollama,
and returns structured scene descriptions.
"""

import base64
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests

from app.config import get_settings
from app.utils.exceptions import InvalidImageError, LLMConnectionError, LLMTimeoutError
from app.utils.logger import get_logger

logger = get_logger(__name__)


# LLaVA prompt for housing inspection analysis
LLAVA_PROMPT = """You are a housing inspection AI.

Describe this image focusing ONLY on:
• structural damage
• safety hazards
• maintenance issues
• signs of mold, leaks, cracks, fire risk
• appliances or utilities
• anything relevant to a tenant complaint

Return STRICT JSON with no additional text:

{
  "scene_summary": "2 sentence description",
  "detected_objects": [],
  "damage_detected": [],
  "safety_hazards": [],
  "cleanliness_level": "clean|average|dirty|unsanitary",
  "indoor_outdoor": "indoor|outdoor|unclear",
  "confidence": 0-100
}"""


class VisionService:
    """
    LLaVA-based image understanding service.
    
    Communicates with Ollama's LLaVA model to analyze housing-related
    evidence images and generate structured scene descriptions.
    
    Attributes:
        base_url: Ollama API base URL.
        model: LLaVA model name.
        timeout: Request timeout in seconds.
        
    Example:
        >>> service = VisionService()
        >>> if service.health_check():
        ...     result = service.analyze_image("/path/to/image.jpg")
        ...     print(result["scene_summary"])
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "llava",
        timeout: int = 90,
    ) -> None:
        """
        Initialize the Vision service.
        
        Args:
            base_url: Ollama API URL. Defaults to settings value.
            model: LLaVA model name. Defaults to "llava".
            timeout: Request timeout in seconds. Defaults to 90.
        """
        settings = get_settings()
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model = model
        self.timeout = timeout
        self.temp_dir = Path(settings.UPLOAD_DIR) / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "VisionService initialized",
            base_url=self.base_url,
            model=self.model,
            timeout=self.timeout,
        )
    
    def health_check(self) -> bool:
        """
        Check if LLaVA model is available in Ollama.
        
        Calls Ollama /api/tags endpoint and verifies that the
        llava model is installed and available.
        
        Returns:
            bool: True if LLaVA is available, False otherwise.
            
        Example:
            >>> service = VisionService()
            >>> service.health_check()
            True
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )
            response.raise_for_status()
            
            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            
            llava_available = self.model.split(":")[0] in model_names
            
            logger.info(
                "Vision health check completed",
                llava_available=llava_available,
                available_models=model_names,
            )
            
            return llava_available
            
        except requests.RequestException as e:
            logger.warning(
                "Vision health check failed",
                error=str(e),
            )
            return False
    
    def download_image(self, url: str) -> str:
        """
        Download an image from URL to temporary storage.
        
        Validates file size (<10MB) and extension (jpg, jpeg, png).
        Returns the local file path for further processing.
        
        Args:
            url: URL of the image to download.
            
        Returns:
            str: Local file path to the downloaded image.
            
        Raises:
            InvalidImageError: If URL is invalid, file too large, or wrong format.
            
        Example:
            >>> service = VisionService()
            >>> path = service.download_image("https://example.com/photo.jpg")
            >>> print(path)
            '/uploads/temp/abc123.jpg'
        """
        settings = get_settings()
        
        logger.info("Downloading image", url=url[:100])
        
        # Parse URL and extract extension
        parsed = urlparse(url)
        path = parsed.path.lower()
        ext = Path(path).suffix or ".jpg"
        
        # Validate extension
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise InvalidImageError(
                f"Invalid file type: {ext}. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        try:
            # Stream download with size limit check
            response = requests.get(
                url,
                stream=True,
                timeout=30,
                headers={"User-Agent": "RentShield-Vision/1.0"},
            )
            response.raise_for_status()
            
            # Check content length header if available
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > settings.MAX_FILE_SIZE:
                raise InvalidImageError(
                    f"File too large. Max size: {settings.MAX_FILE_SIZE // (1024*1024)}MB"
                )
            
            # Create temp file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=ext,
                dir=str(self.temp_dir),
            )
            
            # Download with size tracking
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                total_size += len(chunk)
                if total_size > settings.MAX_FILE_SIZE:
                    temp_file.close()
                    os.unlink(temp_file.name)
                    raise InvalidImageError(
                        f"File too large. Max size: {settings.MAX_FILE_SIZE // (1024*1024)}MB"
                    )
                temp_file.write(chunk)
            
            temp_file.close()
            
            logger.info(
                "Image downloaded successfully",
                path=temp_file.name,
                size_bytes=total_size,
            )
            
            return temp_file.name
            
        except requests.RequestException as e:
            logger.error("Failed to download image", error=str(e), url=url[:100])
            raise InvalidImageError(f"Failed to download image: {str(e)}")
    
    def analyze_image(self, image_path: str) -> dict[str, Any]:
        """
        Analyze an image using LLaVA model.
        
        Sends the image to Ollama's LLaVA model with a housing inspection
        prompt and returns structured JSON analysis.
        
        Args:
            image_path: Local path to the image file.
            
        Returns:
            dict: Structured analysis with keys:
                - scene_summary: 2-sentence description
                - detected_objects: List of objects
                - damage_detected: List of damage types
                - safety_hazards: List of hazards
                - cleanliness_level: clean|average|dirty|unsanitary
                - indoor_outdoor: indoor|outdoor|unclear
                - confidence: 0-100
                
        Raises:
            InvalidImageError: If image file is invalid.
            LLMConnectionError: If Ollama connection fails.
            LLMTimeoutError: If analysis times out.
            
        Example:
            >>> service = VisionService()
            >>> result = service.analyze_image("/path/to/damage.jpg")
            >>> print(result["damage_detected"])
            ['water damage', 'mold growth']
        """
        # Validate file exists
        if not os.path.exists(image_path):
            raise InvalidImageError(f"Image file not found: {image_path}")
        
        logger.info("Analyzing image with LLaVA", path=image_path)
        start_time = time.time()
        
        try:
            # Read and encode image as base64
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Prepare request payload for Ollama
            payload = {
                "model": self.model,
                "prompt": LLAVA_PROMPT,
                "images": [image_data],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1024,
                },
            }
            
            # Send request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            
            result = response.json()
            response_text = result.get("response", "")
            
            # Parse JSON from response
            analysis = self._parse_json_response(response_text)
            
            elapsed = time.time() - start_time
            logger.info(
                "Image analysis completed",
                elapsed_seconds=round(elapsed, 2),
                confidence=analysis.get("confidence", 0),
            )
            
            return analysis
            
        except requests.Timeout:
            logger.error("LLaVA analysis timed out", path=image_path)
            raise LLMTimeoutError(
                f"Image analysis timed out after {self.timeout} seconds"
            )
        except requests.RequestException as e:
            logger.error("LLaVA connection error", error=str(e))
            raise LLMConnectionError(f"Failed to connect to LLaVA: {str(e)}")
    
    def _parse_json_response(self, response_text: str) -> dict[str, Any]:
        """
        Parse JSON from LLaVA response, handling formatting issues.
        
        Args:
            response_text: Raw text response from LLaVA.
            
        Returns:
            dict: Parsed JSON object with validated structure.
        """
        # Try direct parse first
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object pattern
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Return default structure if parsing fails
        logger.warning(
            "Failed to parse LLaVA JSON response, using defaults",
            response_preview=response_text[:200],
        )
        
        return {
            "scene_summary": response_text[:200] if response_text else "Unable to analyze image",
            "detected_objects": [],
            "damage_detected": [],
            "safety_hazards": [],
            "cleanliness_level": "unclear",
            "indoor_outdoor": "unclear",
            "confidence": 30,
        }
    
    def cleanup_temp_file(self, file_path: str) -> None:
        """
        Delete a temporary file.
        
        Args:
            file_path: Path to the file to delete.
        """
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug("Cleaned up temp file", path=file_path)
        except OSError as e:
            logger.warning("Failed to cleanup temp file", path=file_path, error=str(e))
