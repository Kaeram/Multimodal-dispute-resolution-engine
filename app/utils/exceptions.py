"""
Custom exceptions for RentShield AI Analysis Engine.

All exceptions inherit from RentShieldBaseException and include
structured error information for consistent API error responses.
"""

from typing import Any, Optional


class RentShieldBaseException(Exception):
    """
    Base exception for all RentShield errors.
    
    Provides consistent error structure for API responses.
    
    Attributes:
        error_type: Machine-readable error type identifier.
        message: Human-readable error message.
        details: Optional additional error details.
        
    Example:
        >>> raise RentShieldBaseException(
        ...     error_type="general_error",
        ...     message="Something went wrong",
        ...     details={"context": "additional info"}
        ... )
    """
    
    def __init__(
        self,
        error_type: str,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API response."""
        return {
            "error": self.error_type,
            "message": self.message,
            "details": self.details,
        }


class LLMConnectionError(RentShieldBaseException):
    """
    Raised when connection to Ollama LLM fails.
    
    Example:
        >>> raise LLMConnectionError(
        ...     message="Cannot connect to Ollama at localhost:11434"
        ... )
    """
    
    def __init__(
        self,
        message: str = "Failed to connect to LLM service",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            error_type="llm_connection_error",
            message=message,
            details=details,
        )


class LLMTimeoutError(RentShieldBaseException):
    """
    Raised when LLM query times out.
    
    Example:
        >>> raise LLMTimeoutError(timeout_seconds=120)
    """
    
    def __init__(
        self,
        message: str = "LLM query timed out",
        timeout_seconds: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        error_details = details or {}
        if timeout_seconds is not None:
            error_details["timeout_seconds"] = timeout_seconds
        super().__init__(
            error_type="llm_timeout_error",
            message=message,
            details=error_details,
        )


class InvalidImageError(RentShieldBaseException):
    """
    Raised when an uploaded image is invalid.
    
    Example:
        >>> raise InvalidImageError(
        ...     message="File is not a valid image",
        ...     filename="suspicious.exe"
        ... )
    """
    
    def __init__(
        self,
        message: str = "Invalid image file",
        filename: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        error_details = details or {}
        if filename is not None:
            error_details["filename"] = filename
        super().__init__(
            error_type="invalid_image_error",
            message=message,
            details=error_details,
        )


class ExifExtractionError(RentShieldBaseException):
    """
    Raised when EXIF extraction fails.
    
    Example:
        >>> raise ExifExtractionError(
        ...     message="Could not read EXIF data",
        ...     filename="damaged.jpg"
        ... )
    """
    
    def __init__(
        self,
        message: str = "Failed to extract EXIF metadata",
        filename: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        error_details = details or {}
        if filename is not None:
            error_details["filename"] = filename
        super().__init__(
            error_type="exif_extraction_error",
            message=message,
            details=error_details,
        )


class AnalysisError(RentShieldBaseException):
    """
    Raised when case analysis fails.
    
    Example:
        >>> raise AnalysisError(
        ...     message="Failed to analyze dispute case",
        ...     case_id="case-123"
        ... )
    """
    
    def __init__(
        self,
        message: str = "Analysis failed",
        case_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        error_details = details or {}
        if case_id is not None:
            error_details["case_id"] = case_id
        super().__init__(
            error_type="analysis_error",
            message=message,
            details=error_details,
        )


class ValidationError(RentShieldBaseException):
    """
    Raised when request validation fails.
    
    Example:
        >>> raise ValidationError(
        ...     message="Description is required",
        ...     field="description"
        ... )
    """
    
    def __init__(
        self,
        message: str = "Validation failed",
        field: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        error_details = details or {}
        if field is not None:
            error_details["field"] = field
        super().__init__(
            error_type="validation_error",
            message=message,
            details=error_details,
        )
