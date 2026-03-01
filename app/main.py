"""
RentShield AI Analysis Engine - FastAPI Application.

Main application entry point with CORS, error handlers, and startup events.
Run with: uvicorn app.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.endpoints import router
from app.config import get_settings
from app.services.llm_service import OllamaLLMService
from app.utils.exceptions import RentShieldBaseException
from app.utils.logger import (
    clear_request_id,
    get_logger,
    set_request_id,
    setup_logging,
)

# Initialize logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(
        "RentShield AI Analysis Engine starting",
        version=__version__,
    )
    
    # Test LLM connection
    llm_service = OllamaLLMService()
    if llm_service.test_connection():
        logger.info(
            "LLM connection verified",
            model=llm_service.model,
        )
    else:
        logger.warning(
            "LLM connection failed - service will run in degraded mode",
            base_url=llm_service.base_url,
        )
    
    # Create upload directory
    settings = get_settings()
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    logger.info("Upload directory ready", path=str(settings.upload_path))
    
    yield
    
    # Shutdown
    logger.info("RentShield AI Analysis Engine shutting down")


# Initialize FastAPI application
settings = get_settings()

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ============================================================================
# CORS Middleware
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request ID Middleware
# ============================================================================

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Add request ID to all requests for tracing."""
    # Generate or extract request ID
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = set_request_id()
    else:
        set_request_id(request_id)
    
    # Process request
    response = await call_next(request)
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    # Clear context
    clear_request_id()
    
    return response


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(RentShieldBaseException)
async def rentshield_exception_handler(
    request: Request,
    exc: RentShieldBaseException,
) -> JSONResponse:
    """Handle all RentShield custom exceptions."""
    logger.error(
        "RentShield exception",
        error_type=exc.error_type,
        message=exc.message,
        details=exc.details,
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            **exc.to_dict(),
            "request_id": request.headers.get("X-Request-ID", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors."""
    logger.warning(
        "Validation error",
        errors=exc.errors(),
    )
    
    # Format errors for response
    formatted_errors = []
    for error in exc.errors():
        loc = " -> ".join(str(l) for l in error.get("loc", []))
        formatted_errors.append({
            "field": loc,
            "message": error.get("msg", "Unknown error"),
            "type": error.get("type", "unknown"),
        })
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "validation_error",
            "message": "Request validation failed",
            "details": {"errors": formatted_errors},
            "request_id": request.headers.get("X-Request-ID", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle all unhandled exceptions."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "details": {"error": str(exc)},
            "request_id": request.headers.get("X-Request-ID", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# ============================================================================
# Routes
# ============================================================================

# Include API router
app.include_router(router)


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {
        "service": "RentShield AI Analysis Engine",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
