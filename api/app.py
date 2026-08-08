from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
import os
import sys
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.predict import SeverityPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IT Ticket Severity Score Calculator API",
    description="""
    🎫 **IT Ticket Severity Calculator API**
    
    A production-ready REST API for predicting IT ticket severity scores using AI/ML.
    
    ## Features
    - 🌍 **Multilingual Support**: English and Hindi
    - 🎯 **Accurate Predictions**: AI-powered severity scoring (10-100)
    - ⚡ **Real-time Processing**: Instant predictions
    - 📊 **Batch Processing**: Handle multiple tickets
    - 🔍 **Analytics**: Usage statistics and insights
    - 🛡️ **Robust**: Comprehensive error handling
    
    ## Severity Categories
    - **Critical (80-100)**: System-wide outages, data loss
    - **High (60-79)**: Major functionality affected  
    - **Medium (40-59)**: Moderate impact on users
    - **Low (20-39)**: Minor issues, individual users
    - **Minimal (10-19)**: Requests, questions
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models for request/response
class TicketRequest(BaseModel):
    ticket_text: str = Field(
        ..., 
        description="IT ticket text in English or Hindi",
        min_length=1,
        max_length=5000,
        example="Server is down and users cannot access email"
    )
    
    @field_validator('ticket_text')
    @classmethod
    def validate_ticket_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Ticket text cannot be empty or only whitespace')
        return v.strip()

class SeverityResponse(BaseModel):
    severity_score: float = Field(
        ..., 
        description="Severity score between 10-100",
        ge=10,
        le=100,
        example=75.5
    )
    severity_category: str = Field(
        ..., 
        description="Severity category (High, Medium, Low)",
        example="High"
    )
    confidence: float = Field(
        ..., 
        description="Prediction confidence score between 0-1",
        ge=0,
        le=1,
        example=0.85
    )
    detected_language: str = Field(
        ..., 
        description="Detected language of input text",
        example="en"
    )
    processed_text: str = Field(
        ..., 
        description="Preprocessed version of input text",
        example="server down users access email"
    )
    timestamp: str = Field(
        ..., 
        description="Prediction timestamp in ISO format",
        example="2024-01-15T10:30:00.123456"
    )

class BatchTicketRequest(BaseModel):
    tickets: List[str] = Field(
        ...,
        description="List of IT ticket texts",
        max_items=100,
        example=[
            "Server is down and users cannot access email",
            "Printer not working in office",
            "सर्वर डाउन है"
        ]
    )

class BatchSeverityResponse(BaseModel):
    predictions: List[SeverityResponse]
    total_tickets: int
    processing_time_seconds: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_info: Dict[str, Any]

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str

# Global predictor instance
predictor = None

def get_predictor() -> SeverityPredictor:
    """Get or lazily initialize the predictor."""
    global predictor
    if predictor is not None:
        return predictor
    
    logger.info("Initializing severity predictor...")
    model_dir = "models" if os.path.exists("models") else "../models"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    p = SeverityPredictor(model_dir=model_dir)
    p._ensure_embeddings_loaded()
    predictor = p
    logger.info("Severity predictor initialized successfully and ready for inference.")
    return predictor

@app.on_event("startup")
async def startup_event():
    """Startup hook: triggers background model warmup without blocking socket port binding."""
    logger.info("Server startup: immediate socket binding enabled.")
    import asyncio
    asyncio.create_task(asyncio.to_thread(get_predictor))

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """Serve the main web interface."""
    return FileResponse('static/index.html')

@app.get("/api", response_model=dict)
async def api_root():
    """Root endpoint with API information."""
    return {
        "message": "IT Ticket Severity Score Calculator API",
        "version": "1.0.0",
        "description": "Bilingual (English/Hindi) ML service for predicting IT ticket severity scores",
        "endpoints": {
            "predict": "/predict - Predict severity for a single ticket",
            "predict_batch": "/predict/batch - Predict severity for multiple tickets",
            "health": "/health - Health check",
            "docs": "/docs - API documentation"
        }
    }

@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse)
async def health_check():
    """Health check endpoint: returns 200 OK immediately for Render probes."""
    try:
        model_info = predictor.get_model_info() if predictor is not None else {"status": "warming_up"}
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            model_info=model_info
        )
    except Exception as e:
        logger.warning(f"Health check info: {str(e)}")
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            model_info={"status": "initializing"}
        )

@app.post("/predict", response_model=SeverityResponse)
async def predict_severity(request: TicketRequest):
    """
    Predict severity score for a single IT ticket.
    
    - **ticket_text**: The IT ticket text in English or Hindi
    - Returns severity score (10-100), category, and additional metadata
    """
    try:
        p = get_predictor()
        
        # Validate input
        if not request.ticket_text.strip():
            raise HTTPException(status_code=400, detail="Ticket text cannot be empty")
        
        # Make prediction
        result = p.predict_single(request.ticket_text)
        
        # Check for prediction errors
        if 'error' in result:
            logger.error(f"Prediction error: {result['error']}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {result['error']}")
        
        # Validate prediction
        if not p.validate_prediction(result):
            logger.error(f"Invalid prediction result: {result}")
            raise HTTPException(status_code=500, detail="Invalid prediction result")
        
        return SeverityResponse(
            severity_score=result['severity_score'],
            severity_category=result['severity_category'],
            confidence=result.get('confidence', 0.0),
            detected_language=result.get('detected_language', 'unknown'),
            processed_text=result.get('processed_text', request.ticket_text),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict/batch", response_model=BatchSeverityResponse)
async def predict_batch_severity(request: BatchTicketRequest):
    """
    Predict severity scores for multiple IT tickets.
    
    - **tickets**: List of IT ticket texts (max 100)
    - Returns list of predictions with processing time
    """
    try:
        p = get_predictor()
        
        # Validate input
        if not request.tickets:
            raise HTTPException(status_code=400, detail="Tickets list cannot be empty")
        
        if len(request.tickets) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 tickets allowed per batch")
        
        # Filter out empty tickets
        valid_tickets = [ticket.strip() for ticket in request.tickets if ticket.strip()]
        
        if not valid_tickets:
            raise HTTPException(status_code=400, detail="No valid tickets found")
        
        # Record start time
        start_time = datetime.now()
        
        # Make batch predictions
        results = p.predict_batch(valid_tickets)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Convert results to response format
        predictions = []
        for result in results:
            if 'error' not in result and p.validate_prediction(result):
                predictions.append(SeverityResponse(
                    severity_score=result['severity_score'],
                    severity_category=result['severity_category'],
                    confidence=result.get('confidence', 0.0),
                    detected_language=result.get('detected_language', 'unknown'),
                    processed_text=result.get('processed_text', ''),
                    timestamp=datetime.now().isoformat()
                ))
            else:
                # Handle failed predictions
                logger.warning(f"Failed prediction in batch: {result.get('error', 'Unknown error')}")
                predictions.append(SeverityResponse(
                    severity_score=50.0,  # Default score
                    severity_category="Medium",
                    confidence=0.0,
                    detected_language="unknown",
                    processed_text="",
                    timestamp=datetime.now().isoformat()
                ))
        
        return BatchSeverityResponse(
            predictions=predictions,
            total_tickets=len(valid_tickets),
            processing_time_seconds=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/model/info", response_model=dict)
async def get_model_info():
    """Get information about the loaded model."""
    try:
        p = get_predictor()
        model_info = p.get_model_info()
        return {
            "model_info": model_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model info endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/debug/memory", response_model=dict)
async def debug_memory():
    """Get process memory usage."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return predictor.get_memory_usage()

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Add CORS middleware for web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)