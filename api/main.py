"""
FastAPI Service for Smart Ambulance Patient Monitoring

This API provides real-time patient vitals analysis:

Endpoints:
- POST /analyze: Analyze a window of vitals data
  Input: List of vital readings (30-second window recommended)
  Output: Anomaly flag, risk score, confidence, and explanations

- POST /analyze/single: Analyze single-point vitals (for real-time streaming)
  Input: Single vital reading
  Output: Risk score based on point-in-time analysis

- GET /health: Health check endpoint

- GET /model/info: Get model information and configuration

Usage:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    
Example request:
    curl -X POST "http://localhost:8000/analyze" \\
         -H "Content-Type: application/json" \\
         -d '{"vitals": [{"heart_rate": 85, "spo2": 97, "systolic_bp": 120, "diastolic_bp": 80}]}'
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Import our modules
from src.artifact_detection import ArtifactDetector, ArtifactCorrector
from src.anomaly_detection import AnomalyDetector
from src.risk_scoring import RiskCalculator, AlertManager, RiskLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Ambulance Patient Monitoring API",
    description="Real-time patient vitals analysis for ambulance transport",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class VitalReading(BaseModel):
    """Single vital sign reading"""
    heart_rate: Optional[float] = Field(None, ge=0, le=300, description="Heart rate in BPM")
    spo2: Optional[float] = Field(None, ge=0, le=100, description="SpO2 percentage")
    systolic_bp: Optional[float] = Field(None, ge=0, le=300, description="Systolic blood pressure")
    diastolic_bp: Optional[float] = Field(None, ge=0, le=200, description="Diastolic blood pressure")
    motion: Optional[float] = Field(None, ge=0, le=1, description="Motion signal (0-1)")
    timestamp: Optional[str] = Field(None, description="ISO timestamp")
    
    @validator('diastolic_bp')
    def dbp_less_than_sbp(cls, v, values):
        if v is not None and values.get('systolic_bp') is not None:
            if v >= values['systolic_bp']:
                raise ValueError('diastolic_bp must be less than systolic_bp')
        return v


class AnalyzeRequest(BaseModel):
    """Request body for /analyze endpoint"""
    vitals: List[VitalReading] = Field(..., min_length=1, max_length=1000)
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    

class AnalyzeResponse(BaseModel):
    """Response from /analyze endpoint"""
    is_anomaly: bool = Field(..., description="Whether anomaly is detected")
    risk_score: float = Field(..., ge=0, le=100, description="Risk score 0-100")
    risk_level: str = Field(..., description="Risk level category")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    should_alert: bool = Field(..., description="Whether alert should be raised")
    explanations: List[str] = Field(default_factory=list, description="Reasons for the assessment")
    vital_scores: Dict[str, Any] = Field(default_factory=dict, description="Individual vital assessments")
    data_quality: Dict[str, Any] = Field(default_factory=dict, description="Data quality information")
    

class SingleVitalResponse(BaseModel):
    """Response from /analyze/single endpoint"""
    risk_score: float
    risk_level: str
    warnings: List[str] = Field(default_factory=list)
    
    
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str
    feature_count: int
    window_size: int
    training_samples: Optional[int]
    config: Dict[str, Any]


# ============================================================================
# Global State / Model Loading
# ============================================================================

class AppState:
    """Application state containing loaded models"""
    def __init__(self):
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.artifact_detector: ArtifactDetector = ArtifactDetector()
        self.artifact_corrector: ArtifactCorrector = ArtifactCorrector()
        self.risk_calculator: RiskCalculator = RiskCalculator()
        self.alert_manager: AlertManager = AlertManager()
        self.model_loaded: bool = False
        self.training_samples: int = 0
        
    def load_model(self, model_path: str = None):
        """Load pre-trained model if exists, otherwise train on startup"""
        if model_path and os.path.exists(model_path):
            try:
                self.anomaly_detector = AnomalyDetector.load(model_path)
                self.model_loaded = True
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Will train new model.")
                self._train_initial_model()
        else:
            self._train_initial_model()
            
    def _train_initial_model(self):
        """Train a default model on synthetic data"""
        logger.info("Training initial model on synthetic data...")
        
        from src.data_generator import DataGenerator
        
        generator = DataGenerator(seed=42)
        # Generate training data (normal transport)
        train_data = pd.concat([
            generator.generate_normal_transport(30) for _ in range(3)
        ], ignore_index=True)
        
        # Clean artifacts
        train_cleaned, _ = self.artifact_corrector.process_artifacts(train_data)
        
        # Train model
        self.anomaly_detector = AnomalyDetector()
        self.anomaly_detector.fit(train_cleaned)
        
        self.model_loaded = True
        self.training_samples = len(train_cleaned)
        logger.info(f"Model trained on {self.training_samples} samples")
        
        # Save model
        model_dir = Path(__file__).parent.parent / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "anomaly_detector.joblib"
        self.anomaly_detector.save(str(model_path))
        logger.info(f"Model saved to {model_path}")


# Initialize state
state = AppState()


# ============================================================================
# Startup / Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    model_path = Path(__file__).parent.parent / "models" / "anomaly_detector.joblib"
    state.load_model(str(model_path))
    logger.info("API startup complete")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=state.model_loaded,
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information"""
    if not state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return ModelInfoResponse(
        model_type="IsolationForest + Risk Scoring",
        feature_count=len(state.anomaly_detector.feature_names),
        window_size=state.anomaly_detector.config.window_size,
        training_samples=state.training_samples,
        config={
            "contamination": state.anomaly_detector.config.contamination,
            "n_estimators": state.anomaly_detector.config.n_estimators
        }
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_vitals(request: AnalyzeRequest):
    """
    Analyze a window of patient vitals.
    
    Recommended: Send 30 seconds of data (30 readings at 1 Hz).
    Minimum: 10 readings for meaningful analysis.
    
    Returns anomaly detection results, risk score, and clinical explanations.
    """
    if not state.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please wait for initialization."
        )
    
    # Convert to DataFrame
    vitals_dicts = [v.model_dump() for v in request.vitals]
    df = pd.DataFrame(vitals_dicts)
    
    # Handle timestamp
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    
    # Rename columns to match expected format
    column_map = {
        'heart_rate': 'heart_rate',
        'spo2': 'spo2', 
        'systolic_bp': 'systolic_bp',
        'diastolic_bp': 'diastolic_bp',
        'motion': 'motion'
    }
    df = df.rename(columns=column_map)
    
    # Step 1: Detect artifacts
    df_with_artifacts = state.artifact_detector.detect_all_artifacts(df.copy())
    artifact_summary = state.artifact_detector.get_artifact_summary(df_with_artifacts)
    
    # Step 2: Clean artifacts
    df_cleaned, _ = state.artifact_corrector.process_artifacts(df)
    
    # Step 3: Anomaly detection
    anomaly_score = 0.0
    anomaly_confidence = 0.5
    is_anomaly = False
    
    if len(df_cleaned) >= 10:  # Need enough data for features
        try:
            result = state.anomaly_detector.predict_single_window({
                col: df_cleaned[col].dropna().tolist() 
                for col in df_cleaned.columns 
                if col in ['heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp']
            })
            anomaly_score = result['anomaly_score']
            anomaly_confidence = result['confidence']
            is_anomaly = result['is_anomaly']
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
    
    # Step 4: Risk scoring
    risk_result = state.risk_calculator.compute_from_window(
        df_cleaned,
        anomaly_score=anomaly_score,
        anomaly_confidence=anomaly_confidence
    )
    
    # Step 5: Build response
    explanations = risk_result.get('explanations', [])
    if artifact_summary['artifact_percentage'] > 20:
        explanations.append(
            f"⚠️ High artifact rate ({artifact_summary['artifact_percentage']:.0f}%) - "
            "data quality concerns"
        )
    
    return AnalyzeResponse(
        is_anomaly=is_anomaly,
        risk_score=risk_result['risk_score'],
        risk_level=risk_result['risk_level'],
        confidence=risk_result.get('confidence', anomaly_confidence),
        should_alert=risk_result.get('should_alert', False),
        explanations=explanations,
        vital_scores={
            k: v for k, v in risk_result.get('components', {}).items()
        },
        data_quality={
            'artifact_percentage': artifact_summary['artifact_percentage'],
            'samples_analyzed': len(df),
            'motion_artifacts': artifact_summary.get('motion_artifacts', 0)
        }
    )


@app.post("/analyze/single", response_model=SingleVitalResponse)
async def analyze_single_vital(reading: VitalReading):
    """
    Quick assessment of a single vital reading.
    
    Note: For better accuracy, use /analyze with multiple readings.
    Single-point analysis has higher uncertainty.
    """
    warnings = []
    
    # Simple threshold-based assessment for single points
    vitals = {}
    if reading.heart_rate is not None:
        vitals['heart_rate'] = reading.heart_rate
        if reading.heart_rate > 120 or reading.heart_rate < 50:
            warnings.append(f"Abnormal heart rate: {reading.heart_rate:.0f}")
            
    if reading.spo2 is not None:
        vitals['spo2'] = reading.spo2
        if reading.spo2 < 94:
            warnings.append(f"Low SpO2: {reading.spo2:.0f}%")
            
    if reading.systolic_bp is not None:
        vitals['systolic_bp'] = reading.systolic_bp
        if reading.systolic_bp < 90 or reading.systolic_bp > 160:
            warnings.append(f"Abnormal BP: {reading.systolic_bp:.0f}")
            
    if reading.diastolic_bp is not None:
        vitals['diastolic_bp'] = reading.diastolic_bp
    
    # Quick risk calculation
    result = state.risk_calculator.compute_risk_score(
        vitals=vitals,
        trends={},  # No trend data for single point
        anomaly_score=0,
        anomaly_confidence=0.3,  # Low confidence for single point
        motion_level=reading.motion or 0
    )
    
    return SingleVitalResponse(
        risk_score=result['risk_score'],
        risk_level=result['risk_level'],
        warnings=warnings
    )


@app.post("/batch/analyze")
async def batch_analyze(patient_data: Dict[str, List[VitalReading]]):
    """
    Analyze vitals for multiple patients.
    
    Input: Dictionary mapping patient_id to list of vital readings.
    Output: Dictionary mapping patient_id to analyze results.
    """
    results = {}
    
    for patient_id, vitals in patient_data.items():
        try:
            request = AnalyzeRequest(vitals=vitals, patient_id=patient_id)
            result = await analyze_vitals(request)
            results[patient_id] = result.model_dump()
        except Exception as e:
            results[patient_id] = {"error": str(e)}
            
    return results


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
