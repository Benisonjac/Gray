# Smart Ambulance Patient Monitoring System

Gray Mobility - AI/ML Engineer Intern Assignment

## Overview

This project implements a **real-time patient vitals monitoring system** for ambulance transport. It detects early warning signals of patient deterioration and generates actionable alerts for paramedics.

### Key Features

- **Synthetic Data Generation**: Realistic time-series patient vitals (HR, SpO2, BP) with various clinical scenarios
- **Artifact Detection & Handling**: Identifies and corrects motion artifacts, sensor dropouts, and noise
- **Anomaly Detection**: Isolation Forest-based model for detecting patient deterioration patterns
- **Risk Scoring**: Multi-vital triage scoring system with clinical reasoning
- **Alert Management**: Smart alerting with suppression logic to reduce alarm fatigue
- **REST API**: FastAPI service for real-time inference

## Project Structure

```
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI service
├── src/
│   ├── __init__.py
│   ├── data_generator.py    # Synthetic data generation
│   ├── artifact_detection.py # Artifact detection & cleaning
│   ├── anomaly_detection.py  # ML anomaly detection model
│   ├── risk_scoring.py       # Clinical risk scoring
│   └── evaluation.py         # Alert quality metrics
├── scripts/
│   ├── train.py             # Model training script
│   └── inference.py         # Inference/demo script
├── models/                  # Saved models (generated)
├── data/                    # Generated datasets
├── outputs/                 # Training outputs & figures
├── docs/
│   └── safety_analysis.md   # Safety-critical analysis
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create and activate virtual environment
python -m venv env
.\env\Scripts\Activate.ps1  # Windows
source env/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train the Model

```bash
python scripts/train.py --output-dir outputs
```

This will:
- Generate synthetic training data
- Train the anomaly detection model
- Evaluate on test scenarios
- Save model and metrics

### 2. Run Inference Demo

```bash
python scripts/inference.py --demo
```

### 3. Start API Server

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API Documentation: http://localhost:8000/docs

### 4. Example API Request

```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "vitals": [
         {"heart_rate": 85, "spo2": 97, "systolic_bp": 120, "diastolic_bp": 80},
         {"heart_rate": 87, "spo2": 96, "systolic_bp": 118, "diastolic_bp": 78},
         ...
       ]
     }'
```

## Technical Approach

### Data Generation (Part 1A)

Synthetic data simulates realistic ambulance transport scenarios:

| Scenario | Characteristics |
|----------|-----------------|
| Normal Transport | Stable vitals with minor variations |
| Cardiac Distress | Progressive tachycardia, BP changes |
| Respiratory Distress | SpO2 decline, compensatory HR increase |
| Motion Artifacts | High-frequency artifacts from vehicle movement |

**Signals Generated (1 Hz sampling):**
- Heart Rate: 40-200 BPM
- SpO2: 70-100%
- Systolic/Diastolic BP: Simulated
- Motion Signal: 0-1 normalized

### Artifact Detection (Part 1B)

**Before anomaly detection**, artifacts are identified and handled:

1. **Impossible Values**: Out of physiological bounds (HR > 300)
2. **Rate of Change**: Too rapid to be physiological (HR +50 in 1 second)
3. **Motion Correlation**: Vital changes coinciding with high motion
4. **Sensor Dropouts**: Missing data, stuck readings

**Handling Strategy:**
- Short gaps: Linear interpolation
- Motion artifacts: Flagged, excluded from anomaly detection
- Long dropouts: Marked as unreliable

### Anomaly Detection (Part 2A)

**Model: Isolation Forest**

Why this approach:
- Works well with time-series features
- Unsupervised (trains on normal data)
- Interpretable decision boundaries
- Efficient for real-time inference

**Feature Engineering:**
- Window-based features (30-second windows)
- Statistical: mean, std, min, max, coefficient of variation
- Trend: slope, acceleration, R² of fit
- Clinical: Shock Index, MAP, pulse pressure
- Cross-vital correlations

### Risk Scoring (Part 2B)

**Multi-vital weighted scoring (0-100):**

| Risk Level | Score | Action |
|------------|-------|--------|
| Normal | 0-30 | Continue monitoring |
| Elevated | 30-50 | Increase monitoring frequency |
| High | 50-70 | Prepare for intervention |
| Critical | 70-100 | Immediate action required |

**Clinical Features:**
- Shock Index (HR/SBP > 0.9 indicates shock)
- Mean Arterial Pressure (MAP < 65 = inadequate perfusion)
- Pulse Pressure (narrow = concerning)

**Alert Suppression:**
- Motion-correlated changes are suppressed
- Hysteresis prevents alert spam
- Critical alerts always go through

### Evaluation Metrics (Part 3)

**Key Metrics:**
- **Precision**: Of alerts raised, how many were true emergencies?
- **Recall**: Of true emergencies, how many did we detect?
- **False Alert Rate**: Alerts per hour (target: <5)
- **Detection Latency**: Time from deterioration start to alert

**Error Analysis:**
- False Positives: Motion artifacts, threshold boundary cases
- False Negatives: Gradual deterioration, ambiguous vital ranges

### API Service (Part 4)

FastAPI endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze window of vitals |
| `/analyze/single` | POST | Quick single-point assessment |
| `/health` | GET | Health check |
| `/model/info` | GET | Model configuration |

## Evaluation Results

Example metrics from test scenarios:

```
Precision: 72%
Recall: 85%
False Alert Rate: 3.2/hour
Mean Detection Latency: 12.5 seconds
```

See `outputs/reports/metrics.json` after training for full results.

## Limitations & Assumptions

1. **Synthetic Data**: Real ambulance data would have more complexity
2. **Simplified Physiology**: Actual vital sign relationships are more complex
3. **No Patient Context**: Age, medications, condition not considered
4. **Motion Signal**: Synthetic, not from actual accelerometer
5. **Single Algorithm**: Production would use ensemble methods

## Future Improvements

- [ ] Integration with PhysioNet datasets
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Explainability visualizations (SHAP)
- [ ] Drift detection for model monitoring
- [ ] Docker containerization
- [ ] Green corridor ETA logic

## Author

Benison Jacob Benny

## License

This project is for educational/interview purposes.
