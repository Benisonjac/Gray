"""
Inference Script for Smart Ambulance Patient Monitoring System

This script demonstrates model inference on new data:
1. Load trained model
2. Process input data (artifact handling)
3. Run anomaly detection
4. Calculate risk scores
5. Generate alerts

Usage:
    python scripts/inference.py --input INPUT_FILE [--model MODEL_PATH]
    python scripts/inference.py --demo  # Run with synthetic demo data
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import pandas as pd
import numpy as np

from src.data_generator import DataGenerator
from src.artifact_detection import ArtifactCorrector, ArtifactDetector
from src.anomaly_detection import AnomalyDetector
from src.risk_scoring import RiskCalculator, AlertManager


def load_model(model_path: str) -> AnomalyDetector:
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    detector = AnomalyDetector.load(model_path)
    print(f"Model loaded with {len(detector.feature_names)} features")
    return detector


def process_patient_data(
    df: pd.DataFrame,
    detector: AnomalyDetector,
    window_size: int = 30,
    step_size: int = 15
) -> list:
    """
    Process patient data and generate risk assessments.
    
    Args:
        df: Patient vitals DataFrame
        detector: Trained anomaly detector
        window_size: Analysis window size (seconds)
        step_size: Step between windows
        
    Returns:
        List of assessment results
    """
    # Initialize components
    corrector = ArtifactCorrector()
    artifact_detector = ArtifactDetector()
    risk_calculator = RiskCalculator()
    alert_manager = AlertManager()
    
    # Clean artifacts
    df_cleaned, df_flagged = corrector.process_artifacts(df.copy())
    artifact_summary = artifact_detector.get_artifact_summary(df_flagged)
    
    results = []
    
    # Process windows
    for start in range(0, len(df_cleaned) - window_size + 1, step_size):
        end = start + window_size
        window = df_cleaned.iloc[start:end]
        window_original = df_flagged.iloc[start:end]
        
        # Get anomaly prediction
        window_dict = {
            col: window[col].dropna().tolist()
            for col in ['heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp']
            if col in window.columns
        }
        
        try:
            anomaly_result = detector.predict_single_window(window_dict)
        except:
            anomaly_result = {'is_anomaly': False, 'anomaly_score': 0, 'confidence': 0.5}
        
        # Get risk score
        risk_result = risk_calculator.compute_from_window(
            window,
            anomaly_score=anomaly_result.get('anomaly_score', 0),
            anomaly_confidence=anomaly_result.get('confidence', 0.5)
        )
        
        # Check for alert
        alert = alert_manager.should_alert(risk_result)
        
        # Store result
        results.append({
            'window_start': start,
            'window_end': end,
            'is_anomaly': anomaly_result.get('is_anomaly', False),
            'anomaly_score': round(anomaly_result.get('anomaly_score', 0), 3),
            'risk_score': risk_result['risk_score'],
            'risk_level': risk_result['risk_level'],
            'should_alert': alert['should_alert'],
            'alert_type': alert.get('alert_type'),
            'alert_message': alert.get('message', ''),
            'explanations': risk_result.get('explanations', []),
            'data_quality': {
                'artifacts_in_window': window_original['is_artifact'].sum() if 'is_artifact' in window_original.columns else 0
            }
        })
    
    return results


def print_results(results: list, verbose: bool = True):
    """Print inference results"""
    print("\n" + "=" * 70)
    print("INFERENCE RESULTS")
    print("=" * 70)
    
    n_anomalies = sum(1 for r in results if r['is_anomaly'])
    n_alerts = sum(1 for r in results if r['should_alert'])
    avg_risk = np.mean([r['risk_score'] for r in results])
    
    print(f"\nSummary:")
    print(f"  Total windows analyzed: {len(results)}")
    print(f"  Anomalies detected: {n_anomalies}")
    print(f"  Alerts triggered: {n_alerts}")
    print(f"  Average risk score: {avg_risk:.1f}")
    
    # Risk level distribution
    risk_levels = {}
    for r in results:
        level = r['risk_level']
        risk_levels[level] = risk_levels.get(level, 0) + 1
    print(f"\nRisk level distribution:")
    for level, count in sorted(risk_levels.items()):
        print(f"    {level}: {count}")
    
    if verbose:
        # Show alerts
        alerts = [r for r in results if r['should_alert']]
        if alerts:
            print(f"\n{'='*70}")
            print("ALERTS RAISED:")
            print("=" * 70)
            for alert in alerts:
                print(f"\n  Window {alert['window_start']}-{alert['window_end']}:")
                print(f"    Type: {alert['alert_type']}")
                print(f"    Risk Score: {alert['risk_score']}")
                print(f"    Message: {alert['alert_message']}")
        else:
            print("\nNo alerts raised.")
        
        # Show high-risk windows
        high_risk = [r for r in results if r['risk_score'] >= 50 and not r['should_alert']]
        if high_risk:
            print(f"\n{'='*70}")
            print("HIGH RISK WINDOWS (not alerted):")
            print("=" * 70)
            for r in high_risk[:5]:  # Show first 5
                print(f"\n  Window {r['window_start']}-{r['window_end']}:")
                print(f"    Risk Score: {r['risk_score']}")
                print(f"    Level: {r['risk_level']}")
                if r['explanations']:
                    print(f"    Reasons: {'; '.join(r['explanations'])}")


def run_demo():
    """Run inference on demo synthetic data"""
    print("Running inference demo with synthetic data...")
    
    # Import DataGenerator locally for consistency
    from src.data_generator import DataGenerator as DG
    
    # Generate test scenarios
    generator = DG(seed=999)
    
    scenarios = {
        'Normal Transport': generator.generate_normal_transport(10),
        'Cardiac Distress': generator.generate_cardiac_distress(10, 4, 'moderate'),
        'Respiratory Distress': generator.generate_respiratory_distress(10, 5, 'severe'),
        'Motion Artifacts': generator.generate_heavy_motion_artifacts(10)
    }
    
    # Load or train model
    model_path = project_root / "outputs" / "models" / "anomaly_detector.joblib"
    if not model_path.exists():
        model_path = project_root / "models" / "anomaly_detector.joblib"
    
    if model_path.exists():
        detector = load_model(str(model_path))
    else:
        print("Model not found. Training new model...")
        from src.artifact_detection import ArtifactCorrector
        
        gen = DG(seed=42)
        train_data = pd.concat([
            gen.generate_normal_transport(30) for _ in range(3)
        ], ignore_index=True)
        
        corrector = ArtifactCorrector()
        train_cleaned, _ = corrector.process_artifacts(train_data)
        
        detector = AnomalyDetector()
        detector.fit(train_cleaned)
        
        # Save model
        model_path.parent.mkdir(exist_ok=True)
        detector.save(str(model_path))
    
    # Process each scenario
    all_results = {}
    for scenario_name, data in scenarios.items():
        print(f"\n{'='*70}")
        print(f"Processing: {scenario_name}")
        print("=" * 70)
        
        results = process_patient_data(data, detector, window_size=30, step_size=15)
        all_results[scenario_name] = results
        print_results(results, verbose=True)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run inference on patient data")
    parser.add_argument(
        '--input',
        type=str,
        help='Input CSV file with patient vitals'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/anomaly_detector.joblib',
        help='Path to trained model'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo with synthetic data'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file for results'
    )
    args = parser.parse_args()
    
    if args.demo:
        results = run_demo()
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")
        return
    
    if not args.input:
        parser.error("Either --input or --demo is required")
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Load model
    model_path = project_root / args.model
    detector = load_model(str(model_path))
    
    # Process
    results = process_patient_data(df, detector)
    
    # Print results
    print_results(results)
    
    # Save if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
