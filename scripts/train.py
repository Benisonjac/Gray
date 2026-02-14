"""
Training Script for Smart Ambulance Patient Monitoring System

This script:
1. Generates synthetic training data
2. Applies artifact detection and cleaning
3. Trains the anomaly detection model
4. Evaluates model performance
5. Saves the trained model

Usage:
    python scripts/train.py [--output-dir OUTPUT_DIR] [--n-patients N_PATIENTS]
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from src.data_generator import DataGenerator
from src.artifact_detection import ArtifactCorrector, ArtifactDetector
from src.anomaly_detection import AnomalyDetector, AnomalyConfig
from src.risk_scoring import RiskCalculator
from src.evaluation import AlertEvaluator, clinical_error_report, plot_precision_recall_curve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_training_data(
    n_normal: int = 5,
    n_anomaly: int = 0,  # Training should be mostly normal
    duration_minutes: int = 30,
    seed: int = 42
) -> pd.DataFrame:
    """Generate training data (mostly normal transport)"""
    logger.info(f"Generating {n_normal} normal transport scenarios...")
    
    generator = DataGenerator(seed=seed)
    
    all_data = []
    for i in range(n_normal):
        df = generator.generate_normal_transport(duration_minutes)
        df['patient_id'] = i
        all_data.append(df)
        
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Generated {len(combined)} training samples")
    
    return combined


def generate_test_data(
    n_normal: int = 2,
    n_cardiac: int = 2,
    n_respiratory: int = 2,
    n_motion_artifact: int = 2,
    duration_minutes: int = 20,
    seed: int = 123
) -> pd.DataFrame:
    """Generate diverse test data"""
    logger.info("Generating test data with various scenarios...")
    
    generator = DataGenerator(seed=seed)
    
    test_data = generator.generate_mixed_scenario_dataset(
        n_normal=n_normal,
        n_cardiac=n_cardiac,
        n_respiratory=n_respiratory,
        n_motion_artifact=n_motion_artifact,
        duration_minutes=duration_minutes
    )
    
    logger.info(f"Generated {len(test_data)} test samples")
    logger.info(f"Scenarios: {test_data['scenario'].value_counts().to_dict()}")
    
    return test_data


def preprocess_data(df: pd.DataFrame) -> tuple:
    """Apply artifact detection and cleaning"""
    logger.info("Preprocessing data...")
    
    corrector = ArtifactCorrector()
    df_cleaned, df_flagged = corrector.process_artifacts(df.copy())
    
    # Get artifact summary
    detector = ArtifactDetector()
    summary = detector.get_artifact_summary(df_flagged)
    
    logger.info(f"Artifacts detected: {summary['artifact_percentage']:.1f}%")
    logger.info(f"  - Impossible values: {summary['impossible_values']}")
    logger.info(f"  - Motion artifacts: {summary['motion_artifacts']}")
    logger.info(f"  - Dropouts: {summary['dropouts']}")
    
    return df_cleaned, df_flagged


def train_model(df_cleaned: pd.DataFrame, config: AnomalyConfig = None) -> AnomalyDetector:
    """Train the anomaly detection model"""
    logger.info("Training anomaly detection model...")
    
    if config is None:
        config = AnomalyConfig()
    
    detector = AnomalyDetector(config)
    detector.fit(df_cleaned)
    
    logger.info(f"Model trained with {len(detector.feature_names)} features")
    logger.info(f"Features: {detector.feature_names[:10]}...")
    
    return detector


def evaluate_model(
    detector: AnomalyDetector,
    test_cleaned: pd.DataFrame,
    test_flagged: pd.DataFrame
) -> dict:
    """Evaluate model on test data"""
    logger.info("Evaluating model...")
    
    # Get predictions
    predictions = detector.predict(test_cleaned)
    
    y_true = predictions['label'].values
    y_pred = predictions['predicted_anomaly'].values
    y_scores = predictions['anomaly_score'].values
    
    # Evaluate
    evaluator = AlertEvaluator()
    result = evaluator.evaluate(y_true, y_pred, y_scores)
    
    # Error analysis
    fp_analysis = evaluator.analyze_false_positives(test_cleaned, y_true, y_pred)
    fn_analysis = evaluator.analyze_false_negatives(test_cleaned, y_true, y_pred)
    
    # Print report
    report = clinical_error_report(result, fp_analysis, fn_analysis)
    print("\n" + report)
    
    return {
        'result': result,
        'fp_analysis': fp_analysis,
        'fn_analysis': fn_analysis,
        'predictions': predictions
    }


def save_outputs(
    detector: AnomalyDetector,
    evaluation: dict,
    output_dir: Path,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame
):
    """Save model and outputs"""
    logger.info(f"Saving outputs to {output_dir}")
    
    # Create directories
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "models" / "anomaly_detector.joblib"
    detector.save(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Save data
    train_data.to_csv(output_dir / "data" / "train_data.csv", index=False)
    test_data.to_csv(output_dir / "data" / "test_data.csv", index=False)
    
    # Save predictions
    evaluation['predictions'].to_csv(
        output_dir / "data" / "test_predictions.csv", 
        index=False
    )
    
    # Save metrics
    metrics = evaluation['result'].to_dict()
    import json
    with open(output_dir / "reports" / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot precision-recall curve
    y_true = evaluation['predictions']['label'].values
    y_scores = evaluation['predictions']['anomaly_score'].values
    
    if y_true.sum() > 0 and (1 - y_true).sum() > 0:
        plot_precision_recall_curve(
            y_true, y_scores,
            save_path=str(output_dir / "figures" / "precision_recall.png")
        )
    
    logger.info("All outputs saved successfully")


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='outputs',
        help='Output directory for model and results'
    )
    parser.add_argument(
        '--n-train-patients',
        type=int,
        default=5,
        help='Number of normal patients for training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    logger.info("=" * 60)
    logger.info("Smart Ambulance Patient Monitoring - Model Training")
    logger.info("=" * 60)
    
    # Generate data
    train_data = generate_training_data(
        n_normal=args.n_train_patients,
        seed=args.seed
    )
    test_data = generate_test_data(seed=args.seed + 100)
    
    # Preprocess
    train_cleaned, train_flagged = preprocess_data(train_data)
    test_cleaned, test_flagged = preprocess_data(test_data)
    
    # Train
    detector = train_model(train_cleaned)
    
    # Evaluate
    evaluation = evaluate_model(detector, test_cleaned, test_flagged)
    
    # Save
    save_outputs(
        detector, 
        evaluation, 
        output_dir,
        train_data,
        test_data
    )
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {output_dir / 'models' / 'anomaly_detector.joblib'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
