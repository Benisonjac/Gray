"""
Alert Quality Evaluation Module for Smart Ambulance Patient Monitoring

This module evaluates the quality of the anomaly detection and alerting system
using clinically relevant metrics.

Metrics Reported:
1. Precision: Of all alerts raised, how many were true emergencies?
2. Recall (Sensitivity): Of all true emergencies, how many did we detect?
3. False Alert Rate: How many unnecessary alerts per hour?
4. Alert Latency: How long after deterioration starts before we alert?

Clinical Context for Metrics:
- In ambulance setting, FALSE NEGATIVES (missing deterioration) are more 
  dangerous than FALSE POSITIVES (unnecessary alerts)
- However, too many false alerts cause "alert fatigue" - providers ignore them
- Target: Recall > 90%, with acceptable false alert rate (<5 per hour)

Error Analysis:
- What types of deterioration are we missing? (False negatives)
- What causes false alerts? (Motion, sensor issues, etc.)
- How can we improve?
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, precision_recall_curve,
    roc_auc_score
)
import matplotlib.pyplot as plt


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    precision: float
    recall: float
    f1: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float
    false_alerts_per_hour: float
    mean_detection_latency: float
    confusion_matrix: np.ndarray
    
    def to_dict(self) -> Dict:
        return {
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1,
            'specificity': self.specificity,
            'false_positive_rate': self.false_positive_rate,
            'false_negative_rate': self.false_negative_rate,
            'false_alerts_per_hour': self.false_alerts_per_hour,
            'mean_detection_latency_seconds': self.mean_detection_latency,
            'true_negatives': int(self.confusion_matrix[0, 0]),
            'false_positives': int(self.confusion_matrix[0, 1]),
            'false_negatives': int(self.confusion_matrix[1, 0]),
            'true_positives': int(self.confusion_matrix[1, 1])
        }


class AlertEvaluator:
    """
    Evaluates alert system performance with clinical context.
    """
    
    def __init__(self, sampling_rate_hz: float = 1.0):
        """
        Args:
            sampling_rate_hz: Data sampling rate (1 Hz = 1 sample/second)
        """
        self.sampling_rate = sampling_rate_hz
        
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None
    ) -> EvaluationResult:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels (1 = anomaly/deterioration)
            y_pred: Predicted labels
            y_scores: Prediction scores (for threshold analysis)
            timestamps: Timestamps for latency calculation
            
        Returns:
            EvaluationResult with all metrics
        """
        # Basic metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Handle edge cases
        if cm.shape != (2, 2):
            # If only one class present
            if y_true.sum() == 0:  # No positives
                cm = np.array([[len(y_true) - y_pred.sum(), y_pred.sum()],
                              [0, 0]])
            else:  # No negatives
                cm = np.array([[0, 0],
                              [len(y_true) - y_pred.sum(), y_pred.sum()]])
        
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Specificity and rates
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # False alerts per hour
        total_time_hours = len(y_true) / self.sampling_rate / 3600
        false_alerts_per_hour = fp / total_time_hours if total_time_hours > 0 else 0
        
        # Detection latency
        latency = self._compute_detection_latency(y_true, y_pred)
        
        return EvaluationResult(
            precision=precision,
            recall=recall,
            f1=f1,
            specificity=specificity,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            false_alerts_per_hour=false_alerts_per_hour,
            mean_detection_latency=latency,
            confusion_matrix=cm
        )
    
    def _compute_detection_latency(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute average time between deterioration onset and detection.
        
        For each true deterioration event, measure how long until
        first prediction.
        """
        latencies = []
        
        # Find events (transitions from 0 to 1 in y_true)
        true_changes = np.diff(y_true.astype(int))
        event_starts = np.where(true_changes == 1)[0] + 1
        
        for start in event_starts:
            # Find end of this event
            remaining = y_true[start:]
            end_offsets = np.where(remaining == 0)[0]
            end = start + end_offsets[0] if len(end_offsets) > 0 else len(y_true)
            
            # Find first prediction in this window
            predictions_in_event = y_pred[start:end]
            pred_indices = np.where(predictions_in_event == 1)[0]
            
            if len(pred_indices) > 0:
                # Latency = samples until first detection
                latency_samples = pred_indices[0]
                latencies.append(latency_samples / self.sampling_rate)
            # If not detected, we don't add to latency (counted in recall)
        
        return np.mean(latencies) if latencies else float('inf')
    
    def analyze_false_positives(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> List[Dict]:
        """
        Analyze false positive cases.
        
        Args:
            df: Original DataFrame with vitals
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            List of dicts describing each false positive
        """
        fp_indices = np.where((y_pred == 1) & (y_true == 0))[0]
        
        analyses = []
        for idx in fp_indices[:10]:  # Limit to first 10
            window_start = max(0, idx - 15)
            window_end = min(len(df), idx + 15)
            
            window = df.iloc[window_start:window_end]
            
            analysis = {
                'index': int(idx),
                'possible_causes': []
            }
            
            # Check for motion
            if 'motion' in window.columns:
                motion_level = window['motion'].mean()
                if motion_level > 0.3:
                    analysis['possible_causes'].append(
                        f"High motion ({motion_level:.2f})"
                    )
            
            # Check for rapid changes
            for col in ['heart_rate', 'spo2', 'systolic_bp']:
                if col in window.columns:
                    change = window[col].diff().abs().max()
                    if change > 10:
                        analysis['possible_causes'].append(
                            f"Rapid {col} change ({change:.1f})"
                        )
            
            # Check if values themselves are near thresholds
            if 'heart_rate' in window.columns:
                hr = window['heart_rate'].iloc[15] if len(window) > 15 else window['heart_rate'].mean()
                if 95 <= hr <= 105:
                    analysis['possible_causes'].append(
                        f"HR near threshold ({hr:.0f})"
                    )
                    
            if 'spo2' in window.columns:
                spo2 = window['spo2'].iloc[15] if len(window) > 15 else window['spo2'].mean()
                if 93 <= spo2 <= 96:
                    analysis['possible_causes'].append(
                        f"SpO2 near threshold ({spo2:.0f}%)"
                    )
            
            if not analysis['possible_causes']:
                analysis['possible_causes'].append("Unknown - requires investigation")
                
            analyses.append(analysis)
            
        return analyses
    
    def analyze_false_negatives(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> List[Dict]:
        """
        Analyze false negative cases (missed detections).
        
        These are the most dangerous errors in a medical context.
        """
        fn_indices = np.where((y_pred == 0) & (y_true == 1))[0]
        
        analyses = []
        for idx in fn_indices[:10]:
            window_start = max(0, idx - 15)
            window_end = min(len(df), idx + 15)
            
            window = df.iloc[window_start:window_end]
            
            analysis = {
                'index': int(idx),
                'severity': 'unknown',
                'possible_causes': []
            }
            
            # Determine severity of missed event
            if 'spo2' in window.columns:
                min_spo2 = window['spo2'].min()
                if min_spo2 < 85:
                    analysis['severity'] = 'severe'
                elif min_spo2 < 92:
                    analysis['severity'] = 'moderate'
                else:
                    analysis['severity'] = 'mild'
                    
            # Check why it might have been missed
            if 'heart_rate' in window.columns and 'spo2' in window.columns:
                hr_trend = window['heart_rate'].diff().mean()
                spo2_trend = window['spo2'].diff().mean()
                
                if abs(hr_trend) < 0.5:
                    analysis['possible_causes'].append(
                        "Slow HR change - gradual deterioration"
                    )
                if abs(spo2_trend) < 0.2:
                    analysis['possible_causes'].append(
                        "Slow SpO2 change - subtle deterioration"
                    )
                    
            # Check if values are in ambiguous range
            for col, low, high in [('heart_rate', 90, 110), ('spo2', 92, 96)]:
                if col in window.columns:
                    val = window[col].mean()
                    if low <= val <= high:
                        analysis['possible_causes'].append(
                            f"{col} in ambiguous range ({val:.1f})"
                        )
            
            if not analysis['possible_causes']:
                analysis['possible_causes'].append(
                    "Feature extraction may have missed pattern"
                )
                
            analyses.append(analysis)
            
        return analyses


def clinical_error_report(
    result: EvaluationResult,
    fp_analysis: List[Dict],
    fn_analysis: List[Dict]
) -> str:
    """
    Generate a human-readable clinical error report.
    """
    report = []
    report.append("=" * 60)
    report.append("ALERT SYSTEM EVALUATION REPORT")
    report.append("=" * 60)
    
    # Overall performance
    report.append("\n## Overall Performance")
    report.append(f"Precision: {result.precision:.2%}")
    report.append(f"Recall (Sensitivity): {result.recall:.2%}")
    report.append(f"F1 Score: {result.f1:.2%}")
    report.append(f"Specificity: {result.specificity:.2%}")
    
    # Clinical relevance
    report.append("\n## Clinical Metrics")
    report.append(f"False Alert Rate: {result.false_alerts_per_hour:.1f} per hour")
    report.append(f"Mean Detection Latency: {result.mean_detection_latency:.1f} seconds")
    
    # Interpretation
    report.append("\n## Clinical Interpretation")
    
    if result.recall >= 0.9:
        report.append("✓ Sensitivity is adequate (>=90%) - most deteriorations detected")
    else:
        report.append(f"⚠ Sensitivity is below target ({result.recall:.1%} < 90%)")
        report.append("  ACTION: Review false negatives, consider lowering thresholds")
        
    if result.false_alerts_per_hour <= 5:
        report.append("✓ False alert rate is acceptable (<=5 per hour)")
    else:
        report.append(f"⚠ High false alert rate ({result.false_alerts_per_hour:.1f}/hour)")
        report.append("  ACTION: Risk of alert fatigue, improve artifact handling")
    
    # Error analysis
    report.append("\n## False Positive Analysis (Unnecessary Alerts)")
    if fp_analysis:
        for i, fp in enumerate(fp_analysis[:3], 1):
            report.append(f"\n  Case {i} (sample {fp['index']}):")
            for cause in fp['possible_causes']:
                report.append(f"    - {cause}")
    else:
        report.append("  No false positives detected")
        
    report.append("\n## False Negative Analysis (Missed Detections)")
    if fn_analysis:
        for i, fn in enumerate(fn_analysis[:3], 1):
            report.append(f"\n  Case {i} (sample {fn['index']}), severity: {fn['severity']}")
            for cause in fn['possible_causes']:
                report.append(f"    - {cause}")
    else:
        report.append("  No false negatives detected")
    
    # Acceptable vs unacceptable errors
    report.append("\n## Error Tolerance Analysis")
    report.append("\nAcceptable errors in ambulance context:")
    report.append("  - False positives during high motion (expected artifact)")
    report.append("  - False positives near thresholds (conservative is safer)")
    report.append("  - Late detection of very gradual changes (rare in emergencies)")
    
    report.append("\nUNACCEPTABLE errors:")
    report.append("  - Missing severe hypoxia (SpO2 < 85%)")
    report.append("  - Missing rapid deterioration")
    report.append("  - Systematic bias (always missing certain patterns)")
    
    return "\n".join(report)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot precision-recall curve for threshold analysis"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recall, precision, 'b-', linewidth=2)
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve for Anomaly Detection', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Mark clinical target zone (recall > 0.9, precision > 0.5)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Min Precision')
    ax.axvline(x=0.9, color='g', linestyle='--', alpha=0.5, label='Target Recall')
    ax.legend()
    
    # Add F1 iso-lines
    for f1 in [0.3, 0.5, 0.7, 0.9]:
        r = np.linspace(0.01, 1, 100)
        p = f1 * r / (2 * r - f1)
        ax.plot(r[p > 0], p[p > 0], 'gray', alpha=0.3, linestyle=':')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    plt.close()
    return fig


def main():
    """Demo evaluation metrics"""
    from data_generator import DataGenerator
    from artifact_detection import ArtifactCorrector
    from anomaly_detection import AnomalyDetector
    
    print("=== Alert Evaluation Demo ===\n")
    
    # Generate data
    generator = DataGenerator(seed=42)
    
    # Training data (normal)
    train_data = pd.concat([
        generator.generate_normal_transport(30) for _ in range(3)
    ], ignore_index=True)
    
    # Test data (mixed)
    test_data = generator.generate_mixed_scenario_dataset(
        n_normal=2, n_cardiac=2, n_respiratory=2, n_motion_artifact=1,
        duration_minutes=20
    )
    
    # Clean and train
    corrector = ArtifactCorrector()
    train_cleaned, _ = corrector.process_artifacts(train_data)
    test_cleaned, test_flagged = corrector.process_artifacts(test_data)
    
    detector = AnomalyDetector()
    detector.fit(train_cleaned)
    
    # Predict
    predictions = detector.predict(test_cleaned)
    
    # Get actual labels per window
    y_true = predictions['label'].values
    y_pred = predictions['predicted_anomaly'].values
    y_scores = predictions['anomaly_score'].values
    
    # Evaluate
    evaluator = AlertEvaluator(sampling_rate_hz=1.0)
    result = evaluator.evaluate(y_true, y_pred, y_scores)
    
    print("Evaluation Results:")
    for key, value in result.to_dict().items():
        print(f"  {key}: {value}")
    
    # Error analysis
    fp_analysis = evaluator.analyze_false_positives(test_cleaned, y_true, y_pred)
    fn_analysis = evaluator.analyze_false_negatives(test_cleaned, y_true, y_pred)
    
    # Generate report
    report = clinical_error_report(result, fp_analysis, fn_analysis)
    print("\n" + report)
    
    # Save plots
    import os
    os.makedirs('outputs/figures', exist_ok=True)
    
    # Only plot if we have both classes
    if y_true.sum() > 0 and (1 - y_true).sum() > 0:
        plot_precision_recall_curve(
            y_true, y_scores,
            save_path='outputs/figures/precision_recall_curve.png'
        )
        print("\nPrecision-recall curve saved to outputs/figures/")
    
    return result, fp_analysis, fn_analysis


if __name__ == "__main__":
    main()
