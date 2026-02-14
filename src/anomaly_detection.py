"""
Anomaly Detection Model for Smart Ambulance Patient Monitoring

This module implements anomaly detection that identifies early warning signals
of patient deterioration - not just threshold breaches, but trends and patterns
that indicate developing problems.

Approach:
1. Feature Engineering: Extract meaningful features from raw vitals
   - Rolling statistics (mean, std, trend)
   - Rate of change features
   - Cross-vital correlations
   
2. Model Architecture:
   - Primary: Isolation Forest (unsupervised, good for time-series)
   - Secondary: Statistical process control (CUSUM, EWMA)
   - Combined ensemble for robust predictions

3. Key Design Decisions:
   - Windowed approach: 30-second windows with 10-second overlap
   - Multi-scale analysis: Short-term spikes vs long-term trends
   - Artifact-aware: Only uses cleaned data for detection

Why Not Deep Learning?
- Small dataset (synthetic)
- Interpretability critical in medical domain
- Classical methods perform well for this problem size
- Easier to deploy and maintain
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from scipy import stats


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection"""
    # Window parameters
    window_size: int = 30  # seconds
    window_overlap: int = 10  # seconds
    
    # Isolation Forest parameters  
    contamination: float = 0.1  # Expected proportion of anomalies
    n_estimators: int = 100
    random_state: int = 42
    
    # Statistical thresholds
    cusum_threshold: float = 3.0  # Standard deviations
    ewma_span: int = 10  # Seconds for exponential weighted average
    
    # Alert thresholds (can be tuned)
    alert_threshold: float = -0.3  # Isolation Forest decision boundary


class FeatureExtractor:
    """
    Extract features from windowed vital sign data.
    
    Features include:
    - Basic statistics (mean, std, min, max)
    - Trend features (slope, acceleration)
    - Cross-vital relationships
    - Domain-specific clinical features
    """
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.feature_names = []
        
    def extract_basic_stats(self, window: pd.DataFrame) -> Dict[str, float]:
        """Extract basic statistical features from a window"""
        features = {}
        
        vitals = ['heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp']
        
        for vital in vitals:
            if vital in window.columns:
                data = window[vital].dropna()
                if len(data) > 0:
                    features[f'{vital}_mean'] = data.mean()
                    features[f'{vital}_std'] = data.std()
                    features[f'{vital}_min'] = data.min()
                    features[f'{vital}_max'] = data.max()
                    features[f'{vital}_range'] = data.max() - data.min()
                    # Coefficient of variation (normalized variability)
                    features[f'{vital}_cv'] = data.std() / data.mean() if data.mean() != 0 else 0
                else:
                    for stat in ['mean', 'std', 'min', 'max', 'range', 'cv']:
                        features[f'{vital}_{stat}'] = np.nan
                        
        return features
    
    def extract_trend_features(self, window: pd.DataFrame) -> Dict[str, float]:
        """
        Extract trend features (slope, acceleration).
        
        These are critical for early warning - a gradual HR increase
        over 30 seconds is more concerning than a stable elevated HR.
        """
        features = {}
        vitals = ['heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp']
        
        for vital in vitals:
            if vital in window.columns:
                data = window[vital].dropna().values
                if len(data) > 5:
                    # Linear regression for trend
                    x = np.arange(len(data))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)
                    
                    features[f'{vital}_slope'] = slope
                    features[f'{vital}_r_squared'] = r_value ** 2
                    
                    # Second derivative (acceleration)
                    if len(data) > 10:
                        first_half_slope = np.polyfit(x[:len(x)//2], data[:len(data)//2], 1)[0]
                        second_half_slope = np.polyfit(x[len(x)//2:], data[len(data)//2:], 1)[0]
                        features[f'{vital}_acceleration'] = second_half_slope - first_half_slope
                    else:
                        features[f'{vital}_acceleration'] = 0
                else:
                    features[f'{vital}_slope'] = 0
                    features[f'{vital}_r_squared'] = 0
                    features[f'{vital}_acceleration'] = 0
                    
        return features
    
    def extract_clinical_features(self, window: pd.DataFrame) -> Dict[str, float]:
        """
        Extract domain-specific clinical features.
        
        These features encode medical knowledge:
        - Shock Index: HR/SBP (elevated in shock)
        - Pulse Pressure: SBP - DBP (narrow in shock)
        - SpO2/HR relationship (compensatory tachycardia when hypoxic)
        """
        features = {}
        
        # Shock Index
        if 'heart_rate' in window.columns and 'systolic_bp' in window.columns:
            hr = window['heart_rate'].mean()
            sbp = window['systolic_bp'].mean()
            if sbp > 0:
                features['shock_index'] = hr / sbp
                features['shock_index_elevated'] = 1 if features['shock_index'] > 0.9 else 0
            else:
                features['shock_index'] = np.nan
                features['shock_index_elevated'] = 0
                
        # Pulse Pressure
        if 'systolic_bp' in window.columns and 'diastolic_bp' in window.columns:
            sbp = window['systolic_bp'].mean()
            dbp = window['diastolic_bp'].mean()
            features['pulse_pressure'] = sbp - dbp
            features['pulse_pressure_narrow'] = 1 if features['pulse_pressure'] < 25 else 0
            
        # SpO2 with compensatory HR
        if 'spo2' in window.columns and 'heart_rate' in window.columns:
            spo2 = window['spo2'].mean()
            hr = window['heart_rate'].mean()
            # Low SpO2 with high HR suggests respiratory compensation
            features['respiratory_compensation'] = 1 if (spo2 < 94 and hr > 100) else 0
            
        # MAP (Mean Arterial Pressure)
        if 'systolic_bp' in window.columns and 'diastolic_bp' in window.columns:
            sbp = window['systolic_bp'].mean()
            dbp = window['diastolic_bp'].mean()
            features['map'] = dbp + (sbp - dbp) / 3
            features['hypotensive'] = 1 if features['map'] < 65 else 0
            
        return features
    
    def extract_cross_vital_features(self, window: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features that capture relationships between vitals.
        
        In clinical deterioration, multiple vitals change together
        in characteristic patterns.
        """
        features = {}
        
        vitals = ['heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp']
        available = [v for v in vitals if v in window.columns]
        
        # Pairwise correlations within window
        for i, v1 in enumerate(available):
            for v2 in available[i+1:]:
                data1 = window[v1].dropna()
                data2 = window[v2].dropna()
                common_idx = data1.index.intersection(data2.index)
                if len(common_idx) > 5:
                    corr = data1.loc[common_idx].corr(data2.loc[common_idx])
                    features[f'corr_{v1}_{v2}'] = corr
                else:
                    features[f'corr_{v1}_{v2}'] = 0
                    
        return features
    
    def extract_all_features(self, window: pd.DataFrame) -> Dict[str, float]:
        """Extract all features from a window"""
        features = {}
        features.update(self.extract_basic_stats(window))
        features.update(self.extract_trend_features(window))
        features.update(self.extract_clinical_features(window))
        features.update(self.extract_cross_vital_features(window))
        
        # Store feature names
        if not self.feature_names:
            self.feature_names = list(features.keys())
            
        return features
    
    def transform_dataframe(
        self, 
        df: pd.DataFrame, 
        window_size: int = 30,
        step_size: int = 10
    ) -> pd.DataFrame:
        """
        Transform full DataFrame into feature vectors using sliding windows.
        
        Args:
            df: DataFrame with vitals
            window_size: Window size in seconds/samples
            step_size: Step between windows
            
        Returns:
            DataFrame where each row is features from one window
        """
        feature_rows = []
        labels = []
        timestamps = []
        
        for start in range(0, len(df) - window_size + 1, step_size):
            end = start + window_size
            window = df.iloc[start:end]
            
            features = self.extract_all_features(window)
            feature_rows.append(features)
            
            # Get label (majority vote in window)
            if 'label' in df.columns:
                window_labels = window['label'].dropna()
                if len(window_labels) > 0:
                    # For anomaly: if any point in window is anomaly (label=1), mark window as anomaly
                    # Ignore -1 (artifact) labels
                    real_labels = window_labels[window_labels >= 0]
                    if len(real_labels) > 0:
                        labels.append(1 if real_labels.max() == 1 else 0)
                    else:
                        labels.append(0)
                else:
                    labels.append(0)
            else:
                labels.append(0)
                
            if 'timestamp' in df.columns:
                timestamps.append(window['timestamp'].iloc[window_size // 2])
        
        feature_df = pd.DataFrame(feature_rows)
        feature_df['label'] = labels
        if timestamps:
            feature_df['timestamp'] = timestamps
            
        return feature_df


class AnomalyDetector:
    """
    Main anomaly detection model combining Isolation Forest with
    statistical methods.
    """
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        self.config = config or AnomalyConfig()
        self.feature_extractor = FeatureExtractor(self.config.window_size)
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        
    def _create_model(self) -> IsolationForest:
        """Create the Isolation Forest model"""
        return IsolationForest(
            contamination=self.config.contamination,
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            n_jobs=-1
        )
    
    def fit(self, df: pd.DataFrame) -> 'AnomalyDetector':
        """
        Fit the anomaly detection model on training data.
        
        Args:
            df: Training DataFrame with vitals (should be mostly normal data)
            
        Returns:
            Self for chaining
        """
        # Extract features
        feature_df = self.feature_extractor.transform_dataframe(
            df, 
            self.config.window_size,
            self.config.window_overlap
        )
        
        # Store feature names
        self.feature_names = [c for c in feature_df.columns 
                            if c not in ['label', 'timestamp']]
        
        # Get feature matrix
        X = feature_df[self.feature_names].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Fit scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model = self._create_model()
        self.model.fit(X_scaled)
        
        self.is_fitted = True
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies in new data.
        
        Args:
            df: DataFrame with vitals
            
        Returns:
            DataFrame with predictions for each window
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Extract features
        feature_df = self.feature_extractor.transform_dataframe(
            df,
            self.config.window_size,
            self.config.window_overlap
        )
        
        # Get feature matrix
        X = feature_df[self.feature_names].values
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        # Isolation Forest: -1 = anomaly, 1 = normal
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)
        
        # Convert to our format: 1 = anomaly, 0 = normal
        feature_df['predicted_anomaly'] = (predictions == -1).astype(int)
        feature_df['anomaly_score'] = -scores  # Higher = more anomalous
        feature_df['confidence'] = self._compute_confidence(scores)
        
        return feature_df
    
    def _compute_confidence(self, scores: np.ndarray) -> np.ndarray:
        """
        Compute confidence score for predictions.
        
        Confidence is based on how far the score is from the decision boundary.
        """
        # Normalize scores to 0-1 range
        # More negative scores (anomalies) get higher confidence
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score - min_score > 0:
            normalized = (scores - min_score) / (max_score - min_score)
        else:
            normalized = np.ones_like(scores) * 0.5
            
        # Convert to confidence (distance from 0.5 decision boundary)
        confidence = np.abs(normalized - 0.5) * 2
        
        return confidence
    
    def predict_single_window(self, window_data: Dict[str, List[float]]) -> Dict:
        """
        Predict anomaly for a single window of data.
        
        This is used for real-time API inference.
        
        Args:
            window_data: Dict with vital names as keys and lists of values
            
        Returns:
            Dict with prediction results
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Convert to DataFrame
        df = pd.DataFrame(window_data)
        
        # Extract features for single window
        features = self.feature_extractor.extract_all_features(df)
        
        # Create feature vector
        X = np.array([[features.get(f, 0) for f in self.feature_names]])
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        score = self.model.decision_function(X_scaled)[0]
        
        is_anomaly = prediction == -1
        confidence = min(abs(score) / 0.5, 1.0)  # Normalize confidence
        
        return {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(-score),
            'confidence': float(confidence),
            'features': {k: float(v) for k, v in features.items() if not np.isnan(v)}
        }
    
    def save(self, path: str):
        """Save the model to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config,
            'feature_extractor': self.feature_extractor
        }
        joblib.dump(model_data, path)
        
    @classmethod
    def load(cls, path: str) -> 'AnomalyDetector':
        """Load a saved model from disk"""
        model_data = joblib.load(path)
        
        detector = cls(config=model_data['config'])
        detector.model = model_data['model']
        detector.scaler = model_data['scaler']
        detector.feature_names = model_data['feature_names']
        detector.feature_extractor = model_data['feature_extractor']
        detector.is_fitted = True
        
        return detector


class StatisticalAnomalyDetector:
    """
    Statistical methods for anomaly detection as secondary/ensemble.
    
    Methods:
    - CUSUM: Cumulative Sum control chart for detecting shifts
    - EWMA: Exponentially Weighted Moving Average for trend detection
    """
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        self.config = config or AnomalyConfig()
        self.baseline_stats = {}
        
    def fit_baseline(self, df: pd.DataFrame):
        """Compute baseline statistics from normal data"""
        vitals = ['heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp']
        
        for vital in vitals:
            if vital in df.columns:
                data = df[vital].dropna()
                self.baseline_stats[vital] = {
                    'mean': data.mean(),
                    'std': data.std()
                }
                
    def compute_cusum(
        self, 
        values: np.ndarray, 
        target_mean: float,
        std: float,
        threshold: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CUSUM control chart.
        
        CUSUM accumulates deviations from target, detecting sustained shifts
        that might be missed by point-in-time analysis.
        
        Args:
            values: Time series values
            target_mean: Expected mean
            std: Expected standard deviation
            threshold: Detection threshold (in std units)
            
        Returns:
            Tuple of (upper_cusum, lower_cusum)
        """
        if threshold is None:
            threshold = self.config.cusum_threshold
            
        k = 0.5 * std  # Allowable slack
        h = threshold * std  # Decision threshold
        
        upper = np.zeros(len(values))
        lower = np.zeros(len(values))
        
        for i in range(1, len(values)):
            upper[i] = max(0, upper[i-1] + values[i] - target_mean - k)
            lower[i] = min(0, lower[i-1] + values[i] - target_mean + k)
            
        return upper, lower
    
    def compute_ewma(
        self, 
        values: np.ndarray,
        span: int = None
    ) -> np.ndarray:
        """
        Compute EWMA for smoothed trend detection.
        
        Args:
            values: Time series values
            span: EWMA span parameter
            
        Returns:
            EWMA values
        """
        if span is None:
            span = self.config.ewma_span
            
        # Pandas EWMA
        ewma = pd.Series(values).ewm(span=span).mean().values
        return ewma
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies using statistical methods.
        
        Args:
            df: DataFrame with vitals
            
        Returns:
            DataFrame with statistical anomaly scores
        """
        results = df.copy()
        vitals = ['heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp']
        
        for vital in vitals:
            if vital not in df.columns or vital not in self.baseline_stats:
                continue
                
            values = df[vital].fillna(method='ffill').fillna(method='bfill').values
            baseline = self.baseline_stats[vital]
            
            # CUSUM
            upper, lower = self.compute_cusum(
                values, 
                baseline['mean'], 
                baseline['std']
            )
            results[f'{vital}_cusum_upper'] = upper
            results[f'{vital}_cusum_lower'] = lower
            
            # Detect CUSUM threshold breaches
            threshold = self.config.cusum_threshold * baseline['std']
            results[f'{vital}_cusum_alert'] = (
                (upper > threshold) | (lower < -threshold)
            ).astype(int)
            
            # EWMA and its derivative
            ewma = self.compute_ewma(values)
            results[f'{vital}_ewma'] = ewma
            results[f'{vital}_ewma_slope'] = np.gradient(ewma)
            
        # Combined statistical anomaly score
        cusum_alerts = [c for c in results.columns if c.endswith('_cusum_alert')]
        if cusum_alerts:
            results['stat_anomaly_score'] = results[cusum_alerts].sum(axis=1)
            
        return results


def main():
    """Demo anomaly detection"""
    from data_generator import DataGenerator
    from artifact_detection import ArtifactCorrector
    
    print("=== Anomaly Detection Demo ===\n")
    
    # Generate data
    generator = DataGenerator(seed=42)
    
    # Training data (mostly normal)
    print("Generating training data (normal transport)...")
    train_data = pd.concat([
        generator.generate_normal_transport(30) for _ in range(5)
    ], ignore_index=True)
    
    # Test data (includes anomalies)
    print("Generating test data (with anomalies)...")
    test_normal = generator.generate_normal_transport(30)
    test_cardiac = generator.generate_cardiac_distress(30, 10, 'moderate')
    test_respiratory = generator.generate_respiratory_distress(30, 12, 'moderate')
    
    # Clean artifacts
    corrector = ArtifactCorrector()
    train_cleaned, _ = corrector.process_artifacts(train_data)
    
    # Train model
    print("Training anomaly detection model...")
    detector = AnomalyDetector()
    detector.fit(train_cleaned)
    
    # Predict on test data
    print("\nPredicting on test data...")
    
    for name, data in [('Normal', test_normal), 
                       ('Cardiac', test_cardiac), 
                       ('Respiratory', test_respiratory)]:
        cleaned, _ = corrector.process_artifacts(data)
        results = detector.predict(cleaned)
        
        n_windows = len(results)
        n_anomalies = results['predicted_anomaly'].sum()
        avg_score = results['anomaly_score'].mean()
        
        print(f"\n{name} scenario:")
        print(f"  Windows analyzed: {n_windows}")
        print(f"  Anomalies detected: {n_anomalies} ({100*n_anomalies/n_windows:.1f}%)")
        print(f"  Average anomaly score: {avg_score:.3f}")
        
        if 'label' in results.columns:
            true_anomalies = results['label'].sum()
            print(f"  True anomalies in data: {true_anomalies}")
    
    # Save model
    import os
    os.makedirs('models', exist_ok=True)
    detector.save('models/anomaly_detector.joblib')
    print("\nModel saved to models/anomaly_detector.joblib")
    
    return detector


if __name__ == "__main__":
    main()
