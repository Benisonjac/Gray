"""
Data Generation Module for Smart Ambulance Patient Monitoring System

This module generates synthetic but realistic time-series data simulating
patient vitals during ambulance transport. It includes:
- Normal transport scenarios
- Distress/deterioration scenarios  
- Sensor artifacts (motion, noise, dropouts)

Signals Generated (sampled at 1 Hz / 1 second intervals):
- Heart Rate (HR): 40-200 bpm, normal range 60-100
- SpO2: 70-100%, normal range 95-100%
- Systolic Blood Pressure (SBP): 70-200 mmHg, normal 90-140
- Diastolic Blood Pressure (DBP): 40-120 mmHg, normal 60-90
- Motion Signal: 0-1 normalized, represents vehicle/patient movement

Assumptions & Limitations:
1. Vitals are simplified - real ECG/PPG waveforms are more complex
2. Motion signal is synthesized, not from actual accelerometer
3. Correlation between vitals is simplified
4. Does not account for patient age, medications, or conditions
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class ScenarioType(Enum):
    """Types of clinical scenarios that can be simulated"""
    NORMAL_TRANSPORT = "normal_transport"
    CARDIAC_DISTRESS = "cardiac_distress"
    RESPIRATORY_DISTRESS = "respiratory_distress"
    HYPOTENSIVE_CRISIS = "hypotensive_crisis"
    COMBINED_DETERIORATION = "combined_deterioration"
    MOTION_ARTIFACT = "motion_artifact"
    SENSOR_DROPOUT = "sensor_dropout"


@dataclass
class VitalRanges:
    """Normal and abnormal ranges for vital signs"""
    # Heart Rate
    hr_normal: Tuple[int, int] = (60, 100)
    hr_tachycardia: Tuple[int, int] = (100, 180)
    hr_bradycardia: Tuple[int, int] = (35, 60)
    
    # SpO2
    spo2_normal: Tuple[int, int] = (95, 100)
    spo2_hypoxia_mild: Tuple[int, int] = (90, 95)
    spo2_hypoxia_severe: Tuple[int, int] = (70, 90)
    
    # Systolic Blood Pressure
    sbp_normal: Tuple[int, int] = (90, 140)
    sbp_hypertensive: Tuple[int, int] = (140, 200)
    sbp_hypotensive: Tuple[int, int] = (60, 90)
    
    # Diastolic Blood Pressure  
    dbp_normal: Tuple[int, int] = (60, 90)
    dbp_hypertensive: Tuple[int, int] = (90, 120)
    dbp_hypotensive: Tuple[int, int] = (30, 60)


class DataGenerator:
    """
    Generates synthetic patient vitals data for ambulance transport scenarios.
    
    The generator creates realistic time-series data with:
    - Physiological noise and natural variation
    - Temporal correlation (values don't jump randomly)
    - Inter-signal correlation (e.g., HR and BP are related)
    - Motion artifacts that affect sensor readings
    - Sensor dropouts and missing data
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.ranges = VitalRanges()
        
    def _generate_smooth_signal(
        self, 
        duration: int, 
        base_value: float, 
        noise_std: float,
        smoothing_factor: float = 0.9
    ) -> np.ndarray:
        """
        Generate a smooth, temporally correlated signal using autoregressive process.
        
        Args:
            duration: Length of signal in seconds
            base_value: Center value for the signal
            noise_std: Standard deviation of noise
            smoothing_factor: AR(1) coefficient, higher = smoother
            
        Returns:
            Smooth signal array
        """
        signal = np.zeros(duration)
        signal[0] = base_value + self.rng.normal(0, noise_std)
        
        for i in range(1, duration):
            innovation = self.rng.normal(0, noise_std * (1 - smoothing_factor**2)**0.5)
            signal[i] = smoothing_factor * signal[i-1] + (1 - smoothing_factor) * base_value + innovation
            
        return signal
    
    def _generate_trend(
        self, 
        duration: int, 
        start_value: float, 
        end_value: float,
        trend_type: str = "linear"
    ) -> np.ndarray:
        """
        Generate a trend component for deterioration scenarios.
        
        Args:
            duration: Length of trend
            start_value: Starting value
            end_value: Ending value 
            trend_type: "linear", "exponential", or "sigmoid"
            
        Returns:
            Trend array
        """
        t = np.linspace(0, 1, duration)
        
        if trend_type == "linear":
            trend = start_value + (end_value - start_value) * t
        elif trend_type == "exponential":
            # Exponential approach to end value
            trend = end_value - (end_value - start_value) * np.exp(-3 * t)
        elif trend_type == "sigmoid":
            # S-curve transition
            trend = start_value + (end_value - start_value) / (1 + np.exp(-10 * (t - 0.5)))
        else:
            trend = np.linspace(start_value, end_value, duration)
            
        return trend
    
    def _add_motion_artifacts(
        self, 
        signal: np.ndarray, 
        motion: np.ndarray,
        artifact_magnitude: float,
        artifact_type: str = "multiplicative"
    ) -> np.ndarray:
        """
        Add motion-induced artifacts to a signal.
        
        Motion artifacts cause:
        - SpO2 drops (finger probe movement)
        - HR spikes (EMG interference)
        - BP measurement errors
        
        Args:
            signal: Original signal
            motion: Motion intensity signal (0-1)
            artifact_magnitude: Strength of artifact effect
            artifact_type: "multiplicative" or "additive"
            
        Returns:
            Signal with artifacts
        """
        if artifact_type == "multiplicative":
            # Motion reduces signal quality proportionally
            artifact = 1 - artifact_magnitude * motion
            return signal * artifact
        else:
            # Motion adds noise
            artifact = artifact_magnitude * motion * self.rng.normal(0, 1, len(signal))
            return signal + artifact
    
    def _add_sensor_dropout(
        self, 
        signal: np.ndarray,
        dropout_probability: float = 0.01,
        dropout_duration_range: Tuple[int, int] = (1, 5)
    ) -> np.ndarray:
        """
        Add sensor dropout periods (missing data/NaN values).
        
        Args:
            signal: Original signal
            dropout_probability: Probability of dropout starting at each point
            dropout_duration_range: Min/max duration of each dropout
            
        Returns:
            Signal with NaN values for dropouts
        """
        result = signal.copy()
        i = 0
        while i < len(signal):
            if self.rng.random() < dropout_probability:
                duration = self.rng.integers(*dropout_duration_range)
                result[i:i+duration] = np.nan
                i += duration
            else:
                i += 1
        return result
    
    def generate_normal_transport(
        self, 
        duration_minutes: int = 30,
        patient_baseline: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Generate data for normal ambulance transport without clinical events.
        
        Patient remains stable with normal vital variations and minor
        motion artifacts from vehicle movement.
        
        Args:
            duration_minutes: Duration of transport in minutes
            patient_baseline: Optional dict with baseline vitals
            
        Returns:
            DataFrame with timestamp and all vitals
        """
        duration_seconds = duration_minutes * 60
        
        # Set baseline values
        if patient_baseline is None:
            patient_baseline = {
                'hr': self.rng.integers(65, 85),
                'spo2': self.rng.uniform(96, 99),
                'sbp': self.rng.integers(100, 130),
                'dbp': self.rng.integers(65, 85)
            }
        
        # Generate motion signal (vehicle movement)
        # Smooth base motion with occasional bumps
        base_motion = self._generate_smooth_signal(duration_seconds, 0.1, 0.05, 0.95)
        bumps = self.rng.random(duration_seconds) < 0.02  # 2% chance of bump
        bump_magnitude = self.rng.uniform(0.3, 0.8, duration_seconds) * bumps
        motion = np.clip(base_motion + bump_magnitude, 0, 1)
        
        # Generate heart rate
        hr_base = self._generate_smooth_signal(
            duration_seconds, patient_baseline['hr'], 3, 0.95
        )
        hr = self._add_motion_artifacts(hr_base, motion, 0.15, "additive")
        hr = np.clip(hr, 40, 200).astype(int)
        
        # Generate SpO2
        spo2_base = self._generate_smooth_signal(
            duration_seconds, patient_baseline['spo2'], 0.5, 0.98
        )
        # Motion causes SpO2 drops (probe displacement)
        spo2 = self._add_motion_artifacts(spo2_base, motion, 0.08, "multiplicative")
        spo2 = np.clip(spo2, 70, 100)
        
        # Generate Blood Pressure (intermittent, but simulated continuous)
        sbp_base = self._generate_smooth_signal(
            duration_seconds, patient_baseline['sbp'], 5, 0.97
        )
        dbp_base = self._generate_smooth_signal(
            duration_seconds, patient_baseline['dbp'], 3, 0.97
        )
        # Ensure SBP > DBP
        sbp = np.clip(sbp_base, 60, 220).astype(int)
        dbp = np.clip(dbp_base, 30, 130).astype(int)
        dbp = np.minimum(dbp, sbp - 20)  # Maintain pulse pressure
        
        # Add minor dropouts
        spo2 = self._add_sensor_dropout(spo2, 0.005, (1, 3))
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=duration_seconds, freq='s'),
            'heart_rate': hr,
            'spo2': spo2,
            'systolic_bp': sbp,
            'diastolic_bp': dbp,
            'motion': motion,
            'scenario': 'normal_transport',
            'label': 0  # 0 = normal, no anomaly
        })
        
        return df
    
    def generate_cardiac_distress(
        self,
        duration_minutes: int = 30,
        distress_start_minute: int = 10,
        severity: str = "moderate"
    ) -> pd.DataFrame:
        """
        Generate data simulating cardiac distress during transport.
        
        Simulates:
        - Progressive tachycardia 
        - BP changes (often drops in severe cases)
        - SpO2 may decrease if perfusion affected
        
        Args:
            duration_minutes: Total duration
            distress_start_minute: When distress begins
            severity: "mild", "moderate", or "severe"
            
        Returns:
            DataFrame with vitals and anomaly labels
        """
        duration_seconds = duration_minutes * 60
        distress_start = distress_start_minute * 60
        
        # Severity parameters
        severity_params = {
            "mild": {"hr_increase": 25, "bp_drop": 10, "spo2_drop": 2},
            "moderate": {"hr_increase": 45, "bp_drop": 25, "spo2_drop": 5},
            "severe": {"hr_increase": 70, "bp_drop": 45, "spo2_drop": 10}
        }
        params = severity_params[severity]
        
        # Generate baseline normal period
        baseline = self.generate_normal_transport(distress_start_minute)
        
        # Generate distress period
        distress_duration = duration_seconds - distress_start
        
        # HR increases with distress
        hr_trend = self._generate_trend(
            distress_duration,
            baseline['heart_rate'].iloc[-1],
            baseline['heart_rate'].iloc[-1] + params['hr_increase'],
            "sigmoid"
        )
        hr_noise = self._generate_smooth_signal(distress_duration, 0, 5, 0.9)
        hr_distress = np.clip(hr_trend + hr_noise, 40, 200).astype(int)
        
        # BP drops progressively
        sbp_trend = self._generate_trend(
            distress_duration,
            baseline['systolic_bp'].iloc[-1],
            baseline['systolic_bp'].iloc[-1] - params['bp_drop'],
            "exponential"
        )
        sbp_noise = self._generate_smooth_signal(distress_duration, 0, 4, 0.95)
        sbp_distress = np.clip(sbp_trend + sbp_noise, 60, 220).astype(int)
        
        dbp_trend = self._generate_trend(
            distress_duration,
            baseline['diastolic_bp'].iloc[-1],
            baseline['diastolic_bp'].iloc[-1] - params['bp_drop'] * 0.6,
            "exponential"
        )
        dbp_distress = np.clip(dbp_trend, 30, 130).astype(int)
        
        # SpO2 may decrease
        spo2_trend = self._generate_trend(
            distress_duration,
            baseline['spo2'].iloc[-1],
            baseline['spo2'].iloc[-1] - params['spo2_drop'],
            "exponential"
        )
        spo2_noise = self._generate_smooth_signal(distress_duration, 0, 0.5, 0.98)
        spo2_distress = np.clip(spo2_trend + spo2_noise, 70, 100)
        
        # Motion continues normally
        motion_distress = self._generate_smooth_signal(distress_duration, 0.1, 0.05, 0.95)
        motion_distress = np.clip(motion_distress, 0, 1)
        
        # Get baseline spo2 values, filling any NaN
        baseline_spo2 = baseline['spo2'].fillna(baseline['spo2'].median()).values
        
        # Combine baseline and distress periods
        hr_combined = np.concatenate([baseline['heart_rate'].values, hr_distress])[:duration_seconds]
        spo2_combined = np.concatenate([baseline_spo2, spo2_distress])[:duration_seconds]
        sbp_combined = np.concatenate([baseline['systolic_bp'].values, sbp_distress])[:duration_seconds]
        dbp_combined = np.concatenate([baseline['diastolic_bp'].values, dbp_distress])[:duration_seconds]
        motion_combined = np.concatenate([baseline['motion'].values, motion_distress])[:duration_seconds]
        label_combined = np.concatenate([
            np.zeros(distress_start),
            np.ones(distress_duration)
        ])[:duration_seconds].astype(int)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=duration_seconds, freq='s'),
            'heart_rate': hr_combined,
            'spo2': spo2_combined,
            'systolic_bp': sbp_combined,
            'diastolic_bp': dbp_combined,
            'motion': motion_combined,
            'scenario': 'cardiac_distress',
            'label': label_combined
        })
        
        return df
    
    def generate_respiratory_distress(
        self,
        duration_minutes: int = 30,
        distress_start_minute: int = 12,
        severity: str = "moderate"
    ) -> pd.DataFrame:
        """
        Generate data simulating respiratory distress/hypoxia.
        
        Simulates:
        - Progressive SpO2 drop
        - Compensatory tachycardia
        - BP may increase initially (stress response)
        
        Args:
            duration_minutes: Total duration
            distress_start_minute: When distress begins
            severity: "mild", "moderate", or "severe"
            
        Returns:
            DataFrame with vitals and anomaly labels
        """
        duration_seconds = duration_minutes * 60
        distress_start = distress_start_minute * 60
        
        severity_params = {
            "mild": {"spo2_drop": 6, "hr_increase": 15},
            "moderate": {"spo2_drop": 12, "hr_increase": 30},
            "severe": {"spo2_drop": 20, "hr_increase": 50}
        }
        params = severity_params[severity]
        
        # Generate baseline
        baseline = self.generate_normal_transport(distress_start_minute)
        distress_duration = duration_seconds - distress_start
        
        # SpO2 drops significantly
        spo2_trend = self._generate_trend(
            distress_duration,
            97,  # baseline['spo2'].iloc[-1],
            97 - params['spo2_drop'],
            "exponential"
        )
        spo2_noise = self._generate_smooth_signal(distress_duration, 0, 1, 0.95)
        spo2_distress = np.clip(spo2_trend + spo2_noise, 70, 100)
        
        # HR increases as compensation
        hr_trend = self._generate_trend(
            distress_duration,
            baseline['heart_rate'].iloc[-1],
            baseline['heart_rate'].iloc[-1] + params['hr_increase'],
            "exponential"
        )
        hr_distress = np.clip(hr_trend + self.rng.normal(0, 3, distress_duration), 40, 200).astype(int)
        
        # BP may increase initially
        sbp_distress = self._generate_smooth_signal(
            distress_duration, baseline['systolic_bp'].iloc[-1] + 10, 6, 0.95
        ).astype(int)
        dbp_distress = self._generate_smooth_signal(
            distress_duration, baseline['diastolic_bp'].iloc[-1] + 5, 4, 0.95  
        ).astype(int)
        
        motion_distress = self._generate_smooth_signal(distress_duration, 0.1, 0.05, 0.95)
        
        # Combine arrays with proper length handling
        hr_combined = np.concatenate([baseline['heart_rate'].values, hr_distress])[:duration_seconds]
        spo2_combined = np.concatenate([
            np.full(distress_start, baseline['spo2'].median()),
            spo2_distress
        ])[:duration_seconds]
        sbp_combined = np.concatenate([baseline['systolic_bp'].values, sbp_distress])[:duration_seconds]
        dbp_combined = np.concatenate([baseline['diastolic_bp'].values, dbp_distress])[:duration_seconds]
        motion_combined = np.concatenate([baseline['motion'].values, np.clip(motion_distress, 0, 1)])[:duration_seconds]
        label_combined = np.concatenate([
            np.zeros(distress_start),
            np.ones(distress_duration)
        ])[:duration_seconds].astype(int)
        
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=duration_seconds, freq='s'),
            'heart_rate': hr_combined,
            'spo2': spo2_combined,
            'systolic_bp': sbp_combined,
            'diastolic_bp': dbp_combined,
            'motion': motion_combined,
            'scenario': 'respiratory_distress',
            'label': label_combined
        })
        
        return df
    
    def generate_heavy_motion_artifacts(
        self,
        duration_minutes: int = 30,
        high_motion_periods: List[Tuple[int, int]] = None
    ) -> pd.DataFrame:
        """
        Generate data with significant motion artifacts that could be mistaken 
        for clinical events (false positives).
        
        This is critical for testing artifact detection - motion artifacts
        should NOT trigger clinical alerts.
        
        Args:
            duration_minutes: Total duration
            high_motion_periods: List of (start_minute, end_minute) tuples
            
        Returns:
            DataFrame with motion artifacts clearly labeled
        """
        duration_seconds = duration_minutes * 60
        
        if high_motion_periods is None:
            # Default: motion periods scaled to duration
            if duration_minutes >= 25:
                high_motion_periods = [
                    (5, 7),   # Rough road at minute 5-7
                    (12, 15), # Highway bumps
                    (22, 25)  # Urban potholes
                ]
            elif duration_minutes >= 15:
                high_motion_periods = [
                    (3, 5),
                    (8, 10)
                ]
            else:
                high_motion_periods = [
                    (2, 4)
                ]
        
        # Start with normal transport
        df = self.generate_normal_transport(duration_minutes)
        
        # Create artifact mask
        artifact_mask = np.zeros(duration_seconds, dtype=bool)
        
        for start_min, end_min in high_motion_periods:
            start_sec = start_min * 60
            end_sec = min(end_min * 60, duration_seconds)
            artifact_mask[start_sec:end_sec] = True
            
            # Increase motion during these periods
            df.loc[start_sec:end_sec-1, 'motion'] = np.clip(
                self._generate_smooth_signal(end_sec - start_sec, 0.7, 0.15, 0.8),
                0.3, 1.0
            )
            
            # Add SpO2 artifacts (drops due to probe movement)
            spo2_artifact = df.loc[start_sec:end_sec-1, 'spo2'].values
            artifact_drop = self.rng.uniform(5, 15, end_sec - start_sec)
            df.loc[start_sec:end_sec-1, 'spo2'] = np.clip(spo2_artifact - artifact_drop, 75, 100)
            
            # Add HR artifacts (spikes from EMG)
            hr_artifact = df.loc[start_sec:end_sec-1, 'heart_rate'].values.astype(float)
            artifact_spike = self.rng.uniform(-20, 30, end_sec - start_sec)
            df.loc[start_sec:end_sec-1, 'heart_rate'] = np.clip(hr_artifact + artifact_spike, 40, 200).astype(int)
        
        # Label: 0 for normal, -1 for artifact (not a real clinical event)
        df['label'] = 0
        df.loc[artifact_mask, 'label'] = -1  # -1 indicates artifact, not real anomaly
        df['scenario'] = 'motion_artifact_test'
        df['is_artifact'] = artifact_mask.astype(int)
        
        return df
    
    def generate_mixed_scenario_dataset(
        self,
        n_normal: int = 3,
        n_cardiac: int = 2,
        n_respiratory: int = 2,
        n_motion_artifact: int = 2,
        duration_minutes: int = 30
    ) -> pd.DataFrame:
        """
        Generate a mixed dataset with multiple scenarios for training/testing.
        
        Args:
            n_normal: Number of normal transport scenarios
            n_cardiac: Number of cardiac distress scenarios
            n_respiratory: Number of respiratory distress scenarios
            n_motion_artifact: Number of motion artifact scenarios
            duration_minutes: Duration of each scenario
            
        Returns:
            Combined DataFrame with patient_id for identification
        """
        all_data = []
        patient_id = 0
        
        # Normal scenarios
        for i in range(n_normal):
            df = self.generate_normal_transport(duration_minutes)
            df['patient_id'] = patient_id
            all_data.append(df)
            patient_id += 1
            
        # Cardiac distress
        for i in range(n_cardiac):
            severity = self.rng.choice(['mild', 'moderate', 'severe'])
            start_min = self.rng.integers(8, 15)
            df = self.generate_cardiac_distress(duration_minutes, start_min, severity)
            df['patient_id'] = patient_id
            all_data.append(df)
            patient_id += 1
            
        # Respiratory distress
        for i in range(n_respiratory):
            severity = self.rng.choice(['mild', 'moderate', 'severe'])
            start_min = self.rng.integers(8, 15)
            df = self.generate_respiratory_distress(duration_minutes, start_min, severity)
            df['patient_id'] = patient_id
            all_data.append(df)
            patient_id += 1
            
        # Motion artifacts
        for i in range(n_motion_artifact):
            df = self.generate_heavy_motion_artifacts(duration_minutes)
            df['patient_id'] = patient_id
            all_data.append(df)
            patient_id += 1
        
        # Combine and shuffle by patient
        combined = pd.concat(all_data, ignore_index=True)
        
        return combined


def main():
    """Generate and save sample dataset"""
    generator = DataGenerator(seed=42)
    
    # Generate mixed dataset
    print("Generating synthetic patient vitals dataset...")
    dataset = generator.generate_mixed_scenario_dataset(
        n_normal=3,
        n_cardiac=2,
        n_respiratory=2,
        n_motion_artifact=2,
        duration_minutes=30
    )
    
    # Save to CSV
    output_path = "data/synthetic_vitals.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"Total records: {len(dataset)}")
    print(f"Unique patients: {dataset['patient_id'].nunique()}")
    print(f"\nScenario distribution:")
    print(dataset.groupby('scenario')['patient_id'].nunique())
    
    return dataset


if __name__ == "__main__":
    main()
