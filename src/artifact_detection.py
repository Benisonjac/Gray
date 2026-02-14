"""
Artifact Detection Module for Smart Ambulance Patient Monitoring

This module implements artifact detection and signal cleaning BEFORE
anomaly detection. This is critical because:
1. Motion artifacts can cause false positive alerts
2. Sensor dropouts should not trigger clinical alerts
3. Clean data improves anomaly detection accuracy

Artifact Types Handled:
- Motion-induced artifacts (SpO2 drops, HR spikes from vehicle movement)
- Sensor dropouts (missing data, probe disconnection)
- Signal noise (electrical interference, poor contact)
- Physiologically impossible values

Key Insight: Motion artifacts often show:
- Sudden changes that don't correlate across vitals
- High motion signal coinciding with vital changes
- Values returning to baseline quickly after motion stops
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from scipy import signal
from scipy.stats import zscore
import matplotlib.pyplot as plt


@dataclass
class ArtifactDetectionConfig:
    """Configuration for artifact detection thresholds"""
    # Motion correlation threshold
    motion_threshold: float = 0.4  # Motion level above which artifacts are likely
    motion_correlation_window: int = 5  # Seconds to check motion correlation
    
    # Rate of change thresholds (physiologically impossible)
    hr_max_change_per_second: int = 30  # BPM
    spo2_max_change_per_second: float = 5.0  # Percentage
    sbp_max_change_per_second: int = 20  # mmHg
    
    # Absolute bounds (physiologically impossible values)
    hr_bounds: Tuple[int, int] = (30, 220)
    spo2_bounds: Tuple[float, float] = (50.0, 100.0)
    sbp_bounds: Tuple[int, int] = (40, 250)
    dbp_bounds: Tuple[int, int] = (20, 150)
    
    # Signal quality thresholds
    min_consecutive_valid: int = 3  # Minimum valid points to trust
    max_dropout_interpolate: int = 5  # Max seconds to interpolate


class ArtifactDetector:
    """
    Detects and handles artifacts in patient vital signs.
    
    Pipeline:
    1. Detect physiologically impossible values
    2. Detect motion-correlated artifacts
    3. Detect sensor dropouts
    4. Apply appropriate handling (flag, correct, or interpolate)
    """
    
    def __init__(self, config: Optional[ArtifactDetectionConfig] = None):
        self.config = config or ArtifactDetectionConfig()
        self.artifact_log = []
        
    def detect_impossible_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag values outside physiological bounds.
        
        These are definitely artifacts - no human can have HR of 300 or SpO2 of 150.
        
        Args:
            df: DataFrame with vital signs
            
        Returns:
            DataFrame with 'impossible_artifact' column added
        """
        impossible = pd.Series(False, index=df.index)
        
        # Heart rate bounds
        if 'heart_rate' in df.columns:
            hr = df['heart_rate']
            impossible |= (hr < self.config.hr_bounds[0]) | (hr > self.config.hr_bounds[1])
            
        # SpO2 bounds
        if 'spo2' in df.columns:
            spo2 = df['spo2']
            impossible |= (spo2 < self.config.spo2_bounds[0]) | (spo2 > self.config.spo2_bounds[1])
            
        # Blood pressure bounds
        if 'systolic_bp' in df.columns:
            sbp = df['systolic_bp']
            impossible |= (sbp < self.config.sbp_bounds[0]) | (sbp > self.config.sbp_bounds[1])
            
        if 'diastolic_bp' in df.columns:
            dbp = df['diastolic_bp']
            impossible |= (dbp < self.config.dbp_bounds[0]) | (dbp > self.config.dbp_bounds[1])
            # DBP should be less than SBP
            if 'systolic_bp' in df.columns:
                impossible |= (dbp >= df['systolic_bp'])
        
        df['impossible_artifact'] = impossible
        
        n_impossible = impossible.sum()
        if n_impossible > 0:
            self.artifact_log.append(f"Found {n_impossible} impossible values")
            
        return df
    
    def detect_rate_of_change_artifacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect unrealistic rates of change in vital signs.
        
        Example: Heart rate jumping from 70 to 150 in 1 second is
        physiologically impossible and indicates an artifact.
        
        Args:
            df: DataFrame with vital signs
            
        Returns:
            DataFrame with 'roc_artifact' column added
        """
        roc_artifact = pd.Series(False, index=df.index)
        
        if 'heart_rate' in df.columns:
            hr_diff = df['heart_rate'].diff().abs()
            roc_artifact |= hr_diff > self.config.hr_max_change_per_second
            
        if 'spo2' in df.columns:
            spo2_diff = df['spo2'].diff().abs()
            roc_artifact |= spo2_diff > self.config.spo2_max_change_per_second
            
        if 'systolic_bp' in df.columns:
            sbp_diff = df['systolic_bp'].diff().abs()
            roc_artifact |= sbp_diff > self.config.sbp_max_change_per_second
            
        df['roc_artifact'] = roc_artifact
        
        n_roc = roc_artifact.sum()
        if n_roc > 0:
            self.artifact_log.append(f"Found {n_roc} rate-of-change artifacts")
            
        return df
    
    def detect_motion_artifacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect artifacts correlated with motion signal.
        
        Key insight: If a sudden vital sign change occurs exactly when
        motion is high, it's likely an artifact, not a clinical event.
        
        Clinical events:
        - Develop gradually (seconds to minutes)
        - Affect multiple vitals in physiologically consistent ways
        - Persist after initial trigger
        
        Motion artifacts:
        - Occur suddenly during motion
        - May affect only one vital (e.g., SpO2 but not HR)
        - Resolve quickly when motion stops
        
        Args:
            df: DataFrame with vitals and motion column
            
        Returns:
            DataFrame with 'motion_artifact' column added
        """
        if 'motion' not in df.columns:
            df['motion_artifact'] = False
            return df
            
        motion_artifact = pd.Series(False, index=df.index)
        
        # High motion periods
        high_motion = df['motion'] > self.config.motion_threshold
        
        # Check for SpO2 drops during high motion
        if 'spo2' in df.columns:
            spo2_drop = df['spo2'].diff() < -3  # Sudden drop
            motion_artifact |= (spo2_drop & high_motion)
            
        # Check for HR spikes during high motion
        if 'heart_rate' in df.columns:
            hr_change = df['heart_rate'].diff().abs() > 15
            motion_artifact |= (hr_change & high_motion)
            
        # Extend artifact window slightly (artifacts may lag motion)
        window = self.config.motion_correlation_window
        motion_artifact = motion_artifact.rolling(window, center=True, min_periods=1).max().astype(bool)
        
        df['motion_artifact'] = motion_artifact
        
        n_motion = motion_artifact.sum()
        if n_motion > 0:
            self.artifact_log.append(f"Found {n_motion} motion-correlated artifacts")
            
        return df
    
    def detect_dropouts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect sensor dropouts (NaN or constant values).
        
        Args:
            df: DataFrame with vital signs
            
        Returns:
            DataFrame with 'dropout' column added
        """
        dropout = pd.Series(False, index=df.index)
        
        vital_cols = ['heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp']
        for col in vital_cols:
            if col in df.columns:
                # NaN values
                dropout |= df[col].isna()
                
                # Constant values (stuck sensor) - check for 10+ identical consecutive values
                is_constant = df[col].diff() == 0
                constant_streak = is_constant.rolling(10, min_periods=10).sum() == 10
                dropout |= constant_streak
                
        df['dropout'] = dropout
        
        n_dropout = dropout.sum()
        if n_dropout > 0:
            self.artifact_log.append(f"Found {n_dropout} dropout points")
            
        return df
    
    def detect_all_artifacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all artifact detection methods.
        
        Args:
            df: DataFrame with vital signs
            
        Returns:
            DataFrame with artifact columns and combined 'is_artifact' flag
        """
        self.artifact_log = []  # Reset log
        
        df = self.detect_impossible_values(df.copy())
        df = self.detect_rate_of_change_artifacts(df)
        df = self.detect_motion_artifacts(df)
        df = self.detect_dropouts(df)
        
        # Combined artifact flag
        df['is_artifact'] = (
            df['impossible_artifact'] | 
            df['roc_artifact'] | 
            df['motion_artifact'] | 
            df['dropout']
        )
        
        return df
    
    def get_artifact_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of detected artifacts.
        """
        if 'is_artifact' not in df.columns:
            df = self.detect_all_artifacts(df)
            
        total = len(df)
        summary = {
            'total_samples': total,
            'artifact_samples': df['is_artifact'].sum(),
            'artifact_percentage': 100 * df['is_artifact'].sum() / total,
            'impossible_values': df['impossible_artifact'].sum() if 'impossible_artifact' in df.columns else 0,
            'roc_artifacts': df['roc_artifact'].sum() if 'roc_artifact' in df.columns else 0,
            'motion_artifacts': df['motion_artifact'].sum() if 'motion_artifact' in df.columns else 0,
            'dropouts': df['dropout'].sum() if 'dropout' in df.columns else 0,
            'detection_log': self.artifact_log
        }
        return summary


class ArtifactCorrector:
    """
    Corrects or handles detected artifacts in vital signs.
    
    Strategy:
    - Short artifacts (1-5 seconds): Interpolate
    - Motion artifacts: Flag but don't use for anomaly detection
    - Impossible values: Replace with NaN, then interpolate if short
    - Long dropouts: Leave as NaN, flag as unreliable
    """
    
    def __init__(self, max_interpolate_gap: int = 5):
        self.max_interpolate_gap = max_interpolate_gap
        
    def interpolate_short_gaps(
        self, 
        df: pd.DataFrame, 
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Interpolate short gaps in vital signs.
        
        Args:
            df: DataFrame with vitals
            columns: Columns to interpolate (default: all vital columns)
            
        Returns:
            DataFrame with interpolated values
        """
        if columns is None:
            columns = ['heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp']
            
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            # Identify gaps
            is_gap = df[col].isna() | (df.get('is_artifact', False))
            
            # Find gap lengths
            gap_groups = (is_gap != is_gap.shift()).cumsum()
            gap_lengths = is_gap.groupby(gap_groups).transform('sum')
            
            # Only interpolate short gaps
            short_gaps = is_gap & (gap_lengths <= self.max_interpolate_gap)
            
            # Set short gap values to NaN for interpolation
            df.loc[short_gaps, col] = np.nan
            
            # Linear interpolation
            df[col] = df[col].interpolate(method='linear', limit=self.max_interpolate_gap)
            
            # Forward/backward fill for edges
            df[col] = df[col].ffill().bfill()
            
        return df
    
    def apply_median_filter(
        self, 
        df: pd.DataFrame, 
        columns: List[str] = None,
        window: int = 3
    ) -> pd.DataFrame:
        """
        Apply median filter to remove spike artifacts.
        
        Args:
            df: DataFrame with vitals
            columns: Columns to filter
            window: Filter window size (must be odd)
            
        Returns:
            DataFrame with filtered values
        """
        if columns is None:
            columns = ['heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp']
            
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            values = df[col].values
            if not np.isnan(values).all():
                # Handle NaN by interpolating first
                valid = ~np.isnan(values)
                if valid.sum() > window:
                    filtered = signal.medfilt(
                        np.nan_to_num(values, nan=np.nanmedian(values)), 
                        window
                    )
                    df[col] = filtered
                    
        return df
    
    def process_artifacts(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Full artifact processing pipeline.
        
        Returns:
            Tuple of (cleaned_df, original_df_with_artifact_flags)
        """
        # Detect artifacts
        detector = ArtifactDetector()
        df_flagged = detector.detect_all_artifacts(df.copy())
        
        # Correct artifacts
        df_cleaned = df_flagged.copy()
        
        # Replace artifact values with NaN
        vital_cols = ['heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp']
        for col in vital_cols:
            if col in df_cleaned.columns:
                df_cleaned.loc[df_cleaned['is_artifact'], col] = np.nan
        
        # Interpolate short gaps
        df_cleaned = self.interpolate_short_gaps(df_cleaned, vital_cols)
        
        # Apply gentle median filter for remaining noise
        df_cleaned = self.apply_median_filter(df_cleaned, vital_cols, window=3)
        
        # Mark data quality
        df_cleaned['data_quality'] = np.where(
            df_flagged['is_artifact'],
            'corrected',
            'original'
        )
        
        return df_cleaned, df_flagged


def plot_artifact_comparison(
    original_df: pd.DataFrame, 
    cleaned_df: pd.DataFrame,
    vital: str = 'spo2',
    start_idx: int = 0,
    end_idx: int = 500,
    save_path: Optional[str] = None
):
    """
    Plot before/after comparison of artifact correction.
    
    Args:
        original_df: Original data with artifacts
        cleaned_df: Data after artifact correction
        vital: Which vital to plot
        start_idx: Start index for plot
        end_idx: End index for plot
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    time_range = range(start_idx, min(end_idx, len(original_df)))
    
    # Original signal
    ax1 = axes[0]
    ax1.plot(time_range, original_df[vital].iloc[start_idx:end_idx], 
             'b-', alpha=0.7, label='Original')
    if 'is_artifact' in original_df.columns:
        artifact_mask = original_df['is_artifact'].iloc[start_idx:end_idx]
        ax1.scatter(
            [t for t, a in zip(time_range, artifact_mask) if a],
            original_df[vital].iloc[start_idx:end_idx][artifact_mask],
            c='red', s=20, label='Detected Artifact', zorder=5
        )
    ax1.set_ylabel(f'{vital.upper()} (Original)')
    ax1.legend()
    ax1.set_title('Before Artifact Correction')
    ax1.grid(True, alpha=0.3)
    
    # Motion signal
    ax2 = axes[1]
    if 'motion' in original_df.columns:
        ax2.fill_between(time_range, 0, original_df['motion'].iloc[start_idx:end_idx],
                        alpha=0.5, color='orange', label='Motion')
        ax2.axhline(y=0.4, color='red', linestyle='--', label='Motion Threshold')
    ax2.set_ylabel('Motion Signal')
    ax2.legend()
    ax2.set_title('Motion Signal (Artifact Indicator)')
    ax2.grid(True, alpha=0.3)
    
    # Cleaned signal
    ax3 = axes[2]
    ax3.plot(time_range, cleaned_df[vital].iloc[start_idx:end_idx], 
             'g-', alpha=0.7, label='Cleaned')
    ax3.set_ylabel(f'{vital.upper()} (Cleaned)')
    ax3.set_xlabel('Time (seconds)')
    ax3.legend()
    ax3.set_title('After Artifact Correction')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    return fig


def main():
    """Demonstrate artifact detection and correction"""
    from data_generator import DataGenerator
    
    # Generate data with motion artifacts
    generator = DataGenerator(seed=42)
    df = generator.generate_heavy_motion_artifacts(duration_minutes=10)
    
    print("=== Artifact Detection Demo ===\n")
    
    # Detect and correct artifacts
    corrector = ArtifactCorrector()
    df_cleaned, df_flagged = corrector.process_artifacts(df)
    
    # Get summary
    detector = ArtifactDetector()
    summary = detector.get_artifact_summary(df_flagged)
    
    print("Artifact Detection Summary:")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Artifact samples: {summary['artifact_samples']} ({summary['artifact_percentage']:.1f}%)")
    print(f"  - Impossible values: {summary['impossible_values']}")
    print(f"  - Rate of change artifacts: {summary['roc_artifacts']}")
    print(f"  - Motion artifacts: {summary['motion_artifacts']}")
    print(f"  - Dropouts: {summary['dropouts']}")
    
    # Save comparison plot
    import os
    os.makedirs('outputs/figures', exist_ok=True)
    plot_artifact_comparison(
        df_flagged, df_cleaned, 
        vital='spo2',
        start_idx=200, end_idx=600,
        save_path='outputs/figures/artifact_correction_spo2.png'
    )
    
    plot_artifact_comparison(
        df_flagged, df_cleaned,
        vital='heart_rate', 
        start_idx=200, end_idx=600,
        save_path='outputs/figures/artifact_correction_hr.png'
    )
    
    print("\nArtifact correction plots saved to outputs/figures/")
    
    return df_flagged, df_cleaned


if __name__ == "__main__":
    main()
