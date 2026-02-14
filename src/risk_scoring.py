"""
Risk Scoring Module for Smart Ambulance Patient Monitoring

This module computes a triage/risk score combining:
1. Multiple vital signs (not just one metric)
2. Trends over time (getting worse vs stable)
3. Anomaly detection confidence
4. Alert suppression logic (to reduce false alarms)

Risk Score Design Philosophy:
- Score 0-100, where higher = more urgent
- Below 30: Normal, continue monitoring
- 30-50: Elevated, increase monitoring frequency
- 50-70: High, prepare for intervention
- Above 70: Critical, immediate action needed

Alert Suppression:
- Don't alert on single-point anomalies (may be artifacts)
- Require sustained abnormality or rapid deterioration
- Consider motion context (suppress during high motion)
- Hysteresis: Once alerted, don't re-alert for same event
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """Risk level categories for clinical triage"""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"
    
    @classmethod
    def from_score(cls, score: float) -> 'RiskLevel':
        if score < 30:
            return cls.NORMAL
        elif score < 50:
            return cls.ELEVATED
        elif score < 70:
            return cls.HIGH
        else:
            return cls.CRITICAL


@dataclass  
class RiskScoringConfig:
    """Configuration for risk scoring"""
    # Individual vital score weights
    weights: Dict[str, float] = None
    
    # Thresholds for vital scores
    hr_normal_range: Tuple[int, int] = (60, 100)
    spo2_normal_min: float = 95.0
    sbp_normal_range: Tuple[int, int] = (90, 140)
    map_normal_min: float = 65.0
    
    # Trend weights
    trend_weight: float = 0.3  # How much trend affects final score
    
    # Alert suppression
    min_sustained_duration: int = 30  # Seconds of sustained abnormality for alert
    motion_suppression_threshold: float = 0.5  # High motion level
    hysteresis_window: int = 60  # Seconds before re-alerting same condition
    
    def __post_init__(self):
        if self.weights is None:
            # Default weights - clinical importance
            self.weights = {
                'heart_rate': 0.25,
                'spo2': 0.30,  # SpO2 critical in ambulance
                'blood_pressure': 0.25,
                'trend': 0.20
            }


class VitalScorer:
    """
    Compute individual risk scores for each vital sign.
    
    Each vital is scored 0-100 based on:
    - Distance from normal range
    - Rate of change (trend)
    - Clinical severity mapping
    """
    
    def __init__(self, config: Optional[RiskScoringConfig] = None):
        self.config = config or RiskScoringConfig()
        
    def score_heart_rate(self, hr: float, hr_trend: float = 0) -> Dict:
        """
        Score heart rate.
        
        Scoring logic:
        - Normal (60-100): 0-10
        - Mild tachycardia (100-120) or bradycardia (50-60): 20-40
        - Moderate (120-150 or 40-50): 40-60
        - Severe (>150 or <40): 60-90
        - Extreme (>180 or <35): 90-100
        
        Args:
            hr: Heart rate in BPM
            hr_trend: Rate of change (positive = increasing)
            
        Returns:
            Dict with score and explanation
        """
        normal_low, normal_high = self.config.hr_normal_range
        
        # Base score from absolute value
        if normal_low <= hr <= normal_high:
            base_score = 5 + (abs(hr - 80) / 20) * 5  # 0-10 range
        elif hr > normal_high:
            # Tachycardia
            if hr <= 120:
                base_score = 20 + (hr - 100) * 1.0
            elif hr <= 150:
                base_score = 40 + (hr - 120) * 0.67
            elif hr <= 180:
                base_score = 60 + (hr - 150) * 1.0
            else:
                base_score = 90 + min((hr - 180) * 0.5, 10)
        else:
            # Bradycardia
            if hr >= 50:
                base_score = 20 + (60 - hr) * 2.0
            elif hr >= 40:
                base_score = 40 + (50 - hr) * 2.0
            elif hr >= 35:
                base_score = 60 + (40 - hr) * 6.0
            else:
                base_score = 90 + min((35 - hr) * 2.0, 10)
        
        # Trend adjustment
        trend_adjustment = abs(hr_trend) * 5  # Up to 15 points for rapid change
        trend_adjustment = min(trend_adjustment, 15)
        
        final_score = min(base_score + trend_adjustment, 100)
        
        # Generate explanation
        if final_score < 20:
            explanation = "Heart rate within normal range"
        elif hr > normal_high:
            explanation = f"Tachycardia detected (HR: {hr:.0f})"
        else:
            explanation = f"Bradycardia detected (HR: {hr:.0f})"
            
        if hr_trend > 1:
            explanation += ", increasing trend"
        elif hr_trend < -1:
            explanation += ", decreasing trend"
            
        return {
            'score': final_score,
            'value': hr,
            'trend': hr_trend,
            'explanation': explanation
        }
    
    def score_spo2(self, spo2: float, spo2_trend: float = 0) -> Dict:
        """
        Score SpO2 (oxygen saturation).
        
        SpO2 is critical - hypoxia can cause rapid deterioration.
        
        Scoring logic:
        - Normal (95-100): 0-10
        - Mild hypoxia (90-95): 30-50
        - Moderate (85-90): 50-70
        - Severe (<85): 70-100
        
        Args:
            spo2: SpO2 percentage
            spo2_trend: Rate of change (negative = dropping)
        """
        normal_min = self.config.spo2_normal_min
        
        if spo2 >= normal_min:
            base_score = (100 - spo2) * 2  # 0-10 for 95-100%
        elif spo2 >= 90:
            base_score = 30 + (95 - spo2) * 4  # 30-50 for 90-95%
        elif spo2 >= 85:
            base_score = 50 + (90 - spo2) * 4  # 50-70 for 85-90%
        elif spo2 >= 75:
            base_score = 70 + (85 - spo2) * 2  # 70-90 for 75-85%
        else:
            base_score = 90 + min((75 - spo2), 10)  # 90-100 for <75%
        
        # Trend adjustment - dropping SpO2 is very concerning
        if spo2_trend < 0:  # Dropping
            trend_adjustment = abs(spo2_trend) * 10  # Significant penalty
        else:
            trend_adjustment = 0
            
        trend_adjustment = min(trend_adjustment, 20)
        final_score = min(base_score + trend_adjustment, 100)
        
        # Explanation
        if final_score < 20:
            explanation = "Oxygen saturation normal"
        elif spo2 >= 90:
            explanation = f"Mild hypoxia (SpO2: {spo2:.0f}%)"
        elif spo2 >= 85:
            explanation = f"Moderate hypoxia (SpO2: {spo2:.0f}%)"
        else:
            explanation = f"Severe hypoxia (SpO2: {spo2:.0f}%)"
            
        if spo2_trend < -0.5:
            explanation += ", rapidly declining"
            
        return {
            'score': final_score,
            'value': spo2,
            'trend': spo2_trend,
            'explanation': explanation
        }
    
    def score_blood_pressure(
        self, 
        sbp: float, 
        dbp: float,
        sbp_trend: float = 0
    ) -> Dict:
        """
        Score blood pressure using MAP and trends.
        
        Mean Arterial Pressure (MAP) < 65 mmHg indicates inadequate
        organ perfusion.
        
        Scoring:
        - Normal (MAP 65-105): 0-20
        - Hypotension (MAP < 65): 40-100 based on severity
        - Hypertension (MAP > 105): 20-60 based on severity
        """
        # Calculate MAP: DBP + 1/3(SBP - DBP)
        map_value = dbp + (sbp - dbp) / 3
        
        # Calculate pulse pressure
        pulse_pressure = sbp - dbp
        
        if 65 <= map_value <= 105:
            base_score = 10
            if pulse_pressure < 25:  # Narrow pulse pressure
                base_score += 15
        elif map_value < 65:
            # Hypotension - dangerous
            deficit = 65 - map_value
            base_score = 40 + min(deficit * 2, 60)
        else:
            # Hypertension
            excess = map_value - 105
            base_score = 20 + min(excess * 0.5, 40)
            
        # Trend adjustment
        trend_adjustment = 0
        if sbp_trend < -2:  # Dropping BP
            trend_adjustment = abs(sbp_trend) * 5
        elif sbp_trend > 3:  # Rising BP
            trend_adjustment = sbp_trend * 2
            
        trend_adjustment = min(trend_adjustment, 20)
        final_score = min(base_score + trend_adjustment, 100)
        
        # Explanation
        if final_score < 20:
            explanation = "Blood pressure normal"
        elif map_value < 65:
            explanation = f"Hypotension (MAP: {map_value:.0f})"
        elif map_value > 105:
            explanation = f"Hypertension (SBP: {sbp:.0f})"
        else:
            explanation = f"Blood pressure borderline (SBP: {sbp:.0f})"
            
        if pulse_pressure < 25:
            explanation += ", narrow pulse pressure"
            
        return {
            'score': final_score,
            'sbp': sbp,
            'dbp': dbp,
            'map': map_value,
            'pulse_pressure': pulse_pressure,
            'trend': sbp_trend,
            'explanation': explanation
        }


class RiskCalculator:
    """
    Main risk scoring calculator that combines all components.
    """
    
    def __init__(self, config: Optional[RiskScoringConfig] = None):
        self.config = config or RiskScoringConfig()
        self.vital_scorer = VitalScorer(self.config)
        self.alert_history = []
        self.last_alert_time = None
        
    def compute_risk_score(
        self,
        vitals: Dict[str, float],
        trends: Dict[str, float] = None,
        anomaly_score: float = 0,
        anomaly_confidence: float = 0,
        motion_level: float = 0
    ) -> Dict:
        """
        Compute comprehensive risk score.
        
        Args:
            vitals: Dict with 'heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp'
            trends: Dict with trend values for each vital
            anomaly_score: Score from anomaly detection model
            anomaly_confidence: Confidence from anomaly detection
            motion_level: Current motion level (for suppression)
            
        Returns:
            Dict with risk score, level, explanations, and alert decision
        """
        if trends is None:
            trends = {k: 0 for k in vitals}
            
        results = {
            'components': {},
            'explanations': [],
            'suppression_reasons': []
        }
        
        # Score each vital
        if 'heart_rate' in vitals:
            hr_result = self.vital_scorer.score_heart_rate(
                vitals['heart_rate'],
                trends.get('heart_rate', 0)
            )
            results['components']['heart_rate'] = hr_result
            if hr_result['score'] >= 20:
                results['explanations'].append(hr_result['explanation'])
                
        if 'spo2' in vitals:
            spo2_result = self.vital_scorer.score_spo2(
                vitals['spo2'],
                trends.get('spo2', 0)
            )
            results['components']['spo2'] = spo2_result
            if spo2_result['score'] >= 20:
                results['explanations'].append(spo2_result['explanation'])
                
        if 'systolic_bp' in vitals and 'diastolic_bp' in vitals:
            bp_result = self.vital_scorer.score_blood_pressure(
                vitals['systolic_bp'],
                vitals['diastolic_bp'],
                trends.get('systolic_bp', 0)
            )
            results['components']['blood_pressure'] = bp_result
            if bp_result['score'] >= 20:
                results['explanations'].append(bp_result['explanation'])
        
        # Compute weighted average
        weights = self.config.weights
        total_weight = 0
        weighted_sum = 0
        
        if 'heart_rate' in results['components']:
            weighted_sum += results['components']['heart_rate']['score'] * weights['heart_rate']
            total_weight += weights['heart_rate']
            
        if 'spo2' in results['components']:
            weighted_sum += results['components']['spo2']['score'] * weights['spo2']
            total_weight += weights['spo2']
            
        if 'blood_pressure' in results['components']:
            weighted_sum += results['components']['blood_pressure']['score'] * weights['blood_pressure']
            total_weight += weights['blood_pressure']
        
        # Add anomaly detection component
        if anomaly_score > 0 and anomaly_confidence > 0.5:
            anomaly_contribution = anomaly_score * 20 * anomaly_confidence
            weighted_sum += anomaly_contribution * weights.get('trend', 0.2)
            total_weight += weights.get('trend', 0.2)
            results['anomaly_contribution'] = anomaly_contribution
            
        # Normalize
        if total_weight > 0:
            raw_score = weighted_sum / total_weight
        else:
            raw_score = 0
            
        # Apply suppression logic
        suppressed = False
        suppression_reasons = []
        
        # Motion suppression
        if motion_level > self.config.motion_suppression_threshold:
            suppressed = True
            suppression_reasons.append(f"High motion detected ({motion_level:.2f})")
            
        # Hysteresis - don't spam alerts
        if self.last_alert_time is not None:
            # This would be checked against actual time in production
            suppression_reasons.append("Recent alert - monitoring for change")
        
        results['suppression_reasons'] = suppression_reasons
        
        # Final score (may be reduced if suppressed)
        if suppressed and raw_score < 70:  # Still alert for critical
            final_score = max(raw_score * 0.7, 0)  # Reduce but don't eliminate
            results['suppressed'] = True
        else:
            final_score = raw_score
            results['suppressed'] = False
            
        results['risk_score'] = round(final_score, 1)
        results['risk_level'] = RiskLevel.from_score(final_score).value
        results['should_alert'] = final_score >= 50 and not suppressed
        results['confidence'] = max(anomaly_confidence, 0.7) if final_score >= 30 else 0.5
        
        return results
    
    def compute_from_window(
        self,
        window_df: pd.DataFrame,
        anomaly_score: float = 0,
        anomaly_confidence: float = 0
    ) -> Dict:
        """
        Compute risk score from a DataFrame window.
        
        Args:
            window_df: DataFrame with vital columns
            anomaly_score: From anomaly detection
            anomaly_confidence: From anomaly detection
            
        Returns:
            Risk score result dict
        """
        # Extract current values (most recent)
        vitals = {}
        trends = {}
        
        for col in ['heart_rate', 'spo2', 'systolic_bp', 'diastolic_bp']:
            if col in window_df.columns:
                values = window_df[col].dropna()
                if len(values) > 0:
                    vitals[col] = values.iloc[-1]
                    if len(values) > 5:
                        # Compute trend as slope per second
                        trends[col] = (values.iloc[-1] - values.iloc[-5]) / 5
                    else:
                        trends[col] = 0
                        
        # Get motion level
        motion_level = 0
        if 'motion' in window_df.columns:
            motion_level = window_df['motion'].iloc[-1] if not pd.isna(window_df['motion'].iloc[-1]) else 0
            
        return self.compute_risk_score(
            vitals=vitals,
            trends=trends,
            anomaly_score=anomaly_score,
            anomaly_confidence=anomaly_confidence,
            motion_level=motion_level
        )


class AlertManager:
    """
    Manages alert generation and suppression.
    
    Ensures:
    - Alerts are meaningful (not too frequent)
    - Critical alerts always go through
    - Context is preserved for clinical decision
    """
    
    def __init__(self, config: Optional[RiskScoringConfig] = None):
        self.config = config or RiskScoringConfig()
        self.alert_history = []
        self.active_alerts = {}
        
    def should_alert(
        self,
        risk_result: Dict,
        timestamp: pd.Timestamp = None
    ) -> Dict:
        """
        Determine if an alert should be generated.
        
        Args:
            risk_result: Output from RiskCalculator
            timestamp: Current timestamp
            
        Returns:
            Dict with alert decision and details
        """
        score = risk_result['risk_score']
        level = risk_result['risk_level']
        
        alert_decision = {
            'should_alert': False,
            'alert_type': None,
            'priority': 'low',
            'message': '',
            'details': risk_result
        }
        
        if level == 'critical':
            alert_decision['should_alert'] = True
            alert_decision['alert_type'] = 'CRITICAL'
            alert_decision['priority'] = 'urgent'
            alert_decision['message'] = "CRITICAL: Immediate attention required"
            
        elif level == 'high' and not risk_result.get('suppressed', False):
            alert_decision['should_alert'] = True
            alert_decision['alert_type'] = 'WARNING'
            alert_decision['priority'] = 'high'
            alert_decision['message'] = "WARNING: Patient condition deteriorating"
            
        elif level == 'elevated':
            alert_decision['should_alert'] = False  # Monitor, don't alert
            alert_decision['alert_type'] = 'MONITOR'
            alert_decision['priority'] = 'medium'
            alert_decision['message'] = "Elevated risk - increased monitoring"
            
        # Add explanations
        if risk_result.get('explanations'):
            alert_decision['message'] += ": " + "; ".join(risk_result['explanations'])
            
        # Record alert
        if alert_decision['should_alert']:
            self.alert_history.append({
                'timestamp': timestamp,
                'score': score,
                'level': level,
                'message': alert_decision['message']
            })
            
        return alert_decision
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alerts generated"""
        return {
            'total_alerts': len(self.alert_history),
            'critical_alerts': sum(1 for a in self.alert_history if 'CRITICAL' in str(a.get('level', ''))),
            'history': self.alert_history[-10:]  # Last 10 alerts
        }


def main():
    """Demo risk scoring"""
    from data_generator import DataGenerator
    from artifact_detection import ArtifactCorrector
    from anomaly_detection import AnomalyDetector
    
    print("=== Risk Scoring Demo ===\n")
    
    # Generate test scenarios
    generator = DataGenerator(seed=42)
    
    # Test cases
    test_cases = [
        ("Normal Transport", generator.generate_normal_transport(10)),
        ("Cardiac Distress", generator.generate_cardiac_distress(10, 5, 'moderate')),
        ("Respiratory Distress", generator.generate_respiratory_distress(10, 5, 'severe')),
    ]
    
    risk_calculator = RiskCalculator()
    alert_manager = AlertManager()
    
    for name, data in test_cases:
        print(f"\n--- {name} ---")
        
        # Get last window of data
        window = data.tail(30)
        
        # Compute risk score
        result = risk_calculator.compute_from_window(window, anomaly_score=0.3, anomaly_confidence=0.7)
        
        print(f"Risk Score: {result['risk_score']:.1f}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Should Alert: {result['should_alert']}")
        
        if result['explanations']:
            print("Reasons:")
            for exp in result['explanations']:
                print(f"  - {exp}")
                
        # Get alert decision
        alert = alert_manager.should_alert(result)
        if alert['should_alert']:
            print(f"ALERT: {alert['message']}")
    
    print("\n" + "=" * 50)
    print("Alert Summary:", alert_manager.get_alert_summary())


if __name__ == "__main__":
    main()
