"""
=============================================================
AI Interview Analyzer — Score Engine
=============================================================
Calculates the overall Confidence Score and generates
rule-based suggestions using only mathematical formulas
and if-else logic.

NO machine learning. NO AI models.

Confidence Score =
    eye_contact%  × 0.35  +
    posture%      × 0.25  +
    smile%        × 0.15  +
    head_stable%  × 0.15  +
    hand_stable%  × 0.10
=============================================================
"""

from typing import Dict, List, Tuple
import config


class ScoreEngine:
    """
    Computes the weighted confidence score and generates
    actionable feedback using rule-based logic.
    """

    def __init__(self):
        """Load weights and suggestion rules from config."""
        self.weights = config.CONFIDENCE_WEIGHTS
        self.suggestion_rules = config.SUGGESTION_RULES
        self.score_thresholds = config.SCORE_THRESHOLDS

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def calculate_confidence(self,
                             metrics: Dict[str, float]) -> float:
        """
        Calculate the weighted confidence score.

        Args:
            metrics: Dictionary with keys matching
                     CONFIDENCE_WEIGHTS keys.  Each value
                     is a percentage (0–100).
                     Expected keys:
                       - eye_contact
                       - posture
                       - smile
                       - head_stability
                       - hand_stability

        Returns:
            Confidence score as a float (0–100).
        """
        score = 0.0
        for metric_name, weight in self.weights.items():
            value = metrics.get(metric_name, 0.0)
            # Clamp each metric to 0–100
            value = max(0.0, min(100.0, value))
            score += value * weight

        return round(score, 2)

    def get_score_label(self, score: float) -> str:
        """
        Map a numeric score to a descriptive label.

        Args:
            score: Confidence score (0–100).

        Returns:
            "Excellent", "Good", "Average", or "Poor".
        """
        if score >= self.score_thresholds["excellent"]:
            return "Excellent"
        elif score >= self.score_thresholds["good"]:
            return "Good"
        elif score >= self.score_thresholds["average"]:
            return "Average"
        return "Poor"

    def get_score_color(self, score: float) -> str:
        """
        Return a hex colour string for the score level.

        Args:
            score: Confidence score (0–100).

        Returns:
            Hex colour string (e.g. "#10B981").
        """
        if score >= self.score_thresholds["excellent"]:
            return config.CHART_COLORS["secondary"]  # Green
        elif score >= self.score_thresholds["good"]:
            return config.CHART_COLORS["info"]        # Blue
        elif score >= self.score_thresholds["average"]:
            return config.CHART_COLORS["accent"]      # Amber
        return config.CHART_COLORS["danger"]          # Red

    def generate_suggestions(self,
                             metrics: Dict[str, float],
                             blink_rate: float
                             ) -> List[Dict[str, str]]:
        """
        Generate rule-based feedback suggestions.

        Uses if-else logic against thresholds defined in
        config.SUGGESTION_RULES.

        Args:
            metrics: Same dictionary as calculate_confidence().
            blink_rate: Blinks per minute.

        Returns:
            List of dicts, each with:
              - "metric": name of the metric
              - "status": "good" or "needs_improvement"
              - "message": suggestion text
        """
        suggestions: List[Dict[str, str]] = []

        # Check each standard metric
        for metric_name in ["eye_contact", "posture", "smile",
                            "head_stability", "hand_stability"]:
            rule = self.suggestion_rules.get(metric_name)
            if rule is None:
                continue

            value = metrics.get(metric_name, 0.0)
            threshold = rule["threshold"]

            if value < threshold:
                suggestions.append({
                    "metric": metric_name,
                    "status": "needs_improvement",
                    "message": rule["low_msg"],
                })
            else:
                suggestions.append({
                    "metric": metric_name,
                    "status": "good",
                    "message": rule["high_msg"],
                })

        # Check blink rate separately (two-sided threshold)
        blink_rule = self.suggestion_rules.get("blink_rate")
        if blink_rule:
            if blink_rate < blink_rule["threshold_low"]:
                suggestions.append({
                    "metric": "blink_rate",
                    "status": "needs_improvement",
                    "message": blink_rule["low_msg"],
                })
            elif blink_rate > blink_rule["threshold_high"]:
                suggestions.append({
                    "metric": "blink_rate",
                    "status": "needs_improvement",
                    "message": blink_rule["high_msg"],
                })
            else:
                suggestions.append({
                    "metric": "blink_rate",
                    "status": "good",
                    "message": blink_rule["normal_msg"],
                })

        return suggestions

    def build_session_summary(self,
                              metrics: Dict[str, float],
                              blink_rate: float,
                              blink_count: int,
                              duration_seconds: float
                              ) -> Dict:
        """
        Build a complete session summary dictionary suitable
        for storage in the database and PDF report.

        Args:
            metrics:          Metric percentages dict.
            blink_rate:       Blinks per minute.
            blink_count:      Total blinks in the session.
            duration_seconds: Session length in seconds.

        Returns:
            Dictionary containing all scores, the label,
            suggestions, and session metadata.
        """
        confidence = self.calculate_confidence(metrics)
        label = self.get_score_label(confidence)
        suggestions = self.generate_suggestions(
            metrics, blink_rate)

        return {
            "metrics": {
                "eye_contact":    round(metrics.get("eye_contact", 0), 2),
                "posture":        round(metrics.get("posture", 0), 2),
                "smile":          round(metrics.get("smile", 0), 2),
                "head_stability": round(metrics.get("head_stability", 0), 2),
                "hand_stability": round(metrics.get("hand_stability", 0), 2),
            },
            "blink_rate":       round(blink_rate, 2),
            "blink_count":      blink_count,
            "confidence_score": confidence,
            "score_label":      label,
            "duration_seconds": round(duration_seconds, 1),
            "suggestions":      suggestions,
        }
