"""
=============================================================
AI Interview Analyzer — Smile Detector
=============================================================
Detects smiling by analysing the mouth aspect ratio (MAR)
and the ratio of mouth width to face width using MediaPipe
Face Mesh landmarks.

A smile widens the mouth and raises the corners, so:
  • mouth_width / face_width  increases
  • vertical opening may change

Both ratios are compared to configurable thresholds.
=============================================================
"""

from utils.helpers import (
    get_landmark_coords,
    euclidean_distance,
    MetricAccumulator,
)
import config


# ---------------------------------------------------------
# MediaPipe Face Mesh landmark indices for the mouth
# ---------------------------------------------------------
# Outer lip corners
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# Upper and lower lip (for vertical opening)
UPPER_LIP = 13
LOWER_LIP = 14

# Upper and lower lip outer edges (for MAR)
UPPER_LIP_TOP = 12
LOWER_LIP_BOTTOM = 15

# Face width reference points (cheekbones)
LEFT_CHEEK = 234
RIGHT_CHEEK = 454


class SmileDetector:
    """
    Detects smiling using mouth geometry ratios.
    Tracks smile percentage across the session.
    """

    def __init__(self):
        """Initialise thresholds and accumulator."""
        self.mar_threshold = config.SMILE_MAR_THRESHOLD
        self.width_ratio_threshold = config.SMILE_WIDTH_RATIO_THRESHOLD
        self.accumulator = MetricAccumulator()
        self.is_smiling: bool = False
        self.current_mar: float = 0.0
        self.current_width_ratio: float = 0.0

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def detect(self, face_landmarks, frame_w: int,
               frame_h: int) -> bool:
        """
        Analyse one frame for a smile.

        Args:
            face_landmarks: MediaPipe face mesh landmarks.
            frame_w: Frame width in pixels.
            frame_h: Frame height in pixels.

        Returns:
            True if a smile is detected.
        """
        # Mouth corners
        left = get_landmark_coords(
            face_landmarks, MOUTH_LEFT, frame_w, frame_h)
        right = get_landmark_coords(
            face_landmarks, MOUTH_RIGHT, frame_w, frame_h)

        # Upper and lower lip for vertical measurement
        upper = get_landmark_coords(
            face_landmarks, UPPER_LIP_TOP, frame_w, frame_h)
        lower = get_landmark_coords(
            face_landmarks, LOWER_LIP_BOTTOM, frame_w, frame_h)

        # Face width for normalisation
        l_cheek = get_landmark_coords(
            face_landmarks, LEFT_CHEEK, frame_w, frame_h)
        r_cheek = get_landmark_coords(
            face_landmarks, RIGHT_CHEEK, frame_w, frame_h)

        # Calculate mouth width and height
        mouth_width = euclidean_distance(left, right)
        mouth_height = euclidean_distance(upper, lower)
        face_width = euclidean_distance(l_cheek, r_cheek)

        # Mouth Aspect Ratio (width / height)
        if mouth_height == 0:
            self.current_mar = 0.0
        else:
            self.current_mar = mouth_width / mouth_height

        # Mouth-width to face-width ratio
        if face_width == 0:
            self.current_width_ratio = 0.0
        else:
            self.current_width_ratio = mouth_width / face_width

        # Smile if width ratio exceeds threshold
        self.is_smiling = (
            self.current_width_ratio > self.width_ratio_threshold
        )

        self.accumulator.update(self.is_smiling)
        return self.is_smiling

    def get_percentage(self) -> float:
        """Return smile percentage for the session."""
        return self.accumulator.percentage()

    def get_mar(self) -> float:
        """Return the latest Mouth Aspect Ratio."""
        return self.current_mar

    def reset(self) -> None:
        """Reset for a new session."""
        self.accumulator.reset()
        self.is_smiling = False
        self.current_mar = 0.0
        self.current_width_ratio = 0.0
