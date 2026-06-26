"""
=============================================================
AI Interview Analyzer — Posture Detector
=============================================================
Determines whether the user has good posture or is slouching
by analysing:
  1. Shoulder alignment — angle of the line connecting the
     left and right shoulders relative to horizontal.
  2. Neck angle — the forward lean of the head relative
     to the midpoint of the shoulders.

Uses MediaPipe Pose landmarks.
=============================================================
"""

import math
from typing import Tuple, Optional

from utils.helpers import (
    get_landmark_coords,
    angle_from_horizontal,
    angle_between_points,
    midpoint,
    MetricAccumulator,
)
import config


# ---------------------------------------------------------
# MediaPipe Pose landmark indices
# ---------------------------------------------------------
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_HIP = 23
RIGHT_HIP = 24


class PostureDetector:
    """
    Classifies posture as "Good" or "Slouching" using
    shoulder tilt and neck-lean angle.
    """

    def __init__(self):
        """Initialise thresholds and accumulator."""
        self.shoulder_threshold = config.POSTURE_SHOULDER_ANGLE_THRESHOLD
        self.neck_threshold = config.POSTURE_NECK_ANGLE_THRESHOLD
        self.accumulator = MetricAccumulator()

        # Latest frame results
        self.is_good_posture: bool = True
        self.shoulder_angle: float = 0.0
        self.neck_angle: float = 0.0

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def detect(self, pose_landmarks, frame_w: int,
               frame_h: int) -> bool:
        """
        Evaluate posture for one frame.

        Args:
            pose_landmarks: MediaPipe pose landmarks.
            frame_w: Frame width in pixels.
            frame_h: Frame height in pixels.

        Returns:
            True if posture is good, False if slouching.
        """
        # Extract key points
        l_shoulder = get_landmark_coords(
            pose_landmarks, LEFT_SHOULDER, frame_w, frame_h)
        r_shoulder = get_landmark_coords(
            pose_landmarks, RIGHT_SHOULDER, frame_w, frame_h)
        l_ear = get_landmark_coords(
            pose_landmarks, LEFT_EAR, frame_w, frame_h)
        r_ear = get_landmark_coords(
            pose_landmarks, RIGHT_EAR, frame_w, frame_h)
        l_hip = get_landmark_coords(
            pose_landmarks, LEFT_HIP, frame_w, frame_h)
        r_hip = get_landmark_coords(
            pose_landmarks, RIGHT_HIP, frame_w, frame_h)

        # 1. Shoulder tilt — angle with the horizontal
        self.shoulder_angle = abs(
            angle_from_horizontal(l_shoulder, r_shoulder))

        # 2. Neck lean angle
        # Midpoint of shoulders and ears
        shoulder_mid = midpoint(l_shoulder, r_shoulder)
        ear_mid = midpoint(l_ear, r_ear)
        hip_mid = midpoint(l_hip, r_hip)

        # Angle at shoulder_mid between hip_mid and ear_mid
        self.neck_angle = self._compute_neck_angle(
            ear_mid, shoulder_mid, hip_mid)

        # Classify posture
        shoulder_ok = self.shoulder_angle < self.shoulder_threshold
        neck_ok = self.neck_angle < self.neck_threshold

        self.is_good_posture = shoulder_ok and neck_ok
        self.accumulator.update(self.is_good_posture)
        return self.is_good_posture

    def get_percentage(self) -> float:
        """Return good-posture percentage for the session."""
        return self.accumulator.percentage()

    def get_status(self) -> str:
        """Return 'Good Posture' or 'Slouching'."""
        return "Good Posture" if self.is_good_posture else "Slouching"

    def reset(self) -> None:
        """Reset for a new session."""
        self.accumulator.reset()
        self.is_good_posture = True
        self.shoulder_angle = 0.0
        self.neck_angle = 0.0

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    @staticmethod
    def _compute_neck_angle(ear_mid: Tuple[float, float],
                            shoulder_mid: Tuple[float, float],
                            hip_mid: Tuple[float, float]) -> float:
        """
        Compute the forward-lean angle of the neck.

        Measures the angle between:
          • A vertical line from shoulder_mid downward
          • The line from shoulder_mid to ear_mid

        A perfectly upright posture → ~0°.
        Slouching / leaning forward → larger angle.

        Returns:
            Angle in degrees (0–90+).
        """
        # Vector from shoulder midpoint to ear midpoint
        dx = ear_mid[0] - shoulder_mid[0]
        dy = ear_mid[1] - shoulder_mid[1]

        # Vertical reference vector (straight up, negative y)
        # In image coords, y increases downward, so "up" is -y.
        ref_dx = 0
        ref_dy = -1

        # Angle between vectors
        dot = dx * ref_dx + dy * ref_dy
        mag = math.sqrt(dx ** 2 + dy ** 2)

        if mag == 0:
            return 0.0

        cos_angle = max(-1.0, min(1.0, dot / mag))
        return math.degrees(math.acos(cos_angle))
