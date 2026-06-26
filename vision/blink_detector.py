"""
=============================================================
AI Interview Analyzer — Blink Detector
=============================================================
Detects eye blinks using the Eye Aspect Ratio (EAR) method.

EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)

When the eye is open, EAR ≈ 0.25–0.30.
When the eye closes, EAR drops below ~0.21.

A "blink" is registered when EAR stays below the threshold
for a minimum number of consecutive frames.
=============================================================
"""

import time
from typing import Tuple

from utils.helpers import get_landmark_coords, euclidean_distance
import config


# ---------------------------------------------------------
# MediaPipe Face Mesh landmark indices for each eye
# (6 points per eye for the EAR formula)
# ---------------------------------------------------------
# Right eye (subject's right)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# LEFT eye (subject's left)
LEFT_EYE = [362, 385, 387, 263, 373, 380]


class BlinkDetector:
    """
    Counts blinks and computes blink rate using the
    Eye Aspect Ratio (EAR) method.
    """

    def __init__(self):
        """Initialise blink counter and EAR settings."""
        self.ear_threshold = config.EAR_THRESHOLD
        self.consec_frames = config.EAR_CONSEC_FRAMES

        # Internal state
        self._frame_counter: int = 0   # Consecutive low-EAR frames
        self.blink_count: int = 0
        self.current_ear: float = 0.0

        # Timing for blink-rate calculation
        self._start_time: float = time.time()

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def detect(self, face_landmarks, frame_w: int,
               frame_h: int) -> bool:
        """
        Process one frame and return True if a blink was
        just completed.

        Args:
            face_landmarks: MediaPipe face mesh landmarks.
            frame_w: Frame width in pixels.
            frame_h: Frame height in pixels.

        Returns:
            True if a blink was detected on this frame.
        """
        left_ear = self._compute_ear(
            face_landmarks, LEFT_EYE, frame_w, frame_h)
        right_ear = self._compute_ear(
            face_landmarks, RIGHT_EYE, frame_w, frame_h)

        # Average EAR of both eyes
        self.current_ear = (left_ear + right_ear) / 2.0

        blinked = False

        if self.current_ear < self.ear_threshold:
            self._frame_counter += 1
        else:
            # Eye just re-opened after being closed
            if self._frame_counter >= self.consec_frames:
                self.blink_count += 1
                blinked = True
            self._frame_counter = 0

        return blinked

    def get_blink_rate(self) -> float:
        """
        Calculate blinks per minute based on elapsed time.

        Returns:
            Blink rate (blinks/min). Returns 0 if less than
            1 second has elapsed to avoid division issues.
        """
        elapsed = time.time() - self._start_time
        if elapsed < 1.0:
            return 0.0
        return (self.blink_count / elapsed) * 60.0

    def get_blink_count(self) -> int:
        """Return total blink count for the session."""
        return self.blink_count

    def get_ear(self) -> float:
        """Return the most recent average EAR value."""
        return self.current_ear

    def is_blink_rate_normal(self) -> bool:
        """
        Check if the current blink rate is within the
        normal range (15–20 blinks/min).

        Returns:
            True if within normal range.
        """
        rate = self.get_blink_rate()
        return (config.NORMAL_BLINK_RATE_MIN
                <= rate <=
                config.NORMAL_BLINK_RATE_MAX)

    def reset(self) -> None:
        """Reset all counters for a new session."""
        self._frame_counter = 0
        self.blink_count = 0
        self.current_ear = 0.0
        self._start_time = time.time()

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _compute_ear(self, landmarks, eye_indices,
                     fw: int, fh: int) -> float:
        """
        Compute the Eye Aspect Ratio for one eye.

        EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)

        Where p1..p6 are the six eye landmarks in order:
          p1 = outer corner, p4 = inner corner,
          p2/p3 = upper lid, p5/p6 = lower lid.

        Args:
            landmarks:    Face mesh landmarks.
            eye_indices:  List of 6 landmark indices.
            fw, fh:       Frame width and height.

        Returns:
            EAR value (float, typically 0.15–0.35).
        """
        coords = [get_landmark_coords(landmarks, idx, fw, fh)
                  for idx in eye_indices]

        # Vertical distances
        v1 = euclidean_distance(coords[1], coords[5])
        v2 = euclidean_distance(coords[2], coords[4])

        # Horizontal distance
        h = euclidean_distance(coords[0], coords[3])

        if h == 0:
            return 0.0

        return (v1 + v2) / (2.0 * h)
