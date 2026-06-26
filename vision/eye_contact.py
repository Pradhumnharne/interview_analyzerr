"""
=============================================================
AI Interview Analyzer — Eye Contact Detector
=============================================================
Estimates whether the user is looking at the camera by
analysing the position of the iris relative to the eye
corners using MediaPipe Face Mesh iris landmarks.

Approach:
  1. Extract left and right iris centre landmarks.
  2. Extract the inner and outer corners of each eye.
  3. Compute a horizontal gaze ratio (iris position
     between inner and outer corner).
  4. Compute a vertical gaze ratio similarly.
  5. If both ratios are close to 0.5 (centred), the user
     is making eye contact.
=============================================================
"""

from typing import Tuple, Optional

from utils.helpers import (
    get_landmark_coords,
    euclidean_distance,
    MetricAccumulator,
)
import config


# ---------------------------------------------------------
# MediaPipe Face Mesh landmark indices for iris / eye corners
# (with refine_landmarks=True)
# ---------------------------------------------------------
# Left eye (from the subject's perspective)
LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
LEFT_IRIS_CENTER = 473

# Right eye
RIGHT_EYE_INNER = 133
RIGHT_EYE_OUTER = 33
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
RIGHT_IRIS_CENTER = 468


class EyeContactDetector:
    """
    Detects whether the user is maintaining eye contact
    with the camera using iris-to-eye-corner ratios.
    """

    def __init__(self):
        """Initialise the accumulator and thresholds."""
        self.accumulator = MetricAccumulator()
        self.h_threshold = config.EYE_CONTACT_HORIZONTAL_THRESHOLD
        self.v_threshold = config.EYE_CONTACT_VERTICAL_THRESHOLD
        self.is_looking = False  # Latest frame result

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def detect(self, face_landmarks, frame_w: int,
               frame_h: int) -> bool:
        """
        Analyse one frame and return True if eye contact
        is detected.

        Args:
            face_landmarks: MediaPipe face landmarks for one
                            face (multi_face_landmarks[0]).
            frame_w: Frame width in pixels.
            frame_h: Frame height in pixels.

        Returns:
            True if the user appears to be looking at camera.
        """
        # Compute gaze ratios for both eyes
        left_h, left_v = self._gaze_ratio(
            face_landmarks, frame_w, frame_h,
            LEFT_IRIS_CENTER, LEFT_EYE_INNER, LEFT_EYE_OUTER,
            LEFT_EYE_TOP, LEFT_EYE_BOTTOM)

        right_h, right_v = self._gaze_ratio(
            face_landmarks, frame_w, frame_h,
            RIGHT_IRIS_CENTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
            RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)

        # Average both eyes for stability
        avg_h = (left_h + right_h) / 2.0
        avg_v = (left_v + right_v) / 2.0

        # Check if gaze is centred (≈0.5 means looking straight)
        h_centred = abs(avg_h - 0.5) <= self.h_threshold
        v_centred = abs(avg_v - 0.5) <= self.v_threshold

        self.is_looking = h_centred and v_centred
        self.accumulator.update(self.is_looking)
        return self.is_looking

    def get_percentage(self) -> float:
        """Return eye-contact percentage for the session."""
        return self.accumulator.percentage()

    def reset(self) -> None:
        """Reset for a new session."""
        self.accumulator.reset()
        self.is_looking = False

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _gaze_ratio(self, landmarks, fw: int, fh: int,
                    iris_idx: int, inner_idx: int,
                    outer_idx: int, top_idx: int,
                    bottom_idx: int) -> Tuple[float, float]:
        """
        Compute horizontal and vertical gaze ratios for
        one eye.

        Ratio = distance(iris, inner_corner) /
                distance(inner_corner, outer_corner)

        A ratio of ~0.5 means the iris is centred.

        Args:
            landmarks:  Face landmarks.
            fw, fh:     Frame width and height.
            iris_idx:   Iris centre landmark index.
            inner_idx:  Inner eye corner index.
            outer_idx:  Outer eye corner index.
            top_idx:    Top eyelid landmark index.
            bottom_idx: Bottom eyelid landmark index.

        Returns:
            (horizontal_ratio, vertical_ratio) each in 0..1.
        """
        iris = get_landmark_coords(landmarks, iris_idx, fw, fh)
        inner = get_landmark_coords(landmarks, inner_idx, fw, fh)
        outer = get_landmark_coords(landmarks, outer_idx, fw, fh)
        top = get_landmark_coords(landmarks, top_idx, fw, fh)
        bottom = get_landmark_coords(landmarks, bottom_idx, fw, fh)

        # Horizontal ratio
        eye_width = euclidean_distance(inner, outer)
        if eye_width == 0:
            h_ratio = 0.5
        else:
            h_ratio = euclidean_distance(inner, iris) / eye_width

        # Vertical ratio
        eye_height = euclidean_distance(top, bottom)
        if eye_height == 0:
            v_ratio = 0.5
        else:
            v_ratio = euclidean_distance(top, iris) / eye_height

        return h_ratio, v_ratio
