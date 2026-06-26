"""
=============================================================
AI Interview Analyzer — Head Pose Estimator
=============================================================
Estimates the 3-D orientation of the head (yaw, pitch, roll)
using six canonical face landmarks and OpenCV's solvePnP.

Outputs one of:
  • Facing Front
  • Looking Left / Right
  • Looking Up / Down

Also tracks a "head stability" metric — the percentage of
frames where the head was facing front.
=============================================================
"""

import numpy as np
import cv2
from typing import Tuple

from utils.helpers import MetricAccumulator
import config


# ---------------------------------------------------------
# Six canonical 3-D model points for a generic face
# (nose tip, chin, left/right eye corners, mouth corners).
# Units are arbitrary but proportional to a real face.
# ---------------------------------------------------------
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),       # Nose tip
    (0.0,   -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0,  170.0, -135.0),     # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0,  -150.0, -125.0),    # Right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe Face Mesh landmark indices
LANDMARK_INDICES = [1, 152, 263, 33, 287, 57]


class HeadPoseEstimator:
    """
    Estimates head yaw, pitch, and roll angles using
    PnP (Perspective-n-Point) solving with OpenCV.
    """

    def __init__(self):
        """Initialise thresholds and accumulator."""
        self.yaw_threshold = config.HEAD_YAW_THRESHOLD
        self.pitch_threshold = config.HEAD_PITCH_THRESHOLD
        self.accumulator = MetricAccumulator()

        # Latest computed values
        self.yaw: float = 0.0
        self.pitch: float = 0.0
        self.roll: float = 0.0
        self.direction: str = "Facing Front"

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def detect(self, face_landmarks, frame_w: int,
               frame_h: int) -> str:
        """
        Estimate head pose for one frame.

        Args:
            face_landmarks: MediaPipe face mesh landmarks.
            frame_w: Frame width in pixels.
            frame_h: Frame height in pixels.

        Returns:
            Direction string: "Facing Front", "Looking Left",
            "Looking Right", "Looking Up", "Looking Down".
        """
        # Extract 2-D image points
        image_points = self._get_image_points(
            face_landmarks, frame_w, frame_h)

        # Build a simple camera matrix (no lens distortion)
        focal_length = frame_w
        center = (frame_w / 2.0, frame_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0,            center[0]],
            [0,            focal_length, center[1]],
            [0,            0,            1.0],
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            MODEL_POINTS, image_points, camera_matrix,
            dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        if not success:
            return self.direction

        # Convert rotation vector → rotation matrix → Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        self.yaw, self.pitch, self.roll = (
            self._rotation_matrix_to_euler(rotation_mat))

        # Classify direction
        self.direction = self._classify_direction()

        # Track stability (front-facing = stable)
        is_stable = (self.direction == "Facing Front")
        self.accumulator.update(is_stable)

        return self.direction

    def get_angles(self) -> Tuple[float, float, float]:
        """Return the latest (yaw, pitch, roll) in degrees."""
        return self.yaw, self.pitch, self.roll

    def get_stability_percentage(self) -> float:
        """Return head-stability percentage for the session."""
        return self.accumulator.percentage()

    def reset(self) -> None:
        """Reset for a new session."""
        self.accumulator.reset()
        self.yaw = self.pitch = self.roll = 0.0
        self.direction = "Facing Front"

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _get_image_points(self, landmarks, fw: int,
                          fh: int) -> np.ndarray:
        """
        Extract 2-D pixel coordinates for the six canonical
        landmarks.
        """
        points = []
        for idx in LANDMARK_INDICES:
            lm = landmarks.landmark[idx]
            points.append((lm.x * fw, lm.y * fh))
        return np.array(points, dtype=np.float64)

    @staticmethod
    def _rotation_matrix_to_euler(
            rotation_mat: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert a 3×3 rotation matrix to Euler angles
        (yaw, pitch, roll) in degrees.

        Uses the decomposition:
            pitch = asin(-R[2][0])
            yaw   = atan2(R[2][1], R[2][2])
            roll  = atan2(R[1][0], R[0][0])
        """
        sy = np.sqrt(
            rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            pitch = np.degrees(
                np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2]))
            yaw = np.degrees(
                np.arctan2(-rotation_mat[2, 0], sy))
            roll = np.degrees(
                np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0]))
        else:
            pitch = np.degrees(
                np.arctan2(-rotation_mat[1, 2], rotation_mat[1, 1]))
            yaw = np.degrees(
                np.arctan2(-rotation_mat[2, 0], sy))
            roll = 0.0

        return yaw, pitch, roll

    def _classify_direction(self) -> str:
        """
        Map yaw/pitch angles to a human-readable direction.
        """
        if self.yaw < -self.yaw_threshold:
            return "Looking Right"
        elif self.yaw > self.yaw_threshold:
            return "Looking Left"
        elif self.pitch < -self.pitch_threshold:
            return "Looking Down"
        elif self.pitch > self.pitch_threshold:
            return "Looking Up"
        return "Facing Front"
