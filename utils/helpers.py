"""
=============================================================
AI Interview Analyzer — Utility Helpers
=============================================================
Shared helper functions used across all vision modules,
scoring, and reporting.

Includes:
  • Geometric calculations (distance, angle, midpoint)
  • Landmark coordinate extraction from MediaPipe results
  • Drawing overlays on OpenCV frames
  • Timestamp and formatting utilities
=============================================================
"""

import math
import time
from datetime import datetime, timedelta
from typing import Tuple, List, Optional

import cv2
import numpy as np


# ---------------------------------------------------------
# 1. GEOMETRIC CALCULATIONS
# ---------------------------------------------------------

def euclidean_distance(point1: Tuple[float, float],
                       point2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two 2-D points.

    Args:
        point1: (x, y) coordinates of the first point.
        point2: (x, y) coordinates of the second point.

    Returns:
        Distance as a float.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 +
                     (point1[1] - point2[1]) ** 2)


def midpoint(point1: Tuple[float, float],
             point2: Tuple[float, float]) -> Tuple[float, float]:
    """
    Return the midpoint between two 2-D points.

    Args:
        point1: (x, y) of the first point.
        point2: (x, y) of the second point.

    Returns:
        (mid_x, mid_y) as a tuple.
    """
    return ((point1[0] + point2[0]) / 2.0,
            (point1[1] + point2[1]) / 2.0)


def angle_between_points(point1: Tuple[float, float],
                         vertex: Tuple[float, float],
                         point2: Tuple[float, float]) -> float:
    """
    Calculate the angle (in degrees) at *vertex* formed by
    the line segments vertex→point1 and vertex→point2.

    Uses the dot-product formula:
        cos(θ) = (A · B) / (|A| × |B|)

    Args:
        point1: First endpoint.
        vertex:  The vertex of the angle.
        point2: Second endpoint.

    Returns:
        Angle in degrees (0–180).
    """
    # Vectors from vertex to each point
    v1 = (point1[0] - vertex[0], point1[1] - vertex[1])
    v2 = (point2[0] - vertex[0], point2[1] - vertex[1])

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    if mag1 * mag2 == 0:
        return 0.0

    # Clamp to [-1, 1] to avoid math domain errors from
    # floating-point imprecision.
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def angle_from_horizontal(point1: Tuple[float, float],
                          point2: Tuple[float, float]) -> float:
    """
    Calculate the angle (in degrees) that the line segment
    point1→point2 makes with the horizontal axis.

    Useful for checking shoulder tilt in posture detection.

    Args:
        point1: (x, y) of the first point.
        point2: (x, y) of the second point.

    Returns:
        Angle in degrees. Positive = tilted clockwise.
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return math.degrees(math.atan2(dy, dx))


# ---------------------------------------------------------
# 2. LANDMARK EXTRACTION
# ---------------------------------------------------------

def get_landmark_coords(landmarks, index: int,
                        frame_w: int,
                        frame_h: int) -> Tuple[int, int]:
    """
    Extract pixel coordinates for a single MediaPipe landmark.

    MediaPipe returns normalised coordinates (0..1).
    This function converts them to absolute pixel values.

    Args:
        landmarks: MediaPipe landmark list (e.g.,
                   face_mesh_results.multi_face_landmarks[0]).
        index:     Landmark index number.
        frame_w:   Frame width in pixels.
        frame_h:   Frame height in pixels.

    Returns:
        (x, y) in pixel coordinates.
    """
    lm = landmarks.landmark[index]
    return int(lm.x * frame_w), int(lm.y * frame_h)


def get_landmark_coords_3d(landmarks, index: int,
                           frame_w: int,
                           frame_h: int) -> Tuple[int, int, float]:
    """
    Extract pixel coordinates **and** the normalised z-depth
    for a single MediaPipe landmark.

    Args:
        landmarks: MediaPipe landmark list.
        index:     Landmark index number.
        frame_w:   Frame width in pixels.
        frame_h:   Frame height in pixels.

    Returns:
        (x, y, z) where x, y are pixels and z is normalised.
    """
    lm = landmarks.landmark[index]
    return int(lm.x * frame_w), int(lm.y * frame_h), lm.z


def get_multiple_landmark_coords(
        landmarks,
        indices: List[int],
        frame_w: int,
        frame_h: int) -> List[Tuple[int, int]]:
    """
    Extract pixel coordinates for multiple landmarks at once.

    Args:
        landmarks: MediaPipe landmark list.
        indices:   List of landmark index numbers.
        frame_w:   Frame width in pixels.
        frame_h:   Frame height in pixels.

    Returns:
        List of (x, y) tuples in pixel coordinates.
    """
    return [get_landmark_coords(landmarks, idx, frame_w, frame_h)
            for idx in indices]


# ---------------------------------------------------------
# 3. DRAWING HELPERS
# ---------------------------------------------------------

def draw_text_with_bg(frame: np.ndarray,
                      text: str,
                      position: Tuple[int, int],
                      font_scale: float = 0.6,
                      color: Tuple[int, int, int] = (255, 255, 255),
                      bg_color: Tuple[int, int, int] = (0, 0, 0),
                      thickness: int = 1,
                      padding: int = 5) -> None:
    """
    Draw text on a frame with a filled background rectangle
    for readability.

    Args:
        frame:      OpenCV image (BGR).
        text:       String to render.
        position:   (x, y) bottom-left corner of the text.
        font_scale: Font size multiplier.
        color:      Text colour (BGR).
        bg_color:   Background rectangle colour (BGR).
        thickness:  Text stroke thickness.
        padding:    Pixels of padding around the text.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, font_scale, thickness)

    x, y = position
    # Background rectangle
    cv2.rectangle(frame,
                  (x - padding, y - text_h - padding),
                  (x + text_w + padding, y + baseline + padding),
                  bg_color, cv2.FILLED)
    # Text
    cv2.putText(frame, text, (x, y),
                font, font_scale, color, thickness, cv2.LINE_AA)


def draw_status_indicator(frame: np.ndarray,
                          label: str,
                          value: str,
                          position: Tuple[int, int],
                          is_good: bool = True) -> None:
    """
    Draw a coloured status label on the frame.
    Green if *is_good* is True, red otherwise.

    Args:
        frame:    OpenCV image (BGR).
        label:    Metric name  (e.g. "Posture").
        value:    Status text  (e.g. "Good").
        position: (x, y) on the frame.
        is_good:  True → green, False → red.
    """
    color = (0, 200, 0) if is_good else (0, 0, 230)
    bg = (0, 60, 0) if is_good else (0, 0, 60)
    draw_text_with_bg(frame, f"{label}: {value}",
                      position, 0.55, color, bg)


def draw_progress_bar(frame: np.ndarray,
                      position: Tuple[int, int],
                      width: int,
                      height: int,
                      progress: float,
                      color: Tuple[int, int, int] = (0, 200, 0),
                      bg_color: Tuple[int, int, int] = (50, 50, 50)
                      ) -> None:
    """
    Draw a horizontal progress bar on the frame.

    Args:
        frame:    OpenCV image (BGR).
        position: Top-left corner (x, y).
        width:    Total bar width in pixels.
        height:   Bar height in pixels.
        progress: Fill fraction 0.0 – 1.0.
        color:    Fill colour (BGR).
        bg_color: Background colour (BGR).
    """
    x, y = position
    progress = max(0.0, min(1.0, progress))

    # Background
    cv2.rectangle(frame, (x, y), (x + width, y + height),
                  bg_color, cv2.FILLED)
    # Fill
    fill_w = int(width * progress)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height),
                      color, cv2.FILLED)
    # Border
    cv2.rectangle(frame, (x, y), (x + width, y + height),
                  (200, 200, 200), 1)


# ---------------------------------------------------------
# 4. TIME / FORMAT UTILITIES
# ---------------------------------------------------------

def format_elapsed_time(seconds: float) -> str:
    """
    Convert elapsed seconds into MM:SS string.

    Args:
        seconds: Total seconds elapsed.

    Returns:
        Formatted string like "03:45".
    """
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins:02d}:{secs:02d}"


def format_countdown(remaining: float) -> str:
    """
    Convert remaining seconds into MM:SS string.

    Args:
        remaining: Seconds left on the timer.

    Returns:
        Formatted string; "00:00" if time is up.
    """
    if remaining <= 0:
        return "00:00"
    return format_elapsed_time(remaining)


def get_timestamp() -> str:
    """
    Return the current date-time as a readable string.

    Returns:
        e.g. "2026-06-27 02:18:00"
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_date_for_filename() -> str:
    """
    Return a filesystem-safe date-time string for filenames.

    Returns:
        e.g. "2026-06-27_02-18-00"
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ---------------------------------------------------------
# 5. METRIC ACCUMULATOR
# ---------------------------------------------------------

class MetricAccumulator:
    """
    A simple running-average tracker for a single metric.

    Usage:
        acc = MetricAccumulator()
        acc.update(True)   # good frame
        acc.update(False)  # bad frame
        print(acc.percentage())   # → 50.0
    """

    def __init__(self):
        """Initialise counters to zero."""
        self.total_frames: int = 0
        self.positive_frames: int = 0

    def update(self, is_positive: bool) -> None:
        """
        Record one observation.

        Args:
            is_positive: True if the frame meets the
                         "good" criterion for this metric.
        """
        self.total_frames += 1
        if is_positive:
            self.positive_frames += 1

    def percentage(self) -> float:
        """
        Return the positive-frame percentage (0–100).

        Returns:
            Percentage as a float; 0.0 if no frames recorded.
        """
        if self.total_frames == 0:
            return 0.0
        return (self.positive_frames / self.total_frames) * 100.0

    def reset(self) -> None:
        """Reset counters to zero for a new session."""
        self.total_frames = 0
        self.positive_frames = 0
