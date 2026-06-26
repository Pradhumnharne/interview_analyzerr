"""
=============================================================
AI Interview Analyzer — Hand Movement Detector
=============================================================
Tracks hand landmark positions across frames and detects
excessive hand movement using MediaPipe Hands.

Approach:
  1. Record the wrist landmark position each frame.
  2. Compute frame-to-frame displacement (pixels).
  3. If displacement exceeds a threshold → "movement".
  4. Accumulate movements and compute a per-minute rate.
  5. Flag as "excessive" if rate is too high.
=============================================================
"""

import time
from typing import Optional, Tuple, List

from utils.helpers import euclidean_distance, MetricAccumulator
import config


# MediaPipe Hands wrist landmark index
WRIST = 0


class HandMovementDetector:
    """
    Detects and quantifies hand movement frequency using
    frame-to-frame wrist displacement.
    """

    def __init__(self):
        """Initialise movement tracker."""
        self.pixel_threshold = config.HAND_MOVEMENT_PIXEL_THRESHOLD
        self.excessive_rate = config.HAND_EXCESSIVE_MOVEMENT_RATE
        self.accumulator = MetricAccumulator()

        # Previous wrist positions: {hand_index: (x, y)}
        self._prev_positions: dict = {}
        self.movement_count: int = 0
        self._start_time: float = time.time()

        # Latest frame state
        self.is_moving: bool = False
        self.hands_detected: int = 0

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def detect(self, hand_landmarks_list,
               frame_w: int, frame_h: int) -> bool:
        """
        Process one frame of hand landmarks.

        Args:
            hand_landmarks_list: List of hand landmarks from
                MediaPipe (results.multi_hand_landmarks).
                Can be None if no hands detected.
            frame_w: Frame width in pixels.
            frame_h: Frame height in pixels.

        Returns:
            True if significant hand movement was detected
            on this frame.
        """
        if hand_landmarks_list is None:
            self.is_moving = False
            self.hands_detected = 0
            # No hands visible → stable (good)
            self.accumulator.update(True)
            return False

        self.hands_detected = len(hand_landmarks_list)
        frame_movement = False

        for idx, hand_landmarks in enumerate(hand_landmarks_list):
            wrist = hand_landmarks.landmark[WRIST]
            current_pos = (wrist.x * frame_w, wrist.y * frame_h)

            if idx in self._prev_positions:
                displacement = euclidean_distance(
                    self._prev_positions[idx], current_pos)

                if displacement > self.pixel_threshold:
                    frame_movement = True
                    self.movement_count += 1

            self._prev_positions[idx] = current_pos

        # Clean up stale hand entries if fewer hands now
        stale_keys = [k for k in self._prev_positions
                      if k >= self.hands_detected]
        for k in stale_keys:
            del self._prev_positions[k]

        self.is_moving = frame_movement
        # Stable frame = no excessive movement
        self.accumulator.update(not frame_movement)
        return frame_movement

    def get_movement_rate(self) -> float:
        """
        Calculate hand movements per minute.

        Returns:
            Movements per minute (float).
        """
        elapsed = time.time() - self._start_time
        if elapsed < 1.0:
            return 0.0
        return (self.movement_count / elapsed) * 60.0

    def get_stability_percentage(self) -> float:
        """
        Return the percentage of frames with stable
        (non-excessive) hand movement.
        """
        return self.accumulator.percentage()

    def is_excessive(self) -> bool:
        """
        Check if current movement rate exceeds the
        'excessive' threshold.

        Returns:
            True if hand movement is excessive.
        """
        return self.get_movement_rate() > self.excessive_rate

    def get_movement_count(self) -> int:
        """Return total movement count for the session."""
        return self.movement_count

    def reset(self) -> None:
        """Reset all state for a new session."""
        self._prev_positions.clear()
        self.movement_count = 0
        self._start_time = time.time()
        self.is_moving = False
        self.hands_detected = 0
        self.accumulator.reset()
