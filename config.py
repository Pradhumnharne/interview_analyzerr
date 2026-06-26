"""
=============================================================
AI Interview Analyzer - Configuration Module
=============================================================
Central configuration file for all project-wide constants,
thresholds, weights, and default settings.

All tunable parameters are defined here so that no magic
numbers are scattered across the codebase.
=============================================================
"""

import os

# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------
# Base directory of the project (resolves to the folder
# containing this config.py file).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder where generated PDF reports are saved.
REPORTS_DIR = os.path.join(BASE_DIR, "assets", "reports")

# SQLite database file path.
DATABASE_PATH = os.path.join(BASE_DIR, "database", "interview_sessions.db")


# ---------------------------------------------------------
# 2. CAMERA SETTINGS
# ---------------------------------------------------------
# Default camera index (0 = built-in webcam).
CAMERA_INDEX = 0

# Resolution of the captured video frame.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Target frames per second for processing.
TARGET_FPS = 30


# ---------------------------------------------------------
# 3. MEDIAPIPE MODEL SETTINGS
# ---------------------------------------------------------
# Face Mesh configuration.
FACE_MESH_MAX_FACES = 1               # Track only one face
FACE_MESH_REFINE_LANDMARKS = True     # Enable iris landmarks
FACE_MESH_MIN_DETECTION_CONF = 0.5    # Detection confidence
FACE_MESH_MIN_TRACKING_CONF = 0.5     # Tracking confidence

# Pose estimation configuration.
POSE_MIN_DETECTION_CONF = 0.5
POSE_MIN_TRACKING_CONF = 0.5

# Hand detection configuration.
HAND_MAX_HANDS = 2
HAND_MIN_DETECTION_CONF = 0.5
HAND_MIN_TRACKING_CONF = 0.5


# ---------------------------------------------------------
# 4. EYE CONTACT THRESHOLDS
# ---------------------------------------------------------
# Horizontal and vertical gaze ratios that define the
# "looking at camera" zone.  Values closer to 0.5 mean
# the iris is centred in the eye.
EYE_CONTACT_HORIZONTAL_THRESHOLD = 0.35   # Max deviation from centre (left/right)
EYE_CONTACT_VERTICAL_THRESHOLD = 0.30     # Max deviation from centre (up/down)


# ---------------------------------------------------------
# 5. HEAD POSE THRESHOLDS (in degrees)
# ---------------------------------------------------------
HEAD_YAW_THRESHOLD = 15       # Beyond ±15° → Looking Left / Right
HEAD_PITCH_THRESHOLD = 10     # Beyond ±10° → Looking Up / Down
HEAD_ROLL_THRESHOLD = 15      # Beyond ±15° → Head tilted


# ---------------------------------------------------------
# 6. BLINK DETECTION THRESHOLDS
# ---------------------------------------------------------
# Eye Aspect Ratio (EAR) below this value is a blink.
EAR_THRESHOLD = 0.21

# Minimum consecutive frames with EAR < threshold to
# count as a single blink (avoids false positives).
EAR_CONSEC_FRAMES = 2

# Normal blink rate: 15-20 blinks per minute.
NORMAL_BLINK_RATE_MIN = 15
NORMAL_BLINK_RATE_MAX = 20


# ---------------------------------------------------------
# 7. SMILE DETECTION THRESHOLDS
# ---------------------------------------------------------
# Mouth Aspect Ratio (MAR) above this value → smiling.
SMILE_MAR_THRESHOLD = 0.45

# Ratio of mouth width to face width above this → smile.
SMILE_WIDTH_RATIO_THRESHOLD = 0.42


# ---------------------------------------------------------
# 8. POSTURE DETECTION THRESHOLDS
# ---------------------------------------------------------
# Maximum shoulder angle deviation from horizontal (degrees).
POSTURE_SHOULDER_ANGLE_THRESHOLD = 10

# Minimum acceptable neck inclination angle (degrees).
# If the neck leans forward beyond this → slouching.
POSTURE_NECK_ANGLE_THRESHOLD = 25


# ---------------------------------------------------------
# 9. HAND MOVEMENT THRESHOLDS
# ---------------------------------------------------------
# Pixel displacement per frame above this → "movement".
HAND_MOVEMENT_PIXEL_THRESHOLD = 30

# Movements per minute above this → "excessive".
HAND_EXCESSIVE_MOVEMENT_RATE = 40


# ---------------------------------------------------------
# 10. CONFIDENCE SCORE WEIGHTS
# ---------------------------------------------------------
# Rule-based confidence score weights (must sum to 1.0).
CONFIDENCE_WEIGHTS = {
    "eye_contact":    0.35,
    "posture":        0.25,
    "smile":          0.15,
    "head_stability": 0.15,
    "hand_stability": 0.10,
}


# ---------------------------------------------------------
# 11. SESSION / TIMER DEFAULTS
# ---------------------------------------------------------
# Default interview duration in seconds (5 minutes).
DEFAULT_SESSION_DURATION = 300

# Minimum session length to generate a report (seconds).
MIN_SESSION_DURATION = 10


# ---------------------------------------------------------
# 12. DASHBOARD / UI SETTINGS
# ---------------------------------------------------------
# Colour palette for charts and the Streamlit theme.
CHART_COLORS = {
    "primary":   "#4F46E5",   # Indigo
    "secondary": "#10B981",   # Emerald
    "accent":    "#F59E0B",   # Amber
    "danger":    "#EF4444",   # Red
    "info":      "#3B82F6",   # Blue
    "bg_dark":   "#1E1E2E",   # Dark background
    "text":      "#E2E8F0",   # Light text
}

# Score colour coding thresholds.
SCORE_THRESHOLDS = {
    "excellent": 80,   # ≥ 80 → green
    "good":      60,   # ≥ 60 → blue
    "average":   40,   # ≥ 40 → amber
    "poor":       0,   # <  40 → red
}


# ---------------------------------------------------------
# 13. PDF REPORT SETTINGS
# ---------------------------------------------------------
PDF_TITLE = "AI Interview Analyzer — Session Report"
PDF_AUTHOR = "AI Interview Analyzer"
PDF_PAGE_SIZE = "A4"


# ---------------------------------------------------------
# 14. SUGGESTION RULES
# ---------------------------------------------------------
# If a metric falls below its threshold, the corresponding
# suggestion text is shown to the user.
SUGGESTION_RULES = {
    "eye_contact": {
        "threshold": 60,
        "low_msg":  "You looked away from the camera frequently. Try to maintain steady eye contact.",
        "high_msg": "Great eye contact! You appeared confident and engaged.",
    },
    "posture": {
        "threshold": 60,
        "low_msg":  "Your posture needs improvement. Sit up straight with shoulders back.",
        "high_msg": "Good posture throughout the session. Well done!",
    },
    "smile": {
        "threshold": 30,
        "low_msg":  "Try to smile occasionally to appear friendly and approachable.",
        "high_msg": "Nice! Your smile made you look warm and confident.",
    },
    "head_stability": {
        "threshold": 60,
        "low_msg":  "Your head moved around a lot. Keep your head steady and face the camera.",
        "high_msg": "Your head was stable, showing composure.",
    },
    "hand_stability": {
        "threshold": 50,
        "low_msg":  "Reduce unnecessary hand movements. Keep gestures purposeful.",
        "high_msg": "Good hand control. Your gestures were measured.",
    },
    "blink_rate": {
        "threshold_low":  10,
        "threshold_high": 30,
        "low_msg":   "You blinked very infrequently — this can seem intense. Blink naturally.",
        "high_msg":  "You blinked excessively, which may signal nervousness. Try to relax.",
        "normal_msg": "Your blink rate was within the normal range.",
    },
}
