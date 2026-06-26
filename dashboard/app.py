"""
=============================================================
AI Interview Analyzer — Streamlit Dashboard
=============================================================
Main application entry point.

Pages:
  🏠 Home           — Welcome screen and quick stats
  🎥 Start Interview — Live webcam analysis session
  📊 Dashboard       — Post-session performance charts
  📂 Previous Reports — Browse & download past sessions
  ⚙  Settings        — Configure session duration & thresholds
=============================================================

Run with:
    streamlit run dashboard/app.py
=============================================================
"""

import sys
import os
import time

# ---------------------------------------------------------
# Ensure project root is on sys.path so that absolute
# imports (config, vision.*, etc.) work correctly.
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import pandas as pd

import config
from vision.eye_contact import EyeContactDetector
from vision.head_pose import HeadPoseEstimator
from vision.blink_detector import BlinkDetector
from vision.smile_detector import SmileDetector
from vision.posture_detector import PostureDetector
from vision.hand_detector import HandMovementDetector
from scoring.score_engine import ScoreEngine
from database.database import SessionDatabase
from reports.report_generator import ReportGenerator
from reports.charts import ChartGenerator
from utils.helpers import (
    format_elapsed_time, format_countdown,
    draw_text_with_bg, draw_status_indicator,
    draw_progress_bar, get_timestamp,
)


# =============================================================
# PAGE CONFIG  (must be the first Streamlit command)
# =============================================================
st.set_page_config(
    page_title="AI Interview Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================
# CUSTOM CSS
# =============================================================
def inject_css():
    """Inject custom CSS for a polished dark-themed UI."""
    st.markdown("""
    <style>
        /* ---- Global ---- */
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
            border-right: 1px solid #2a2a4a;
        }
        /* ---- Metric cards ---- */
        .metric-card {
            background: linear-gradient(135deg, #1e1e3f, #2a2a5a);
            border: 1px solid #3a3a6a;
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(79, 70, 229, 0.15);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 30px rgba(79, 70, 229, 0.3);
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #4F46E5, #10B981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #a0aec0;
            margin-top: 4px;
        }
        /* ---- Hero header ---- */
        .hero-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #4F46E5, #7C3AED, #10B981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0;
        }
        .hero-sub {
            color: #94a3b8;
            text-align: center;
            font-size: 1.1rem;
            margin-top: 5px;
        }
        /* ---- Suggestion cards ---- */
        .suggestion-good {
            background: linear-gradient(135deg, #064e3b, #065f46);
            border-left: 4px solid #10B981;
            border-radius: 8px;
            padding: 12px 16px;
            margin: 6px 0;
            color: #d1fae5;
        }
        .suggestion-bad {
            background: linear-gradient(135deg, #450a0a, #7f1d1d);
            border-left: 4px solid #EF4444;
            border-radius: 8px;
            padding: 12px 16px;
            margin: 6px 0;
            color: #fecaca;
        }
        /* ---- Status pill ---- */
        .status-pill {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .pill-green { background: #065f46; color: #6ee7b7; }
        .pill-red   { background: #7f1d1d; color: #fca5a5; }
        .pill-blue  { background: #1e3a5f; color: #93c5fd; }
        .pill-amber { background: #78350f; color: #fcd34d; }
    </style>
    """, unsafe_allow_html=True)


# =============================================================
# SESSION STATE INITIALISATION
# =============================================================
def init_session_state():
    """Set default values for Streamlit session state."""
    defaults = {
        "page": "Home",
        "is_running": False,
        "session_data": None,
        "duration": config.DEFAULT_SESSION_DURATION,
        "show_landmarks": True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =============================================================
# SIDEBAR NAVIGATION
# =============================================================
def render_sidebar():
    """Render the navigation sidebar."""
    with st.sidebar:
        st.markdown('<p class="hero-title" style="font-size:1.6rem">'
                    '🎯 Interview Analyzer</p>',
                    unsafe_allow_html=True)
        st.markdown("---")

        pages = {
            "🏠 Home": "Home",
            "🎥 Start Interview": "Interview",
            "📊 Dashboard": "Dashboard",
            "📂 Previous Reports": "Reports",
            "⚙️ Settings": "Settings",
        }

        for label, page_id in pages.items():
            if st.button(label, key=f"nav_{page_id}",
                         use_container_width=True):
                st.session_state["page"] = page_id

        st.markdown("---")
        db = SessionDatabase()
        count = db.get_session_count()
        st.caption(f"📁 {count} session(s) saved")
        st.caption(f"⏱ Default: {config.DEFAULT_SESSION_DURATION}s")


# =============================================================
# PAGE: HOME
# =============================================================
def page_home():
    """Welcome / landing page."""
    st.markdown('<p class="hero-title">AI Interview Analyzer</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">'
                'Analyse your body language during mock interviews '
                'using real-time computer vision.</p>',
                unsafe_allow_html=True)
    st.markdown("")

    cols = st.columns(3)
    features = [
        ("👁️", "Eye Contact", "Track gaze direction and eye contact %"),
        ("🧠", "Head Pose", "Detect head orientation in real time"),
        ("😊", "Smile Detection", "Measure smile frequency"),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        col.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:2.5rem">{icon}</div>'
            f'<div class="metric-value" style="font-size:1.3rem">{title}</div>'
            f'<div class="metric-label">{desc}</div></div>',
            unsafe_allow_html=True)

    st.markdown("")
    cols2 = st.columns(3)
    features2 = [
        ("🧍", "Posture Check", "Identify slouching vs good posture"),
        ("🖐️", "Hand Movement", "Flag excessive hand gestures"),
        ("📊", "Confidence Score", "Get a rule-based overall rating"),
    ]
    for col, (icon, title, desc) in zip(cols2, features2):
        col.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:2.5rem">{icon}</div>'
            f'<div class="metric-value" style="font-size:1.3rem">{title}</div>'
            f'<div class="metric-label">{desc}</div></div>',
            unsafe_allow_html=True)

    st.markdown("")
    st.markdown("")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("🚀  Start Interview Session",
                      use_container_width=True, type="primary"):
            st.session_state["page"] = "Interview"
            st.rerun()


# =============================================================
# PAGE: START INTERVIEW
# =============================================================
def _is_cloud_environment() -> bool:
    """Detect if running on Streamlit Cloud (no local webcam)."""
    import platform
    # Streamlit Cloud runs on Linux containers
    # Check for common cloud indicators
    return (os.environ.get("STREAMLIT_SHARING_MODE") is not None
            or os.environ.get("HOSTNAME", "").startswith("streamlit")
            or (platform.system() == "Linux"
                and os.path.exists("/mount/src")))


def page_interview():
    """Live webcam interview analysis page."""
    st.markdown('<p class="hero-title" style="font-size:2rem">'
                '🎥 Live Interview Session</p>',
                unsafe_allow_html=True)
    st.markdown("")

    # Cloud environment notice
    if _is_cloud_environment():
        st.warning(
            "⚠️ **Live webcam analysis requires running the app locally.**\n\n"
            "Streamlit Cloud does not have access to your camera. "
            "To use the live interview feature:\n\n"
            "```bash\n"
            "git clone https://github.com/Pradhumnharne/interview_analyzerr.git\n"
            "cd interview_analyzerr\n"
            "pip install -r requirements.txt\n"
            "streamlit run dashboard/app.py\n"
            "```\n\n"
            "You can still view the **Dashboard**, **Previous Reports**, "
            "and **Settings** pages on the cloud."
        )
        st.markdown("")

    # Controls
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        duration = st.number_input(
            "Session Duration (seconds)",
            min_value=10, max_value=1800,
            value=st.session_state["duration"], step=10)
        st.session_state["duration"] = duration
    with col_b:
        show_landmarks = st.checkbox(
            "Show Landmarks", value=st.session_state["show_landmarks"])
        st.session_state["show_landmarks"] = show_landmarks
    with col_c:
        st.markdown("")
        start_btn = st.button("▶️  Start", use_container_width=True,
                              type="primary")

    if start_btn:
        _run_interview(duration, show_landmarks)


def _run_interview(duration: int, show_landmarks: bool):
    """
    Core interview loop — captures webcam frames, runs all
    detectors, and shows live metrics.
    """
    # ----- Initialise MediaPipe models -----
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=config.FACE_MESH_MAX_FACES,
        refine_landmarks=config.FACE_MESH_REFINE_LANDMARKS,
        min_detection_confidence=config.FACE_MESH_MIN_DETECTION_CONF,
        min_tracking_confidence=config.FACE_MESH_MIN_TRACKING_CONF,
    )
    pose = mp_pose.Pose(
        min_detection_confidence=config.POSE_MIN_DETECTION_CONF,
        min_tracking_confidence=config.POSE_MIN_TRACKING_CONF,
    )
    hands = mp_hands.Hands(
        max_num_hands=config.HAND_MAX_HANDS,
        min_detection_confidence=config.HAND_MIN_DETECTION_CONF,
        min_tracking_confidence=config.HAND_MIN_TRACKING_CONF,
    )

    # ----- Initialise detectors -----
    eye_det = EyeContactDetector()
    head_det = HeadPoseEstimator()
    blink_det = BlinkDetector()
    smile_det = SmileDetector()
    posture_det = PostureDetector()
    hand_det = HandMovementDetector()

    # ----- Open webcam -----
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        st.error("❌ Could not access the webcam. "
                 "Please check your camera.")
        return

    # ----- UI placeholders -----
    frame_placeholder = st.empty()
    metrics_placeholder = st.empty()
    timer_placeholder = st.empty()
    stop_placeholder = st.empty()

    stop_btn = stop_placeholder.button(
        "⏹  Stop Session", type="secondary",
        use_container_width=True)

    start_time = time.time()

    # ----- Main loop -----
    try:
        while True:
            elapsed = time.time() - start_time
            remaining = duration - elapsed

            if remaining <= 0 or stop_btn:
                break

            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to read frame from webcam.")
                break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ------- MediaPipe processing -------
            face_results = face_mesh.process(rgb_frame)
            pose_results = pose.process(rgb_frame)
            hand_results = hands.process(rgb_frame)

            # ------- Run detectors -------
            eye_contact = False
            head_dir = "N/A"
            blinked = False
            smiling = False
            good_posture = True
            hand_moving = False

            if face_results.multi_face_landmarks:
                fl = face_results.multi_face_landmarks[0]
                eye_contact = eye_det.detect(fl, frame_w, frame_h)
                head_dir = head_det.detect(fl, frame_w, frame_h)
                blinked = blink_det.detect(fl, frame_w, frame_h)
                smiling = smile_det.detect(fl, frame_w, frame_h)

                # Draw face mesh landmarks
                if show_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, fl,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=(
                            mp_drawing_styles
                            .get_default_face_mesh_contours_style()
                        ),
                    )

            if pose_results.pose_landmarks:
                good_posture = posture_det.detect(
                    pose_results.pose_landmarks, frame_w, frame_h)

                if show_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, pose_results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                            color=(80, 110, 200), thickness=2,
                            circle_radius=2),
                        mp_drawing.DrawingSpec(
                            color=(80, 200, 120), thickness=2),
                    )

            hand_moving = hand_det.detect(
                hand_results.multi_hand_landmarks,
                frame_w, frame_h)

            if hand_results.multi_hand_landmarks and show_landmarks:
                for hlm in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hlm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                            color=(200, 80, 80), thickness=2,
                            circle_radius=2),
                        mp_drawing.DrawingSpec(
                            color=(200, 150, 80), thickness=2),
                    )

            # ------- Draw overlays -------
            y_offset = 30
            draw_status_indicator(
                frame, "Eye Contact",
                "Yes" if eye_contact else "No",
                (10, y_offset), eye_contact)
            y_offset += 30

            draw_status_indicator(
                frame, "Head", head_dir,
                (10, y_offset),
                head_dir == "Facing Front")
            y_offset += 30

            draw_status_indicator(
                frame, "Blinks",
                str(blink_det.get_blink_count()),
                (10, y_offset), True)
            y_offset += 30

            draw_status_indicator(
                frame, "Smile",
                "Yes" if smiling else "No",
                (10, y_offset), smiling)
            y_offset += 30

            draw_status_indicator(
                frame, "Posture",
                posture_det.get_status(),
                (10, y_offset), good_posture)
            y_offset += 30

            draw_status_indicator(
                frame, "Hands",
                "Moving" if hand_moving else "Stable",
                (10, y_offset), not hand_moving)

            # Timer overlay (top-right)
            timer_text = f"Time: {format_elapsed_time(elapsed)} / {format_countdown(float(duration))}"
            draw_text_with_bg(frame, timer_text,
                              (frame_w - 270, 30), 0.6,
                              (255, 255, 255), (40, 40, 40))

            # Progress bar
            progress = min(1.0, elapsed / duration)
            draw_progress_bar(frame, (frame_w - 270, 55),
                              250, 12, progress,
                              (79, 70, 229), (50, 50, 50))

            # ------- Display frame -------
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB",
                                    use_container_width=True)

            # ------- Live metrics sidebar -------
            with metrics_placeholder.container():
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("👁 Eye Contact",
                           f"{eye_det.get_percentage():.1f}%")
                mc2.metric("😊 Smile",
                           f"{smile_det.get_percentage():.1f}%")
                mc3.metric("🧍 Posture",
                           f"{posture_det.get_percentage():.1f}%")

                mc4, mc5, mc6 = st.columns(3)
                mc4.metric("🧠 Head Stability",
                           f"{head_det.get_stability_percentage():.1f}%")
                mc5.metric("🖐 Hand Stability",
                           f"{hand_det.get_stability_percentage():.1f}%")
                mc6.metric("👀 Blink Rate",
                           f"{blink_det.get_blink_rate():.1f}/min")

            timer_placeholder.progress(progress,
                                       text=f"⏱ {format_countdown(remaining)} remaining")

    finally:
        cap.release()
        face_mesh.close()
        pose.close()
        hands.close()

    # ----- Session complete — build summary -----
    total_elapsed = time.time() - start_time
    if total_elapsed < config.MIN_SESSION_DURATION:
        st.warning("⚠️ Session too short for a meaningful report.")
        return

    metrics = {
        "eye_contact":    eye_det.get_percentage(),
        "posture":        posture_det.get_percentage(),
        "smile":          smile_det.get_percentage(),
        "head_stability": head_det.get_stability_percentage(),
        "hand_stability": hand_det.get_stability_percentage(),
    }

    engine = ScoreEngine()
    session_data = engine.build_session_summary(
        metrics=metrics,
        blink_rate=blink_det.get_blink_rate(),
        blink_count=blink_det.get_blink_count(),
        duration_seconds=total_elapsed,
    )

    # Generate PDF
    report_gen = ReportGenerator()
    pdf_path = report_gen.generate(session_data)
    session_data["pdf_path"] = pdf_path

    # Save to database
    db = SessionDatabase()
    db.save_session(session_data)

    # Store in session state for dashboard
    st.session_state["session_data"] = session_data

    st.success("✅ Session complete! Navigate to **Dashboard** "
               "to view your results.")
    st.balloons()

    # Auto-navigate to dashboard
    st.session_state["page"] = "Dashboard"
    st.rerun()


# =============================================================
# PAGE: DASHBOARD
# =============================================================
def page_dashboard():
    """Post-session performance dashboard."""
    st.markdown('<p class="hero-title" style="font-size:2rem">'
                '📊 Performance Dashboard</p>',
                unsafe_allow_html=True)
    st.markdown("")

    data = st.session_state.get("session_data")

    if data is None:
        # Try loading the most recent session
        db = SessionDatabase()
        sessions = db.get_recent_sessions(1)
        if sessions:
            data = sessions[0]
            # Reconstruct metrics dict for chart generation
            if "metrics" not in data:
                data["metrics"] = {
                    "eye_contact":    data.get("eye_contact", 0),
                    "posture":        data.get("posture", 0),
                    "smile":          data.get("smile", 0),
                    "head_stability": data.get("head_stability", 0),
                    "hand_stability": data.get("hand_stability", 0),
                }
            # Reconstruct suggestions if needed
            if "suggestions" not in data:
                engine = ScoreEngine()
                data["suggestions"] = engine.generate_suggestions(
                    data["metrics"], data.get("blink_rate", 0))

    if data is None:
        st.info("ℹ️ No session data available. "
                "Complete an interview first!")
        return

    metrics = data.get("metrics", {})
    confidence = data.get("confidence_score", 0)
    label = data.get("score_label", "N/A")

    # ----- Confidence score hero -----
    engine = ScoreEngine()
    score_color = engine.get_score_color(confidence)

    st.markdown(
        f'<div class="metric-card" style="max-width:400px; margin:auto">'
        f'<div class="metric-label">CONFIDENCE SCORE</div>'
        f'<div class="metric-value" style="font-size:3rem">'
        f'{confidence:.1f}%</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True)
    st.markdown("")

    # ----- Metric cards -----
    cols = st.columns(5)
    metric_icons = {
        "eye_contact": "👁️",
        "posture": "🧍",
        "smile": "😊",
        "head_stability": "🧠",
        "hand_stability": "🖐️",
    }
    for col, (key, value) in zip(cols, metrics.items()):
        icon = metric_icons.get(key, "📈")
        label_text = key.replace("_", " ").title()
        col.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:1.8rem">{icon}</div>'
            f'<div class="metric-value">{value:.1f}%</div>'
            f'<div class="metric-label">{label_text}</div></div>',
            unsafe_allow_html=True)

    st.markdown("")

    # ----- Additional metrics -----
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("👀 Blink Count",
                 data.get("blink_count", 0))
    col_b.metric("💨 Blink Rate",
                 f"{data.get('blink_rate', 0):.1f} /min")
    col_c.metric("⏱ Duration",
                 f"{data.get('duration_seconds', 0):.0f}s")

    st.markdown("---")

    # ----- Charts -----
    chart_gen = ChartGenerator()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🕸️ Radar Chart")
        radar_path = chart_gen.create_radar_chart(
            metrics, "dashboard_radar.png")
        if os.path.exists(radar_path):
            st.image(radar_path, use_container_width=True)

    with col2:
        st.subheader("📊 Bar Chart")
        bar_path = chart_gen.create_bar_chart(
            metrics, "dashboard_bar.png")
        if os.path.exists(bar_path):
            st.image(bar_path, use_container_width=True)

    # Gauge chart
    st.markdown("")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.subheader("🎯 Confidence Gauge")
        gauge_path = chart_gen.create_gauge_chart(
            confidence, "Confidence", "dashboard_gauge.png")
        if os.path.exists(gauge_path):
            st.image(gauge_path, use_container_width=True)

    st.markdown("---")

    # ----- Suggestions -----
    st.subheader("💡 Suggestions & Feedback")
    suggestions = data.get("suggestions", [])
    for s in suggestions:
        status = s.get("status", "")
        msg = s.get("message", "")
        css_class = "suggestion-good" if status == "good" else "suggestion-bad"
        icon = "✅" if status == "good" else "⚠️"
        st.markdown(
            f'<div class="{css_class}">{icon} {msg}</div>',
            unsafe_allow_html=True)

    st.markdown("")

    # ----- PDF Download -----
    pdf_path = data.get("pdf_path", "")
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📥 Download PDF Report", f,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
                use_container_width=True)


# =============================================================
# PAGE: PREVIOUS REPORTS
# =============================================================
def page_reports():
    """Browse and view past interview sessions."""
    st.markdown('<p class="hero-title" style="font-size:2rem">'
                '📂 Previous Reports</p>',
                unsafe_allow_html=True)
    st.markdown("")

    db = SessionDatabase()
    sessions = db.get_all_sessions()

    if not sessions:
        st.info("ℹ️ No previous sessions found. "
                "Complete an interview to see reports here.")
        return

    # Summary table
    df = pd.DataFrame([{
        "ID": s["id"],
        "Date": s["timestamp"],
        "Duration (s)": s.get("duration_seconds", 0),
        "Confidence": f"{s.get('confidence_score', 0):.1f}%",
        "Rating": s.get("score_label", "N/A"),
    } for s in sessions])

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Session detail viewer
    session_ids = [s["id"] for s in sessions]
    selected_id = st.selectbox(
        "Select a session to view details",
        session_ids,
        format_func=lambda x: f"Session #{x}")

    if selected_id:
        session = db.get_session_by_id(selected_id)
        if session:
            _render_session_detail(session)

    # Trend chart
    if len(sessions) >= 2:
        st.markdown("---")
        st.subheader("📈 Score Trend")
        chart_gen = ChartGenerator()
        trend_path = chart_gen.create_trend_chart(
            sessions, "trend_chart.png")
        if trend_path and os.path.exists(trend_path):
            st.image(trend_path, use_container_width=True)

    # Delete session
    st.markdown("---")
    del_id = st.number_input("Delete Session (enter ID)",
                             min_value=0, step=1, value=0)
    if del_id > 0:
        if st.button("🗑️ Delete", type="secondary"):
            if db.delete_session(del_id):
                st.success(f"Session #{del_id} deleted.")
                st.rerun()
            else:
                st.error("Session not found.")


def _render_session_detail(session: dict):
    """Render a detailed view of a single session."""
    cols = st.columns(5)
    metric_keys = ["eye_contact", "posture", "smile",
                   "head_stability", "hand_stability"]
    icons = ["👁️", "🧍", "😊", "🧠", "🖐️"]
    for col, key, icon in zip(cols, metric_keys, icons):
        val = session.get(key, 0)
        label = key.replace("_", " ").title()
        col.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:1.5rem">{icon}</div>'
            f'<div class="metric-value">{val:.1f}%</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True)

    # Suggestions
    suggestions = session.get("suggestions", [])
    if suggestions:
        st.markdown("**Feedback:**")
        for s in suggestions:
            status = s.get("status", "")
            msg = s.get("message", "")
            css = "suggestion-good" if status == "good" else "suggestion-bad"
            ico = "✅" if status == "good" else "⚠️"
            st.markdown(
                f'<div class="{css}">{ico} {msg}</div>',
                unsafe_allow_html=True)

    # PDF download
    pdf_path = session.get("pdf_path", "")
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📥 Download PDF",
                f, file_name=os.path.basename(pdf_path),
                mime="application/pdf")


# =============================================================
# PAGE: SETTINGS
# =============================================================
def page_settings():
    """Configuration page."""
    st.markdown('<p class="hero-title" style="font-size:2rem">'
                '⚙️ Settings</p>',
                unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎥 Camera Settings")
        st.number_input("Camera Index",
                        min_value=0, max_value=10,
                        value=config.CAMERA_INDEX,
                        key="setting_cam_idx",
                        help="0 = default webcam")
        st.number_input("Frame Width",
                        value=config.FRAME_WIDTH,
                        key="setting_fw")
        st.number_input("Frame Height",
                        value=config.FRAME_HEIGHT,
                        key="setting_fh")

    with col2:
        st.subheader("⏱ Session Settings")
        new_dur = st.number_input(
            "Default Duration (seconds)",
            min_value=10, max_value=1800,
            value=st.session_state["duration"],
            step=10, key="setting_dur")
        st.session_state["duration"] = new_dur

        st.subheader("🎯 Confidence Weights")
        st.json(config.CONFIDENCE_WEIGHTS)

    st.markdown("---")
    st.subheader("📂 Paths")
    st.code(f"Reports:  {config.REPORTS_DIR}")
    st.code(f"Database: {config.DATABASE_PATH}")

    st.markdown("---")
    st.subheader("🔍 Detection Thresholds")
    thresh_df = pd.DataFrame({
        "Parameter": [
            "EAR Threshold (blink)",
            "Smile Width Ratio",
            "Head Yaw Threshold (°)",
            "Head Pitch Threshold (°)",
            "Shoulder Angle Threshold (°)",
            "Hand Movement Pixel Threshold",
        ],
        "Value": [
            config.EAR_THRESHOLD,
            config.SMILE_WIDTH_RATIO_THRESHOLD,
            config.HEAD_YAW_THRESHOLD,
            config.HEAD_PITCH_THRESHOLD,
            config.POSTURE_SHOULDER_ANGLE_THRESHOLD,
            config.HAND_MOVEMENT_PIXEL_THRESHOLD,
        ]
    })
    st.dataframe(thresh_df, use_container_width=True,
                 hide_index=True)


# =============================================================
# MAIN ROUTER
# =============================================================
def main():
    """Application entry point and page router."""
    inject_css()
    init_session_state()
    render_sidebar()

    page = st.session_state["page"]

    if page == "Home":
        page_home()
    elif page == "Interview":
        page_interview()
    elif page == "Dashboard":
        page_dashboard()
    elif page == "Reports":
        page_reports()
    elif page == "Settings":
        page_settings()
    else:
        page_home()


if __name__ == "__main__":
    main()
