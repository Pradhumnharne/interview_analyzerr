# AI Interview Analyzer

> **Analyse your body language during mock interviews using real-time computer vision — no microphone, no speech analysis, no AI models.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green?style=for-the-badge&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red?style=for-the-badge&logo=streamlit)

---

## 📋 Project Overview

AI Interview Analyzer is an intermediate-level MCA final-year project that uses **only webcam video** to evaluate a candidate's body language during mock interviews.

**No ML models are trained.** All scoring is **rule-based** using mathematical formulas and thresholds.

### Tech Stack

| Library     | Purpose                        |
|-------------|--------------------------------|
| OpenCV      | Webcam capture & frame drawing |
| MediaPipe   | Face Mesh, Pose, Hands         |
| NumPy       | Numerical computations         |
| Streamlit   | Web dashboard UI               |
| Matplotlib  | Charts (radar, bar, gauge)     |
| Pandas      | Data tables                    |
| ReportLab   | PDF report generation          |
| SQLite      | Session history storage        |

---

## ✨ Features

| #  | Feature                | Description                                          |
|----|------------------------|------------------------------------------------------|
| 1  | Live Webcam Feed       | Real-time camera with MediaPipe landmark overlays    |
| 2  | Face Detection         | MediaPipe Face Mesh with 468+ landmarks              |
| 3  | Eye Contact Detection  | Iris-to-eye-corner gaze ratio analysis               |
| 4  | Head Pose Detection    | PnP-based yaw/pitch/roll estimation                  |
| 5  | Blink Counter          | Eye Aspect Ratio (EAR) method                        |
| 6  | Smile Detection        | Mouth-width-to-face-width ratio                      |
| 7  | Posture Detection      | Shoulder alignment + neck lean angle                 |
| 8  | Hand Movement          | Frame-to-frame wrist displacement tracking           |
| 9  | Session Timer          | Elapsed time + countdown with progress bar           |
| 10 | Confidence Score       | Weighted formula (eye 35%, posture 25%, etc.)        |
| 11 | Performance Dashboard  | Radar chart, bar chart, gauge, metric cards           |
| 12 | Suggestions            | Rule-based if-else feedback (no AI)                  |
| 13 | PDF Report             | Professional A4 report with charts and scores        |
| 14 | Session History        | SQLite storage with trend analysis                   |

---

## 📁 Project Structure

```
interview_analyzerr/
│
├── config.py                   # Central configuration & thresholds
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── vision/                     # Computer vision modules
│   ├── eye_contact.py          # Eye contact detector
│   ├── head_pose.py            # Head pose estimator (PnP)
│   ├── blink_detector.py       # Blink counter (EAR method)
│   ├── smile_detector.py       # Smile detector (mouth ratio)
│   ├── posture_detector.py     # Posture classifier
│   └── hand_detector.py        # Hand movement tracker
│
├── scoring/                    # Score calculation
│   └── score_engine.py         # Weighted confidence score + suggestions
│
├── dashboard/                  # Streamlit UI
│   └── app.py                  # Main application (5 pages)
│
├── database/                   # Data persistence
│   └── database.py             # SQLite CRUD operations
│
├── reports/                    # Report generation
│   ├── charts.py               # Matplotlib chart generator
│   └── report_generator.py     # ReportLab PDF builder
│
├── utils/                      # Shared utilities
│   └── helpers.py              # Geometry, drawing, formatting
│
└── assets/
    └── reports/                # Generated PDF reports
```

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd interview_analyzerr
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run dashboard/app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🎮 How to Use

1. **Home** — Overview of features. Click "Start Interview Session".
2. **Start Interview** — Set duration, click Start. The webcam opens with live overlays showing eye contact, head direction, posture, smile, blinks, and hand status.
3. **Dashboard** — After the session ends, view your radar chart, bar chart, confidence gauge, individual scores, and suggestions. Download the PDF report.
4. **Previous Reports** — Browse all past sessions in a table. View details, download PDFs, or track your score trend over time.
5. **Settings** — View and adjust camera index, session duration, and detection thresholds.

---

## 🎯 Confidence Score Formula

The confidence score is computed using a **weighted average** — no machine learning:

```
Confidence Score = (Eye Contact % × 0.35)
                 + (Posture %     × 0.25)
                 + (Smile %       × 0.15)
                 + (Head Stability × 0.15)
                 + (Hand Stability × 0.10)
```

| Score Range | Rating    |
|-------------|-----------|
| 80 – 100    | Excellent |
| 60 – 79     | Good      |
| 40 – 59     | Average   |
| 0 – 39      | Poor      |

---

## 🧠 Detection Methods

### Eye Contact
Uses **iris landmarks** (MediaPipe Face Mesh with `refine_landmarks=True`). Computes the horizontal and vertical gaze ratio — if the iris is centred in the eye socket, the user is looking at the camera.

### Head Pose
Uses **OpenCV solvePnP** with 6 canonical face landmarks mapped to a 3D face model. Extracts yaw, pitch, and roll angles to classify direction.

### Blink Detection
Implements the **Eye Aspect Ratio (EAR)** formula. When EAR drops below 0.21 for ≥2 consecutive frames, a blink is counted.

### Smile Detection
Computes the ratio of **mouth width to face width**. A wider ratio indicates a smile.

### Posture
Measures **shoulder tilt** (angle from horizontal) and **neck lean** (forward head angle). Both must be within thresholds for "Good Posture".

### Hand Movement
Tracks **wrist landmark displacement** between frames. Movements above a pixel threshold are counted, and a per-minute rate is computed.

---

## 📊 Generated Reports

Each session produces a **professional PDF** containing:
- Session date, time, and duration
- Confidence score with rating
- Individual metric table
- Radar chart, bar chart, and gauge chart
- Colour-coded suggestions

Reports are saved to `assets/reports/` and can be downloaded from the dashboard.

---

## ⚙️ Configuration

All thresholds and weights are centralised in `config.py`:

| Parameter                     | Default | Description                          |
|-------------------------------|---------|--------------------------------------|
| `EAR_THRESHOLD`               | 0.21    | Blink detection sensitivity          |
| `SMILE_WIDTH_RATIO_THRESHOLD` | 0.42    | Smile detection threshold            |
| `HEAD_YAW_THRESHOLD`          | 15°     | Left/right head turn limit           |
| `HEAD_PITCH_THRESHOLD`        | 10°     | Up/down head tilt limit              |
| `POSTURE_SHOULDER_ANGLE_THRESHOLD` | 10° | Shoulder tilt tolerance         |
| `HAND_MOVEMENT_PIXEL_THRESHOLD`   | 30px  | Min displacement for "movement" |
| `DEFAULT_SESSION_DURATION`    | 300s    | Default interview length             |

---

## 🛑 What This Project Does NOT Use

- ❌ TensorFlow / Keras
- ❌ DeepFace
- ❌ Whisper / Speech Recognition
- ❌ NLP / LLM APIs
- ❌ OpenAI / HuggingFace
- ❌ Sentence Transformers
- ❌ Audio Processing
- ❌ Microphone

---

## 📄 License

This project is developed as an academic MCA final-year project.

---

## 👤 Author

MCA Final Year Project — AI Interview Analyzer
