"""
=============================================================
AI Interview Analyzer — Chart Generator
=============================================================
Creates Matplotlib charts for the performance dashboard
and PDF report:
  • Radar chart  — multi-metric overview
  • Bar chart    — individual metric comparison
  • Gauge chart  — confidence score dial
  • Trend chart  — historical score progression

All charts are returned as file paths (saved as PNG) or
as Matplotlib Figure objects for direct embedding in
Streamlit.
=============================================================
"""

import os
import math
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config


class ChartGenerator:
    """
    Generates publication-quality Matplotlib charts for
    interview session metrics.
    """

    def __init__(self):
        """Set up chart styling defaults."""
        self.colors = config.CHART_COLORS
        self.output_dir = config.REPORTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        # Apply a dark style globally
        plt.style.use("dark_background")

    # ---------------------------------------------------------
    # 1. RADAR CHART
    # ---------------------------------------------------------

    def create_radar_chart(self,
                           metrics: Dict[str, float],
                           filename: Optional[str] = None
                           ) -> str:
        """
        Create a radar (spider) chart showing all metrics.

        Args:
            metrics: Dict of metric_name → percentage (0–100).
            filename: Output PNG filename (optional).

        Returns:
            Absolute path to the saved PNG file.
        """
        labels = list(metrics.keys())
        values = list(metrics.values())
        num_vars = len(labels)

        # Compute angles for each axis
        angles = [n / float(num_vars) * 2 * math.pi
                  for n in range(num_vars)]
        values += values[:1]      # Close the polygon
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw=dict(polar=True))
        fig.patch.set_facecolor(self.colors["bg_dark"])
        ax.set_facecolor(self.colors["bg_dark"])

        # Draw the filled radar
        ax.plot(angles, values, "o-", linewidth=2,
                color=self.colors["primary"])
        ax.fill(angles, values, alpha=0.25,
                color=self.colors["primary"])

        # Axis labels
        display_labels = [self._format_label(l) for l in labels]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(display_labels, size=10,
                           color=self.colors["text"])

        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"],
                           size=8, color="#888888")
        ax.set_title("Performance Overview",
                     size=14, color=self.colors["text"],
                     pad=20, fontweight="bold")

        # Grid styling
        ax.spines["polar"].set_color("#444444")
        ax.grid(color="#444444", linestyle="--", linewidth=0.5)

        path = self._save(fig, filename or "radar_chart.png")
        plt.close(fig)
        return path

    # ---------------------------------------------------------
    # 2. BAR CHART
    # ---------------------------------------------------------

    def create_bar_chart(self,
                         metrics: Dict[str, float],
                         filename: Optional[str] = None
                         ) -> str:
        """
        Create a horizontal bar chart for metric comparison.

        Args:
            metrics: Dict of metric_name → percentage (0–100).
            filename: Output PNG filename (optional).

        Returns:
            Absolute path to the saved PNG file.
        """
        labels = [self._format_label(k) for k in metrics.keys()]
        values = list(metrics.values())

        # Colour each bar by score level
        bar_colors = [self._score_color(v) for v in values]

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(self.colors["bg_dark"])
        ax.set_facecolor(self.colors["bg_dark"])

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=bar_colors,
                       edgecolor="#333333", height=0.6)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color=self.colors["text"],
                           fontsize=11)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Score (%)", color=self.colors["text"],
                      fontsize=11)
        ax.set_title("Metric Breakdown",
                     color=self.colors["text"],
                     fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", colors=self.colors["text"])

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=10,
                    color=self.colors["text"])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#555555")
        ax.spines["left"].set_color("#555555")
        ax.grid(axis="x", color="#333333", linestyle="--",
                linewidth=0.5)

        plt.tight_layout()
        path = self._save(fig, filename or "bar_chart.png")
        plt.close(fig)
        return path

    # ---------------------------------------------------------
    # 3. GAUGE CHART (confidence score)
    # ---------------------------------------------------------

    def create_gauge_chart(self,
                           score: float,
                           label: str = "Confidence",
                           filename: Optional[str] = None
                           ) -> str:
        """
        Create a semi-circular gauge chart for the
        confidence score.

        Args:
            score:    Score value (0–100).
            label:    Label below the score.
            filename: Output PNG filename (optional).

        Returns:
            Absolute path to the saved PNG file.
        """
        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor(self.colors["bg_dark"])
        ax.set_facecolor(self.colors["bg_dark"])

        # Background arc (grey)
        theta_bg = np.linspace(math.pi, 0, 100)
        r = 1.0
        ax.plot(r * np.cos(theta_bg), r * np.sin(theta_bg),
                linewidth=20, color="#333333", solid_capstyle="round")

        # Foreground arc (coloured by score)
        fill_count = max(1, int(score))
        theta_fg = np.linspace(math.pi,
                               math.pi - (score / 100) * math.pi,
                               fill_count)
        color = self._score_color(score)
        ax.plot(r * np.cos(theta_fg), r * np.sin(theta_fg),
                linewidth=20, color=color, solid_capstyle="round")

        # Score text in the centre
        ax.text(0, 0.2, f"{score:.0f}%",
                ha="center", va="center",
                fontsize=28, fontweight="bold",
                color=self.colors["text"])
        ax.text(0, -0.1, label,
                ha="center", va="center",
                fontsize=12, color="#888888")

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.4, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")

        plt.tight_layout()
        path = self._save(fig, filename or "gauge_chart.png")
        plt.close(fig)
        return path

    # ---------------------------------------------------------
    # 4. TREND CHART (historical sessions)
    # ---------------------------------------------------------

    def create_trend_chart(self,
                           sessions: List[Dict],
                           filename: Optional[str] = None
                           ) -> str:
        """
        Plot confidence score trend over past sessions.

        Args:
            sessions: List of session dicts (must contain
                      'confidence_score' and 'timestamp').
            filename: Output PNG filename (optional).

        Returns:
            Absolute path to the saved PNG file.
        """
        if not sessions:
            return ""

        # Sort chronologically
        sorted_sessions = sorted(sessions,
                                 key=lambda s: s.get("timestamp", ""))
        scores = [s.get("confidence_score", 0)
                  for s in sorted_sessions]
        labels = [s.get("timestamp", "")[-8:-3]  # HH:MM
                  for s in sorted_sessions]

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor(self.colors["bg_dark"])
        ax.set_facecolor(self.colors["bg_dark"])

        ax.plot(labels, scores, "o-", color=self.colors["primary"],
                linewidth=2, markersize=8,
                markerfacecolor=self.colors["secondary"])
        ax.fill_between(range(len(scores)), scores,
                        alpha=0.15, color=self.colors["primary"])

        ax.set_ylim(0, 105)
        ax.set_ylabel("Confidence Score (%)",
                      color=self.colors["text"], fontsize=11)
        ax.set_xlabel("Session", color=self.colors["text"],
                      fontsize=11)
        ax.set_title("Score Trend",
                     color=self.colors["text"],
                     fontsize=14, fontweight="bold")
        ax.tick_params(colors=self.colors["text"])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#555555")
        ax.spines["left"].set_color("#555555")
        ax.grid(color="#333333", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        path = self._save(fig, filename or "trend_chart.png")
        plt.close(fig)
        return path

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _save(self, fig, filename: str) -> str:
        """Save figure and return the absolute path."""
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        return path

    @staticmethod
    def _format_label(key: str) -> str:
        """Convert 'eye_contact' → 'Eye Contact'."""
        return key.replace("_", " ").title()

    def _score_color(self, value: float) -> str:
        """Return a hex colour based on score value."""
        thresholds = config.SCORE_THRESHOLDS
        if value >= thresholds["excellent"]:
            return self.colors["secondary"]
        elif value >= thresholds["good"]:
            return self.colors["info"]
        elif value >= thresholds["average"]:
            return self.colors["accent"]
        return self.colors["danger"]
