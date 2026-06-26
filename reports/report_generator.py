"""
=============================================================
AI Interview Analyzer — PDF Report Generator
=============================================================
Generates a professional multi-page PDF report using
ReportLab, containing:
  • Session summary (date, duration, confidence score)
  • Individual metric scores
  • Radar and bar chart images
  • Rule-based suggestions
=============================================================
"""

import os
from datetime import datetime
from typing import Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

import config
from reports.charts import ChartGenerator
from utils.helpers import get_date_for_filename


class ReportGenerator:
    """
    Builds a professional PDF report for one interview
    session using ReportLab.
    """

    def __init__(self):
        """Initialise the chart generator and styles."""
        self.chart_gen = ChartGenerator()
        self.output_dir = config.REPORTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        # Base stylesheet
        self.styles = getSampleStyleSheet()
        self._register_custom_styles()

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def generate(self, session_data: Dict) -> str:
        """
        Generate a complete PDF report.

        Args:
            session_data: Dictionary returned by
                ScoreEngine.build_session_summary().

        Returns:
            Absolute path to the generated PDF file.
        """
        # File path
        timestamp = get_date_for_filename()
        filename = f"interview_report_{timestamp}.pdf"
        filepath = os.path.join(self.output_dir, filename)

        # Build the document
        doc = SimpleDocTemplate(
            filepath, pagesize=A4,
            topMargin=20 * mm, bottomMargin=20 * mm,
            leftMargin=20 * mm, rightMargin=20 * mm,
            title=config.PDF_TITLE,
            author=config.PDF_AUTHOR,
        )

        # Assemble the story (list of flowables)
        story = []
        story += self._build_header()
        story += self._build_session_info(session_data)
        story += self._build_scores_table(session_data)
        story += self._build_charts(session_data)
        story += self._build_suggestions(session_data)
        story += self._build_footer()

        doc.build(story)
        return filepath

    # ---------------------------------------------------------
    # Section builders
    # ---------------------------------------------------------

    def _build_header(self) -> List:
        """Title and subtitle."""
        elements = []
        elements.append(
            Paragraph("AI Interview Analyzer",
                      self.styles["CustomTitle"]))
        elements.append(Spacer(1, 4 * mm))
        elements.append(
            Paragraph("Session Performance Report",
                      self.styles["CustomSubtitle"]))
        elements.append(Spacer(1, 3 * mm))
        elements.append(HRFlowable(
            width="100%", thickness=1,
            color=colors.HexColor("#4F46E5")))
        elements.append(Spacer(1, 6 * mm))
        return elements

    def _build_session_info(self, data: Dict) -> List:
        """Date, duration, confidence score summary."""
        elements = []
        now = datetime.now().strftime("%B %d, %Y  •  %I:%M %p")
        duration_min = data.get("duration_seconds", 0) / 60.0
        score = data.get("confidence_score", 0)
        label = data.get("score_label", "N/A")

        info_text = (
            f"<b>Date:</b> {now}<br/>"
            f"<b>Duration:</b> {duration_min:.1f} minutes<br/>"
            f"<b>Confidence Score:</b> {score:.1f}% — {label}"
        )
        elements.append(
            Paragraph(info_text, self.styles["CustomBody"]))
        elements.append(Spacer(1, 8 * mm))
        return elements

    def _build_scores_table(self, data: Dict) -> List:
        """Table of individual metric scores."""
        elements = []
        elements.append(
            Paragraph("Metric Scores", self.styles["Heading2"]))
        elements.append(Spacer(1, 3 * mm))

        metrics = data.get("metrics", {})
        table_data = [["Metric", "Score (%)", "Rating"]]

        for key, value in metrics.items():
            label = key.replace("_", " ").title()
            rating = self._get_rating(value)
            table_data.append([label, f"{value:.1f}", rating])

        # Add blink rate row
        blink_rate = data.get("blink_rate", 0)
        table_data.append([
            "Blink Rate", f"{blink_rate:.1f} /min",
            "Normal" if 10 <= blink_rate <= 30 else "Abnormal"
        ])

        table = Table(table_data, colWidths=[55 * mm, 35 * mm, 35 * mm])
        table.setStyle(TableStyle([
            # Header row
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#4F46E5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 11),
            # Body rows
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 10),
            ("TEXTCOLOR", (0, 1), (-1, -1),
             colors.HexColor("#333333")),
            # Alternating row colours
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#F8F9FA"),
              colors.HexColor("#E9ECEF")]),
            # Grid
            ("GRID", (0, 0), (-1, -1), 0.5,
             colors.HexColor("#CCCCCC")),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 10 * mm))
        return elements

    def _build_charts(self, data: Dict) -> List:
        """Generate and embed radar and bar chart images."""
        elements = []
        elements.append(
            Paragraph("Visual Analysis", self.styles["Heading2"]))
        elements.append(Spacer(1, 4 * mm))

        metrics = data.get("metrics", {})

        # Radar chart
        radar_path = self.chart_gen.create_radar_chart(
            metrics, "report_radar.png")
        if os.path.exists(radar_path):
            elements.append(
                Image(radar_path, width=4.5 * inch,
                      height=4.5 * inch))
            elements.append(Spacer(1, 6 * mm))

        # Bar chart
        bar_path = self.chart_gen.create_bar_chart(
            metrics, "report_bar.png")
        if os.path.exists(bar_path):
            elements.append(
                Image(bar_path, width=5.5 * inch,
                      height=2.8 * inch))
            elements.append(Spacer(1, 6 * mm))

        # Gauge chart
        score = data.get("confidence_score", 0)
        gauge_path = self.chart_gen.create_gauge_chart(
            score, "Confidence", "report_gauge.png")
        if os.path.exists(gauge_path):
            elements.append(
                Image(gauge_path, width=3 * inch,
                      height=2.2 * inch))
            elements.append(Spacer(1, 8 * mm))

        return elements

    def _build_suggestions(self, data: Dict) -> List:
        """Render the suggestions section."""
        elements = []
        elements.append(
            Paragraph("Suggestions & Feedback",
                      self.styles["Heading2"]))
        elements.append(Spacer(1, 3 * mm))

        suggestions = data.get("suggestions", [])

        if not suggestions:
            elements.append(
                Paragraph("No suggestions available.",
                          self.styles["CustomBody"]))
        else:
            for s in suggestions:
                status = s.get("status", "")
                icon = "✓" if status == "good" else "⚠"
                msg = s.get("message", "")
                color_hex = ("#10B981" if status == "good"
                             else "#EF4444")
                text = (f'<font color="{color_hex}">'
                        f'{icon}</font>  {msg}')
                elements.append(
                    Paragraph(text, self.styles["CustomBody"]))
                elements.append(Spacer(1, 2 * mm))

        elements.append(Spacer(1, 8 * mm))
        return elements

    def _build_footer(self) -> List:
        """Report footer with timestamp."""
        elements = []
        elements.append(HRFlowable(
            width="100%", thickness=0.5,
            color=colors.HexColor("#CCCCCC")))
        elements.append(Spacer(1, 3 * mm))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(
            Paragraph(
                f"Generated by AI Interview Analyzer  •  {now}",
                self.styles["CustomFooter"]))
        return elements

    # ---------------------------------------------------------
    # Style registration
    # ---------------------------------------------------------

    def _register_custom_styles(self) -> None:
        """Add custom paragraph styles for the report."""
        self.styles.add(ParagraphStyle(
            name="CustomTitle",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=colors.HexColor("#4F46E5"),
            alignment=TA_CENTER,
            spaceAfter=4 * mm,
        ))
        self.styles.add(ParagraphStyle(
            name="CustomSubtitle",
            fontName="Helvetica",
            fontSize=13,
            textColor=colors.HexColor("#666666"),
            alignment=TA_CENTER,
        ))
        self.styles.add(ParagraphStyle(
            name="CustomBody",
            fontName="Helvetica",
            fontSize=11,
            textColor=colors.HexColor("#333333"),
            leading=16,
        ))
        self.styles.add(ParagraphStyle(
            name="CustomFooter",
            fontName="Helvetica",
            fontSize=9,
            textColor=colors.HexColor("#999999"),
            alignment=TA_CENTER,
        ))

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _get_rating(value: float) -> str:
        """Map a 0–100 score to a text rating."""
        if value >= 80:
            return "Excellent"
        elif value >= 60:
            return "Good"
        elif value >= 40:
            return "Average"
        return "Needs Work"
