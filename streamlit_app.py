"""
=============================================================
AI Interview Analyzer — Streamlit Cloud Entry Point
=============================================================
This file serves as the root-level entry point for
Streamlit Cloud deployment. It simply imports and runs
the main dashboard application.
=============================================================
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dashboard.app import main

if __name__ == "__main__":
    main()
