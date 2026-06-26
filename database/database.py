"""
=============================================================
AI Interview Analyzer — Database Module
=============================================================
Manages SQLite storage for interview session history.

Tables:
  sessions — stores session summaries (scores, duration,
             timestamp, suggestions, PDF path).

Provides CRUD operations through the SessionDatabase class.
=============================================================
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional

import config


class SessionDatabase:
    """
    SQLite database wrapper for interview session history.
    Thread-safe: creates a new connection per method call
    (suitable for Streamlit's execution model).
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialise the database, creating the file and
        table if they don't exist.

        Args:
            db_path: Path to the SQLite file.  Defaults to
                     config.DATABASE_PATH.
        """
        self.db_path = db_path or config.DATABASE_PATH

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Create table on first run
        self._create_table()

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def save_session(self, session_data: Dict) -> int:
        """
        Insert a new session record.

        Args:
            session_data: Dictionary returned by
                ScoreEngine.build_session_summary().

        Returns:
            The rowid of the inserted record.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                """
                INSERT INTO sessions
                    (timestamp, duration_seconds,
                     eye_contact, posture, smile,
                     head_stability, hand_stability,
                     blink_rate, blink_count,
                     confidence_score, score_label,
                     suggestions, pdf_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    session_data.get("duration_seconds", 0),
                    session_data["metrics"]["eye_contact"],
                    session_data["metrics"]["posture"],
                    session_data["metrics"]["smile"],
                    session_data["metrics"]["head_stability"],
                    session_data["metrics"]["hand_stability"],
                    session_data.get("blink_rate", 0),
                    session_data.get("blink_count", 0),
                    session_data.get("confidence_score", 0),
                    session_data.get("score_label", "N/A"),
                    json.dumps(session_data.get("suggestions", [])),
                    session_data.get("pdf_path", ""),
                )
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_all_sessions(self) -> List[Dict]:
        """
        Retrieve all sessions ordered by most recent first.

        Returns:
            List of session dictionaries.
        """
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY id DESC"
            ).fetchall()
            return [self._row_to_dict(row) for row in rows]
        finally:
            conn.close()

    def get_session_by_id(self, session_id: int) -> Optional[Dict]:
        """
        Retrieve a single session by its ID.

        Args:
            session_id: Primary key of the session.

        Returns:
            Session dict or None if not found.
        """
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,)
            ).fetchone()
            return self._row_to_dict(row) if row else None
        finally:
            conn.close()

    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve the N most recent sessions.

        Args:
            limit: Number of sessions to return.

        Returns:
            List of session dictionaries.
        """
        conn = self._connect()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [self._row_to_dict(row) for row in rows]
        finally:
            conn.close()

    def delete_session(self, session_id: int) -> bool:
        """
        Delete a session by its ID.

        Args:
            session_id: Primary key of the session.

        Returns:
            True if a row was deleted, False otherwise.
        """
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE id = ?",
                (session_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_session_count(self) -> int:
        """Return the total number of stored sessions."""
        conn = self._connect()
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM sessions"
            ).fetchone()[0]
            return count
        finally:
            conn.close()

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Create and return a new SQLite connection."""
        return sqlite3.connect(self.db_path)

    def _create_table(self) -> None:
        """Create the sessions table if it doesn't exist."""
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp         TEXT NOT NULL,
                    duration_seconds  REAL DEFAULT 0,
                    eye_contact       REAL DEFAULT 0,
                    posture           REAL DEFAULT 0,
                    smile             REAL DEFAULT 0,
                    head_stability    REAL DEFAULT 0,
                    hand_stability    REAL DEFAULT 0,
                    blink_rate        REAL DEFAULT 0,
                    blink_count       INTEGER DEFAULT 0,
                    confidence_score  REAL DEFAULT 0,
                    score_label       TEXT DEFAULT 'N/A',
                    suggestions       TEXT DEFAULT '[]',
                    pdf_path          TEXT DEFAULT ''
                )
            """)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict:
        """
        Convert a sqlite3.Row to a plain dictionary,
        deserialising the JSON suggestions field.
        """
        d = dict(row)
        # Parse the JSON suggestions string
        if "suggestions" in d and isinstance(d["suggestions"], str):
            try:
                d["suggestions"] = json.loads(d["suggestions"])
            except json.JSONDecodeError:
                d["suggestions"] = []
        return d
