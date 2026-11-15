"""Attendance recording and attendance-record queries."""

import sqlite3
from datetime import datetime

from database import get_connection


def mark_attendance(student_id: str, confidence: float) -> tuple[bool, str]:
    """Record Present once per student per day; return whether a new record was made."""
    now = datetime.now()
    attendance_date = now.date().isoformat()
    attendance_time = now.strftime("%H:%M:%S")
    try:
        with get_connection() as connection:
            student = connection.execute(
                "SELECT name, roll_number FROM students WHERE student_id = ?", (student_id,)
            ).fetchone()
            if student is None:
                return False, "Recognized student profile was not found."

            connection.execute(
                """
                INSERT INTO attendance
                    (student_id, attendance_date, attendance_time, status, confidence, created_at)
                VALUES (?, ?, ?, 'Present', ?, ?)
                """,
                (student_id, attendance_date, attendance_time, confidence, now.isoformat(timespec="seconds")),
            )
            connection.execute(
                """
                INSERT INTO recognition_logs
                    (student_id, recognized_name, log_date, log_time, result, confidence, status)
                VALUES (?, ?, ?, ?, 'Recognized', ?, 'Attendance marked')
                """,
                (student_id, student["name"], attendance_date, attendance_time, confidence),
            )
        return True, f"Present marked: {student['name']} ({student['roll_number']})."
    except sqlite3.IntegrityError:
        return False, "Attendance already recorded for today."
    except sqlite3.Error:
        return False, "Database error: attendance could not be recorded."


def get_attendance_records(limit: int | None = 200) -> list[dict]:
    """Return attendance records joined with student details, newest first."""
    query = """
        SELECT a.attendance_id, s.student_id, s.roll_number, s.name, s.class_name,
               s.division, s.department, a.attendance_date, a.attendance_time,
               a.status, a.confidence
        FROM attendance AS a
        JOIN students AS s ON s.student_id = a.student_id
        ORDER BY a.attendance_date DESC, a.attendance_time DESC
    """
    parameters = () if limit is None else (limit,)
    if limit is not None:
        query += " LIMIT ?"
    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]
