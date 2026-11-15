"""Filtered reports, absence analysis, attendance summaries, and export helpers."""

from io import BytesIO

import pandas as pd

from database import get_connection


def get_filter_options() -> dict[str, list]:
    """Return available values for report filter controls."""
    with get_connection() as connection:
        students = connection.execute(
            "SELECT student_id, name, roll_number FROM students ORDER BY name COLLATE NOCASE"
        ).fetchall()
        classes = connection.execute("SELECT DISTINCT class_name FROM students ORDER BY class_name").fetchall()
        divisions = connection.execute("SELECT DISTINCT division FROM students ORDER BY division").fetchall()
        departments = connection.execute("SELECT DISTINCT department FROM students ORDER BY department").fetchall()
    return {
        "students": [dict(row) for row in students],
        "classes": [row[0] for row in classes],
        "divisions": [row[0] for row in divisions],
        "departments": [row[0] for row in departments],
    }


def _build_student_filters(student_id=None, class_name=None, division=None, department=None) -> tuple[list[str], list]:
    clauses, parameters = [], []
    for column, value in (
        ("s.student_id", student_id),
        ("s.class_name", class_name),
        ("s.division", division),
        ("s.department", department),
    ):
        if value:
            clauses.append(f"{column} = ?")
            parameters.append(value)
    return clauses, parameters


def get_filtered_attendance(
    attendance_date=None, student_id=None, class_name=None, division=None, department=None
) -> list[dict]:
    """Return attendance records matching the chosen report filters."""
    clauses, parameters = _build_student_filters(student_id, class_name, division, department)
    if attendance_date:
        clauses.append("a.attendance_date = ?")
        parameters.append(attendance_date)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    query = f"""
        SELECT s.student_id, s.roll_number, s.name, s.class_name, s.division, s.department,
               a.attendance_date, a.attendance_time, a.status, a.confidence
        FROM attendance AS a JOIN students AS s ON s.student_id = a.student_id
        {where}
        ORDER BY a.attendance_date DESC, a.attendance_time DESC
    """
    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def get_absent_students(attendance_date: str, class_name=None, division=None, department=None) -> list[dict]:
    """List registered students without a Present entry on a selected date."""
    clauses, parameters = _build_student_filters(None, class_name, division, department)
    clauses.append(
        "NOT EXISTS (SELECT 1 FROM attendance AS a WHERE a.student_id = s.student_id "
        "AND a.attendance_date = ? AND a.status = 'Present')"
    )
    parameters.append(attendance_date)
    where = " WHERE " + " AND ".join(clauses)
    query = f"""
        SELECT s.student_id, s.roll_number, s.name, s.class_name, s.division, s.department
        FROM students AS s {where}
        ORDER BY s.name COLLATE NOCASE
    """
    with get_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]


def get_student_attendance_summary(class_name=None, division=None, department=None) -> list[dict]:
    """Calculate each selected student's Present days divided by all recorded working days."""
    clauses, parameters = _build_student_filters(None, class_name, division, department)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    with get_connection() as connection:
        working_days = connection.execute("SELECT COUNT(DISTINCT attendance_date) FROM attendance").fetchone()[0]
        rows = connection.execute(
            f"""
            SELECT s.student_id, s.roll_number, s.name, s.class_name, s.division, s.department,
                   COUNT(a.attendance_id) AS present_days
            FROM students AS s
            LEFT JOIN attendance AS a ON a.student_id = s.student_id AND a.status = 'Present'
            {where}
            GROUP BY s.student_id, s.roll_number, s.name, s.class_name, s.division, s.department
            ORDER BY s.name COLLATE NOCASE
            """,
            parameters,
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["total_working_days"] = working_days
        item["attendance_percentage"] = round((item["present_days"] / working_days) * 100, 1) if working_days else 0.0
        result.append(item)
    return result


def to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")


def to_excel_bytes(dataframe: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Attendance Report")
    return buffer.getvalue()
