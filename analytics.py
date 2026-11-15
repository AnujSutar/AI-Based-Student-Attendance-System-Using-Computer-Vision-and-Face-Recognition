"""Dashboard metrics and chart data derived from the local attendance database."""

from datetime import date

from database import get_connection


def get_dashboard_metrics(selected_date: date | None = None) -> dict[str, float | int]:
    """Calculate summary values for one selected attendance date."""
    selected_date = selected_date or date.today()
    selected_date_text = selected_date.isoformat()
    with get_connection() as connection:
        total_students = connection.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        present_today = connection.execute(
            "SELECT COUNT(*) FROM attendance WHERE attendance_date = ? AND status = 'Present'",
            (selected_date_text,),
        ).fetchone()[0]
        total_records = connection.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
    absent_today = max(0, total_students - present_today)
    attendance_percentage = round((present_today / total_students) * 100, 1) if total_students else 0.0
    return {
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": absent_today,
        "attendance_percentage": attendance_percentage,
        "total_records": total_records,
    }


def get_daily_attendance_trend() -> list[dict]:
    """Return present-student count by recorded attendance date."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT attendance_date AS date, COUNT(*) AS present_count
            FROM attendance
            WHERE status = 'Present'
            GROUP BY attendance_date
            ORDER BY attendance_date
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_student_attendance_percentages() -> list[dict]:
    """Calculate present days and percentage for every registered student."""
    with get_connection() as connection:
        working_days = connection.execute(
            "SELECT COUNT(DISTINCT attendance_date) FROM attendance"
        ).fetchone()[0]
        rows = connection.execute(
            """
            SELECT s.student_id, s.roll_number, s.name,
                   COUNT(a.attendance_id) AS present_days
            FROM students AS s
            LEFT JOIN attendance AS a ON a.student_id = s.student_id AND a.status = 'Present'
            GROUP BY s.student_id, s.roll_number, s.name
            ORDER BY s.name COLLATE NOCASE
            """
        ).fetchall()
    result = []
    for row in rows:
        record = dict(row)
        record["attendance_percentage"] = round((record["present_days"] / working_days) * 100, 1) if working_days else 0.0
        result.append(record)
    return result


def get_monthly_attendance_trend() -> list[dict]:
    """Return total Present records grouped by calendar month."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT substr(attendance_date, 1, 7) AS month, COUNT(*) AS present_count
            FROM attendance
            WHERE status = 'Present'
            GROUP BY substr(attendance_date, 1, 7)
            ORDER BY month
            """
        ).fetchall()
    return [dict(row) for row in rows]
