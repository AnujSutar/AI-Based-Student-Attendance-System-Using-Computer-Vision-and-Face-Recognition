"""Student profile validation and database operations."""

import re
import sqlite3
from datetime import datetime
from pathlib import Path

from database import get_connection
from config import FACE_DATA_DIR


REQUIRED_FIELDS = ("student_id", "roll_number", "name", "class_name", "division", "department")


def validate_student(student: dict[str, str]) -> list[str]:
    """Return user-friendly validation errors for a student profile."""
    errors = []
    for field in REQUIRED_FIELDS:
        if not student.get(field, "").strip():
            errors.append(f"{field.replace('_', ' ').title()} is required.")

    email = student.get("email", "").strip()
    if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        errors.append("Enter a valid email address or leave it blank.")
    return errors


def add_student(student: dict[str, str]) -> tuple[bool, str]:
    """Save one student and return a success flag plus a message."""
    errors = validate_student(student)
    if errors:
        return False, " ".join(errors)

    clean_student = {key: value.strip() for key, value in student.items()}
    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO students
                    (student_id, roll_number, name, class_name, division, department, email, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_student["student_id"],
                    clean_student["roll_number"],
                    clean_student["name"],
                    clean_student["class_name"],
                    clean_student["division"],
                    clean_student["department"],
                    clean_student.get("email") or None,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
        return True, "Student registered successfully. Capture face data in Stage 3."
    except sqlite3.IntegrityError as error:
        if "students.student_id" in str(error):
            return False, "This Student ID is already registered."
        if "students.roll_number" in str(error):
            return False, "This Roll Number is already registered."
        return False, "The student could not be saved because a unique value already exists."
    except sqlite3.Error:
        return False, "Database error: the student could not be saved. Please try again."


def get_students(search_text: str = "") -> list[dict]:
    """Return student records, optionally filtered by a simple search query."""
    search_pattern = f"%{search_text.strip()}%"
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT student_id, roll_number, name, class_name, division, department, email, created_at
            FROM students
            WHERE student_id LIKE ? OR roll_number LIKE ? OR name LIKE ?
               OR class_name LIKE ? OR division LIKE ? OR department LIKE ?
            ORDER BY name COLLATE NOCASE
            """,
            (search_pattern,) * 6,
        ).fetchall()
    return [dict(row) for row in rows]


def get_student_options() -> list[dict]:
    """Return short student records used by the face-capture selector."""
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT student_id, roll_number, name FROM students ORDER BY name COLLATE NOCASE"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_student(student_id: str) -> tuple[bool, str]:
    """Safely delete a student, physical face files, and cascade delete database records."""
    try:
        with get_connection() as connection:
            # Fetch physical file paths before deletion
            rows = connection.execute(
                "SELECT file_path FROM face_samples WHERE student_id = ?", (student_id,)
            ).fetchall()

            # Delete physical image files safely
            face_root = FACE_DATA_DIR.resolve()
            for row in rows:
                path = Path(row["file_path"]).resolve()
                if path.is_relative_to(face_root):
                    try:
                        if path.exists():
                            path.unlink()
                    except OSError:
                        pass

            # Delete the student's dedicated folder if it is empty
            student_folder = face_root / student_id
            if student_folder.exists() and student_folder.is_dir():
                try:
                    student_folder.rmdir()
                except OSError:
                    pass

            # Delete student database record
            cursor = connection.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
            if cursor.rowcount == 0:
                return False, "Student not found."

        return True, "Student profile and all associated data deleted successfully."
    except sqlite3.Error as error:
        return False, f"Database error: {str(error)}"