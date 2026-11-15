"""SQLite database setup and common database helpers."""

import hashlib
import hmac
import os
import sqlite3
from datetime import datetime

from config import (
    CAPTURED_FACES_DIR,
    DATABASE_DIR,
    DATABASE_PATH,
    DEFAULT_ADMIN_PASSWORD,
    DEFAULT_ADMIN_USERNAME,
    FACE_DATA_DIR,
    MODEL_DIR,
    REPORTS_DIR,
)


def create_directories() -> None:
    """Create local folders that the application needs."""
    for directory in (DATABASE_DIR, FACE_DATA_DIR, MODEL_DIR, CAPTURED_FACES_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with foreign-key protection enabled."""
    create_directories()
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Create a salted PBKDF2 password hash for local admin authentication."""
    salt = salt or os.urandom(16)
    password_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return f"{salt.hex()}${password_hash.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Check a password against a PBKDF2 hash without exposing plain text."""
    salt_hex, expected_hash = stored_hash.split("$", maxsplit=1)
    actual_hash = hash_password(password, bytes.fromhex(salt_hex)).split("$", maxsplit=1)[1]
    return hmac.compare_digest(actual_hash, expected_hash)


def initialize_database() -> None:
    """Create all Stage 1 tables and seed one development admin account."""
    create_directories()
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS admin (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                roll_number TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                class_name TEXT NOT NULL,
                division TEXT NOT NULL,
                department TEXT NOT NULL,
                email TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS face_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                file_path TEXT NOT NULL UNIQUE,
                captured_at TEXT NOT NULL,
                quality_score REAL NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                attendance_time TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('Present', 'Absent')),
                confidence REAL,
                created_at TEXT NOT NULL,
                UNIQUE(student_id, attendance_date),
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                recognized_name TEXT,
                log_date TEXT NOT NULL,
                log_time TEXT NOT NULL,
                result TEXT NOT NULL,
                confidence REAL,
                status TEXT NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS application_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL
            );
            """
        )

        existing_admin = connection.execute(
            "SELECT id FROM admin WHERE username = ?", (DEFAULT_ADMIN_USERNAME,)
        ).fetchone()
        if existing_admin is None:
            connection.execute(
                "INSERT INTO admin (username, password_hash, created_at) VALUES (?, ?, ?)",
                (DEFAULT_ADMIN_USERNAME, hash_password(DEFAULT_ADMIN_PASSWORD), datetime.now().isoformat()),
            )


def get_table_counts() -> dict[str, int]:
    """Return a small database health summary for the Stage 1 screen."""
    with get_connection() as connection:
        return {
            "Students": connection.execute("SELECT COUNT(*) FROM students").fetchone()[0],
            "Face samples": connection.execute("SELECT COUNT(*) FROM face_samples").fetchone()[0],
            "Attendance records": connection.execute("SELECT COUNT(*) FROM attendance").fetchone()[0],
            "Recognition logs": connection.execute("SELECT COUNT(*) FROM recognition_logs").fetchone()[0],
        }
