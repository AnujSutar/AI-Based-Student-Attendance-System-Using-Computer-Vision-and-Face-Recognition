
"""Persistent application settings stored locally in SQLite."""

from config import (
    CAMERA_INDEX,
    DUPLICATE_INTERVAL_MINUTES,
    LOW_ATTENDANCE_THRESHOLD,
    MAX_RECOGNITION_DISTANCE,
)
from database import get_connection


DEFAULT_SETTINGS = {
    "app_name": "AI-Based Student Attendance System",
    "camera_index": str(CAMERA_INDEX),
    "max_recognition_distance": str(MAX_RECOGNITION_DISTANCE),
    "duplicate_interval_minutes": str(DUPLICATE_INTERVAL_MINUTES),
    "low_attendance_threshold": str(LOW_ATTENDANCE_THRESHOLD),
    "campus_latitude": "",
    "campus_longitude": "",
    "campus_radius_meters": "150",
}


def initialize_settings() -> None:
    with get_connection() as connection:
        for key, value in DEFAULT_SETTINGS.items():
            connection.execute(
                "INSERT OR IGNORE INTO application_settings "
                "(setting_key, setting_value) VALUES (?, ?)",
                (key, value),
            )


def get_settings() -> dict[str, str]:
    initialize_settings()
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT setting_key, setting_value FROM application_settings"
        ).fetchall()

    stored = {row["setting_key"]: row["setting_value"] for row in rows}
    return {**DEFAULT_SETTINGS, **stored}


def update_settings(settings: dict[str, str]) -> tuple[bool, str]:
    """Validate and persist configurable settings."""
    try:
        camera_index = int(settings["camera_index"])
        max_distance = float(settings["max_recognition_distance"])
        duplicate_interval = int(settings["duplicate_interval_minutes"])
        low_threshold = float(settings["low_attendance_threshold"])

        latitude_text = str(settings.get("campus_latitude", "")).strip()
        longitude_text = str(settings.get("campus_longitude", "")).strip()
        radius = float(settings.get("campus_radius_meters", "150"))
    except (KeyError, TypeError, ValueError):
        return False, "Enter valid values for all settings."

    if (
        camera_index < 0
        or not 0 < max_distance <= 100
        or duplicate_interval < 1
        or not 0 <= low_threshold <= 100
    ):
        return False, "Enter valid values for all settings."

    if bool(latitude_text) != bool(longitude_text):
        return False, "Enter both campus latitude and longitude."

    if latitude_text and longitude_text:
        try:
            latitude = float(latitude_text)
            longitude = float(longitude_text)
        except ValueError:
            return False, "Campus coordinates must be numeric."

        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            return False, "Enter valid campus coordinates."

    if not 10 <= radius <= 5000:
        return False, "Campus radius must be between 10 and 5000 metres."

    values = {
        "app_name": str(
            settings.get("app_name", DEFAULT_SETTINGS["app_name"])
        ).strip() or DEFAULT_SETTINGS["app_name"],
        "camera_index": str(camera_index),
        "max_recognition_distance": str(max_distance),
        "duplicate_interval_minutes": str(duplicate_interval),
        "low_attendance_threshold": str(low_threshold),
        "campus_latitude": latitude_text,
        "campus_longitude": longitude_text,
        "campus_radius_meters": str(radius),
    }

    with get_connection() as connection:
        for key, value in values.items():
            connection.execute(
                "INSERT INTO application_settings "
                "(setting_key, setting_value) VALUES (?, ?) "
                "ON CONFLICT(setting_key) DO UPDATE SET "
                "setting_value = excluded.setting_value",
                (key, value),
            )

    return True, "Settings saved successfully."
