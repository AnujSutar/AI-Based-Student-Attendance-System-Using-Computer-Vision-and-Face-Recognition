"""Central configuration for the attendance application."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "attendance.db"
FACE_DATA_DIR = BASE_DIR / "face_data"
MODEL_DIR = BASE_DIR / "model_data"
MODEL_PATH = MODEL_DIR / "lbph_model.yml"
LABELS_PATH = MODEL_DIR / "labels.json"
CAPTURED_FACES_DIR = BASE_DIR / "captured_faces"
REPORTS_DIR = BASE_DIR / "reports"

APP_NAME = "AI-Based Student Attendance System"
CAMERA_INDEX = 0
RECOGNITION_THRESHOLD = 0.70
MAX_RECOGNITION_DISTANCE = 60.0
DUPLICATE_INTERVAL_MINUTES = 1440
LOW_ATTENDANCE_THRESHOLD = 75

CLASS_OPTIONS = ("FE", "SE", "TE", "BE")
DEPARTMENT_OPTIONS = ("AI&DS", "AI&ML", "Computer", "Civil", "Mechanical", "E&TC")
MIN_FACE_SAMPLES = 3
MAX_FACE_SAMPLES = 3

# The first run creates this local development account. Change it before a demo.
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"
