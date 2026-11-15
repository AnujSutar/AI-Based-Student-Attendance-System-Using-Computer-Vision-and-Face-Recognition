"""Local face detection, quality checks, sample storage, and LBPH recognition."""

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import (
    FACE_DATA_DIR,
    LABELS_PATH,
    MAX_FACE_SAMPLES,

    MIN_FACE_SAMPLES,
    MODEL_PATH,
)
from database import get_connection


def get_face_detector() -> cv2.CascadeClassifier:
    """Load OpenCV's bundled Haar-cascade face detector."""
    cascade_path = Path(cv2.data.haarcascades) / \
        "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError("The OpenCV face detector could not be loaded.")
    return detector


def count_face_samples(student_id: str) -> int:
    with get_connection() as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM face_samples WHERE student_id = ?", (student_id,)
        ).fetchone()[0]


def get_face_sample_summary() -> list[dict]:
    """Return sample counts so the operator can see training readiness."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT s.student_id, s.roll_number, s.name, COUNT(fs.id) AS sample_count
            FROM students AS s
            LEFT JOIN face_samples AS fs ON fs.student_id = s.student_id
            GROUP BY s.student_id, s.roll_number, s.name
            ORDER BY s.name COLLATE NOCASE
            """
        ).fetchall()
    return [dict(row) for row in rows]


def delete_face_samples(student_id: str) -> tuple[bool, str]:
    """Permanently remove one student's saved face photos and sample records."""
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT file_path FROM face_samples WHERE student_id = ?", (
                student_id,)
        ).fetchall()

    if not rows:
        return False, "No saved face data was found for this student."

    face_root = FACE_DATA_DIR.resolve()
    paths = [Path(row["file_path"]).resolve() for row in rows]
    invalid_paths = [
        path for path in paths if not path.is_relative_to(face_root)]
    if invalid_paths:
        return False, "Face data could not be deleted because an unsafe file path was detected."

    failed_paths = []
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except OSError:
            failed_paths.append(path.name)
    if failed_paths:
        return False, "Some local face photos could not be deleted. Close any app using them and try again."

    with get_connection() as connection:
        connection.execute(
            "DELETE FROM face_samples WHERE student_id = ?", (student_id,))

    # Remove the now-empty per-student folder when it is safe to do so.
    for parent in {path.parent for path in paths}:
        if parent.is_relative_to(face_root):
            try:
                parent.rmdir()
            except OSError:
                pass
    return True, "Face data deleted successfully. This student must be captured again before recognition."


def _create_lbph_recognizer():
    """Create OpenCV's local LBPH recognizer."""
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "LBPH recognition is unavailable. Install opencv-contrib-python and restart the app.")
    return cv2.face.LBPHFaceRecognizer_create()


def train_recognizer() -> tuple[bool, str]:
    """Train and save an offline LBPH model from students with three samples."""
    try:
        recognizer = _create_lbph_recognizer()
    except RuntimeError as error:
        return False, str(error)

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT fs.student_id, fs.file_path, s.name, s.roll_number
            FROM face_samples AS fs JOIN students AS s ON s.student_id = fs.student_id
            ORDER BY fs.student_id, fs.id
            """
        ).fetchall()

    by_student: dict[str, list] = {}
    for row in rows:
        by_student.setdefault(row["student_id"], []).append(row)
    eligible = {student_id: samples for student_id,
                samples in by_student.items() if len(samples) >= MIN_FACE_SAMPLES}
    if not eligible:
        return False, f"Capture {MIN_FACE_SAMPLES} face samples for at least one student before training."

    images: list[np.ndarray] = []
    labels: list[int] = []
    label_map: dict[str, dict[str, str]] = {}
    for numeric_label, student_id in enumerate(sorted(eligible)):
        student_samples = eligible[student_id]
        label_map[str(numeric_label)] = {
            "student_id": student_id,
            "name": student_samples[0]["name"],
            "roll_number": student_samples[0]["roll_number"],
        }
        for sample in student_samples:
            image = cv2.imread(sample["file_path"], cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(cv2.resize(image, (200, 200)))
                labels.append(numeric_label)
    if not images:
        return False, "No readable face samples were found. Capture the face data again."

    recognizer.train(images, np.array(labels, dtype=np.int32))
    recognizer.write(str(MODEL_PATH))
    LABELS_PATH.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
    return True, f"Recognition model trained with {len(images)} samples from {len(label_map)} student(s)."


def load_recognizer():
    """Load the saved local model and its student-label map."""
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, None, "Train the recognition model before starting the camera."
    try:
        recognizer = _create_lbph_recognizer()
        recognizer.read(str(MODEL_PATH))
        return recognizer, json.loads(LABELS_PATH.read_text(encoding="utf-8")), None
    except (RuntimeError, cv2.error, json.JSONDecodeError):
        return None, None, "The saved recognition model could not be loaded. Train it again."


def recognize_faces(
    frame: np.ndarray,
    recognizer,
    label_map: dict,
    max_distance: float,
) -> list[dict]:
    """Detect every face in a frame and return its recognition result."""
    detector = get_face_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=6, minSize=(90, 90))
    results = []
    for x, y, width, height in faces:
        face = cv2.resize(gray[y: y + height, x: x + width], (200, 200))
        predicted_label, distance = recognizer.predict(face)
        similarity = round(max(0.0, 100.0 - float(distance)), 1)
        student = label_map.get(str(predicted_label))
        if student and distance <= max_distance:
            result = {**student, "result": "Recognized",
                      "similarity": similarity}
        elif student and distance <= max_distance + 15:
            result = {**student, "result": "Recognition uncertain",
                      "similarity": similarity}
        else:
            result = {"student_id": None, "name": "Unknown",
                      "roll_number": "—", "result": "Unknown", "similarity": similarity}
        result["box"] = (int(x), int(y), int(width), int(height))
        results.append(result)
    return results


def assess_and_save_face(image_bytes: bytes, student_id: str) -> tuple[bool, str, float | None]:
    """Validate one camera image, crop its face, and store it locally for training."""
    sample_count = count_face_samples(student_id)
    if sample_count >= MAX_FACE_SAMPLES:
        return False, f"This student already has {MAX_FACE_SAMPLES} face samples.", None

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return False, "The captured image could not be read. Please take another photo.", None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        detector = get_face_detector()
    except RuntimeError:
        return (
            False,
            "Face detection is not available in this OpenCV installation. "
            "Reinstall opencv-contrib-python using the command in the project instructions.",
            None,
        )
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=6, minSize=(90, 90))

    if len(faces) == 0:
        return False, "No clear face detected. Face the camera and improve the lighting.", None
    if len(faces) > 1:
        return False, "Multiple faces detected. Ensure only the selected student is in the frame.", None

    x, y, width, height = faces[0]
    face = gray[y: y + height, x: x + width]
    brightness = float(face.mean())
    sharpness = float(cv2.Laplacian(face, cv2.CV_64F).var())

    if brightness < 45:
        return False, "The image is too dark. Move to a better-lit area and try again.", None
    if brightness > 220:
        return False, "The image is overexposed. Reduce direct light and try again.", None
    if sharpness < 45:
        return False, "The face image is blurry. Hold still and capture another image.", None

    face = cv2.resize(face, (200, 200))
    quality_score = round(
        min(100, (sharpness / 4) + min(brightness, 180) / 3), 1)
    student_folder = FACE_DATA_DIR / student_id
    student_folder.mkdir(parents=True, exist_ok=True)
    filename = f"face_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    output_path = student_folder / filename

    if not cv2.imwrite(str(output_path), face):
        return False, "The face sample could not be stored. Please try again.", None

    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO face_samples (student_id, file_path, captured_at, quality_score)
                VALUES (?, ?, ?, ?)
                """,
                (student_id, str(output_path), datetime.now().isoformat(
                    timespec="seconds"), quality_score),
            )
    except Exception:
        output_path.unlink(missing_ok=True)
        return False, "The sample was not saved to the database. Please try again.", None

    current_total = sample_count + 1
    return True, f"Face sample {current_total}/{MAX_FACE_SAMPLES} saved (quality score: {quality_score}).", quality_score
