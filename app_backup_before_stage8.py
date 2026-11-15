"""Stage 1 application entry point."""

import time

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    APP_NAME,
    CAMERA_INDEX,
    CLASS_OPTIONS,
    DATABASE_PATH,
    DEPARTMENT_OPTIONS,
    MAX_FACE_SAMPLES,
    MIN_FACE_SAMPLES,
    LOW_ATTENDANCE_THRESHOLD,
)
from database import get_table_counts, initialize_database
from attendance_manager import get_attendance_records, mark_attendance
from analytics import (
    get_daily_attendance_trend,
    get_dashboard_metrics,
    get_monthly_attendance_trend,
    get_student_attendance_percentages,
)
from report_manager import (
    get_absent_students,
    get_filter_options,
    get_filtered_attendance,
    get_student_attendance_summary,
    to_csv_bytes,
    to_excel_bytes,
)
from face_engine import (
    assess_and_save_face,
    count_face_samples,
    delete_face_samples,
    get_face_sample_summary,
    load_recognizer,
    recognize_faces,
    train_recognizer,
)
from student_manager import add_student, get_student_options, get_students


st.set_page_config(page_title=APP_NAME, page_icon="🎓", layout="wide")
initialize_database()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard", "Register Student", "Students", "Capture Face", "Face Data",
        "Live Recognition", "Live Attendance", "Attendance Records", "Reports",
    ],
)


def show_dashboard() -> None:
    st.title("🎓 AI-Based Student Attendance System")
    st.caption("Attendance dashboard and analytics")
    selected_date = st.date_input("Dashboard date")
    metrics = get_dashboard_metrics(selected_date)

    metric_columns = st.columns(5)
    metric_columns[0].metric("Total Students", metrics["total_students"])
    metric_columns[1].metric("Present", metrics["present_today"])
    metric_columns[2].metric("Absent", metrics["absent_today"])
    metric_columns[3].metric("Attendance %", f"{metrics['attendance_percentage']}%")
    metric_columns[4].metric("Total Records", metrics["total_records"])

    if metrics["total_students"] == 0:
        st.info("Register students to begin building attendance analytics.")
        return

    distribution_data = pd.DataFrame(
        {"Status": ["Present", "Absent"], "Students": [metrics["present_today"], metrics["absent_today"]]}
    )
    daily_trend = pd.DataFrame(get_daily_attendance_trend())
    student_percentages = pd.DataFrame(get_student_attendance_percentages())
    monthly_trend = pd.DataFrame(get_monthly_attendance_trend())

    left, right = st.columns(2)
    with left:
        st.subheader("Present vs Absent")
        st.plotly_chart(
            px.pie(distribution_data, names="Status", values="Students", color="Status",
                   color_discrete_map={"Present": "#16a34a", "Absent": "#dc2626"}, hole=0.45),
            use_container_width=True,
        )
    with right:
        st.subheader("Daily Attendance Trend")
        if daily_trend.empty:
            st.info("No attendance trend is available yet.")
        else:
            st.plotly_chart(
                px.line(daily_trend, x="date", y="present_count", markers=True,
                        labels={"date": "Date", "present_count": "Present Students"}),
                use_container_width=True,
            )

    left, right = st.columns(2)
    with left:
        st.subheader("Student-wise Attendance Percentage")
        st.plotly_chart(
            px.bar(student_percentages, x="name", y="attendance_percentage", color="attendance_percentage",
                   color_continuous_scale="Blues", labels={"name": "Student", "attendance_percentage": "Attendance %"}),
            use_container_width=True,
        )
    with right:
        st.subheader("Monthly Attendance Trend")
        if monthly_trend.empty:
            st.info("No monthly attendance trend is available yet.")
        else:
            st.plotly_chart(
                px.bar(monthly_trend, x="month", y="present_count", labels={"month": "Month", "present_count": "Present Records"}),
                use_container_width=True,
            )

    st.caption(f"Local SQLite database: {DATABASE_PATH}")


def show_registration() -> None:
    st.title("Register Student")
    st.caption("Save the student profile first. Face samples will be captured in Stage 3.")

    with st.form("student_registration_form", clear_on_submit=True):
        left, right = st.columns(2)
        with left:
            student_id = st.text_input("Student ID *", placeholder="STU001")
            roll_number = st.text_input("Roll Number *", placeholder="2026A001")
            name = st.text_input("Student Name *", placeholder="Full name")
            email = st.text_input("Email (optional)", placeholder="student@example.com")
        with right:
            class_name = st.selectbox("Class *", CLASS_OPTIONS)
            division = st.text_input("Division *", placeholder="A")
            department = st.selectbox("Department *", DEPARTMENT_OPTIONS)

        submitted = st.form_submit_button("Save Student", type="primary")
        if submitted:
            success, message = add_student(
                {
                    "student_id": student_id,
                    "roll_number": roll_number,
                    "name": name,
                    "class_name": class_name,
                    "division": division,
                    "department": department,
                    "email": email,
                }
            )
            if success:
                st.success(message)
            else:
                st.error(message)


def show_students() -> None:
    st.title("Students")
    search_text = st.text_input("Search by name, ID, roll number, class, division, or department")
    students = get_students(search_text)
    if not students:
        st.info("No matching students found. Register a student to begin.")
        return

    table = pd.DataFrame(students).rename(
        columns={
            "student_id": "Student ID",
            "roll_number": "Roll Number",
            "name": "Name",
            "class_name": "Class",
            "division": "Division",
            "department": "Department",
            "email": "Email",
            "created_at": "Registered At",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(students)} student(s).")


def show_capture_face() -> None:
    st.title("Capture Face Samples")
    st.caption("Capture three clear samples for each student. Only one face should be visible.")
    students = get_student_options()
    if not students:
        st.warning("Register a student before capturing face samples.")
        return

    student_lookup = {
        f"{student['name']} — {student['roll_number']} ({student['student_id']})": student
        for student in students
    }
    selected_label = st.selectbox("Select student", list(student_lookup))
    student = student_lookup[selected_label]
    current_count = count_face_samples(student["student_id"])
    st.progress(current_count / MAX_FACE_SAMPLES, text=f"Samples captured: {current_count}/{MAX_FACE_SAMPLES}")
    st.info(
        "Take three photos with slightly different natural angles and expressions. Keep your face clear, "
        "well lit, and alone in the frame."
    )

    captured_image = st.camera_input("Take a face photo", key=f"camera_{student['student_id']}")
    if captured_image is not None:
        image_hash = hash(captured_image.getvalue())
        processed_key = f"processed_{student['student_id']}"
        if st.session_state.get(processed_key) != image_hash:
            st.session_state[processed_key] = image_hash
            success, message, _ = assess_and_save_face(captured_image.getvalue(), student["student_id"])
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    st.divider()
    st.subheader("Or upload photos from this computer")
    uploaded_images = st.file_uploader(
        "Choose up to 3 clear face photos",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"upload_{student['student_id']}",
    )
    if uploaded_images:
        st.caption("Only new photos are processed. The app still stores a maximum of 3 samples per student.")
        if len(uploaded_images) > 3:
            st.warning("Choose no more than 3 photos at one time.")
        elif st.button("Validate and save uploaded photos", key=f"save_uploads_{student['student_id']}"):
            processed_key = f"uploaded_hashes_{student['student_id']}"
            processed_hashes = st.session_state.setdefault(processed_key, set())
            saved_count = 0
            for uploaded_image in uploaded_images:
                image_bytes = uploaded_image.getvalue()
                image_hash = hash(image_bytes)
                if image_hash in processed_hashes:
                    continue
                processed_hashes.add(image_hash)
                success, message, _ = assess_and_save_face(image_bytes, student["student_id"])
                if success:
                    saved_count += 1
                    st.success(f"{uploaded_image.name}: {message}")
                else:
                    st.error(f"{uploaded_image.name}: {message}")
            if saved_count:
                new_total = count_face_samples(student["student_id"])
                st.info(
                    f"Upload complete: {saved_count} photo(s) saved. "
                    f"This student now has {new_total}/{MAX_FACE_SAMPLES} face samples."
                )

    if current_count >= MIN_FACE_SAMPLES:
        st.success("This student has enough samples for the recognition-training stage.")


def show_face_data() -> None:
    st.title("Face Data")
    st.caption("Face samples are stored only on this computer for this college prototype.")
    rows = get_face_sample_summary()
    table = pd.DataFrame(rows).rename(
        columns={
            "student_id": "Student ID",
            "roll_number": "Roll Number",
            "name": "Student Name",
            "sample_count": "Saved Face Samples",
        }
    )
    if table.empty:
        st.info("No student profiles are available yet.")
        return
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(f"Each student needs {MIN_FACE_SAMPLES} face samples before recognition training.")

    students_with_face_data = [row for row in rows if row["sample_count"] > 0]
    if students_with_face_data:
        st.divider()
        st.subheader("Delete face data")
        st.warning("This permanently removes the selected student's saved face photos. Their student profile remains.")
        options = {
            f"{row['name']} — {row['roll_number']} ({row['sample_count']} sample(s))": row["student_id"]
            for row in students_with_face_data
        }

        def reset_deletion_confirmation() -> None:
            st.session_state["confirm_face_data_deletion"] = False

        selected_label = st.selectbox(
            "Student whose face data will be deleted",
            list(options),
            key="face_data_delete_student",
            on_change=reset_deletion_confirmation,
        )
        confirmed = st.checkbox(
            "I understand that this deletes the selected student's face data permanently.",
            key="confirm_face_data_deletion",
        )
        if st.button("Delete selected face data", type="primary", disabled=not confirmed):
            success, message = delete_face_samples(options[selected_label])
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)


def show_live_recognition() -> None:
    st.title("Live Face Recognition")
    st.caption("Stage 4: identify known faces with the laptop webcam. Attendance is not recorded yet.")
    st.info(
        "This uses local LBPH face recognition. It compares live face patterns with the three "
        "saved samples; no face data is sent online."
    )

    left, right = st.columns([1, 2])
    with left:
        if st.button("Train / retrain recognition model", type="primary"):
            success, message = train_recognizer()
            if success:
                st.success(message)
            else:
                st.error(message)
    with right:
        st.caption("Retrain after adding or deleting face samples.")

    duration = st.slider("Live camera duration (seconds)", min_value=5, max_value=30, value=15)
    if st.button("Start live recognition"):
        recognizer, label_map, error = load_recognizer()
        if error:
            st.error(error)
            return

        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            st.error("Webcam unavailable. Check camera permissions and close other apps using the camera.")
            return

        camera_view = st.empty()
        status_box = st.empty()
        recognized_names: set[str] = set()
        end_time = time.time() + duration
        try:
            while time.time() < end_time:
                success, frame = camera.read()
                if not success:
                    st.error("A camera frame could not be read. Try reconnecting the webcam.")
                    break
                results = recognize_faces(frame, recognizer, label_map)
                for result in results:
                    x, y, width, height = result["box"]
                    color = (0, 180, 0) if result["result"] == "Recognized" else (0, 165, 255)
                    if result["result"] == "Unknown":
                        color = (0, 0, 255)
                    label = f"{result['name']} | {result['similarity']}%"
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                    cv2.putText(frame, label, (x, max(y - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if result["result"] == "Recognized":
                        recognized_names.add(f"{result['name']} ({result['roll_number']})")
                camera_view.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                status_box.caption(f"Scanning... {max(0, int(end_time - time.time()))} second(s) remaining.")
        except RuntimeError as error:
            st.error(str(error))
        finally:
            camera.release()

        if recognized_names:
            st.success("Recognized during this session: " + ", ".join(sorted(recognized_names)))
        else:
            st.warning("No high-confidence registered face was recognized during this session.")
        st.info("Live recognition session complete. Stage 5 will add automatic attendance marking.")


def show_live_attendance() -> None:
    st.title("Live Attendance")
    st.caption("Recognized students are marked Present automatically, once per day.")
    st.warning(
        "Only green, high-confidence recognitions are marked. Orange uncertain and red unknown faces are not recorded."
    )
    duration = st.slider("Attendance camera duration (seconds)", min_value=5, max_value=30, value=15, key="attendance_duration")
    if st.button("Start attendance camera", type="primary"):
        recognizer, label_map, error = load_recognizer()
        if error:
            st.error(error)
            return

        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            st.error("Webcam unavailable. Check camera permissions and close other apps using the camera.")
            return

        camera_view = st.empty()
        status_box = st.empty()
        result_box = st.empty()
        handled_student_ids: set[str] = set()
        session_messages: list[str] = []
        end_time = time.time() + duration
        try:
            while time.time() < end_time:
                success, frame = camera.read()
                if not success:
                    st.error("A camera frame could not be read. Try reconnecting the webcam.")
                    break
                results = recognize_faces(frame, recognizer, label_map)
                for result in results:
                    x, y, width, height = result["box"]
                    color = (0, 180, 0) if result["result"] == "Recognized" else (0, 165, 255)
                    if result["result"] == "Unknown":
                        color = (0, 0, 255)
                    label = f"{result['name']} | {result['similarity']}%"
                    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                    cv2.putText(frame, label, (x, max(y - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    student_id = result["student_id"]
                    if result["result"] == "Recognized" and student_id and student_id not in handled_student_ids:
                        handled_student_ids.add(student_id)
                        created, message = mark_attendance(student_id, result["similarity"])
                        session_messages.append(message)

                camera_view.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                status_box.caption(f"Attendance scanning... {max(0, int(end_time - time.time()))} second(s) remaining.")
                if session_messages:
                    result_box.info("Latest result: " + session_messages[-1])
        except RuntimeError as error:
            st.error(str(error))
        finally:
            camera.release()

        if session_messages:
            st.success("Attendance session complete.")
            for message in session_messages:
                st.write(f"- {message}")
        else:
            st.warning("No high-confidence registered face was detected; no attendance was marked.")


def show_attendance_records() -> None:
    st.title("Attendance Records")
    records = get_attendance_records()
    if not records:
        st.info("No attendance has been recorded yet. Use Live Attendance to mark students present.")
        return
    table = pd.DataFrame(records).rename(
        columns={
            "attendance_id": "Attendance ID", "student_id": "Student ID", "roll_number": "Roll Number",
            "name": "Student Name", "class_name": "Class", "attendance_date": "Date",
            "attendance_time": "Time", "status": "Status", "confidence": "Recognition Confidence",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(records)} most recent attendance record(s).")


def show_reports() -> None:
    st.title("Attendance Reports")
    st.caption("Filter attendance records, identify absences, and download reports.")
    options = get_filter_options()
    if not options["students"]:
        st.info("Register students before creating reports.")
        return

    with st.expander("Report filters", expanded=True):
        first, second, third = st.columns(3)
        with first:
            use_date = st.checkbox("Filter by date", value=False)
            selected_date = st.date_input("Attendance date")
            class_name = st.selectbox("Class", ["All"] + options["classes"])
        with second:
            division = st.selectbox("Division", ["All"] + options["divisions"])
            department = st.selectbox("Department", ["All"] + options["departments"])
        with third:
            student_labels = {"All": None}
            student_labels.update(
                {f"{student['name']} — {student['roll_number']}": student["student_id"] for student in options["students"]}
            )
            selected_student_label = st.selectbox("Student", list(student_labels))
            low_threshold = st.number_input(
                "Low attendance threshold (%)", min_value=0, max_value=100,
                value=LOW_ATTENDANCE_THRESHOLD, step=5,
            )

    selected_date_text = selected_date.isoformat() if use_date else None
    selected_student = student_labels[selected_student_label]
    class_filter = None if class_name == "All" else class_name
    division_filter = None if division == "All" else division
    department_filter = None if department == "All" else department

    records = get_filtered_attendance(
        selected_date_text, selected_student, class_filter, division_filter, department_filter
    )
    report_table = pd.DataFrame(records).rename(
        columns={
            "student_id": "Student ID", "roll_number": "Roll Number", "name": "Student Name",
            "class_name": "Class", "division": "Division", "department": "Department",
            "attendance_date": "Date", "attendance_time": "Time", "status": "Status",
            "confidence": "Recognition Confidence",
        }
    )

    st.subheader("Filtered attendance records")
    if report_table.empty:
        st.info("No attendance records match these filters.")
    else:
        st.dataframe(report_table, use_container_width=True, hide_index=True)
        download_left, download_right = st.columns(2)
        with download_left:
            st.download_button(
                "Download CSV", to_csv_bytes(report_table), "attendance_report.csv", "text/csv"
            )
        with download_right:
            st.download_button(
                "Download Excel", to_excel_bytes(report_table), "attendance_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    st.divider()
    absent_date = selected_date.isoformat()
    absent_students = pd.DataFrame(get_absent_students(absent_date, class_filter, division_filter, department_filter)).rename(
        columns={"student_id": "Student ID", "roll_number": "Roll Number", "name": "Student Name",
                 "class_name": "Class", "division": "Division", "department": "Department"}
    )
    st.subheader(f"Absent students — {absent_date}")
    if absent_students.empty:
        st.success("No absent students for the selected scope.")
    else:
        st.dataframe(absent_students, use_container_width=True, hide_index=True)
        st.caption(f"Absent students: {len(absent_students)}")

    summary = pd.DataFrame(get_student_attendance_summary(class_filter, division_filter, department_filter)).rename(
        columns={"student_id": "Student ID", "roll_number": "Roll Number", "name": "Student Name",
                 "class_name": "Class", "present_days": "Present Days", "total_working_days": "Working Days",
                 "attendance_percentage": "Attendance %"}
    )
    st.subheader("Low attendance")
    if summary.empty or summary["Working Days"].max() == 0:
        st.info("Low-attendance analysis becomes available after attendance is recorded on at least one date.")
    else:
        low_attendance = summary[summary["Attendance %"] < low_threshold]
        if low_attendance.empty:
            st.success(f"No students are below the {low_threshold}% threshold.")
        else:
            st.dataframe(low_attendance, use_container_width=True, hide_index=True)


if page == "Dashboard":
    show_dashboard()
elif page == "Register Student":
    show_registration()
elif page == "Students":
    show_students()
elif page == "Capture Face":
    show_capture_face()
elif page == "Face Data":
    show_face_data()
elif page == "Live Recognition":
    show_live_recognition()
elif page == "Live Attendance":
    show_live_attendance()
else:
    show_attendance_records()
