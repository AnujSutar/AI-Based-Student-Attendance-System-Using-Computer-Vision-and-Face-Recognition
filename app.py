"""Stage 8 application entry point."""

from student_manager import add_student, get_student_options, get_students, delete_student
from face_engine import (
    assess_and_save_face,
    count_face_samples,
    delete_face_samples,
    get_face_sample_summary,
    load_recognizer,
    recognize_faces,
    train_recognizer,
)
from report_manager import (
    get_absent_students,
    get_filter_options,
    get_filtered_attendance,
    get_student_attendance_summary,
    to_csv_bytes,
    to_excel_bytes,
)
from analytics import (
    get_daily_attendance_trend,
    get_dashboard_metrics,
    get_monthly_attendance_trend,
    get_student_attendance_percentages,
)
from attendance_manager import get_attendance_records, mark_attendance
from config import (
    CLASS_OPTIONS,
    DATABASE_PATH,
    DEPARTMENT_OPTIONS,
    MAX_FACE_SAMPLES,
    MIN_FACE_SAMPLES,
)
from settings_manager import get_settings, update_settings
from auth import authenticate, change_password
import time
import math
from streamlit_geolocation import streamlit_geolocation
import cv2
import pandas as pd
import plotly.express as px
import streamlit as st
import pydeck as pdk
from pydeck.types import String
import folium
from streamlit_folium import st_folium
from folium.plugins import MeasureControl

# Initialize the database early so settings can be read for the page config
from database import get_table_counts, initialize_database
initialize_database()


# Apply dynamic settings to the page configuration
initial_settings = get_settings()
st.set_page_config(
    page_title=initial_settings.get(
        "app_name", "AI-Based Student Attendance System"),
    page_icon="🎓",
    layout="wide"
)


# --- AUTHENTICATION STATE MANAGEMENT ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""

if not st.session_state.get("is_authenticated", False):

    st.markdown(
        """
        <style>
        /* Overall login page */
        .login-wrapper {
            max-width: 520px;
            margin: 90px auto 0 auto;
        }

        .login-title {
            text-align: center;
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .login-subtitle {
            text-align: center;
            color: #9a9a9a;
            font-size: 16px;
            margin-bottom: 30px;
        }

        /* Center Streamlit input widgets */
        div[data-testid="stTextInput"] {
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
        }

        div[data-testid="stButton"] {
            max-width: 500px;
            margin-left: auto;
            margin-right: auto;
        }

        div[data-testid="stTextInput"] input {
            height: 45px;
            border-radius: 8px;
        }

        div[data-testid="stButton"] button {
            height: 45px;
            border-radius: 8px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="login-wrapper">
            <div class="login-title">🔐 Admin Login</div>
            <div class="login-subtitle">
                Please log in to access the AI Attendance System.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    username = st.text_input(
        "Username",
        key="login_username",
    )

    password = st.text_input(
        "Password",
        type="password",
        key="login_password",
    )

    if st.button("Login", type="primary", use_container_width=True):
        if not username or not password:
            st.warning("Please enter username and password.")
        elif authenticate(username, password):
            st.session_state["is_authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.stop()

# --- NAVIGATION & LOGOUT ---
# --- MODERN SIDEBAR NAVIGATION ---

st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #171923 0%, #0f1015 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] > div {
    padding: 1.3rem 1rem;
}

.sidebar-brand {
    padding: 5px 8px 18px 8px;
}

.sidebar-title {
    font-size: 21px;
    font-weight: 700;
    color: white;
}

.sidebar-subtitle {
    font-size: 12px;
    color: #8f94a3;
    margin-top: 3px;
}

.sidebar-user {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 20px;
}

.sidebar-user-label {
    font-size: 11px;
    color: #858a99;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

.sidebar-user-name {
    font-size: 15px;
    font-weight: 600;
    color: white;
    margin-top: 4px;
}

.sidebar-heading {
    font-size: 11px;
    color: #858a99;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 10px 8px 8px;
}

div[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 3px;
}

div[data-testid="stSidebar"] div[role="radiogroup"] > label {
    border-radius: 9px;
    padding: 8px 10px;
    transition: 0.2s;
}

div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(255,255,255,0.07);
}

div[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 14px;
    color: #d8dbe5;
}

div[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none;
}

.sidebar-divider {
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 20px 0 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------- BRAND ----------

st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-title">🎓 AI Attendance</div>
    <div class="sidebar-subtitle">Student Management System</div>
</div>
""", unsafe_allow_html=True)


# ---------- USER ----------

username = st.session_state.get("username", "admin")

st.sidebar.markdown(f"""
<div class="sidebar-user">
    <div class="sidebar-user-label">Logged in as</div>
    <div class="sidebar-user-name">👤 {username}</div>
</div>
""", unsafe_allow_html=True)


# ---------- NAVIGATION ----------

st.sidebar.markdown(
    '<div class="sidebar-heading">Main Navigation</div>',
    unsafe_allow_html=True
)

navigation = {
    "🏠  Dashboard": "Dashboard",
    "👨‍🎓  Register Student": "Register Student",
    "👥  Students": "Students",
    "📷  Capture Face": "Capture Face",
    "🗂️  Face Data": "Face Data",
    "👁️  Live Recognition": "Live Recognition",
    "✅  Live Attendance": "Live Attendance",
    "📋  Attendance Records": "Attendance Records",
    "📊  Reports": "Reports",
    "⚙️  Settings": "Settings",
}

selected_label = st.sidebar.radio(
    "Go to",
    list(navigation.keys()),
    label_visibility="collapsed",
)

# Keep the original page names for your existing routing
page = navigation[selected_label]


# ---------- LOGOUT ----------

st.sidebar.markdown(
    '<div class="sidebar-divider"></div>',
    unsafe_allow_html=True
)

if st.sidebar.button("🚪  Logout", use_container_width=True):
    st.session_state["is_authenticated"] = False
    st.session_state["username"] = ""
    st.rerun()

# --- DASHBOARD STYLING ---

st.markdown("""
<style>

/* Dashboard header */
.dashboard-header {
    padding: 8px 0 20px 0;
}

.dashboard-title {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 4px;
}

.dashboard-subtitle {
    color: #94a3b8;
    font-size: 15px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #1e293b, #111827);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px;
    min-height: 125px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.20);
}

.metric-label {
    color: #94a3b8;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 10px;
}

.metric-value {
    color: #f8fafc;
    font-size: 30px;
    font-weight: 700;
}

.metric-icon {
    font-size: 22px;
    margin-bottom: 8px;
}

/* Section headings */
.dashboard-section {
    font-size: 21px;
    font-weight: 650;
    margin-top: 28px;
    margin-bottom: 12px;
}

/* Chart containers */
.chart-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 8px 12px 4px 12px;
}

/* Hide Streamlit decoration */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- CUSTOM UI STYLE ----------------

# ---------------- DASHBOARD STYLES ----------------

st.markdown(
    """
    <style>

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .dashboard-header {
        padding: 8px 0 20px 0;
    }

    .dashboard-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .dashboard-subtitle {
        color: #94a3b8;
        font-size: 15px;
    }

    .metric-card {
        background: linear-gradient(145deg, #1e293b, #111827);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.20);
    }

    .metric-label {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 30px;
        font-weight: 700;
    }

    .metric-icon {
        font-size: 22px;
        margin-bottom: 8px;
    }

    .dashboard-section {
        font-size: 21px;
        font-weight: 650;
        margin-top: 28px;
        margin-bottom: 12px;
    }

    .chart-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 8px 12px 4px 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


def show_dashboard() -> None:
    settings = get_settings()

    # ---------------- HEADER ----------------

    st.markdown("""
    <div class="dashboard-header">
        <div class="dashboard-title">Dashboard</div>
        <div class="dashboard-subtitle">
            Welcome back, Admin 👋 &nbsp; | &nbsp;
            Here's today's attendance overview.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- DATE ----------------

    selected_date = st.date_input(
        "Dashboard date",
        key="dashboard_date"
    )

    metrics = get_dashboard_metrics(selected_date)

    # ---------------- METRIC CARDS ----------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">👨‍🎓</div>
            <div class="metric-label">TOTAL STUDENTS</div>
            <div class="metric-value">
                {metrics["total_students"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">✅</div>
            <div class="metric-label">PRESENT TODAY</div>
            <div class="metric-value">
                {metrics["present_today"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">❌</div>
            <div class="metric-label">ABSENT TODAY</div>
            <div class="metric-value">
                {metrics["absent_today"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📊</div>
            <div class="metric-label">ATTENDANCE RATE</div>
            <div class="metric-value">
                {metrics["attendance_percentage"]}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- EMPTY DATABASE ----------------

    if metrics["total_students"] == 0:
        st.info(
            "Register students to begin building attendance analytics."
        )
        return

    # ---------------- DATA ----------------

    distribution_data = pd.DataFrame({
        "Status": ["Present", "Absent"],
        "Students": [
            metrics["present_today"],
            metrics["absent_today"]
        ]
    })

    daily_trend = pd.DataFrame(
        get_daily_attendance_trend()
    )

    student_percentages = pd.DataFrame(
        get_student_attendance_percentages()
    )

    monthly_trend = pd.DataFrame(
        get_monthly_attendance_trend()
    )

    # ---------------- SECTION ----------------

    st.markdown(
        '<div class="dashboard-section">Attendance Overview</div>',
        unsafe_allow_html=True
    )

    # ---------------- FIRST ROW ----------------

    left, right = st.columns(2)

    with left:

        st.markdown(
            '<div class="chart-card">',
            unsafe_allow_html=True
        )

        st.subheader("Present vs Absent")

        fig = px.pie(
            distribution_data,
            names="Status",
            values="Students",
            color="Status",
            color_discrete_map={
                "Present": "#16a34a",
                "Absent": "#dc2626"
            },
            hole=0.55
        )

        fig.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc")
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            '<div class="chart-card">',
            unsafe_allow_html=True
        )

        st.subheader("Daily Attendance Trend")

        if daily_trend.empty:

            st.info(
                "No attendance trend is available yet."
            )

        else:

            fig = px.line(
                daily_trend,
                x="date",
                y="present_count",
                markers=True,
                labels={
                    "date": "Date",
                    "present_count": "Present Students"
                }
            )

            fig.update_layout(
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc")
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False}
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # ---------------- SECOND ROW ----------------

    st.markdown(
        '<div class="dashboard-section">Attendance Analytics</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
            '<div class="chart-card">',
            unsafe_allow_html=True
        )

        st.subheader("Student-wise Attendance Percentage")

        if student_percentages.empty:

            st.info(
                "No student attendance data is available yet."
            )

        else:

            fig = px.bar(
                student_percentages,
                x="name",
                y="attendance_percentage",
                color="attendance_percentage",
                color_continuous_scale="Blues",
                labels={
                    "name": "Student",
                    "attendance_percentage": "Attendance %"
                }
            )

            fig.update_layout(
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc")
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False}
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            '<div class="chart-card">',
            unsafe_allow_html=True
        )

        st.subheader("Monthly Attendance Trend")

        if monthly_trend.empty:

            st.info(
                "No monthly attendance trend is available yet."
            )

        else:

            fig = px.line(
                monthly_trend,
                x="month",
                y="present_count",
                markers=True,
                labels={
                    "month": "Month",
                    "present_count": "Present Students"
                }
            )

            fig.update_layout(
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc")
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False}
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


def show_registration() -> None:
    st.title("Register Student")
    st.caption(
        "Save the student profile first. Face samples will be captured in Stage 3.")

    with st.form("student_registration_form", clear_on_submit=True):
        left, right = st.columns(2)
        with left:
            student_id = st.text_input("Student ID *", placeholder="STU001")
            roll_number = st.text_input(
                "Roll Number *", placeholder="2026A001")
            name = st.text_input("Student Name *", placeholder="Full name")
            email = st.text_input("Email (optional)",
                                  placeholder="student@example.com")
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
    search_text = st.text_input(
        "Search by name, ID, roll number, class, division, or department")
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

    st.divider()
    st.subheader("Delete Student Profile")
    st.warning("This action permanently removes the selected student, their saved face samples, and their entire attendance history.")

    options = {
        f"{student['name']} — {student['roll_number']} ({student['student_id']})": student["student_id"]
        for student in students
    }

    def reset_deletion_confirmation() -> None:
        st.session_state["confirm_student_deletion"] = False

    selected_label = st.selectbox(
        "Select a student to delete",
        list(options),
        key="student_delete_select",
        on_change=reset_deletion_confirmation,
    )

    confirmed = st.checkbox(
        "I confirm that I want to delete this student and all associated data permanently.",
        key="confirm_student_deletion",
    )

    if st.button("Delete student", type="primary", disabled=not confirmed):
        success, message = delete_student(options[selected_label])
        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)


def show_face_data() -> None:
    st.title("Face Data")
    st.caption(
        "Face samples are stored only on this computer for this college prototype.")
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
    st.caption(
        f"Each student needs {MIN_FACE_SAMPLES} face samples before recognition training.")

    students_with_face_data = [row for row in rows if row["sample_count"] > 0]
    if students_with_face_data:
        st.divider()
        st.subheader("Delete face data")
        st.warning(
            "This permanently removes the selected student's saved face photos. Their student profile remains.")
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


def show_capture_face() -> None:
    st.title("Capture Face")
    st.caption(
        "Stage 3: capture three face samples for a registered student."
    )

    students = get_student_options()

    if not students:
        st.warning("No students are registered yet. Register a student first.")
        return

    options = {
        f"{student['name']} — {student['roll_number']} ({student['student_id']})":
        student["student_id"]
        for student in students
    }

    selected_label = st.selectbox(
        "Select Student",
        list(options.keys()),
    )
    student_id = options[selected_label]

    current_count = count_face_samples(student_id)

    st.info(
        f"Current face samples: {current_count}/{MAX_FACE_SAMPLES}"
    )

    if current_count >= MAX_FACE_SAMPLES:
        st.success(
            "This student already has the maximum number of face samples."
        )
        st.info("Go to Live Recognition and retrain the recognition model.")
        return

    camera_image = st.camera_input(
        "Take a face sample"
    )

    if camera_image is not None:
        if st.button("Save Face Sample", type="primary"):
            success, message, quality_score = assess_and_save_face(
                camera_image.getvalue(),
                student_id,
            )

            if success:
                st.success(message)

                if quality_score is not None:
                    st.caption(
                        f"Image quality score: {quality_score}"
                    )

                st.rerun()
            else:
                st.error(message)


def show_live_recognition() -> None:
    st.title("Live Face Recognition")
    st.caption(
        "Stage 4: identify known faces with the laptop webcam. Attendance is not recorded yet.")
    st.info(
        "This uses local LBPH face recognition. It compares live face patterns with the three "
        "saved samples; no face data is sent online."
    )

    # Fetch dynamic settings
    settings = get_settings()
    camera_index = int(settings.get("camera_index", 0))
    max_rec_distance = float(settings.get("max_recognition_distance", 60.0))

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

    duration = st.slider("Live camera duration (seconds)",
                         min_value=5, max_value=30, value=15)
    if st.button("Start live recognition"):
        recognizer, label_map, error = load_recognizer()
        if error:
            st.error(error)
            return

        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            st.error(
                "Webcam unavailable. Check your settings camera index, check permissions, and close other apps.")
            return

        camera_view = st.empty()
        status_box = st.empty()
        recognized_names: set[str] = set()
        end_time = time.time() + duration
        try:
            while time.time() < end_time:
                success, frame = camera.read()
                if not success:
                    st.error(
                        "A camera frame could not be read. Try reconnecting the webcam.")
                    break

                # Pass the dynamically loaded max_rec_distance
                results = recognize_faces(
                    frame, recognizer, label_map, max_distance=max_rec_distance)

                for result in results:
                    x, y, width, height = result["box"]
                    color = (0, 180, 0) if result["result"] == "Recognized" else (
                        0, 165, 255)
                    if result["result"] == "Unknown":
                        color = (0, 0, 255)
                    label = f"{result['name']} | {result['similarity']}%"
                    cv2.rectangle(
                        frame, (x, y), (x + width, y + height), color, 2)
                    cv2.putText(frame, label, (x, max(y - 10, 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if result["result"] == "Recognized":
                        recognized_names.add(
                            f"{result['name']} ({result['roll_number']})")
                camera_view.image(cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                status_box.caption(
                    f"Scanning... {max(0, int(end_time - time.time()))} second(s) remaining.")
        except RuntimeError as error:
            st.error(str(error))
        finally:
            camera.release()

        if recognized_names:
            st.success("Recognized during this session: " +
                       ", ".join(sorted(recognized_names)))
        else:
            st.warning(
                "No high-confidence registered face was recognized during this session.")
        st.info(
            "Live recognition session complete. Stage 5 will add automatic attendance marking.")


def calculate_distance_meters(
    latitude1: float,
    longitude1: float,
    latitude2: float,
    longitude2: float,
) -> float:
    """Calculate the approximate distance between two GPS coordinates."""
    earth_radius = 6_371_000

    lat1 = math.radians(latitude1)
    lat2 = math.radians(latitude2)
    delta_lat = math.radians(latitude2 - latitude1)
    delta_lon = math.radians(longitude2 - longitude1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    a = max(0.0, min(1.0, a))
    return 2 * earth_radius * math.asin(math.sqrt(a))


def show_live_attendance() -> None:
    st.title("Live Attendance")
    st.caption(
        "Recognized students are marked Present automatically, once per day."
    )
    st.warning(
        "Only green, high-confidence recognitions are marked. "
        "Orange uncertain and red unknown faces are not recorded."
    )

    # Fetch dynamic settings
    settings = get_settings()
    camera_index = int(settings.get("camera_index", 0))
    max_rec_distance = float(
        settings.get("max_recognition_distance", 60.0)
    )

    duration = st.slider(
        "Attendance camera duration (seconds)",
        min_value=5,
        max_value=30,
        value=15,
        key="attendance_duration",
    )

    # Verify the laptop's reported location against the saved campus location
    location = streamlit_geolocation()
    location_verified = False
    location_message = "Location has not been verified."

    campus_latitude = settings.get("campus_latitude", "").strip()
    campus_longitude = settings.get("campus_longitude", "").strip()
    campus_radius = float(settings.get("campus_radius_meters", 150))

    if not campus_latitude or not campus_longitude:
        location_message = "Campus coordinates are not configured."
    elif location and location.get("latitude") is not None:
        try:
            current_latitude = float(location["latitude"])
            current_longitude = float(location["longitude"])
            accuracy = float(location.get("accuracy") or 0)

            distance = calculate_distance_meters(
                current_latitude,
                current_longitude,
                float(campus_latitude),
                float(campus_longitude),
            )

            # Conservatively account for reported location uncertainty
            if accuracy > 0 and distance + accuracy <= campus_radius:
                location_verified = True
                location_message = (
                    f"Location verified: approximately {distance:.0f} m "
                    f"from campus; reported accuracy ±{accuracy:.0f} m."
                )
            else:
                location_message = (
                    f"Location not verified. Estimated distance: "
                    f"{distance:.0f} m; reported accuracy: "
                    f"{accuracy:.0f} m; allowed radius: "
                    f"{campus_radius:.0f} m."
                )
        except (TypeError, ValueError):
            location_message = "Invalid location data. Try again."
    else:
        location_message = (
            "Location unavailable. Click the location button above "
            "and allow browser location permission."
        )

    if location_verified:
        st.success(location_message)
    else:
        st.warning(location_message)

    # Campus map using Folium
    if campus_latitude and campus_longitude:
        campus_lat = float(campus_latitude)
        campus_lon = float(campus_longitude)

        st.subheader("Campus Zone and Laptop Location")

        campus_location = [campus_lat, campus_lon]
        map_center = campus_location
        map_zoom = 16

        laptop_location = None

        if location and location.get("latitude") is not None:
            laptop_location = [
                float(location["latitude"]),
                float(location["longitude"]),
            ]

            distance_between = calculate_distance_meters(
                campus_lat,
                campus_lon,
                laptop_location[0],
                laptop_location[1],
            )

            if distance_between > 2000:
                map_center = [
                    (campus_lat + laptop_location[0]) / 2,
                    (campus_lon + laptop_location[1]) / 2,
                ]
                map_zoom = 5

        campus_map = folium.Map(
            location=map_center,
            zoom_start=map_zoom,
            tiles="OpenStreetMap",
        )

        MeasureControl(
            position="topleft",
            primary_length_unit="meters",
            secondary_length_unit="kilometers",
        ).add_to(campus_map)

        folium.plugins.MeasureControl(
            position="topleft",
            primary_length_unit="meters",
            secondary_length_unit="kilometers",
        ).add_to(campus_map)

        # Blue circle: 400 m campus zone
        folium.Circle(
            location=campus_location,
            radius=float(campus_radius),
            color="blue",
            weight=3,
            fill=True,
            fill_color="blue",
            fill_opacity=0.08,
            tooltip=f"Campus boundary: {campus_radius:.0f} m",
        ).add_to(campus_map)

        # College marker
        folium.Marker(
            location=campus_location,
            tooltip="College",
            icon=folium.Icon(color="blue", icon="education", prefix="fa"),
        ).add_to(campus_map)

        # Laptop marker only; no radius around laptop
        if laptop_location is not None:
            folium.Marker(
                location=laptop_location,
                tooltip="Laptop current location",
                icon=folium.Icon(color="green", icon="laptop", prefix="fa"),
            ).add_to(campus_map)

        st_folium(
            campus_map,
            height=500,
            use_container_width=True,
            returned_objects=[],
            key="campus_location_map",
        )

        st.caption(
            f"Blue circle: campus boundary ({campus_radius:.0f} m). "
            "Blue marker: college. Green marker: laptop."
        )

    if st.button("Start attendance camera", type="primary"):
        if not location_verified:
            st.error(
                "Attendance blocked: verify the campus location first."
            )
            return

        recognizer, label_map, error = load_recognizer()

        if error:
            st.error(error)
            return

        camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        if not camera.isOpened():
            camera.release()
            st.error(
                "Webcam unavailable. Check your settings camera index, "
                "check permissions, and close other apps."
            )
            return

        camera_view = st.empty()
        status_box = st.empty()
        result_box = st.empty()

        handled_student_ids: set[str] = set()
        session_messages: list[str] = []

        # Require repeated recognition before marking attendance
        current_student_id = None
        recognition_streak = 0
        REQUIRED_STREAK = 5

        end_time = time.time() + duration

        try:
            while time.time() < end_time:
                success, frame = camera.read()

                if not success:
                    st.error(
                        "A camera frame could not be read. "
                        "Try reconnecting the webcam."
                    )
                    break

                results = recognize_faces(
                    frame,
                    recognizer,
                    label_map,
                    max_distance=max_rec_distance,
                )

                for result in results:
                    x, y, width, height = result["box"]

                    color = (
                        (0, 180, 0)
                        if result["result"] == "Recognized"
                        else (0, 165, 255)
                    )

                    if result["result"] == "Unknown":
                        color = (0, 0, 255)

                    label = f"{result['name']} | {result['similarity']}%"

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + width, y + height),
                        color,
                        2,
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x, max(y - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                    student_id = result["student_id"]

                    if result["result"] == "Recognized" and student_id:

                        if student_id == current_student_id:
                            recognition_streak += 1
                        else:
                            current_student_id = student_id
                            recognition_streak = 1

                        # Mark attendance only after repeated recognition
                        if (
                            recognition_streak >= REQUIRED_STREAK
                            and student_id not in handled_student_ids
                        ):
                            handled_student_ids.add(student_id)

                            created, message = mark_attendance(
                                student_id,
                                result["similarity"],
                            )

                            session_messages.append(message)

                    else:
                        # Unknown or uncertain face breaks the streak
                        current_student_id = None
                        recognition_streak = 0

                camera_view.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )

                status_box.caption(
                    f"Attendance scanning... "
                    f"{max(0, int(end_time - time.time()))} "
                    f"second(s) remaining."
                )

                if session_messages:
                    result_box.info(
                        "Latest result: " + session_messages[-1]
                    )

        except RuntimeError as error:
            st.error(str(error))

        finally:
            camera.release()

        if session_messages:
            st.success("Attendance session complete.")

            for message in session_messages:
                st.write(f"- {message}")
        else:
            st.warning(
                "No high-confidence registered face was detected; "
                "no attendance was marked."
            )

    st.title("Attendance Reports")
    st.caption(
        "Filter attendance records, identify absences, and download reports.")
    options = get_filter_options()
    if not options["students"]:
        st.info("Register students before creating reports.")
        return

    # Fetch dynamic low attendance threshold
    settings = get_settings()
    default_low_threshold = float(
        settings.get("low_attendance_threshold", 75.0))

    with st.expander("Report filters", expanded=True):
        first, second, third = st.columns(3)
        with first:
            use_date = st.checkbox("Filter by date", value=False)
            selected_date = st.date_input("Attendance date")
            class_name = st.selectbox("Class", ["All"] + options["classes"])
        with second:
            division = st.selectbox("Division", ["All"] + options["divisions"])
            department = st.selectbox(
                "Department", ["All"] + options["departments"])
        with third:
            student_labels = {"All": None}
            student_labels.update(
                {f"{student['name']} — {student['roll_number']}": student["student_id"]
                    for student in options["students"]}
            )
            selected_student_label = st.selectbox(
                "Student", list(student_labels))

            # Using configured value as default
            low_threshold = st.number_input(
                "Low attendance threshold (%)", min_value=0.0, max_value=100.0,
                value=default_low_threshold, step=5.0,
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
                "Download CSV", to_csv_bytes(
                    report_table), "attendance_report.csv", "text/csv"
            )
        with download_right:
            st.download_button(
                "Download Excel", to_excel_bytes(
                    report_table), "attendance_report.xlsx",
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
        st.dataframe(absent_students,
                     use_container_width=True, hide_index=True)
        st.caption(f"Absent students: {len(absent_students)}")

    summary = pd.DataFrame(get_student_attendance_summary(class_filter, division_filter, department_filter)).rename(
        columns={"student_id": "Student ID", "roll_number": "Roll Number", "name": "Student Name",
                 "class_name": "Class", "present_days": "Present Days", "total_working_days": "Working Days",
                 "attendance_percentage": "Attendance %"}
    )
    st.subheader("Low attendance")
    if summary.empty or summary["Working Days"].max() == 0:
        st.info(
            "Low-attendance analysis becomes available after attendance is recorded on at least one date.")
    else:
        low_attendance = summary[summary["Attendance %"] < low_threshold]
        if low_attendance.empty:
            st.success(
                f"No students are below the {low_threshold}% threshold.")
        else:
            st.dataframe(low_attendance, use_container_width=True,
                         hide_index=True)


def show_attendance_records() -> None:
    st.title("Attendance Records")

    records = get_attendance_records()

    if not records:
        st.info(
            "No attendance has been recorded yet. "
            "Use Live Attendance to mark students present."
        )
        return

    table = pd.DataFrame(records).rename(
        columns={
            "attendance_id": "Attendance ID",
            "student_id": "Student ID",
            "roll_number": "Roll Number",
            "name": "Student Name",
            "class_name": "Class",
            "attendance_date": "Date",
            "attendance_time": "Time",
            "status": "Status",
            "confidence": "Recognition Confidence",
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Showing {len(records)} most recent attendance record(s)."
    )


def show_reports() -> None:
    st.title("Attendance Reports")
    st.caption(
        "Filter attendance records, identify absences, and download reports."
    )

    options = get_filter_options()

    if not options["students"]:
        st.info("Register students before creating reports.")
        return

    # Fetch dynamic low attendance threshold
    settings = get_settings()
    default_low_threshold = float(
        settings.get("low_attendance_threshold", 75.0)
    )

    with st.expander("Report filters", expanded=True):
        first, second, third = st.columns(3)

        with first:
            use_date = st.checkbox("Filter by date", value=False)
            selected_date = st.date_input("Attendance date")
            class_name = st.selectbox(
                "Class",
                ["All"] + options["classes"],
            )

        with second:
            division = st.selectbox(
                "Division",
                ["All"] + options["divisions"],
            )

            department = st.selectbox(
                "Department",
                ["All"] + options["departments"],
            )

        with third:
            student_labels = {"All": None}

            student_labels.update(
                {
                    f"{student['name']} — {student['roll_number']}":
                    student["student_id"]
                    for student in options["students"]
                }
            )

            selected_student_label = st.selectbox(
                "Student",
                list(student_labels),
            )

            low_threshold = st.number_input(
                "Low attendance threshold (%)",
                min_value=0.0,
                max_value=100.0,
                value=default_low_threshold,
                step=5.0,
            )

    selected_date_text = (
        selected_date.isoformat()
        if use_date
        else None
    )

    selected_student = student_labels[selected_student_label]

    class_filter = (
        None
        if class_name == "All"
        else class_name
    )

    division_filter = (
        None
        if division == "All"
        else division
    )

    department_filter = (
        None
        if department == "All"
        else department
    )

    records = get_filtered_attendance(
        selected_date_text,
        selected_student,
        class_filter,
        division_filter,
        department_filter,
    )

    report_table = pd.DataFrame(records).rename(
        columns={
            "student_id": "Student ID",
            "roll_number": "Roll Number",
            "name": "Student Name",
            "class_name": "Class",
            "division": "Division",
            "department": "Department",
            "attendance_date": "Date",
            "attendance_time": "Time",
            "status": "Status",
            "confidence": "Recognition Confidence",
        }
    )

    st.subheader("Filtered attendance records")

    if report_table.empty:
        st.info("No attendance records match these filters.")
    else:
        st.dataframe(
            report_table,
            use_container_width=True,
            hide_index=True,
        )

        download_left, download_right = st.columns(2)

        with download_left:
            st.download_button(
                "Download CSV",
                to_csv_bytes(report_table),
                "attendance_report.csv",
                "text/csv",
            )

        with download_right:
            st.download_button(
                "Download Excel",
                to_excel_bytes(report_table),
                "attendance_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    st.divider()

    absent_date = selected_date.isoformat()

    absent_students = pd.DataFrame(
        get_absent_students(
            absent_date,
            class_filter,
            division_filter,
            department_filter,
        )
    ).rename(
        columns={
            "student_id": "Student ID",
            "roll_number": "Roll Number",
            "name": "Student Name",
            "class_name": "Class",
            "division": "Division",
            "department": "Department",
        }
    )

    st.subheader(
        f"Absent students — {absent_date}"
    )

    if absent_students.empty:
        st.success(
            "No absent students for the selected scope."
        )
    else:
        st.dataframe(
            absent_students,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"Absent students: {len(absent_students)}"
        )

    summary = pd.DataFrame(
        get_student_attendance_summary(
            class_filter,
            division_filter,
            department_filter,
        )
    ).rename(
        columns={
            "student_id": "Student ID",
            "roll_number": "Roll Number",
            "name": "Student Name",
            "class_name": "Class",
            "present_days": "Present Days",
            "total_working_days": "Working Days",
            "attendance_percentage": "Attendance %",
        }
    )

    st.subheader("Low attendance")

    if summary.empty or summary["Working Days"].max() == 0:
        st.info(
            "Low-attendance analysis becomes available after "
            "attendance is recorded on at least one date."
        )
    else:
        low_attendance = summary[
            summary["Attendance %"] < low_threshold
        ]

        if low_attendance.empty:
            st.success(
                f"No students are below the "
                f"{low_threshold}% threshold."
            )
        else:
            st.dataframe(
                low_attendance,
                use_container_width=True,
                hide_index=True,
            )


def show_settings() -> None:
    st.title("System Settings")
    settings = get_settings()

    st.subheader("Application Configuration")
    with st.form("settings_form"):
        app_name = st.text_input("Application Name", value=settings.get(
            "app_name", "AI-Based Student Attendance System"))
        camera_index = st.number_input("Camera Index", value=int(
            settings.get("camera_index", 0)), step=1)
        max_dist = st.number_input("Maximum Recognition Distance", value=float(
            settings.get("max_recognition_distance", 60.0)), step=1.0)

        st.caption("ℹ️ Note on Duplicate Attendance: The database strictly enforces one attendance record per student per day. This UI interval setting saves cleanly but does not currently override the database's strict protection rules.")
        duplicate_interval = st.number_input("Duplicate Attendance Interval (minutes)", value=int(
            settings.get("duplicate_interval_minutes", 1440)), step=1)

        low_threshold = st.number_input("Low Attendance Threshold (%)", value=float(
            settings.get("low_attendance_threshold", 75.0)), step=1.0, min_value=0.0, max_value=100.0)

        st.divider()
        st.subheader("Campus Location Verification")
        st.caption(
            "Attendance will require the laptop's reported location "
            "to be within the configured campus radius."
        )

        campus_latitude = st.text_input(
            "Campus Latitude",
            value=settings.get("campus_latitude", ""),
            placeholder="Example: 19.0760",
        )
        campus_longitude = st.text_input(
            "Campus Longitude",
            value=settings.get("campus_longitude", ""),
            placeholder="Example: 72.8777",
        )
        campus_radius = st.number_input(
            "Allowed Campus Radius (metres)",
            min_value=10.0,
            max_value=5000.0,
            value=float(settings.get("campus_radius_meters", 150)),
            step=10.0,
        )

        if st.form_submit_button("Save Settings", type="primary"):
            success, message = update_settings({
                "app_name": app_name,
                "camera_index": str(camera_index),
                "max_recognition_distance": str(max_dist),
                "duplicate_interval_minutes": str(duplicate_interval),
                "low_attendance_threshold": str(low_threshold),
                "campus_latitude": campus_latitude,
                "campus_longitude": campus_longitude,
                "campus_radius_meters": str(campus_radius),
            })
            if success:
                st.success(message)
                time.sleep(1)
                st.rerun()
            else:
                st.error(message)

    st.divider()

    st.subheader("Admin Account")
    with st.form("password_change_form", clear_on_submit=True):
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input(
            "Confirm New Password", type="password")

        if st.form_submit_button("Change Password"):
            if not current_password or not new_password or not confirm_password:
                st.error("Please fill in all password fields.")
            elif new_password != confirm_password:
                st.error("New password and confirmation do not match.")
            else:
                success, message = change_password(
                    st.session_state["username"], current_password, new_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)


# Explicit Page Routing
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
elif page == "Attendance Records":
    show_attendance_records()
elif page == "Reports":
    show_reports()
elif page == "Settings":
    show_settings()
