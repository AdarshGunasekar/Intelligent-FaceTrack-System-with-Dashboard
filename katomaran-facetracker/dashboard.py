import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime
import subprocess
import sys

# -----------------------------
# CONFIGS
# -----------------------------
DB_PATH = "data/visitor_log.db"
INSIG_SCRIPT = "insig_bt.py"
VENV_PYTHON = os.path.join(os.getcwd(), "venv", "Scripts", "python.exe")  # Windows path

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Face Tracker Dashboard", layout="wide")
st.title("📊 Face Tracker Dashboard")

# -----------------------------
# CAMERA CONTROL
# -----------------------------
st.subheader("🎥 Camera Control")
cam_col1, cam_col2,cam_col3 = st.columns(3)

camera_status = st.session_state.get("camera_running", False)

with cam_col1:
    if st.button("▶️ Start Camera and Track Faces"):
        st.session_state["camera_running"] = True
        with st.spinner("Running face tracker..."):
            result = subprocess.run([VENV_PYTHON, INSIG_SCRIPT], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Face tracking completed. Logs updated.")
            else:
                st.error("Face tracking script failed to run.")
                st.code(result.stderr or "Unknown error")

with cam_col2:
    if st.button("⏹ Stop Camera"):
        st.session_state["camera_running"] = False
        st.warning("Camera tracking manually stopped.")

# -----------------------------
# REFRESH CONTROL
# -----------------------------
with cam_col3:
    if st.button("🔄 Refresh Dashboard Data"):
        st.rerun()


st.markdown("---")

# -----------------------------
# RTSP CONTROL
# -----------------------------

st.subheader("📡 RTSP Stream Input")

with st.form("rtsp_form"):
    rtsp_url = st.text_input("Enter RTSP Stream URL (e.g., rtsp://...)", "")
    submitted = st.form_submit_button("▶️ Start RTSP Face Tracking")

    if submitted and rtsp_url:
        st.session_state["camera_running"] = True
        with st.spinner("Running face tracker on RTSP stream..."):
            result = subprocess.run([VENV_PYTHON, INSIG_SCRIPT, rtsp_url], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("RTSP face tracking completed. Logs updated.")
            else:
                st.error("Face tracking script failed to run.")
                st.code(result.stderr or "Unknown error")

# Stop RTSP tracking manually
if st.session_state.get("camera_running", False):
    if st.button("⏹ Stop RTSP Stream"):
        st.session_state["camera_running"] = False
        st.warning("RTSP tracking manually stopped.")


st.markdown("---")



# -----------------------------
# VIDEO FILE UPLOAD & TRACKING
# -----------------------------
st.subheader("📼 Upload Video for Face Tracking")
video_file = st.file_uploader("Upload a .mp4 file", type=["mp4"])

if video_file is not None:
    video_path = os.path.join("data", "uploaded_video.mp4")
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.success("Video uploaded successfully!")

    vid_col1, vid_col2 = st.columns(2)

    with vid_col1:
        if st.button("▶️ Run Face Tracker on Uploaded Video"):
            st.session_state["video_tracking"] = True
            with st.spinner("Processing uploaded video..."):
                result = subprocess.run([VENV_PYTHON, INSIG_SCRIPT, video_path], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Face tracking on video completed. Logs updated.")
                else:
                    st.error("Tracking script failed on uploaded video.")
                    st.code(result.stderr or "Unknown error")
            st.session_state["video_tracking"] = False

    with vid_col2:
        if st.button("⏹ Stop Video Tracker"):
            st.session_state["video_tracking"] = False
            st.warning("Video tracking manually stopped. You may re-run to resume.")



st.markdown("---")

# -----------------------------
# LOAD DATA FROM DB
# -----------------------------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

try:
    df = pd.read_sql_query("SELECT face_id, event_type, timestamp, image_path FROM visits", conn)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)

    # -----------------------------
    # METRICS
    # -----------------------------
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Unique Visitors", df['face_id'].nunique())
    with col2:
        st.metric("Total Events Logged", len(df))

    st.markdown("---")

    # -----------------------------
    # RECENT VISITS
    # -----------------------------
    st.subheader("🧾 Recent Visit Logs")
    recent_df = df.sort_values(by="timestamp", ascending=False).head(10)

    for _, row in recent_df.iterrows():
        with st.container():
            c1, c2 = st.columns([1, 4])
            with c1:
                if os.path.exists(row['image_path']):
                    st.image(row['image_path'], width=100)
                else:
                    st.write("Image not found.")
            with c2:
                st.markdown(f"**Visitor ID**: `{row['face_id']}`")
                st.markdown(f"**Event**: `{row['event_type']}`")
                st.markdown(f"**Timestamp**: `{row['timestamp']}`")

    st.markdown("---")

    # -----------------------------
    # FILTERS AND VISUALS
    # -----------------------------
    with st.expander("🔍 Filter & Visualize Logs"):
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()

        if pd.notna(min_date) and pd.notna(max_date):
            start_date = st.date_input("Start Date", value=min_date.date(), min_value=min_date.date(), max_value=max_date.date())
            end_date = st.date_input("End Date", value=max_date.date(), min_value=min_date.date(), max_value=max_date.date())
        else:
            st.warning("No valid timestamps to filter.")
            start_date = end_date = datetime.today().date()

        filtered_df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]

        selected_events = st.multiselect("Filter by Event Type", df['event_type'].unique(), default=list(df['event_type'].unique()))
        filtered_df = filtered_df[filtered_df['event_type'].isin(selected_events)]

        st.write(f"Showing {len(filtered_df)} logs between {start_date} and {end_date}.")
        st.dataframe(filtered_df)

        # -----------------------------
        # PLOT
        # -----------------------------
        st.subheader("📈 Event Count Over Time")
        plot_df = filtered_df.copy()
        plot_df['date'] = plot_df['timestamp'].dt.date
        plot_counts = plot_df.groupby(['date', 'event_type']).size().unstack(fill_value=0)
        st.bar_chart(plot_counts)

except Exception as e:
    st.error(f"Error loading from database: {e}")

conn.close()








