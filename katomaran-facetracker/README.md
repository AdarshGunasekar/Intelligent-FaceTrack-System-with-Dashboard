# 🎯 Intelligent Face Tracker with Auto-Logging & Dashboard
<br>


This project is a real-time face tracking and visitor logging system using YOLOv8 for face detection, InsightFace for recognition, and ByteTrack for tracking. It logs all events (entry, re-detection, exit) with cropped images and timestamps into a structured SQLite database. A Streamlit dashboard enables easy monitoring and filtering.

---
<br>
<br>
<br>

## 🛠️ Setup Instructions

### 1. Clone the Repository

<pre>git clone https://github.com/your-username/katomaran-facetracker.git
cd katomaran-facetracker</pre>

### 2. Create & Activate Virtual Environment

<pre>python -m venv venv
venv\Scripts\activate  # For Windows</pre>

### 3. Install Requirements

<pre>pip install -r requirements.txt</pre>

### 4. Download InsightFace Models (auto on first run)

<pre>from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)</pre>
<br>
<br>
<br>



# 📚 Assumptions Made


Faces detected by YOLOv8 are assumed to be frontal and clear enough for InsightFace.

Face recognition is based on cosine similarity threshold from normed_embedding.

SQLite database is used and stores logs locally in data/visitor_log.db.

Each day has its own log/image folder: logs/entries/YYYY-MM-DD/.

CPU-only mode (CPUExecutionProvider) is used for deployment simplicity.
<br>
<br>
<br>


# 🧠 Application Architecture

 ## Architecture Diagram

![alt text](architecture.png)
<br>
<br>
<br>


# Sample config.json

{
  
    "DB_PATH": "data/visitor_log.db",
    "FACE_DB_PATH": "data/face_db.pkl",
    "SIMILARITY_THRESHOLD": 0.6,
    "DETECTION_SKIP_FRAMES": 5,
    "LOG_DIR": "logs",
    "YOLO_MODEL_PATH": "yolov8n-face.pt",
    "MIN_CONFIDENCE": 0.5

}
<br>
<br>
<br>


# Project Structure

```mermaid
graph TD
    A[KATOMARAN-FACETRACKER] --> B[ByteTrack]
    A --> C[data/]
    C --> C1(face_db.pkl)
    C --> C2(visitor_log.db)
    A --> D[logs/]
    D --> D1[entries/]
    D1 --> D11(2025-06-22)
    D1 --> D12(2025-06-23)
    D --> D2(events.log)
    A --> E[src/]
    E --> E1(config.py)
    E --> E2(detector.py)
    E --> E3(face_database.py)
    E --> E4(logger.py)
```

<br>
<br>
<br>
# Streamlit Dashboard

## Run the dashboard with:

<pre>streamlit run dashboard.py</pre>


### Filter logs by date or event type

### Visualize event counts with bar chart

### View cropped face images and timestamps

### Launch face tracker from UI
<br>
<br>
<br>


# Demo Video

<pre>https://www.loom.com/share/132b5701196346c3a84df327ff99220e?sid=56585562-984c-49a0-9708-d5ed4afaec59</pre>

<br>
<br>
<br>




# Sample Output

Database (SQLite): data/visitor_log.db

Table: visits(face_id, event_type, timestamp, image_path)

Face Crops: Saved under logs/entries/YYYY-MM-DD/

Log File: Plain text logs in logs/events.log
<br>
<br>
<br>



# How to Run

1.Start Face Tracker

<pre>python insig_bt.py</pre>

2.Launch Dashboard

<pre>streamlit run dashboard.py</pre>
<br>
<br>
<br>



# Key Features

🔍 YOLOv8 for robust face detection

🧠 InsightFace for fast face recognition

🧾 Automatic event logging to SQLite

📷 Cropped face images stored and visualized

📊 Dashboard filters and bar chart

🧠 Redetection + exit logic using ByteTrack

<br>
<br>
<br>


# 📌 Final Note

This project is a part of a hackathon run by https://katomaran.com