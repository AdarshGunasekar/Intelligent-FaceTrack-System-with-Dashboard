import sqlite3
from datetime import datetime
import os
from src.config import Config

config = Config()
DB_PATH = config.DB_PATH
LOG_DIR = config.LOG_DIR
EVENT_LOG_PATH = os.path.join(LOG_DIR, "events.log")

def init_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id TEXT,
            event_type TEXT,
            timestamp TEXT,
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

    # Init event log file
    with open(EVENT_LOG_PATH, "a") as f:
        f.write(f"\n[{datetime.now()}] Logger initialized\n")

def log_visit(face_id, event_type, image_path):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO visits (face_id, event_type, timestamp, image_path)
        VALUES (?, ?, ?, ?)
    ''', (face_id, event_type, datetime.now().isoformat(), image_path))
    conn.commit()
    conn.close()

    write_event(f"{event_type.upper()} | ID: {face_id} | Image: {image_path}")

def write_event(message):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "events.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:  # 🔥 force UTF-8
        f.write(f"[{timestamp}] {message}\n")

