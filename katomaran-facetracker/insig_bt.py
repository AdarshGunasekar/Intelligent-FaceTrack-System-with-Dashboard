import numpy as np
np.float = float  # Patch for ByteTrack compatibility

import cv2
import os
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis

from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

from src.face_database import FaceDatabase
from src.logger import init_logger, log_visit, write_event
from src.config import Config
import sys

# Load config
config = Config()

# Config params
CONF_THRESHOLD = config.MIN_CONFIDENCE
SKIP_FRAMES = config.DETECTION_SKIP_FRAMES
YOLO_MODEL_PATH = config.YOLO_MODEL_PATH
LOG_DIR = config.LOG_DIR
VIDEO_SOURCE = sys.argv[1] if len(sys.argv) > 1 else 0
MARGIN = 45 # pixels

# Init
print("[INFO] Initializing YOLOv8 face detector...")
write_event("YOLOv8 face detector initialized")
yolo_model = YOLO(YOLO_MODEL_PATH)

print("[INFO] Initializing InsightFace recognizer...")
write_event("InsightFace recognizer initialized")
insightface_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
insightface_app.prepare(ctx_id=0)

print("[INFO] Initializing Face Database + Logger...")
face_db = FaceDatabase()
init_logger()
write_event("Face database and logger initialized")

# Prepare log directory
today = datetime.now().strftime("%Y-%m-%d")
entry_dir = os.path.join(LOG_DIR, "entries", today)
os.makedirs(entry_dir, exist_ok=True)

# Init ByteTrack
from types import SimpleNamespace
tracker_args = SimpleNamespace(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    frame_rate=30,
    mot20=False
)
tracker = BYTETracker(tracker_args)
timer = Timer()

# Start video
cap = cv2.VideoCapture(VIDEO_SOURCE)
frame_id = 0
last_seen = {}
exit_logged = set()
EXIT_THRESHOLD = 50

print("[INFO] Starting face tracking...")
write_event("Face tracking started")

while True:
    ret, frame = cap.read()
    if not ret:
        write_event("Video stream ended")
        break

    frame_id += 1
    timer.tic()

    if frame_id % SKIP_FRAMES != 0:
        cv2.imshow("Face Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            write_event("User terminated stream with 'q'")
            break
        continue

    results = yolo_model.predict(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        conf = float(box.conf)
        if conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Add margin
        x1 = max(0, x1 - MARGIN)
        y1 = max(0, y1 - MARGIN)
        x2 = min(frame.shape[1], x2 + MARGIN)
        y2 = min(frame.shape[0], y2 + MARGIN)

        detections.append([x1, y1, x2, y2, conf])
    print(f"[DEBUG] Detections: {detections}")

    dets = np.array(detections, dtype=np.float32) if detections else np.empty((0, 5), dtype=np.float32)
    tracks = tracker.update(dets, [frame.shape[0], frame.shape[1]], frame.shape)

    for track in tracks:
        x1, y1, x2, y2 = map(int, track.tlbr)
        track_id = track.track_id

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            print(f"[WARNING] Empty crop for track {track_id}, skipping.")
            write_event(f"[WARNING] Empty face crop for track {track_id}, skipping.")
            continue

        timestamp = datetime.now().strftime("%H%M%S_%f")
        face_filename = os.path.join(entry_dir, f"{track_id}_{timestamp}.jpg")
        cv2.imwrite(face_filename, face_crop)
        print(f"[DEBUG] Saved: {face_filename}")
        write_event(f"Face crop saved for track {track_id} → {face_filename}")

        # Save debug image and run insightface directly on the original crop
        debug_path = os.path.join(entry_dir, f"debug_{track_id}_{timestamp}.jpg")
        cv2.imwrite(debug_path, face_crop)
        print(f"[DEBUG] Saved debug crop: {debug_path}")

        faces = insightface_app.get(face_crop)
        if not faces:
            print(f"[WARNING] No face found in original crop for track {track_id}")
            write_event(f"[WARNING] No face found in original crop for track {track_id}")
            continue

        embedding = faces[0].normed_embedding
        print("Embedding norm:", np.linalg.norm(embedding))

        visitor_id = face_db.match(embedding)
        if visitor_id in exit_logged:
            exit_logged.remove(visitor_id)

        if visitor_id is None:
            visitor_id = face_db.register(embedding)
            log_visit(visitor_id, "entry", face_filename)
            write_event(f"[ENTRY] New visitor registered → ID: {visitor_id}")
        else:
            face_db.update_last_seen(visitor_id)
            log_visit(visitor_id, "re-detection", face_filename)
            write_event(f"[REDETECTION] Visitor {visitor_id} seen again")

        last_seen[visitor_id] = frame_id

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{visitor_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for vid, last_frame in last_seen.items():
        if frame_id - last_frame > EXIT_THRESHOLD and vid not in exit_logged:
            print(f"[INFO] EXIT detected for {vid}")
            write_event(f"[EXIT] Visitor {vid} left the frame")

            exit_filename = os.path.join(entry_dir, f"{vid}_exit_{frame_id}.jpg")
            cv2.imwrite(exit_filename, frame)

            log_visit(vid, "exit", exit_filename)
            exit_logged.add(vid)

    cv2.putText(frame, f"Frame: {frame_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    unique_count = face_db.get_visitor_count()
    cv2.putText(frame, f"Unique Visitors: {unique_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Face Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        write_event("User terminated stream with 'q'")
        break

cap.release()
write_event(f"Total unique visitors this session: {face_db.get_visitor_count()}")
cv2.destroyAllWindows()
