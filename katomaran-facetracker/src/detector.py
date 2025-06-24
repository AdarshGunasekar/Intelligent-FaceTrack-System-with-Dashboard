# src/detector.py
# # src/detector.py
# from ultralytics import YOLO
# import cv2
# import numpy as np

# class FaceDetector:
#     def __init__(self, model_path="yolov8n-face.pt", conf_threshold=0.5):
#         self.model = YOLO(model_path)
#         self.conf_threshold = conf_threshold

#     def detect_faces(self, frame):
#         results = self.model.predict(source=frame, verbose=False)[0]
#         detections = []
#         for box in results.boxes:
#             conf = float(box.conf)
#             if conf < self.conf_threshold:
#                 continue
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             detections.append((x1, y1, x2, y2))
#         return detections

# src/detector.py
from ultralytics import YOLO
import cv2
import numpy as np

class FaceDetector:
    def __init__(self, model_path="yolov8n-face.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    # In detector.py, modify detect_faces() to:
    def detect_faces(self, frame):
        results = self.model.predict(source=frame, verbose=False)[0]
        print("YOLO RAW OUTPUT:", results.boxes.xyxy)  # Add this debug line
        detections = []
        for box in results.boxes:
            conf = float(box.conf)
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append([x1, y1, x2, y2, conf])
        return detections

