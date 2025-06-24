import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Corrected path
img_path = r"logs\entries\2025-06-23\debug_1_181215_818908.jpg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found or unreadable at: {img_path}")

# Init insightface
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0)

# Resize to 112x112
resized = cv2.resize(img, (112, 112))

# Run face detection
faces = app.get(resized)

print(f"Faces found: {len(faces)}")
if faces:
    embedding = faces[0].normed_embedding
    print("Embedding norm:", np.linalg.norm(embedding))
else:
    print("[ERROR] InsightFace could not detect a face")
