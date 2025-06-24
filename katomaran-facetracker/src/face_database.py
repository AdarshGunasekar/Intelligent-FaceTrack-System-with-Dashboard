import pickle
import os
import numpy as np
from datetime import datetime
from src.config import Config

config = Config()

class FaceDatabase:
    def __init__(self):
        self.db = {}
        self.load()

    def load(self):
        if os.path.exists(config.FACE_DB_PATH):
            with open(config.FACE_DB_PATH, 'rb') as f:
                self.db = pickle.load(f)

    def save(self):
        with open(config.FACE_DB_PATH, 'wb') as f:
            pickle.dump(self.db, f)

    def match(self, new_embedding):
        for name, data in self.db.items():
            known_emb = np.array(data["embedding"])
            dist = np.linalg.norm(known_emb - new_embedding)
            if dist < config.SIMILARITY_THRESHOLD:
                return name
        return None

    def register(self, embedding):
        face_id = f"visitor_{len(self.db) + 1:04d}"  # 4-digit zero-padded ID
        self.db[face_id] = {
            "embedding": embedding.tolist(),
            "first_seen": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat()
        }
        self.save()
        return face_id

    def update_last_seen(self, face_id):
        if face_id in self.db:
            self.db[face_id]["last_seen"] = datetime.now().isoformat()
            self.save()

    def get_visitor_count(self):
        return len(self.db)
