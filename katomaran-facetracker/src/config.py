# import json
# import os

# class Config:
#     def __init__(self, config_path="config.json"):
#         with open(config_path, 'r') as f:
#             self.cfg = json.load(f)

#     def get(self, key, default=None):
#         return self.cfg.get(key, default)

#     def __getitem__(self, key):
#         return self.cfg[key]
    
# FACE_DB_PATH = "data/face_db.pkl"
# SIMILARITY_THRESHOLD = 0.6

import json
import os

class Config:
    def __init__(self, config_path="config.json"):
        with open(config_path, 'r') as f:
            self.cfg = json.load(f)

    def __getitem__(self, key):
        return self.cfg[key]

    def __getattr__(self, name):
        try:
            return self.cfg[name]
        except KeyError:
            raise AttributeError(f"Config key '{name}' not found")


