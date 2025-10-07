# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
import cv2

class ArcFaceFeatureExtractor:
    def __init__(self, model_dir=None, ctx_id=0):
        """
        Initialize ArcFace model (via InsightFace)
        """
        self.app = FaceAnalysis(name='buffalo_l', root=model_dir or './models')
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print("✅ ArcFace model loaded successfully")

    def compute_facial_encodings(self, image_paths):
        """
        Calculate ArcFace embeddings for a list of image paths

        Args:
            image_paths: list of image file paths

        Returns:
            facial_encodings: dict { image_path: embedding_vector }
        """
        facial_encodings = {}
        print("Calculating facial feature encodings using ArcFace...")
        
        for img_path in tqdm(image_paths):
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Cannot read {img_path}")
                continue

            faces = self.app.get(img)
            if len(faces) == 0:
                print(f"⚠️ No face detected in {img_path}")
                continue

            # Usually take the largest face
            main_face = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])
            emb = main_face.embedding

            # normalize (ArcFace already L2-normalized, but just to be safe)
            emb = emb / np.linalg.norm(emb)

            facial_encodings[img_path] = emb

        print(f"✅ Completed facial encoding for {len(facial_encodings)} images.")
        return facial_encodings
