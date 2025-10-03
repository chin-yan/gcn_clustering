"""
Feature Extractor Replacement: FaceNet -> ArcFace
This script provides a drop-in replacement for FaceNet feature extraction
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle

# ============================================================================
# ArcFace Feature Extractor
# ============================================================================

class ArcFaceExtractor:
    """
    ArcFace feature extractor using insightface
    Compatible with FaceNet output format (512-d embeddings)
    """
    
    def __init__(self, model_name='buffalo_l', ctx_id=0, det_size=(640, 640)):
        """
        Args:
            model_name: Model name in insightface
                - 'buffalo_l': Recommended, balanced accuracy and speed
                - 'buffalo_s': Smaller, faster
                - 'antelopev2': Higher accuracy but slower
            ctx_id: GPU id, -1 for CPU
            det_size: Face detection size, larger = more accurate but slower
        """
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface not installed. Install with:\n"
                "pip install insightface onnxruntime-gpu\n"
                "or for CPU only:\n"
                "pip install insightface onnxruntime"
            )
        
        print(f"Initializing ArcFace model: {model_name}")
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        print("ArcFace model loaded successfully")
    
    def extract_single(self, image_path, return_largest=True):
        """
        Extract feature from a single image
        
        Args:
            image_path: Path to image file
            return_largest: If True, return the largest face. 
                          If False, return all faces.
        
        Returns:
            If return_largest=True: embedding array of shape (512,) or None if no face
            If return_largest=False: list of embedding arrays
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Failed to read image {image_path}")
            return None if return_largest else []
        
        # Detect and extract features
        faces = self.app.get(img)
        
        if len(faces) == 0:
            return None if return_largest else []
        
        if return_largest:
            # Return the largest face (by bbox area)
            largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            return largest_face.embedding
        else:
            # Return all faces
            return [face.embedding for face in faces]
    
    def extract_from_array(self, img_array):
        """
        Extract feature from a numpy array (already loaded image)
        
        Args:
            img_array: numpy array of shape (H, W, 3) in BGR format
        
        Returns:
            embedding array of shape (512,) or None if no face
        """
        faces = self.app.get(img_array)
        
        if len(faces) == 0:
            return None
        
        # Return the largest face
        largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        return largest_face.embedding
    
    def extract_batch(self, image_paths, batch_size=32, show_progress=True):
        """
        Extract features from multiple images
        
        Args:
            image_paths: List of image paths
            batch_size: Not used (kept for API compatibility), 
                       insightface processes images one by one
            show_progress: Show progress bar
        
        Returns:
            embeddings: numpy array of shape (N, 512)
            valid_indices: indices of images that had faces detected
        """
        embeddings = []
        valid_indices = []
        
        iterator = tqdm(enumerate(image_paths), total=len(image_paths), 
                       desc="Extracting ArcFace features") if show_progress else enumerate(image_paths)
        
        for idx, img_path in iterator:
            embedding = self.extract_single(img_path)
            if embedding is not None:
                embeddings.append(embedding)
                valid_indices.append(idx)
        
        if len(embeddings) == 0:
            print("Warning: No faces detected in any images")
            return np.array([]), []
        
        return np.array(embeddings), valid_indices


# ============================================================================
# FaceNet Feature Extractor (for comparison/fallback)
# ============================================================================

class FaceNetExtractor:
    """
    FaceNet feature extractor (original implementation)
    Kept for comparison or fallback
    """
    
    def __init__(self, model_path):
        """
        Args:
            model_path: Path to FaceNet model file (.pb)
        """
        import tensorflow as tf
        
        print(f"Loading FaceNet model from {model_path}")
        
        # Load the model
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        
        self.sess = tf.Session(graph=self.graph)
        
        # Get input and output tensors
        self.images_placeholder = self.graph.get_tensor_by_name("input:0")
        self.embeddings = self.graph.get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")
        
        print("FaceNet model loaded successfully")
    
    def extract_single(self, image_path, face_crop=None):
        """
        Extract feature from a single face image
        
        Args:
            image_path: Path to face image (should be pre-cropped to face)
            face_crop: Not used, kept for API compatibility
        
        Returns:
            embedding array of shape (512,) or (128,) depending on model
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Preprocess (FaceNet expects specific size and normalization)
        img = cv2.resize(img, (160, 160))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - 127.5) / 128.0
        img = np.expand_dims(img, axis=0)
        
        # Extract features
        feed_dict = {
            self.images_placeholder: img,
            self.phase_train_placeholder: False
        }
        embedding = self.sess.run(self.embeddings, feed_dict=feed_dict)
        
        return embedding[0]
    
    def extract_batch(self, image_paths, batch_size=32, show_progress=True):
        """Extract features from multiple images"""
        embeddings = []
        valid_indices = []
        
        iterator = tqdm(enumerate(image_paths), total=len(image_paths),
                       desc="Extracting FaceNet features") if show_progress else enumerate(image_paths)
        
        for idx, img_path in iterator:
            embedding = self.extract_single(img_path)
            if embedding is not None:
                embeddings.append(embedding)
                valid_indices.append(idx)
        
        return np.array(embeddings), valid_indices


# ============================================================================
# Unified Feature Extraction Interface
# ============================================================================

def extract_features(
    image_paths,
    output_path,
    method='arcface',
    model_path=None,
    batch_size=32,
    save_indices=True
):
    """
    Extract features using specified method
    
    Args:
        image_paths: List of image paths or path to directory
        output_path: Path to save features (.npy or .pkl)
        method: 'arcface' or 'facenet'
        model_path: Path to model (only needed for facenet)
        batch_size: Batch size for processing
        save_indices: If True, also save the valid indices
    
    Returns:
        features: numpy array of shape (N, 512)
        valid_indices: list of valid indices
    """
    # Handle directory input
    if isinstance(image_paths, (str, Path)):
        image_dir = Path(image_paths)
        if image_dir.is_dir():
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_paths = [
                str(p) for p in image_dir.rglob('*') 
                if p.suffix.lower() in extensions
            ]
            print(f"Found {len(image_paths)} images in {image_dir}")
    
    # Initialize extractor
    if method.lower() == 'arcface':
        extractor = ArcFaceExtractor()
    elif method.lower() == 'facenet':
        if model_path is None:
            raise ValueError("model_path must be provided for FaceNet")
        extractor = FaceNetExtractor(model_path)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Extract features
    print(f"\nExtracting features using {method.upper()}...")
    features, valid_indices = extractor.extract_batch(
        image_paths,
        batch_size=batch_size,
        show_progress=True
    )
    
    print(f"\nExtracted {len(features)} features from {len(image_paths)} images")
    print(f"Success rate: {len(features)/len(image_paths)*100:.1f}%")
    
    # Save features
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path, features)
    print(f"Features saved to {output_path}")
    
    # Save indices if requested
    if save_indices:
        indices_path = output_path.parent / f"{output_path.stem}_indices.npy"
        np.save(indices_path, valid_indices)
        print(f"Valid indices saved to {indices_path}")
    
    return features, valid_indices


# ============================================================================
# Comparison Utility
# ============================================================================

def compare_extractors(image_paths, facenet_model_path, n_samples=100):
    """
    Compare FaceNet and ArcFace on the same images
    Useful for understanding the feature quality difference
    
    Args:
        image_paths: List of image paths
        facenet_model_path: Path to FaceNet model
        n_samples: Number of samples to compare
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Sample images if too many
    if len(image_paths) > n_samples:
        indices = np.random.choice(len(image_paths), n_samples, replace=False)
        image_paths = [image_paths[i] for i in indices]
    
    print("Extracting features with FaceNet...")
    facenet = FaceNetExtractor(facenet_model_path)
    facenet_features, facenet_indices = facenet.extract_batch(image_paths)
    
    print("\nExtracting features with ArcFace...")
    arcface = ArcFaceExtractor()
    arcface_features, arcface_indices = arcface.extract_batch(image_paths)
    
    # Compare on common images
    common_indices = list(set(facenet_indices) & set(arcface_indices))
    print(f"\nCommon detections: {len(common_indices)}/{len(image_paths)}")
    
    if len(common_indices) < 10:
        print("Too few common detections for meaningful comparison")
        return
    
    # Get features for common images
    facenet_common = facenet_features[[facenet_indices.index(i) for i in common_indices]]
    arcface_common = arcface_features[[arcface_indices.index(i) for i in common_indices]]
    
    # Compute similarity matrices
    facenet_sim = cosine_similarity(facenet_common)
    arcface_sim = cosine_similarity(arcface_common)
    
    # Statistics (excluding diagonal)
    mask = ~np.eye(len(facenet_sim), dtype=bool)
    
    print("\n=== Similarity Statistics ===")
    print(f"FaceNet:")
    print(f"  Mean: {facenet_sim[mask].mean():.3f}")
    print(f"  Std:  {facenet_sim[mask].std():.3f}")
    
    print(f"\nArcFace:")
    print(f"  Mean: {arcface_sim[mask].mean():.3f}")
    print(f"  Std:  {arcface_sim[mask].std():.3f}")
    
    # Typically, ArcFace should have:
    # - Lower mean similarity (better separation)
    # - Similar or higher std (maintains within-class similarity)


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    """
    Example usage scenarios
    """
    
    # ========================================================================
    # Scenario 1: Extract features from a directory of face images
    # ========================================================================
    
    # Using ArcFace (RECOMMENDED)
    features, indices = extract_features(
        image_paths='path/to/face/images',  # directory with face images
        output_path='features/arcface_features.npy',
        method='arcface'
    )
    
    # ========================================================================
    # Scenario 2: Extract features from a list of image paths
    # ========================================================================
    
    # Suppose you already have a list of face image paths
    face_image_paths = [
        'path/to/face1.jpg',
        'path/to/face2.jpg',
        # ... more paths
    ]
    
    extractor = ArcFaceExtractor()
    features, valid_indices = extractor.extract_batch(face_image_paths)
    
    # Save
    np.save('features/my_features.npy', features)
    
    # ========================================================================
    # Scenario 3: Extract from a single image
    # ========================================================================
    
    extractor = ArcFaceExtractor()
    embedding = extractor.extract_single('path/to/single_face.jpg')
    
    if embedding is not None:
        print(f"Extracted embedding shape: {embedding.shape}")  # (512,)
    
    # ========================================================================
    # Scenario 4: Compare FaceNet vs ArcFace (optional)
    # ========================================================================
    
    compare_extractors(
        image_paths=face_image_paths,
        facenet_model_path='models/20180402-114759/20180402-114759.pb',
        n_samples=50
    )
    
    # ========================================================================
    # Scenario 5: Integration with existing GCN clustering pipeline
    # ========================================================================
    
    # Step 1: Extract ArcFace features
    print("Step 1: Extracting features...")
    features, valid_indices = extract_features(
        image_paths='data/video1/faces',
        output_path='features/video1_arcface.npy',
        method='arcface'
    )
    
    # Step 2: Load features and run GCN clustering
    # (This is where your existing gin-clustering code continues)
    print("Step 2: Running GCN clustering...")
    # from your_gcn_module import cluster_with_gcn
    # clusters = cluster_with_gcn(features, k1=20, k2=5)
    
    # Step 3: Run diagnostics to verify improvement
    print("Step 3: Running diagnostics...")
    from feature_diagnostics import FeatureQualityDiagnostics
    
    diagnostics = FeatureQualityDiagnostics(
        features=features,
        labels=None,
        video_name="video1_arcface"
    )
    diagnostics.run_full_diagnostics()