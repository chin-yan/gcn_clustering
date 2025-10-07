# -*- coding: utf-8 -*-
"""
enhanced_face_retrieval.py (ArcFace / InsightFace version)

Replaces TensorFlow FaceNet pipeline with InsightFace (ArcFace) for detection + embedding.
Keeps original retrieval, Annoy, temporal weighting, visualization logic.
"""

import os
import cv2
import numpy as np
import pickle
from annoy import AnnoyIndex
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from collections import deque
from random import shuffle

# InsightFace
from insightface.app import FaceAnalysis

# ------------------------------------------------------------------
# Helper: ArcFace feature extractor + detection (InsightFace)
# ------------------------------------------------------------------
class ArcFaceFeatureExtractor:
    def __init__(self, model_root=None, ctx_id=0, det_size=(640, 640), name='buffalo_l'):
        """
        Initialize InsightFace FaceAnalysis (detect + embedding).
        model_root: local model root for insightface; if None, will use default.
        ctx_id: GPU id (0..) or -1 for CPU
        name: model name; 'buffalo_l' is a common choice with 512-d embeddings
        """
        model_root = model_root or './models'
        print(f"Loading InsightFace model (name={name}, root={model_root}) ...")
        self.app = FaceAnalysis(name=name, root=model_root)
        # ctx_id = 0 for GPU; -1 for CPU. If user wants GPU they must ensure correct install.
        try:
            self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        except Exception as e:
            print("Warning: app.prepare raised an exception. Retrying with ctx_id=-1 (CPU).")
            self.app = FaceAnalysis(name=name, root=model_root)
            self.app.prepare(ctx_id=-1, det_size=det_size)
        print("✅ InsightFace model loaded successfully")

    def detect_and_crop_faces(self, frame_path, out_dir, frame_idx, img_name_prefix="retrieval_frame"):
        """
        Detect faces in one frame, crop them and save to out_dir with standardized filenames.
        Returns list of saved face image paths for this frame.
        """
        img = cv2.imread(frame_path)
        if img is None:
            return []

        faces = self.app.get(img)
        saved_paths = []

        for i, f in enumerate(faces):
            # bounding box: [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, f.bbox.astype(int))
            # add small padding
            h, w = img.shape[:2]
            pad = int(0.12 * max(x2 - x1, y2 - y1))
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(w - 1, x2 + pad)
            y2p = min(h - 1, y2 + pad)

            crop = img[y1p:y2p, x1p:x2p]
            # resize to 160x160 as expected downstream visualization / quality scores
            crop_resized = cv2.resize(crop, (160, 160))
            face_filename = f"{img_name_prefix}_{frame_idx:06d}_face_{i}.jpg"
            face_path = os.path.join(out_dir, face_filename)
            cv2.imwrite(face_path, crop_resized)
            saved_paths.append(face_path)

        return saved_paths

    def compute_facial_encodings(self, image_paths, require_l2norm=True):
        """
        Compute ArcFace embeddings for list of image paths.
        Returns dict: { path: embedding(np.array) }
        """
        facial_encodings = {}
        print("Calculating facial feature encodings using ArcFace...")
        for img_path in tqdm(image_paths):
            img = cv2.imread(img_path)
            if img is None:
                # skip unreadable images
                continue
            faces = self.app.get(img)
            if len(faces) == 0:
                # No face detected (should be rare since we used detector earlier)
                continue
            # choose the face with largest bbox
            main_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            emb = main_face.embedding
            emb = np.array(emb, dtype=np.float32)
            if require_l2norm:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
            facial_encodings[img_path] = emb
        print(f"✅ Completed facial encoding for {len(facial_encodings)} images.")
        return facial_encodings

# ------------------------------------------------------------------
# Existing utilities (slightly adapted)
# ------------------------------------------------------------------
def load_centers_data(centers_data_path):
    print("Loading cluster center data...")
    with open(centers_data_path, 'rb') as f:
        centers_data = pickle.load(f)
    if 'cluster_centers' not in centers_data:
        raise ValueError("Missing cluster center information in the data")
    centers, center_paths = centers_data['cluster_centers']
    # Ensure centers are numpy arrays and L2-normalized for angular Annoy
    centers = [np.array(c, dtype=np.float32) for c in centers]
    centers = [c / np.linalg.norm(c) if np.linalg.norm(c) > 0 else c for c in centers]
    print(f"Successfully loaded {len(centers)} cluster centers")
    return centers, center_paths, centers_data

def build_annoy_index(centers, n_trees=15):
    print("Step 1: Building improved Annoy index...")
    if len(centers) == 0:
        raise ValueError("No centers to build index")
    embedding_size = centers[0].shape[0]
    annoy_index = AnnoyIndex(embedding_size, 'angular')  # angular ~ cosine
    for i, center in enumerate(centers):
        annoy_index.add_item(i, center.tolist())
    print(f"Using {n_trees} trees to build random forest index")
    annoy_index.build(n_trees)
    return annoy_index

def extract_frames_and_faces(video_path, output_dir, extractor: ArcFaceFeatureExtractor, interval=5):
    """
    Extract frames from video and detect faces using InsightFace detector.
    Returns list of face image paths and frames paths.
    """
    print("Extracting frames from video with smaller interval...")
    frames_dir = os.path.join(output_dir, 'retrieval_frames')
    faces_dir = os.path.join(output_dir, 'retrieval_faces')

    for dir_path in [frames_dir, faces_dir]:
        os.makedirs(dir_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames_paths = []
    saved_face_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame_path = os.path.join(frames_dir, f"retrieval_frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames_paths.append(frame_path)

            # detect faces in this frame and save crops
            face_paths = extractor.detect_and_crop_faces(frame_path, faces_dir, frame_count)
            saved_face_paths.extend(face_paths)

        frame_count += 1

    cap.release()
    print(f"Extracted {len(frames_paths)} frames and {len(saved_face_paths)} face crops")
    return saved_face_paths, frames_paths

def extract_frame_info(image_path):
    basename = os.path.basename(image_path)
    import re
    match = re.match(r'retrieval_frame_(\d+)_face_(\d+)\.jpg', basename)
    if match:
        frame_num = int(match.group(1))
        face_idx = int(match.group(2))
        return frame_num, face_idx
    match = re.match(r'retrieval_frame_(\d+)_sideface_(\d+)\.jpg', basename)
    if match:
        frame_num = int(match.group(1))
        face_idx = int(match.group(2))
        return frame_num, face_idx
    return -1, -1

def compute_face_quality(face_path):
    img = cv2.imread(face_path)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian) / 100.0
    sharpness = min(1.0, sharpness)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist / (hist.sum() + 1e-9)
    non_zero_bins = np.count_nonzero(hist_norm > 0.0005)
    contrast = non_zero_bins / 256.0
    height, width = gray.shape
    face_area = height * width
    face_size_score = min(1.0, face_area / (160.0 * 160.0))
    quality_score = 0.35 * sharpness + 0.35 * contrast + 0.3 * face_size_score
    quality_score = min(1.0, quality_score * 1.2)
    return quality_score

# ------------------------------------------------------------------
# Search, visualization, retrieval logic (kept largely as original)
# ------------------------------------------------------------------
def search_similar_faces_with_temporal(annoy_index, facial_encodings, center_paths,
                                      frame_info_dict=None, n_results=5,
                                      similarity_threshold=0.5, temporal_weight=0.2):
    print("Steps 3 and 4: Performing similar face search with temporal consistency...")
    if frame_info_dict is None:
        frame_info_dict = {}
        for path in facial_encodings.keys():
            frame_num, _ = extract_frame_info(path)
            frame_info_dict[path] = frame_num

    retrieval_results = {}
    frame_results = {}
    recent_matches = {}
    window_size = 5

    # prepare sorted paths by frame
    sorted_paths = sorted(facial_encodings.keys(),
                         key=lambda p: frame_info_dict[p] if frame_info_dict[p] > 0 else float('inf'))

    # init recent windows for all centers we might query
    for i in range(len(center_paths)):
        recent_matches[i] = deque(maxlen=window_size)

    for path in tqdm(sorted_paths):
        frame_num = frame_info_dict[path]
        encoding = facial_encodings[path]
        # ensure normalized
        if np.linalg.norm(encoding) > 0:
            encoding = encoding / np.linalg.norm(encoding)

        quality = compute_face_quality(path)

        # Annoy returns angular distance when metric='angular'
        nearest_indices, distances = annoy_index.get_nns_by_vector(encoding.tolist(), n_results * 2, include_distances=True)

        # Convert angular distance -> cosine similarity approximation
        # For small angles, angular distance d satisfies: cosine_sim approx = 1 - d^2/2
        similarities = [max(0.0, 1 - (d * d / 2)) for d in distances]

        adjusted_similarities = []
        for i, (idx, sim) in enumerate(zip(nearest_indices, similarities)):
            temporal_boost = 0.0
            if len(recent_matches[idx]) > 0:
                temporal_boost = sum(recent_matches[idx]) / len(recent_matches[idx])
            adjusted_sim = (1 - temporal_weight) * sim + temporal_weight * temporal_boost
            adjusted_similarities.append((idx, adjusted_sim, sim))

        adjusted_similarities.sort(key=lambda x: x[1], reverse=True)

        filtered_results = [
            (idx, adj_sim, orig_sim)
            for idx, adj_sim, orig_sim in adjusted_similarities[:n_results]
            if adj_sim > similarity_threshold
        ]

        for idx, adj_sim, orig_sim in filtered_results:
            if idx not in retrieval_results:
                retrieval_results[idx] = []
            retrieval_results[idx].append({
                'path': path,
                'frame': frame_num,
                'adjusted_similarity': adj_sim,
                'original_similarity': orig_sim,
                'quality': quality
            })
            recent_matches[idx].append(quality * orig_sim)

        if frame_num not in frame_results:
            frame_results[frame_num] = []

        if filtered_results:
            best_idx, best_adj_sim, best_orig_sim = filtered_results[0]
            frame_results[frame_num].append({
                'path': path,
                'center_id': best_idx,
                'similarity': best_orig_sim,
                'adjusted_similarity': best_adj_sim,
                'quality': quality
            })

    for idx in retrieval_results:
        retrieval_results[idx] = sorted(retrieval_results[idx], key=lambda x: x['adjusted_similarity'], reverse=True)

    return retrieval_results, frame_results

def visualize_retrieval_results(retrieval_results, center_paths, output_dir, max_results=10):
    print("Visualizing retrieval results...")
    vis_dir = os.path.join(output_dir, 'retrieval_visualization')
    os.makedirs(vis_dir, exist_ok=True)
    active_centers = [idx for idx in retrieval_results if len(retrieval_results[idx]) > 0]

    for idx in active_centers:
        if idx >= len(center_paths):
            continue
        center_path = center_paths[idx]
        center_name = os.path.basename(center_path)
        display_results = retrieval_results[idx][:max_results]
        n_results = len(display_results)
        if n_results == 0:
            continue
        n_cols = 4
        n_rows = (n_results + n_cols - 1) // n_cols + 1
        fig = plt.figure(figsize=(15, 3 * n_rows))
        ax_center = plt.subplot2grid((n_rows, n_cols), (0, 1), colspan=2)
        try:
            center_img = plt.imread(center_path)
            ax_center.imshow(center_img)
            ax_center.set_title(f'Cluster Center {idx}', fontsize=14)
            ax_center.axis('off')
        except Exception as e:
            print(f"Unable to load center image {center_path}: {e}")
        for i, result in enumerate(display_results):
            row = 1 + i // n_cols
            col = i % n_cols
            ax = plt.subplot2grid((n_rows, n_cols), (row, col))
            try:
                img = plt.imread(result['path'])
                ax.imshow(img)
                title_parts = [
                    f"Sim: {result['original_similarity']:.3f}",
                    f"Adj: {result['adjusted_similarity']:.3f}",
                    f"Qual: {result['quality']:.2f}"
                ]
                if 'frame' in result:
                    title_parts.append(f"Frame: {result['frame']}")
                ax.set_title('\n'.join(title_parts), fontsize=9)
                ax.axis('off')
            except Exception as e:
                print(f"Unable to load image {result['path']}: {e}")
        plt.suptitle(f'Retrieval Results for Center {idx} ({n_results} matches)', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(vis_dir, f'retrieval_center_{idx}.png'), dpi=200)
        plt.close()

    # Multi-page summary (keeps your original logic)
    if active_centers:
        centers_per_page = 16
        total_pages = (len(active_centers) + centers_per_page - 1) // centers_per_page
        for page_num in range(total_pages):
            start_idx = page_num * centers_per_page
            end_idx = min(start_idx + centers_per_page, len(active_centers))
            page_centers = active_centers[start_idx:end_idx]
            n_centers = len(page_centers)
            if n_centers <= 4:
                n_cols = n_centers
                n_rows = 1
            elif n_centers <= 8:
                n_cols = 4
                n_rows = 2
            elif n_centers <= 12:
                n_cols = 4
                n_rows = 3
            else:
                n_cols = 4
                n_rows = 4
            fig_width = min(16, 4 * n_cols)
            fig_height = min(12, 3.5 * n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
            if n_centers == 1:
                axes = [axes]
            elif n_rows == 1:
                if n_cols == 1:
                    axes = [axes]
                else:
                    axes = list(axes) if hasattr(axes, '__iter__') else [axes]
            else:
                axes = axes.flatten()
            for i, center_idx in enumerate(page_centers):
                if center_idx >= len(center_paths):
                    continue
                center_path = center_paths[center_idx]
                n_matches = len(retrieval_results[center_idx])
                try:
                    center_img = plt.imread(center_path)
                    axes[i].imshow(center_img)
                    axes[i].set_title(f'Center {center_idx}\n{n_matches} matches', fontsize=11, pad=10)
                    axes[i].axis('off')
                    for spine in axes[i].spines.values():
                        spine.set_edgecolor('lightgray')
                        spine.set_linewidth(1)
                except Exception as e:
                    print(f"Unable to load center image {center_path}: {e}")
                    axes[i].text(0.5, 0.5, f'Center {center_idx}\n{n_matches} matches\n(Image Error)',
                               transform=axes[i].transAxes, ha='center', va='center',
                               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    axes[i].axis('off')
            for i in range(len(page_centers), len(axes)):
                axes[i].axis('off')
                axes[i].set_visible(False)
            if total_pages == 1:
                page_title = f'Active Centers Summary ({len(active_centers)} centers found matches)'
            else:
                page_title = f'Active Centers Summary - Page {page_num + 1}/{total_pages}\n({len(page_centers)} centers on this page, {len(active_centers)} total)'
            plt.suptitle(page_title, fontsize=14, y=0.95)
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)
            if total_pages == 1:
                filename = 'active_centers_summary.png'
            else:
                filename = f'active_centers_summary_page_{page_num + 1:02d}_of_{total_pages:02d}.png'
            save_path = os.path.join(vis_dir, filename)
            try:
                plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
                print(f"✅ Saved: {filename}")
            except Exception as e:
                print(f"⚠️ Failed to save {filename} with DPI 200: {e}")
                try:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
                    print(f"✅ Saved: {filename} (DPI 150)")
                except Exception as e2:
                    print(f"⚠️ Failed to save {filename} with DPI 150: {e2}")
                    try:
                        plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
                        print(f"✅ Saved: {filename} (DPI 100)")
                    except Exception as e3:
                        print(f"❌ Failed to save {filename} even with DPI 100: {e3}")
            plt.close()

        if total_pages > 1:
            index_content = f"""# Active Centers Summary Index

Total active centers: {len(active_centers)}
Total summary pages: {total_pages}
Centers per page: up to {centers_per_page}

## Summary Pages:
"""
            for page_num in range(total_pages):
                start_idx = page_num * centers_per_page
                end_idx = min(start_idx + centers_per_page, len(active_centers))
                page_centers = active_centers[start_idx:end_idx]
                filename = f'active_centers_summary_page_{page_num + 1:02d}_of_{total_pages:02d}.png'
                index_content += f"\n### Page {page_num + 1}: {filename}\n"
                index_content += f"Centers: {', '.join(map(str, page_centers))}\n"
                index_content += f"Count: {len(page_centers)} centers\n"
            index_content += f"\n## Statistics:\n"
            index_content += f"- Total matches found: {sum(len(retrieval_results[idx]) for idx in active_centers)}\n"
            index_content += f"- Average matches per center: {sum(len(retrieval_results[idx]) for idx in active_centers) / len(active_centers):.1f}\n"
            top_centers = sorted(active_centers, key=lambda x: len(retrieval_results[x]), reverse=True)[:5]
            index_content += f"- Top 5 centers by matches: {', '.join(f'Center {idx} ({len(retrieval_results[idx])} matches)' for idx in top_centers)}\n"
            index_path = os.path.join(vis_dir, 'summary_index.md')
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(index_content)
            print(f"✅ Summary index saved: {index_path}")

    print(f"✅ Visualization completed. Results saved in: {vis_dir}")
    if active_centers:
        total_matches = sum(len(retrieval_results[idx]) for idx in active_centers)
        print(f"📊 Summary: {len(active_centers)} active centers, {total_matches} total matches")
        if len(active_centers) > 16:
            pages_created = (len(active_centers) + 15) // 16
            print(f"📄 Created {pages_created} summary pages to display all centers")

def visualize_frame_results(frame_results, center_paths, output_dir):
    print("Visualizing frame-based results...")
    vis_dir = os.path.join(output_dir, 'frame_visualization')
    os.makedirs(vis_dir, exist_ok=True)
    frames = sorted(frame_results.keys())
    if not frames:
        print("No frame results to visualize")
        return
    center_matches = {}
    for frame in frames:
        for match in frame_results[frame]:
            center_id = match['center_id']
            if center_id not in center_matches:
                center_matches[center_id] = []
            center_matches[center_id].append((frame, match['similarity']))
    for center_id, matches in center_matches.items():
        if center_id >= len(center_paths) or len(matches) < 3:
            continue
        match_frames, match_similarities = zip(*matches)
        plt.figure(figsize=(12, 6))
        plt.scatter(match_frames, match_similarities, alpha=0.7, s=30)
        plt.plot(match_frames, match_similarities, 'b-', alpha=0.3)
        try:
            center_img = plt.imread(center_paths[center_id])
            inset_ax = plt.axes([0.05, 0.7, 0.2, 0.2])
            inset_ax.imshow(center_img)
            inset_ax.axis('off')
        except Exception as e:
            print(f"Unable to load center image {center_paths[center_id]}: {e}")
        plt.title(f'Center {center_id} Appearances Over Time ({len(matches)} matches)')
        plt.xlabel('Frame Number')
        plt.ylabel('Similarity Score')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'center_{center_id}_timeline.png'), dpi=200)
        plt.close()
    plt.figure(figsize=(15, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(center_matches)))
    for i, (center_id, matches) in enumerate(center_matches.items()):
        if len(matches) < 3:
            continue
        match_frames, match_similarities = zip(*matches)
        sc = plt.scatter(match_frames, [i] * len(match_frames), c=match_similarities,
                   cmap='viridis', alpha=0.7, s=50, vmin=0, vmax=1)
        plt.text(frames[0] - (frames[-1] - frames[0]) * 0.05, i, f'Center {center_id}',
                va='center', ha='right')
    plt.title('Timeline of Character Appearances')
    plt.xlabel('Frame Number')
    plt.yticks([])
    plt.xlim(frames[0] - (frames[-1] - frames[0]) * 0.05, frames[-1] * 1.01)
    plt.colorbar(sc, label='Similarity Score')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'character_timeline.png'), dpi=200)
    plt.close()

# ------------------------------------------------------------------
# Main pipeline (ArcFace)
# ------------------------------------------------------------------
def enhanced_face_retrieval(video_path, centers_data_path, output_dir, model_dir,
                           frame_interval=5, batch_size=64, n_trees=15, n_results=10,
                           similarity_threshold=0.45, temporal_weight=0.2, ctx_id=-1):
    """
    Main function for enhanced face retrieval using ArcFace (InsightFace)
    - ctx_id: GPU id (0...) or -1 for CPU
    - similarity_threshold: recommended ~0.40-0.5 for cosine-like similarity (tune on your data)
    """
    centers, center_paths, centers_data = load_centers_data(centers_data_path)
    annoy_index = build_annoy_index(centers, n_trees)

    # Initialize ArcFace extractor (detection + embedding)
    extractor = ArcFaceFeatureExtractor(model_root=model_dir, ctx_id=ctx_id)

    # Extract frames and detect faces (produces face crops)
    face_paths, frame_paths = extract_frames_and_faces(video_path, output_dir, extractor, frame_interval)

    # Compute face encodings (ArcFace embeddings)
    # We don't use TensorFlow session; compute via extractor
    facial_encodings = extractor.compute_facial_encodings(face_paths)

    # Prepare frame information for temporal consistency
    frame_info_dict = {}
    for path in facial_encodings.keys():
        frame_num, _ = extract_frame_info(path)
        frame_info_dict[path] = frame_num

    # Search for similar faces with temporal consistency
    retrieval_results, frame_results = search_similar_faces_with_temporal(
        annoy_index, facial_encodings, center_paths, frame_info_dict,
        n_results, similarity_threshold, temporal_weight
    )

    # Save retrieval results
    results_dir = os.path.join(output_dir, 'retrieval')
    os.makedirs(results_dir, exist_ok=True)
    results_data = {
        'by_center': retrieval_results,
        'by_frame': frame_results
    }
    results_path = os.path.join(results_dir, 'enhanced_retrieval_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results_data, f)

    # Visualize retrieval results
    visualize_retrieval_results(retrieval_results, center_paths, output_dir, n_results)
    visualize_frame_results(frame_results, center_paths, output_dir)

    print("Enhanced face retrieval (ArcFace) completed!")
    return retrieval_results, frame_results