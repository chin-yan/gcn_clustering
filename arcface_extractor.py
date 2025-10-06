import os
import sys



import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# 全局模型實例
_session = None
_input_size = (112, 112)  # ArcFace 標準輸入大小

def load_arcface_model(model_path=r"C:\Users\VIPLAB\Desktop\Yan\gcn_clustering-master\models\arcface\arcface.onnx", use_gpu=True):
    """
    載入 ArcFace ONNX 模型
    
    Args:
        model_path: ONNX 模型路徑
        use_gpu: 是否使用 GPU
        
    Returns:
        onnxruntime.InferenceSession
    """
    global _session
    
    if _session is None:
        # 如果沒有指定路徑，使用預設路徑
        if model_path is None:
            possible_paths = [
                'models/arcface/arcface_r100.onnx',
                'models/arcface/arcface_r50.onnx',
                'models/arcface/model.onnx',
                './arcface_r100.onnx',
                './arcface.onnx',
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    "ArcFace model not found. Please download the model and place it at:\n"
                    "  models/arcface/arcface_r100.onnx\n\n"
                    "Download from:\n"
                    "  https://github.com/onnx/models/tree/main/vision/body_analysis/arcface\n"
                    "  or\n"
                    "  https://huggingface.co/models?search=arcface"
                )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading ArcFace model from: {model_path}")
        
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("Attempting to use GPU...")
        else:
            providers = ['CPUExecutionProvider']
            print("Using CPU mode...")
                        
        try:
            # 載入模型
            _session = ort.InferenceSession(model_path, providers=providers)
            
            # 顯示模型信息
            actual_provider = _session.get_providers()[0]
            print(f"✅ ArcFace model loaded successfully")
            print(f"   Provider: {actual_provider}")
            print(f"   Model path: {model_path}")
            
            # 顯示輸入輸出信息
            input_info = _session.get_inputs()[0]
            output_info = _session.get_outputs()[0]
            print(f"   Input: {input_info.name}, shape: {input_info.shape}")
            print(f"   Output: {output_info.name}, shape: {output_info.shape}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # 如果 GPU 失敗，嘗試 CPU
            if use_gpu:
                print("Falling back to CPU...")
                _session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                print("✅ Model loaded with CPU")
    
    return _session


def preprocess_face(img, input_size=(112, 112)):
    """
    預處理人臉圖片
    
    ArcFace 標準預處理流程:
    1. Resize to 112x112
    2. BGR to RGB
    3. Transpose to CHW format (Channel, Height, Width)
    4. Normalize to [-1, 1]
    5. Add batch dimension
    
    Args:
        img: BGR 格式的圖片 (OpenCV 格式)
        input_size: 目標大小 (height, width)
        
    Returns:
        預處理後的 numpy array, shape (1, 3, 112, 112)
    """
    # Step 1: Resize
    if img.shape[0] != input_size[0] or img.shape[1] != input_size[1]:
        img = cv2.resize(img, (input_size[1], input_size[0]))
    
    # Step 2: BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Step 3: Transpose to CHW
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    
    # Step 4: Add batch dimension
    img = np.expand_dims(img, axis=0)  # (C, H, W) -> (1, C, H, W)
    
    return img


def extract_single_feature(session, img, input_name):
    """
    提取單張圖片的特徵
    
    Args:
        session: ONNX session
        img: 預處理後的圖片
        input_name: 模型輸入名稱
        
    Returns:
        特徵向量 (512-d)
    """
    try:
        # ONNX 推理
        embedding = session.run(None, {input_name: img})[0]
        
        # 取出 batch 維度
        embedding = embedding[0]
        
        # L2 正規化
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        else:
            return None
        
        return embedding
        
    except Exception as e:
        print(f"Inference error: {e}")
        return None


def extract_features_from_paths(face_paths, batch_size=32, use_gpu=True, model_path=None):
    """
    從人臉圖片路徑列表提取特徵
    
    Args:
        face_paths: 人臉圖片路徑列表
        batch_size: 批次大小
        use_gpu: 是否使用 GPU
        model_path: 模型路徑（可選）
        
    Returns:
        dict: {圖片路徑: 特徵向量} 的字典
    """
    print(f"Extracting ArcFace features from {len(face_paths)} face images...")
    
    # 載入模型
    session = load_arcface_model(model_path=model_path, use_gpu=use_gpu)
    input_name = session.get_inputs()[0].name
    
    facial_encodings = {}
    successful = 0
    failed = 0
    failed_paths = []
    
    # 批次處理
    for i in tqdm(range(0, len(face_paths), batch_size), desc="Extracting features"):
        batch_paths = face_paths[i:i+batch_size]
        batch_images = []
        valid_paths = []
        
        # 準備批次
        for path in batch_paths:
            try:
                # 讀取圖片
                img = cv2.imread(path)
                if img is None:
                    failed += 1
                    failed_paths.append(path)
                    continue
                
                # 預處理
                processed_img = preprocess_face(img)
                batch_images.append(processed_img)
                valid_paths.append(path)
                
            except Exception as e:
                failed += 1
                failed_paths.append(path)
                continue
        
        if len(batch_images) == 0:
            continue
        
        # 批次推理
        try:
            # 合併批次
            batch_array = np.vstack(batch_images)
            
            # ONNX 推理
            embeddings = session.run(None, {input_name: batch_array})[0]
            
            # 處理結果
            for path, embedding in zip(valid_paths, embeddings):
                # L2 正規化
                norm = np.linalg.norm(embedding)
                if norm > 1e-8:
                    embedding = embedding / norm
                    facial_encodings[path] = embedding
                    successful += 1
                else:
                    failed += 1
                    failed_paths.append(path)
                    
        except Exception as e:
            # 批次失敗，嘗試單個處理
            for path, img_data in zip(valid_paths, batch_images):
                embedding = extract_single_feature(session, img_data, input_name)
                if embedding is not None:
                    facial_encodings[path] = embedding
                    successful += 1
                else:
                    failed += 1
                    failed_paths.append(path)
    
    # 打印結果
    print(f"✅ Feature extraction completed: {successful} successful, {failed} failed")
    
    if failed > 0:
        failure_rate = failed / len(face_paths) * 100
        print(f"   Failure rate: {failure_rate:.1f}%")
        
        if failure_rate > 10:
            print(f"⚠️ High failure rate! First 3 failed images:")
            for path in failed_paths[:3]:
                print(f"      {path}")
                try:
                    img = cv2.imread(path)
                    if img is None:
                        print(f"         → Cannot read image")
                    else:
                        print(f"         → Image shape: {img.shape}, dtype: {img.dtype}")
                except Exception as e:
                    print(f"         → Error: {e}")
    
    if len(facial_encodings) > 0:
        sample_feature = list(facial_encodings.values())[0]
        print(f"   Feature dimension: {sample_feature.shape[0]}")
        print(f"   Feature norm: {np.linalg.norm(sample_feature):.3f}")
        print(f"   Feature range: [{sample_feature.min():.3f}, {sample_feature.max():.3f}]")
    else:
        print("❌ ERROR: No features extracted!")
    
    return facial_encodings


def extract_features_batch(session, face_images, use_gpu=True):
    """
    批次提取特徵（用於 video annotation）
    
    Args:
        session: ONNX session（如果為 None 會自動載入）
        face_images: numpy array of face images, shape (N, H, W, 3), BGR format
        use_gpu: 是否使用 GPU
        
    Returns:
        numpy array: 特徵矩陣, shape (N, feature_dim)
    """
    if session is None:
        session = load_arcface_model(use_gpu=use_gpu)
    
    input_name = session.get_inputs()[0].name
    
    # 預處理所有圖片
    processed_images = []
    for img in face_images:
        try:
            processed_img = preprocess_face(img)
            processed_images.append(processed_img)
        except:
            # 失敗則添加零數據
            processed_images.append(np.zeros((1, 3, 112, 112), dtype=np.float32))
    
    # 合併批次
    batch_array = np.vstack(processed_images)
    
    try:
        # ONNX 推理
        embeddings = session.run(None, {input_name: batch_array})[0]
        
        # L2 正規化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms
        
        return embeddings
        
    except Exception as e:
        print(f"Batch inference failed: {e}")
        # 返回零向量
        feature_dim = 512  # ArcFace 標準輸出維度
        return np.zeros((len(face_images), feature_dim), dtype=np.float32)


# ============================================================================
# 測試和工具函數
# ============================================================================

def test_on_image(image_path, model_path=None):
    """
    測試單張圖片的特徵提取
    
    Args:
        image_path: 圖片路徑
        model_path: 模型路徑
    """
    print(f"Testing feature extraction on: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # 載入模型
    session = load_arcface_model(model_path=model_path, use_gpu=True)
    input_name = session.get_inputs()[0].name
    
    # 讀取圖片
    img = cv2.imread(image_path)
    print(f"Image shape: {img.shape}")
    
    # 預處理
    processed = preprocess_face(img)
    print(f"Preprocessed shape: {processed.shape}")
    
    # 提取特徵
    embedding = extract_single_feature(session, processed, input_name)
    
    if embedding is not None:
        print(f"✅ Feature extraction successful!")
        print(f"   Feature shape: {embedding.shape}")
        print(f"   Feature norm: {np.linalg.norm(embedding):.6f}")
        print(f"   Feature stats:")
        print(f"      Mean: {embedding.mean():.6f}")
        print(f"      Std: {embedding.std():.6f}")
        print(f"      Min: {embedding.min():.6f}")
        print(f"      Max: {embedding.max():.6f}")
        return embedding
    else:
        print(f"❌ Feature extraction failed")
        return None


def compare_two_faces(image_path1, image_path2, model_path=None):
    """
    比較兩張人臉的相似度
    
    Args:
        image_path1: 第一張圖片路徑
        image_path2: 第二張圖片路徑
        model_path: 模型路徑
    """
    print(f"Comparing faces:")
    print(f"  Image 1: {image_path1}")
    print(f"  Image 2: {image_path2}")
    
    # 提取特徵
    encodings = extract_features_from_paths(
        [image_path1, image_path2],
        batch_size=2,
        use_gpu=True,
        model_path=model_path
    )
    
    if len(encodings) == 2:
        feat1 = encodings[image_path1]
        feat2 = encodings[image_path2]
        
        # 計算餘弦相似度
        similarity = np.dot(feat1, feat2)
        
        print(f"\n✅ Comparison result:")
        print(f"   Cosine similarity: {similarity:.6f}")
        
        if similarity > 0.7:
            print(f"   → Same person (high confidence)")
        elif similarity > 0.5:
            print(f"   → Likely same person")
        elif similarity > 0.3:
            print(f"   → Uncertain")
        else:
            print(f"   → Different person")
        
        return similarity
    else:
        print(f"❌ Failed to extract features from one or both images")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ArcFace Feature Extractor - Standalone Version')
    parser.add_argument('--test_image', type=str, help='Test single image')
    parser.add_argument('--compare', nargs=2, metavar=('IMG1', 'IMG2'), help='Compare two images')
    parser.add_argument('--model', type=str, default=None, help='Path to ArcFace ONNX model')
    
    args = parser.parse_args()
    
    if args.test_image:
        # 測試單張圖片
        test_on_image(args.test_image, args.model)
        
    elif args.compare:
        # 比較兩張圖片
        compare_two_faces(args.compare[0], args.compare[1], args.model)
        
    else:
        # 默認：測試模型載入
        print("ArcFace Feature Extractor - Standalone Version")
        print("=" * 60)
        
        try:
            session = load_arcface_model(model_path=args.model, use_gpu=True)
            print("\n✅ Model loaded successfully!")
            print("\nUsage examples:")
            print("  # Test single image:")
            print(f"  python {__file__} --test_image path/to/face.jpg")
            print("\n  # Compare two faces:")
            print(f"  python {__file__} --compare face1.jpg face2.jpg")
            print("\n  # Specify model path:")
            print(f"  python {__file__} --model path/to/model.onnx --test_image face.jpg")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nPlease download ArcFace model:")
            print("  1. Download from: https://github.com/onnx/models/tree/main/vision/body_analysis/arcface")
            print("  2. Place at: models/arcface/arcface_r100.onnx")