#!/usr/bin/env python3
# enroll_deepface.py - Optimized enrollment
import os
import pickle
from deepface import DeepFace
import cv2
import numpy as np
from imutils import paths

EMPLOYEES_DIR = "employees"
ENC_FILE = "encodings_deepface.pickle"
MODEL_NAME = "ArcFace"


def detect_and_crop_dnn(img_bgr, net, conf_threshold=0.5):
    """Detect faces using OpenCV DNN (faster and more reliable)."""
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    crops = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crop = img_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
    return crops


def get_embedding(face_bgr):
    """Compute embedding with preprocessing."""
    # Resize for consistency
    face_bgr = cv2.resize(face_bgr, (160, 160))
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    emb = DeepFace.represent(face_rgb, model_name=MODEL_NAME, enforce_detection=False, detector_backend='skip')
    if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], dict) and 'embedding' in emb[0]:
        emb_vec = np.array(emb[0]['embedding'])
    elif isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], (list, tuple, np.ndarray)):
        emb_vec = np.array(emb[0])
    else:
        emb_vec = np.array(emb)
    return emb_vec.flatten().astype(np.float32)


def enroll():
    """Enroll all employees with optimized detection."""
    if not os.path.exists(EMPLOYEES_DIR):
        print(f"[ERROR] Directory '{EMPLOYEES_DIR}' not found.")
        print(f"[INFO] Create '{EMPLOYEES_DIR}' and add subfolders for each employee with images.")
        return

    print("[INFO] Loading model...")
    model = DeepFace.build_model(MODEL_NAME)
    
    # Load face detector
    print("[INFO] Loading face detector...")
    model_file = "res10_300x300_ssd_iter_140000.caffemodel"
    config_file = "deploy.prototxt"
    if not os.path.exists(model_file):
        print("[INFO] Downloading face detector...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            model_file)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            config_file)
    face_net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    
    known_embeddings = []
    known_names = []

    folders = [d for d in os.listdir(EMPLOYEES_DIR) if os.path.isdir(os.path.join(EMPLOYEES_DIR, d))]
    if not folders:
        print("[ERROR] No employee folders found.")
        print("[INFO] Add employees/<name> with images inside.")
        return

    for person in folders:
        person_path = os.path.join(EMPLOYEES_DIR, person)
        image_paths = list(paths.list_images(person_path))
        if not image_paths:
            print(f"[WARN] No images for {person}, skipping.")
            continue
        print(f"[ENROLL] {person} - {len(image_paths)} images")
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [SKIP] failed to read {img_path}")
                continue
            
            # Enhance image quality
            img = cv2.resize(img, None, fx=1.0, fy=1.0, interpolation=cv2.INTER_CUBIC)
            
            crops = detect_and_crop_dnn(img, face_net, conf_threshold=0.5)
            if not crops:
                # Fallback: try DeepFace detection
                try:
                    emb = DeepFace.represent(img, model_name=MODEL_NAME, enforce_detection=True, detector_backend='opencv')
                    if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], dict) and 'embedding' in emb[0]:
                        emb_vec = np.array(emb[0]['embedding'])
                    elif isinstance(emb, list) and len(emb) > 0:
                        emb_vec = np.array(emb[0])
                    else:
                        emb_vec = np.array(emb)
                    known_embeddings.append(emb_vec.flatten().astype(np.float32))
                    known_names.append(person)
                    print(f"  [+] enrolled via fallback")
                except:
                    print(f"  [-] no face: {os.path.basename(img_path)}")
                continue
            
            for i, face_crop in enumerate(crops):
                try:
                    emb = get_embedding(face_crop)
                    known_embeddings.append(emb)
                    known_names.append(person)
                    print(f"  [+] face {i+1} enrolled")
                except Exception as e:
                    print(f"  [-] error: {e}")

    if len(known_names) == 0:
        print("[ERROR] No faces enrolled. Check image quality.")
        return

    data = {"embeddings": known_embeddings, "names": known_names, "model_name": MODEL_NAME}
    with open(ENC_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"\n[SUCCESS] Saved {len(known_names)} embeddings to {ENC_FILE}")
    print(f"[INFO] Employees enrolled: {set(known_names)}")
    for name in set(known_names):
        count = known_names.count(name)
        print(f"  - {name}: {count} encodings")


if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZED EMPLOYEE ENROLLMENT")
    print("=" * 60)
    enroll()
    print("=" * 60)
