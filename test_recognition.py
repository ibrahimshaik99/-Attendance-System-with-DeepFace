import cv2
import pickle
import numpy as np
from deepface import DeepFace

ENC_FILE = "encodings_deepface.pickle"
MODEL_NAME = "ArcFace"
TOLERANCE = 0.50

def cosine_distance(a, b):
    a = a.flatten()
    b = b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return 1.0 - (np.dot(a, b) / denom)

def load_encodings():
    with open(ENC_FILE, "rb") as f:
        data = pickle.load(f)
    embeddings = [np.array(x, dtype=np.float32) for x in data["embeddings"]]
    names = data["names"]
    return embeddings, names

def recognize_face(face_crop, embeddings_db, names_db):
    face_crop = cv2.resize(face_crop, (112, 112))
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    try:
        emb = DeepFace.represent(face_rgb, model_name=MODEL_NAME, enforce_detection=False, detector_backend='skip')
        if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], dict) and 'embedding' in emb[0]:
            emb_vec = np.array(emb[0]['embedding'], dtype=np.float32).flatten()
        elif isinstance(emb, list) and len(emb) > 0:
            emb_vec = np.array(emb[0], dtype=np.float32).flatten()
        else:
            emb_vec = np.array(emb, dtype=np.float32).flatten()
    except:
        return "Unknown", 1.0

    distances = [cosine_distance(emb_vec, db_emb) for db_emb in embeddings_db]
    best_idx = np.argmin(distances)
    best_dist = distances[best_idx]

    if best_dist <= TOLERANCE:
        return names_db[best_idx], best_dist
    return "Unknown", best_dist

# Load encodings
embeddings_db, names_db = load_encodings()
print("Loaded names:", set(names_db))

# Test on Ramu Annayya's images
for i in [1,2]:
    img_path = f"employees/Ramu Annayya/{i}.jpeg"
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read {img_path}")
        continue

    # Detect face
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for j in range(detections.shape[2]):
        confidence = detections[0, 0, j, 2]
        if confidence > 0.5:
            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            face_crop = img[y1:y2, x1:x2]
            if face_crop.size > 0:
                recognized, dist = recognize_face(face_crop, embeddings_db, names_db)
                print(f"Image {i}: Recognized as {recognized} with dist {dist:.3f}")
                break
