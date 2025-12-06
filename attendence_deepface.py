# attendance_deepface.py - Optimized Real-time IP Camera Face Recognition
import cv2
import os
import time
import pickle
from datetime import datetime, time as dtime
import numpy as np
import pandas as pd
import pytz
from collections import defaultdict, deque
from threading import Thread, Lock
from queue import Queue
from deepface import DeepFace

# ========== CONFIG =======================
CAMERA_URL = "http://191.168.x.xx:8080/video"
ENC_FILE = "encodings_deepface.pickle"
EMPLOYEES_DIR = "employees"
OUTPUT_DIR = "output"
CAMERA_LABEL = "IP_CAMERA"
TIMEZONE = "Asia/Kolkata"
MODEL_NAME = "ArcFace"
TOLERANCE = 0.60  # Increased for Madhu Annayya recognition
FRAME_RESIZE_WIDTH = 320  # Smaller for speed
PROCESS_EVERY_N_FRAMES = 2  # Process more frequently
MIN_DETECTION_COOLDOWN = 10  # Reduced cooldown
MIN_FACE_SIZE = 40  # Lower minimum face size
EARLIEST_CHECKIN = dtime(5, 0, 0)
LATEST_CHECKOUT = dtime(22, 0, 0)
CHECKIN_DEADLINE = dtime(11, 0, 0)
CHECKOUT_TIME = dtime(18, 0, 0)
# ============================

tz = pytz.timezone(TIMEZONE)

def now_local():
    return datetime.now(tz)

def now_utc():
    return datetime.utcnow()

def date_str(dt=None):
    if dt is None:
        dt = now_local()
    return dt.strftime("%Y-%m-%d")

def time_str(dt=None):
    if dt is None:
        dt = now_local()
    return dt.strftime("%H:%M:%S")

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_encodings():    
    if not os.path.exists(ENC_FILE):
        raise FileNotFoundError(f"[ERROR] {ENC_FILE} not found. Run enroll_deepface.py first.")
    with open(ENC_FILE, "rb") as f:
        data = pickle.load(f)
    embeddings = [np.array(x, dtype=np.float32) for x in data["embeddings"]]
    names = data["names"]
    return embeddings, names

def cosine_distance(a, b):
    a = a.flatten()
    b = b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return 1.0 - (np.dot(a, b) / denom)

# Threaded video capture for responsiveness
class VideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = Lock()
    
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
    
    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed, self.frame = grabbed, frame
            
    def read(self):
        with self.lock:
            return self.grabbed, self.frame.copy() if self.frame is not None else None
    
    def stop(self):
        self.stopped = True
        self.stream.release()

# Optimized face detection using OpenCV DNN (faster than RetinaFace)
def detect_faces_dnn(img_bgr, net, conf_threshold=0.3):
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    return boxes

def crop_face(img, box, margin=0.2):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    mx, my = int(w * margin), int(h * margin)
    sx1 = max(0, x1 - mx)
    sy1 = max(0, y1 - my)
    sx2 = min(img.shape[1], x2 + mx)
    sy2 = min(img.shape[0], y2 + my)
    return img[sy1:sy2, sx1:sx2]

def load_or_create_attendance_df(date):
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, f"attendance_{date}.xlsx")
    if os.path.exists(path):
        df = pd.read_excel(path)
    else:
        df = pd.DataFrame(columns=["Employee", "Check-in", "Check-out", "LateMinutes", "LateStatus", "LastSeen", "Camera"])
    return df, path

def save_attendance(df, path):
    try:
        df.to_excel(path, index=False)
    except PermissionError:
        alt_path = path.replace('.xlsx', f'_{int(time.time())}.xlsx')
        print(f"[WARN] File locked, saving to {alt_path}")
        df.to_excel(alt_path, index=False)
        path = alt_path
        #original_path = path
    
    csv_path = path.replace(".xlsx", ".csv")
    df.to_csv(csv_path, index=False)
    
    try:
        from openpyxl import load_workbook
        from openpyxl.styles import PatternFill
        wb = load_workbook(path)
        ws = wb.active
        header = [cell.value for cell in ws[1]]
        if "LateStatus" in header:
            col_idx = header.index("LateStatus") + 1
            yellow = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
            for row in range(2, ws.max_row + 1):
                cell = ws.cell(row=row, column=col_idx)
                if cell.value and str(cell.value).strip().lower() == "late":
                    cell.fill = yellow
        wb.save(path)
    except:
        pass
    
    print(f"[SAVE] Attendance saved to {path}")

def update_record(df, name, checkin_time, checkout_time, camera):
    if name in df["Employee"].values:
        idx = df.index[df["Employee"] == name][0]
    else:
        idx = len(df)
        df.loc[idx, "Employee"] = name
        df.loc[idx, "Check-in"] = ""
        df.loc[idx, "Check-out"] = ""
        df.loc[idx, "LateMinutes"] = 0
        df.loc[idx, "LateStatus"] = ""
        df.loc[idx, "LastSeen"] = ""
        df.loc[idx, "Camera"] = camera
    
    changed = False
    
    if checkin_time:
        prev = df.at[idx, "Check-in"]
        if pd.isna(prev) or prev == "":
            df.at[idx, "Check-in"] = checkin_time
            changed = True
            try:
                h, m, s = [int(x) for x in checkin_time.split(":")]
                secs = h * 3600 + m * 60 + s
                work_start_secs = 11 * 3600
                late_mins = max(0, int((secs - work_start_secs) / 60))
                df.at[idx, "LateMinutes"] = late_mins
                deadline_secs = CHECKIN_DEADLINE.hour * 3600 + CHECKIN_DEADLINE.minute * 60
                df.at[idx, "LateStatus"] = "Late" if secs > deadline_secs else "OnTime"
            except:
                df.at[idx, "LateMinutes"] = 0
                df.at[idx, "LateStatus"] = ""
    
    if checkout_time:
        prev_co = df.at[idx, "Check-out"]
        if pd.isna(prev_co) or prev_co == "":
            df.at[idx, "Check-out"] = checkout_time
            changed = True
        else:
            try:
                if datetime.strptime(checkout_time, "%H:%M:%S") > datetime.strptime(str(prev_co), "%H:%M:%S"):
                    df.at[idx, "Check-out"] = checkout_time
                    changed = True
            except:
                pass
    
    df.at[idx, "LastSeen"] = time_str()
    df.at[idx, "Camera"] = camera
    return df, changed

# Cache for embeddings to avoid recomputation
embedding_cache = {}

def recognize_face_fast(face_crop, embeddings_db, names_db):
    if face_crop.shape[0] < MIN_FACE_SIZE or face_crop.shape[1] < MIN_FACE_SIZE:
        return "Unknown", 1.0
    
    # Resize to standard size for speed
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
    
    # Compute distances to all enrolled faces
    distances = [cosine_distance(emb_vec, db_emb) for db_emb in embeddings_db]
    best_idx = np.argmin(distances)
    best_dist = distances[best_idx]
    
    # Group distances by employee name to get best match per person
    employee_distances = {}
    for idx, dist in enumerate(distances):
        emp_name = names_db[idx]
        if emp_name not in employee_distances:
            employee_distances[emp_name] = []
        employee_distances[emp_name].append(dist)
    
    # Get average distance per employee
    employee_avg_dist = {name: np.mean(dists) for name, dists in employee_distances.items()}
    
    # Find best matching employee
    best_employee = min(employee_avg_dist, key=employee_avg_dist.get)
    best_avg_dist = employee_avg_dist[best_employee]
    
    # Get second best for margin check
    sorted_employees = sorted(employee_avg_dist.items(), key=lambda x: x[1])
    second_best_dist = sorted_employees[1][1] if len(sorted_employees) > 1 else 1.0
    margin = second_best_dist - best_avg_dist
    
    # Accept if distance is below threshold, regardless of margin
    if best_avg_dist <= TOLERANCE:
        print(f"[MATCH] {best_employee} (dist={best_avg_dist:.3f}, margin={margin:.3f})")
        return best_employee, best_avg_dist
    else:
        print(f"[REJECT] No match: best={best_employee} dist={best_avg_dist:.3f} > threshold={TOLERANCE}")
        return "Unknown", best_avg_dist

def check_and_reenroll():
    """Check if new employees added and reenroll if needed."""
    if not os.path.exists(EMPLOYEES_DIR):
        return False
    folders = [d for d in os.listdir(EMPLOYEES_DIR) if os.path.isdir(os.path.join(EMPLOYEES_DIR, d))]
    if not os.path.exists(ENC_FILE):
        return True  # Need to enroll
    try:
        with open(ENC_FILE, "rb") as f:
            data = pickle.load(f)
        enrolled_names = set(data["names"])
        current_names = set(folders)
        if current_names != enrolled_names:
            print(f"[INFO] New employees detected: {current_names - enrolled_names}")
            return True
    except:
        return True
    return False

def main():
    print("=" * 60)
    print("OPTIMIZED IP CAMERA ATTENDANCE SYSTEM")
    print("=" * 60)

    # Check for new employees and reenroll if needed
    if check_and_reenroll():
        print("[INFO] Re-enrolling employees...")
        import subprocess
        result = subprocess.run(["python", "enroll_deepface.py"], capture_output=True, text=True)
        if result.returncode != 0:
            print("[ERROR] Re-enrollment failed:", result.stderr)
            return
        print("[INFO] Re-enrollment completed.")

    print(f"[1/5] Loading encodings from {ENC_FILE}...")
    embeddings_db, names_db = load_encodings()
    print(f"[INFO] Loaded {len(names_db)} encodings for {len(set(names_db))} employees")
    for name in set(names_db):
        print(f"      - {name}")
    
    # Load DeepFace model
    print(f"\n[2/5] Loading {MODEL_NAME} model...")
    model = DeepFace.build_model(MODEL_NAME)
    print("[INFO] Model loaded")
    
    # Load OpenCV DNN face detector (faster than RetinaFace)
    print(f"\n[3/5] Loading face detector...")
    model_file = "res10_300x300_ssd_iter_140000.caffemodel"
    config_file = "deploy.prototxt"
    
    # Download if not exists
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
    print("[INFO] Face detector loaded")
    print("")
    
    # Connect to IP camera with threading
    print(f"\n[4/5] Connecting to camera: {CAMERA_URL}")
    vs = VideoStream(CAMERA_URL).start()
    time.sleep(2.0)  # Warm up
    
    ret, test_frame = vs.read()
    if not ret or test_frame is None:
        print(f"[ERROR] Cannot connect to {CAMERA_URL}")
        return
    
    print("[INFO] Camera connected (threaded)")
    
    # Initialize tracking
    print(f"\n[5/5] Starting attendance tracking...")
    current_date = date_str()
    df, out_path = load_or_create_attendance_df(current_date)
    last_seen = defaultdict(lambda: datetime(1970, 1, 1, tzinfo=tz))
    last_recognition = {}  # Track last recognition per face position
    frame_count = 0
    save_lock = Lock()
    print("\n" + "=" * 60)
    print("SYSTEM READY - Monitoring (optimized)")
    print("Press 'q' to quit")
    print("=" * 60 + "\n")
    
    try:
        while True:
            ret, frame = vs.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Resize for faster processing
            h, w = frame.shape[:2]
            if w > FRAME_RESIZE_WIDTH:
                scale = FRAME_RESIZE_WIDTH / float(w)
                frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
            else:
                frame_small = frame.copy()
            
            # Process every N frames
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                boxes = detect_faces_dnn(frame_small, face_net, conf_threshold=0.3)

                # Calculate scale to map boxes back to original frame
                scale = float(w) / FRAME_RESIZE_WIDTH

                for box in boxes:
                    x1, y1, x2, y2 = box
                    # Scale box back to original frame coordinates
                    orig_x1 = int(x1 * scale)
                    orig_y1 = int(y1 * scale)
                    orig_x2 = int(x2 * scale)
                    orig_y2 = int(y2 * scale)

                    face_crop = crop_face(frame, (orig_x1, orig_y1, orig_x2, orig_y2), margin=0.2)
                    if face_crop.size == 0 or face_crop.shape[0] < MIN_FACE_SIZE or face_crop.shape[1] < MIN_FACE_SIZE:
                        continue

                    # Fast recognition (no consensus delay)
                    recognized, dist = recognize_face_fast(face_crop, embeddings_db, names_db)
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = box
                    color = (0, 255, 0) if recognized != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame_small, (x1, y1), (x2, y2), color, 2)
                    label = f"{recognized} ({dist:.2f})"
                    cv2.putText(frame_small, label, (x1, max(15, y1 - 10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Instant Excel update when recognized
                    if recognized != "Unknown":
                        now_dt = now_local()
                        curtime = now_dt.time()
                        secs_since_last = (now_dt - last_seen[recognized]).total_seconds()
                        
                        if secs_since_last >= MIN_DETECTION_COOLDOWN:
                            last_seen[recognized] = now_dt
                            
                            checkin_time = None
                            checkout_time = None
                            
                            if recognized in df["Employee"].values:
                                r_idx = df.index[df["Employee"] == recognized][0]
                                prev_ci = df.at[r_idx, "Check-in"]
                            else:
                                prev_ci = ""
                            
                            if (pd.isna(prev_ci) or prev_ci == "") and (EARLIEST_CHECKIN <= curtime <= LATEST_CHECKOUT):
                                checkin_time = time_str(now_dt)
                            
                            if curtime >= CHECKOUT_TIME:
                                checkout_time = time_str(now_dt)
                            
                            df, changed = update_record(df, recognized, checkin_time, checkout_time, CAMERA_LABEL)
                            
                            if changed:
                                # Instant save in background
                                Thread(target=save_attendance, args=(df.copy(), out_path), daemon=True).start()
                                status = []
                                if checkin_time:
                                    status.append(f"CHECK-IN: {checkin_time}")
                                if checkout_time:
                                    status.append(f"CHECK-OUT: {checkout_time}")
                                print(f"[✓] {recognized} | {' | '.join(status)} | Dist: {dist:.3f}")
            
            cv2.imshow("Attendance - Press 'q' to quit", frame_small)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[QUIT] User quit")
                break
            
            new_date = date_str()
            if new_date != current_date:
                save_attendance(df, out_path)
                current_date = new_date
                df, out_path = load_or_create_attendance_df(current_date)
                print(f"\n[INFO] New date: {current_date}")
    
    except KeyboardInterrupt:
        print("\n[QUIT] Interrupted")
    
    finally:
        save_attendance(df, out_path)
        vs.stop()
        cv2.destroyAllWindows()
        print(f"\n[SUCCESS] Saved to {out_path}")
        print("=" * 60)

if __name__ == "__main__":
    main()
