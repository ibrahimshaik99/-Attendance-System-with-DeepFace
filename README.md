# Attendance System with DeepFace

A complete facial recognition-based attendance system using DeepFace, RetinaFace, and ArcFace embeddings.

## Features

- **Facial Recognition**: Uses ArcFace embeddings for accurate face recognition
- **Face Detection**: RetinaFace for robust face detection
- **Attendance Logging**: Automatic check-in/check-out tracking to Excel/CSV
- **Demo Mode**: Test without camera
- **Live Camera Mode**: Stream-based real-time recognition

## Project Structure

```
attendence_deepface/
├── attendence_deepface.py      # Main attendance tracking script
├── enroll_deepface.py          # Employee enrollment script
├── employees/                  # Employee face images directory
│   └── shaik ibrahim/          # Example employee folder
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── output/                     # Generated attendance Excel/CSV files
│   ├── attendance_2025-11-24.xlsx
│   └── attendance_2025-11-24.csv
├── encodings_deepface.pickle   # Serialized face embeddings database
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

### 1. Create Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

Main packages:
- `opencv-python` - Image processing
- `deepface` - Face embeddings
- `retinaface` - Face detection
- `tensorflow` - Deep learning backend
- `tf-keras` - Keras compatibility layer
- `pandas` - Excel/CSV handling
- `imutils` - Image utilities
- `pytz` - Timezone handling

## Usage

### Step 1: Enroll Employees

Prepare employee images:

1. Create folder: `employees/<employee_name>/`
2. Add 1+ clear face images per employee (JPG/PNG)
3. Ensure images have good lighting and frontal face orientation

Run enrollment:

```powershell
.\venv\Scripts\python.exe .\enroll_deepface.py
```

**Output**: `encodings_deepface.pickle` (serialized embeddings database)

### Step 2: Run Attendance System

#### Option A: Demo Mode (Recommended for Testing)

```powershell
$env:ATTENDANCE_MODE="demo"
.\venv\Scripts\python.exe .\attendence_deepface.py
```

This generates sample attendance records for all enrolled employees without requiring a camera.

#### Option B: Live Camera Mode

Set your camera URL in `attendence_deepface.py`:

```python
CAMERA_URL = "http://192.168.1.x:8080/video"  # Phone IP Webcam or USB camera
```

Run:

```powershell
.\venv\Scripts\python.exe .\attendence_deepface.py
```

Press `q` to quit.

**Output**: 
- `output/attendance_YYYY-MM-DD.xlsx` - Excel attendance sheet
- `output/attendance_YYYY-MM-DD.csv` - CSV backup

## Configuration

Edit `attendence_deepface.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODE` | "demo" | "demo" or "camera" |
| `CAMERA_URL` | "http://191.168.1.174:8080/video" | Camera stream URL |
| `TOLERANCE` | 0.40 | Face match threshold (lower = stricter) |
| `EARLIEST_CHECKIN` | 05:00:00 | Earliest allowed check-in time |
| `LATEST_CHECKOUT` | 22:00:00 | Latest allowed check-out time |
| `MIN_DETECTION_COOLDOWN` | 20 | Seconds between same person check-ins |
| `TIMEZONE` | "Asia/Kolkata" | Timezone for timestamps |

## Excel Output Format

Generated attendance sheets contain:

| Column | Description | Example |
|--------|-------------|---------|
| Employee | Employee name | shaik ibrahim |
| Check-in | First detection time (HH:MM:SS) | 08:30:45 |
| Check-out | Last detection time (HH:MM:SS) | 17:15:30 |
| LateMinutes | Minutes late from 09:00 | 0 (if before 09:00) |
| LastSeen | Last detection timestamp | 17:15:30 |
| Camera | Camera label | PhoneCam |

## Troubleshooting

### "Encodings file not found"
- Run `enroll_deepface.py` first
- Ensure `employees/<name>/` folders exist with images

### "Cannot open camera URL"
- Verify phone/camera is on same network
- Use demo mode instead: `$env:ATTENDANCE_MODE="demo"`
- Check URL format (http, not https for local streams)

### Low recognition accuracy
- Add more images per employee (5-10 best practice)
- Improve lighting in images
- Reduce `TOLERANCE` value (e.g., 0.35)
- Ensure frontal face orientation

### Models not downloading
- Check internet connection
- Models auto-download to `~/.deepface/weights/`
- First run may take 2-3 minutes

## Performance Notes

- **Speed**: Demo mode ~5 sec, Camera mode ~20-30 FPS (GPU recommended)
- **Accuracy**: ArcFace achieves 99.83% accuracy on aligned faces
- **Storage**: Each employee embedding ~4 KB

## API Keys / Dependencies

No API keys required. Uses only open-source models:
- RetinaFace (face detection)
- ArcFace (embeddings)
- TensorFlow/Keras (deep learning)

## License

This project uses open-source packages. See `requirements.txt` for individual licenses.

## Support

For issues:
1. Check error message in console
2. Verify virtual environment is activated
3. Ensure all dependencies installed: `pip install -r requirements.txt`
4. Check image paths and permissions
5. Try demo mode to isolate camera issues

## Example Workflow

```bash
# 1. Activate venv
.\venv\Scripts\Activate.ps1

# 2. Create employee folder with images
mkdir employees\john_doe
# Add john_doe.jpg to folder

# 3. Enroll employees
python enroll_deepface.py
# Output: encodings_deepface.pickle

# 4. Test with demo mode
$env:ATTENDANCE_MODE="demo"
python attendence_deepface.py
# Output: output\attendance_2025-11-24.xlsx

# 5. View Excel file to verify format
```

Enjoy accurate, automated attendance tracking!
