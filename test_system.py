# Test script to diagnose system issues
import os
import sys
import subprocess

print("="*60)
print("SYSTEM DIAGNOSTIC TEST")
print("="*60)
print()

# Test 1: Python executable
print("[1/6] Python Executable")
print(f"  Path: {sys.executable}")
print(f"  Version: {sys.version}")
print()

# Test 2: Working directory
print("[2/6] Working Directory")
print(f"  Path: {os.getcwd()}")
print()

# Test 3: Required files
print("[3/6] Required Files")
files = {
    'enroll_deepface.py': 'Enrollment script',
    'attendence_deepface.py': 'Attendance script',
    'app.py': 'Flask web app',
    'templates/index.html': 'Frontend HTML',
    'encodings_deepface.pickle': 'Face encodings (optional)'
}

for file, desc in files.items():
    exists = os.path.exists(file)
    status = "✓ Found" if exists else "✗ NOT FOUND"
    print(f"  {status}: {file} ({desc})")
print()

# Test 4: Directories
print("[4/6] Required Directories")
dirs = ['employees', 'output', 'templates', 'static']
for d in dirs:
    exists = os.path.exists(d)
    status = "✓ Exists" if exists else "✗ Missing"
    print(f"  {status}: {d}/")
print()

# Test 5: Python packages
print("[5/6] Python Packages")
packages = ['flask', 'pandas', 'cv2', 'deepface', 'openpyxl']
for pkg in packages:
    try:
        if pkg == 'cv2':
            import cv2
        else:
            __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} - NOT INSTALLED")
print()

# Test 6: Subprocess test
print("[6/6] Subprocess Test")
try:
    result = subprocess.run(
        [sys.executable, '--version'],
        capture_output=True,
        text=True,
        timeout=5
    )
    print(f"  ✓ Subprocess works")
    print(f"  Output: {result.stdout.strip()}")
except Exception as e:
    print(f"  ✗ Subprocess failed: {e}")
print()

print("="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
print()

# Recommendations
print("RECOMMENDATIONS:")
print()

if not os.path.exists('enroll_deepface.py'):
    print("  ✗ enroll_deepface.py not found - Cannot run enrollment")
    
if not os.path.exists('attendence_deepface.py'):
    print("  ✗ attendence_deepface.py not found - Cannot run attendance")
    
if not os.path.exists('encodings_deepface.pickle'):
    print("  ⚠ encodings_deepface.pickle not found - Run enrollment first")
    
if not os.path.exists('employees') or not os.listdir('employees'):
    print("  ⚠ No employees found - Add employees before enrollment")

try:
    import flask
    import pandas
    import cv2
    import deepface
    print("  ✓ All required packages installed")
except ImportError as e:
    print(f"  ✗ Missing package: {e}")
    print("  Run: pip install Flask pandas opencv-python deepface openpyxl")

print()
print("To start web server: python app.py")
print()
