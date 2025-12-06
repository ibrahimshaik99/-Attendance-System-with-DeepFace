# Check enrollment status
import os
import pickle

ENC_FILE = "encodings_deepface.pickle"
EMPLOYEES_DIR = "employees"

print("="*60)
print("ENROLLMENT STATUS CHECK")
print("="*60)
print()

# Check employees folder
print("[1] Employee Folders:")
if os.path.exists(EMPLOYEES_DIR):
    folders = [d for d in os.listdir(EMPLOYEES_DIR) if os.path.isdir(os.path.join(EMPLOYEES_DIR, d))]
    for folder in sorted(folders):
        folder_path = os.path.join(EMPLOYEES_DIR, folder)
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  ✓ {folder}: {len(images)} images")
else:
    print("  ✗ employees/ folder not found!")
print()

# Check encodings file
print("[2] Encodings File:")
if os.path.exists(ENC_FILE):
    with open(ENC_FILE, "rb") as f:
        data = pickle.load(f)
    
    embeddings = data["embeddings"]
    names = data["names"]
    
    print(f"  ✓ {ENC_FILE} exists")
    print(f"  ✓ Total encodings: {len(embeddings)}")
    print()
    
    # Count per employee
    print("[3] Enrolled Employees:")
    unique_names = sorted(set(names))
    for name in unique_names:
        count = names.count(name)
        print(f"  ✓ {name}: {count} encodings")
    print()
    
    # Check for Madhu Annayya specifically
    print("[4] Madhu Annayya Status:")
    madhu_variants = [n for n in unique_names if 'madhu' in n.lower()]
    if madhu_variants:
        for variant in madhu_variants:
            count = names.count(variant)
            print(f"  ✓ Found: '{variant}' with {count} encodings")
    else:
        print("  ✗ NOT FOUND in encodings!")
        print("  → Check folder name: should be 'madhu annayya' (lowercase)")
else:
    print(f"  ✗ {ENC_FILE} not found!")
    print("  → Run: python enroll_deepface.py")

print()
print("="*60)
print("RECOMMENDATIONS:")
print("="*60)

if not os.path.exists(ENC_FILE):
    print("  1. Run enrollment: python enroll_deepface.py")
elif 'madhu_variants' in locals() and not madhu_variants:
    print("  1. Check folder name is exactly: 'madhu annayya' (lowercase)")
    print("  2. Ensure folder has images")
    print("  3. Re-run enrollment: python enroll_deepface.py")
else:
    print("  ✓ All employees enrolled correctly")
    print("  → If still not recognizing, increase TOLERANCE to 0.60")

print()
