#!/usr/bin/env python3
import os
import pickle
import pandas as pd

print("=" * 60)
print("SYSTEM VERIFICATION REPORT")
print("=" * 60)

# Check files
files = {
    'attendence_deepface.py': 'Main attendance script',
    'enroll_deepface.py': 'Enrollment script',
    'encodings_deepface.pickle': 'Face embeddings database',
    'README.md': 'Full documentation',
    'QUICK_START.md': 'Quick start guide',
    'requirements.txt': 'Dependencies'
}

print("\n[FILES]")
for fname, desc in files.items():
    exists = 'OK' if os.path.exists(fname) else 'MISSING'
    print(f'  [{exists:7s}] {fname:30s} - {desc}')

# Check encodings
print("\n[ENCODINGS DATABASE]")
if os.path.exists('encodings_deepface.pickle'):
    with open('encodings_deepface.pickle', 'rb') as f:
        data = pickle.load(f)
    n_emb = len(data['embeddings'])
    n_people = len(set(data['names']))
    print(f'  [OK] Total embeddings: {n_emb}')
    print(f'  [OK] Unique employees: {n_people}')
    print(f'  [OK] Employees: {set(data["names"])}')
else:
    print('  [MISSING] Encodings database not found')

# Check output
print("\n[OUTPUT FILES]")
if os.path.exists('output'):
    xlsx_files = [f for f in os.listdir('output') if f.endswith('.xlsx')]
    if xlsx_files:
        for xlsx in xlsx_files:
            df = pd.read_excel(os.path.join('output', xlsx))
            print(f'  [OK] {xlsx}')
            print(f'       Rows: {len(df)}, Columns: {len(df.columns)}')
    else:
        print('  [INFO] No attendance files yet (run system first)')
else:
    print('  [INFO] Output directory will be created on first run')

# Check employees directory
print("\n[EMPLOYEES DIRECTORY]")
if os.path.exists('employees'):
    emp_dirs = [d for d in os.listdir('employees') if os.path.isdir(os.path.join('employees', d))]
    if emp_dirs:
        for emp in emp_dirs:
            emp_path = os.path.join('employees', emp)
            images = [f for f in os.listdir(emp_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            print(f'  [OK] {emp}: {len(images)} images')
    else:
        print('  [INFO] Empty (add employee folders with images)')
else:
    print('  [INFO] Directory exists but empty')

print("\n" + "=" * 60)
print("STATUS: READY TO USE")
print("=" * 60)
print("\nNext steps:")
print("1. Add employee photos to employees/<name>/")
print("2. Run: python enroll_deepface.py")
print("3. Run: python attendence_deepface.py")
