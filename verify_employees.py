# Verify employee discrimination - Test if system can distinguish between similar faces
import os
import pickle
import numpy as np
from deepface import DeepFace
import cv2

ENC_FILE = "encodings_deepface.pickle"
MODEL_NAME = "ArcFace"

def cosine_distance(a, b):
    a = a.flatten()
    b = b.flatten()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return 1.0 - (np.dot(a, b) / denom)

def load_encodings():
    if not os.path.exists(ENC_FILE):
        print(f"[ERROR] {ENC_FILE} not found. Run enroll_deepface.py first.")
        return None, None
    with open(ENC_FILE, "rb") as f:
        data = pickle.load(f)
    embeddings = [np.array(x, dtype=np.float32) for x in data["embeddings"]]
    names = data["names"]
    return embeddings, names

def analyze_discrimination():
    print("="*60)
    print("EMPLOYEE DISCRIMINATION ANALYSIS")
    print("="*60)
    print()
    
    embeddings, names = load_encodings()
    if embeddings is None:
        return
    
    # Get unique employees
    unique_names = sorted(set(names))
    print(f"[INFO] Found {len(unique_names)} unique employees:")
    for name in unique_names:
        count = names.count(name)
        print(f"  - {name}: {count} encodings")
    print()
    
    # Calculate average embeddings per employee
    employee_avg_embeddings = {}
    for name in unique_names:
        indices = [i for i, n in enumerate(names) if n == name]
        avg_emb = np.mean([embeddings[i] for i in indices], axis=0)
        employee_avg_embeddings[name] = avg_emb
    
    # Calculate inter-employee distances
    print("="*60)
    print("INTER-EMPLOYEE DISTANCES (Lower = More Similar)")
    print("="*60)
    print()
    
    employees_list = list(employee_avg_embeddings.keys())
    for i, emp1 in enumerate(employees_list):
        for emp2 in employees_list[i+1:]:
            dist = cosine_distance(employee_avg_embeddings[emp1], employee_avg_embeddings[emp2])
            status = "⚠ VERY SIMILAR" if dist < 0.30 else "✓ DISTINGUISHABLE"
            print(f"{emp1} <-> {emp2}")
            print(f"  Distance: {dist:.4f} {status}")
            print()
    
    # Calculate intra-employee distances (consistency)
    print("="*60)
    print("INTRA-EMPLOYEE CONSISTENCY (Lower = More Consistent)")
    print("="*60)
    print()
    
    for name in unique_names:
        indices = [i for i, n in enumerate(names) if n == name]
        if len(indices) < 2:
            print(f"{name}: Only 1 encoding (need more images)")
            continue
        
        embs = [embeddings[i] for i in indices]
        distances = []
        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                distances.append(cosine_distance(embs[i], embs[j]))
        
        avg_dist = np.mean(distances)
        max_dist = np.max(distances)
        min_dist = np.min(distances)
        
        print(f"{name}:")
        print(f"  Avg distance: {avg_dist:.4f}")
        print(f"  Min distance: {min_dist:.4f}")
        print(f"  Max distance: {max_dist:.4f}")
        print(f"  Consistency: {'✓ GOOD' if avg_dist < 0.20 else '⚠ VARIABLE'}")
        print()
    
    # Recommendations
    print("="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print()
    
    # Check if any employees are too similar
    min_inter_dist = float('inf')
    similar_pair = None
    for i, emp1 in enumerate(employees_list):
        for emp2 in employees_list[i+1:]:
            dist = cosine_distance(employee_avg_embeddings[emp1], employee_avg_embeddings[emp2])
            if dist < min_inter_dist:
                min_inter_dist = dist
                similar_pair = (emp1, emp2)
    
    if min_inter_dist < 0.30:
        print(f"⚠ WARNING: {similar_pair[0]} and {similar_pair[1]} are very similar!")
        print(f"  Distance: {min_inter_dist:.4f}")
        print(f"  Recommended TOLERANCE: 0.35 or lower")
        print(f"  Current TOLERANCE: Check attendence_deepface.py")
        print()
        print("  Solutions:")
        print("  1. Lower TOLERANCE in attendence_deepface.py (line 23)")
        print("  2. Add more diverse images for each employee")
        print("  3. Ensure good lighting and clear face shots")
        print("  4. Re-enroll with better quality images")
    elif min_inter_dist < 0.40:
        print(f"⚠ CAUTION: {similar_pair[0]} and {similar_pair[1]} are somewhat similar")
        print(f"  Distance: {min_inter_dist:.4f}")
        print(f"  Recommended TOLERANCE: 0.40 or lower")
        print(f"  System should work but monitor for misidentifications")
    else:
        print(f"✓ All employees are well-distinguished")
        print(f"  Minimum distance: {min_inter_dist:.4f}")
        print(f"  Recommended TOLERANCE: 0.45 or lower")
        print(f"  System should work reliably")
    
    print()
    print("="*60)

if __name__ == "__main__":
    analyze_discrimination()
