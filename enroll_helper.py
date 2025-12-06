import os
import subprocess

def add_employee_and_enroll(employee_name, employee_images_dir):
    """
    Add a new employee folder with images, then run enrollment to update encodings.
    
    Args:
        employee_name (str): Name of the employee (folder name).
        employee_images_dir (str): Path to directory containing employee images.
    """
    employees_dir = "employees"
    new_employee_dir = os.path.join(employees_dir, employee_name)
    if not os.path.exists(new_employee_dir):
        os.makedirs(new_employee_dir)

    # Copy provided images into new employee folder
    # Assumes employee_images_dir contains only the relevant images
    for file in os.listdir(employee_images_dir):
        src_path = os.path.join(employee_images_dir, file)
        if os.path.isfile(src_path):
            dest_path = os.path.join(new_employee_dir, file)
            with open(src_path, "rb") as src_file, open(dest_path, "wb") as dest_file:
                dest_file.write(src_file.read())
            
    print(f"[INFO] Added images for employee '{employee_name}' into {new_employee_dir}")

    # Run enroll_deepface.py to update encodings
    print("[INFO] Running enrollment to update embeddings...")
    result = subprocess.run(["python", "enroll_deepface.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("[SUCCESS] Enrollment completed.")
        print(result.stdout)
    else:
        print("[ERROR] Enrollment failed:")
        print(result.stderr)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Add new employee and enroll")
    parser.add_argument("name", help="Employee name (folder name)")
    parser.add_argument("images_dir", help="Directory containing employee images")
    args = parser.parse_args()

    add_employee_and_enroll(args.name, args.images_dir)
