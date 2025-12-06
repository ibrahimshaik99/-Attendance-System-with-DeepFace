# Flask Web Application for Attendance System
from flask import Flask, render_template, request, jsonify, send_file
import os
import subprocess
import pandas as pd
from datetime import datetime
import json
import shutil
from werkzeug.utils import secure_filename
import threading
import pickle
import sys
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'employees'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Global variables
attendance_process = None
enrollment_process = None

# Error handler
@app.errorhandler(Exception)
def handle_error(e):
    print(f"Error: {e}")
    traceback.print_exc()
    return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/test')
def test_api():
    return jsonify({
        'status': 'ok',
        'python': sys.executable,
        'cwd': os.getcwd(),
        'enroll_exists': os.path.exists('enroll_deepface.py'),
        'attendance_exists': os.path.exists('attendence_deepface.py'),
        'encodings_exists': os.path.exists('encodings_deepface.pickle')
    })

@app.route('/api/stats')
def get_stats():
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = f"output/attendance_{today}.xlsx"
        
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            total = len(df)
            present = len(df[df['Check-in'].notna()])
            late = len(df[df['LateStatus'] == 'Late'])
            ontime = len(df[df['LateStatus'] == 'OnTime'])
        else:
            total = present = late = ontime = 0
        
        # Get enrolled employees
        enrolled = 0
        if os.path.exists('encodings_deepface.pickle'):
            with open('encodings_deepface.pickle', 'rb') as f:
                data = pickle.load(f)
                enrolled = len(set(data['names']))
        
        return jsonify({
            'total_employees': enrolled,
            'present_today': present,
            'late_today': late,
            'ontime_today': ontime
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance/today')
def get_today_attendance():
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = f"output/attendance_{today}.xlsx"
        
        if os.path.exists(file_path):
            df = pd.read_excel(file_path)
            # Fill NaN values with empty string for display
            df = df.fillna('')
            records = df.to_dict('records')
            return jsonify(records)
        return jsonify([])
    except Exception as e:
        print(f"Error reading attendance: {e}")
        return jsonify([])

@app.route('/api/attendance/history')
def get_attendance_history():
    try:
        files = [f for f in os.listdir('output') if f.endswith('.xlsx')]
        history = []
        for file in sorted(files, reverse=True)[:7]:
            df = pd.read_excel(f"output/{file}")
            date = file.replace('attendance_', '').replace('.xlsx', '')
            history.append({
                'date': date,
                'total': len(df),
                'present': len(df[df['Check-in'].notna()]),
                'late': len(df[df['LateStatus'] == 'Late'])
            })
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/employees')
def get_employees():
    try:
        employees = []
        if os.path.exists('employees'):
            for emp_name in os.listdir('employees'):
                emp_path = os.path.join('employees', emp_name)
                if os.path.isdir(emp_path):
                    images = [f for f in os.listdir(emp_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    employees.append({
                        'name': emp_name,
                        'images': len(images)
                    })
        return jsonify(employees)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/employee/add', methods=['POST'])
def add_employee():
    try:
        name = request.form.get('name')
        if not name:
            return jsonify({'error': 'Name is required'}), 400
        
        emp_folder = os.path.join(app.config['UPLOAD_FOLDER'], name)
        os.makedirs(emp_folder, exist_ok=True)
        
        files = request.files.getlist('images')
        saved = 0
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file.save(os.path.join(emp_folder, filename))
                saved += 1
        
        return jsonify({'success': True, 'message': f'Added {name} with {saved} images'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/employee/delete/<name>', methods=['DELETE'])
def delete_employee(name):
    try:
        emp_folder = os.path.join(app.config['UPLOAD_FOLDER'], name)
        if os.path.exists(emp_folder):
            shutil.rmtree(emp_folder)
            return jsonify({'success': True, 'message': f'Deleted {name}'})
        return jsonify({'error': 'Employee not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/enroll', methods=['POST'])
def run_enrollment():
    global enrollment_process
    try:
        if enrollment_process and enrollment_process.poll() is None:
            return jsonify({'error': 'Enrollment already running'}), 400
        
        # Get Python executable
        import sys
        python_exe = sys.executable
        script_path = os.path.join(os.getcwd(), 'enroll_deepface.py')
        
        if not os.path.exists(script_path):
            return jsonify({'error': 'enroll_deepface.py not found'}), 404
        
        # Start enrollment in new console window
        enrollment_process = subprocess.Popen(
            [python_exe, script_path],
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0,
            cwd=os.getcwd()
        )
        return jsonify({'success': True, 'message': 'Enrollment started in new window'})
    except Exception as e:
        print(f"Enrollment error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance/start', methods=['POST'])
def start_attendance():
    global attendance_process
    try:
        if attendance_process and attendance_process.poll() is None:
            return jsonify({'error': 'Attendance system already running'}), 400
        
        # Get Python executable
        import sys
        python_exe = sys.executable
        script_path = os.path.join(os.getcwd(), 'attendence_deepface.py')
        
        if not os.path.exists(script_path):
            return jsonify({'error': 'attendence_deepface.py not found'}), 404
        
        # Check if encodings exist
        if not os.path.exists('encodings_deepface.pickle'):
            return jsonify({'error': 'No encodings found. Run enrollment first.'}), 400
        
        # Start attendance system in new console window
        attendance_process = subprocess.Popen(
            [python_exe, script_path],
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0,
            cwd=os.getcwd()
        )
        return jsonify({'success': True, 'message': 'Attendance system started in new window'})
    except Exception as e:
        
        print(f"Attendance start error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance/stop', methods=['POST'])
def stop_attendance():
    global attendance_process
    try:
        if attendance_process:
            attendance_process.terminate()
            attendance_process.wait(timeout=5)
            attendance_process = None
            return jsonify({'success': True, 'message': 'Attendance system stopped'})
        return jsonify({'error': 'Attendance system not running'}), 400
    except subprocess.TimeoutExpired:
        attendance_process.kill()
        attendance_process = None
        return jsonify({'success': True, 'message': 'Attendance system force stopped'})
    except Exception as e:
        print(f"Stop error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/attendance/status')
def attendance_status():
    global attendance_process
    running = attendance_process is not None and attendance_process.poll() is None
    return jsonify({'running': running})

@app.route('/api/download/<date>')
def download_attendance(date):
    try:
        file_path = f"output/attendance_{date}.xlsx"
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("ATTENDANCE SYSTEM - WEB SERVER")
    print("="*60)
    print(f"Python: {sys.executable}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"")
    
    os.makedirs('employees', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("[INFO] Directories created")
    print(f"[INFO] Enroll script: {'Found' if os.path.exists('enroll_deepface.py') else 'NOT FOUND'}")
    print(f"[INFO] Attendance script: {'Found' if os.path.exists('attendence_deepface.py') else 'NOT FOUND'}")
    print(f"[INFO] Encodings: {'Found' if os.path.exists('encodings_deepface.pickle') else 'Not found (run enrollment)'}")
    print(f"")
    print("[INFO] Starting web server on http://localhost:5000")
    print("[INFO] Press Ctrl+C to stop")
    print("="*60)
    print("")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
