# test_camera.py - Quick test to verify IP camera connection
import cv2

# Change this to your camera IP
CAMERA_URL = "http://191.168.1.174:8080/video"

print("=" * 60)
print("TESTING IP CAMERA CONNECTION")
print("=" * 60)
print(f"\nCamera URL: {CAMERA_URL}")
print("\nAttempting to connect...")

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("\n[ERROR] Cannot connect to camera!")
    print("\nTroubleshooting:")
    print("1. Check if IP address is correct")
    print("2. Ensure camera app is running")
    print("3. Verify phone/camera is on same WiFi network")
    print("4. Try opening URL in browser: " + CAMERA_URL.replace("/video", ""))
    exit(1)

print("[SUCCESS] Camera connected!\n")
print("Displaying live feed...")
print("Press 'q' to quit\n")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame")
        break
    
    frame_count += 1
    
    # Display frame info
    h, w = frame.shape[:2]
    cv2.putText(frame, f"Frame: {frame_count} | Size: {w}x{h}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Camera Test - Press 'q' to quit", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n[SUCCESS] Camera test completed!")
print("=" * 60)
