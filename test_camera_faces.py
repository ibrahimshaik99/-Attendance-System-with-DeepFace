import cv2
import numpy as np

cap = cv2.VideoCapture('http://191.168.x.xx:8080/video')
ret, frame = cap.read()
if ret:
    h, w = frame.shape[:2]
    print(f'Frame size: {w}x{h}')
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    print(f'Detections shape: {detections.shape}')
    faces = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            faces += 1
    print(f'Faces detected: {faces}')
else:
    print('Failed to read frame')
cap.release()
