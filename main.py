from flask import Flask, render_template, Response
import cv2
import torch
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model_path = "Models/Violence.pt"
labels = ["non_violence", "violence"]
model = YOLO(model_path)

def detect_objects(frame, model, conf_threshold=0.25, relevant_classes=None):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img, conf=conf_threshold)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            if relevant_classes is None or int(cls) in relevant_classes:
                detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item(), cls.item()])

    return np.array(detections)

def draw_detections(frame, detections, labels):
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        if int(cls) < len(labels):
            label = f"{labels[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def gen_frames():
    relevant_classes = [0, 1]
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        success, frame = cap.read()
        if not success:
            break
        detections = detect_objects(frame, model, relevant_classes=relevant_classes)
        frame = draw_detections(frame, detections, labels)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
