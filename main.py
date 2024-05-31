import cv2
import torch
import numpy as np
from ultralytics import YOLO

def load_yolo_model(model_path):
    model = YOLO(model_path)
    return model

def detect_objects(frame, model, conf_threshold=0.25, relevant_classes=None):
    # Preprocess the frame for YOLOv8 model
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
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Draw label
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def main(model_path, labels):
    model = load_yolo_model(model_path)
    relevant_classes = [0, 1]
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detect_objects(frame, model, relevant_classes=relevant_classes)
        frame = draw_detections(frame, detections, labels)
        cv2.imshow('Violence Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = "Models/yolov8_violence_detection.pt"
    labels = ["non_violence", "violence"]
    main(model_path, labels)
