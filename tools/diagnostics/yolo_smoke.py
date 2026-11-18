
import torch
from ultralytics import YOLO

print("Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("YOLO model loaded!")

device = 0 if torch.cuda.is_available() else "cpu"

print("Using device:", device)

results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device=device,
    imgsz=640,
    conf=0.4,
    save=True
)

print("\nDetection Results:")

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            "Object:", model.names[class_id],
            "| Confidence:", round(confidence, 2)
        )

print("\nYOLO test completed!")
