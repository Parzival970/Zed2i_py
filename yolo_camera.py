import cv2
import torch
from ultralytics import YOLO

model_path = "/home/omni/yolo11n.pt"
model = YOLO(model_path)

# Open webcam (0 is default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
    results = model(frame_tensor.unsqueeze(0))

    # Draw bounding boxes on detected objects
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
            label = f"{model.names[int(box.cls[0])]}: {box.conf[0]:.2f}"  # Class & confidence
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the webcam feed with detections
    cv2.imshow("YOLO Webcam", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
