from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO detection model
model = YOLO(r"C:\Users\mehra\Downloads\Detection_of_Potholes\model_yolo12.pt")
class_names = model.names

cap = cv2.VideoCapture("pothole1.mp4")
count = 0

while True:
    ret, img = cap.read()
    if not ret:
        break

    count += 1
    # Optional: comment out this line if you want all frames
    if count % 3 != 0:
         continue

    # Run YOLO prediction (lower conf to catch more potholes)
    results = model.predict(img, conf=0.25, imgsz=640)

    # Draw bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = float(box.conf)
            cls_id = int(box.cls)
            label = f"{class_names[cls_id]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Pothole Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
