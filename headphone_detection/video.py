# For Webcam
import cv2
from ultralytics import YOLO

model = YOLO('headphone_detection/best (2).pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    results.boxes = results.boxes[results.boxes.conf > 0.1]
    annotated_frame = results.plot()
    
    cv2.imshow("Headphone Detection", cv2.resize(annotated_frame, (700, 700)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()