import cv2
from ultralytics import YOLO

model = YOLO("license_plate_detection/best.pt")  

cap = cv2.VideoCapture("license_plate_detection/assets/Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1) (1).mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    annotated = results.plot()

    cv2.imshow("YOLOv8 Video", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()