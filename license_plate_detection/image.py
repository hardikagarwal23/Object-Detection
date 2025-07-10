import cv2
from ultralytics import YOLO

model = YOLO('license_plate_detection/best.pt') 

image_path = 'HSRP-Installation.webp' 
image = cv2.imread(image_path)

results = model(image)[0]
annotated = results.plot()

cv2.imshow("License Plate Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()