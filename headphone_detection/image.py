import cv2
from ultralytics import YOLO

model = YOLO('headphone_detection/best (2).pt')   

image_path = 'headphone_detection/assets/images (6).jpg'
image = cv2.imread(image_path)

results = model(image)[0]
results.boxes = results.boxes[results.boxes.conf > 0.5]
annotated = results.plot()

cv2.imshow("Headphone Detection", cv2.resize(annotated, (700, 700)))
cv2.waitKey(0)
cv2.destroyAllWindows()
