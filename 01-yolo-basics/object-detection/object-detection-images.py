import cv2
from ultralytics import YOLO

model = YOLO("yolo8n.pt")
results = model.predict("./assets/image1.jpg")

# Get the result image with detections
result_img = results[0].plot()


cv2.imwrite("./output/detected-img.jpg", result_img)

cv2.imshow("Detections", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()