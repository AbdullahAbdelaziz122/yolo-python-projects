import math
import cv2
from kalmanfilter import KalmanFilter
import os
from  ultralytics import YOLO


model = YOLO("model_weights/best.pt")
kf = KalmanFilter()



root = os.path.dirname(os.path.abspath(__file__))
vidPath = os.path.join(root, "./videos/demo1.mp4")
cap = cv2.VideoCapture(vidPath)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

output_path = os.path.join(root,"output", "output.mp4")
output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 1, (frame_width, frame_height))

CLASS_NAMES = ["Basketball"]
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            # anchor box pos
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            # center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 5, (0, 0,255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            conf = math.ceil(box.conf[0] * 100) / 100

            cls = int(box.cls[0])
            class_name = CLASS_NAMES[cls]

            label = f'{class_name}{conf}'

            predicted = kf.predict(cx, cy)
            cv2.circle(frame, (predicted[0], predicted[1]), 13, (255, 0,0), 20)


            

    resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    output.write(frame)
    cv2.imshow("frame", resized_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
