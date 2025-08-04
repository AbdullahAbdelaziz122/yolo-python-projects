import cv2
from ultralytics import YOLO
import os
import math

def detect_objects_on_video(
    model_path="yolov8n.pt",
    video_source=1,
    output_dir="./output",
    output_filename="video.mp4",
    class_names=None,
    show_window=True,
    resize_fx=0.6,
    resize_fy=0.6,
):
    if class_names is None:
        class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                      "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                      "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                      "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                      "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                      "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                      "teddy bear", "hair drier", "toothbrush"
                      ]

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, stream=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                conf = math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                class_name = class_names[cls]
                label = f'{class_name}{conf}'
                text_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + text_size[0], y1 - text_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 0], -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        output.write(frame)
        if show_window:
            resize_frame = cv2.resize(frame, (0, 0), fx=resize_fx, fy=resize_fy, interpolation=cv2.INTER_AREA)
            cv2.imshow('Frame', resize_frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

    cap.release()
    output.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    # Camera: video_source = 1
    # video : video_source = path
    detect_objects_on_video(video_source = 1, output_filename="output.mp4")

