import cv2
from ultralytics import YOLO
import os
import math
import argparse


def detect_objects_on_video(
    model_path="yolov8n.pt",
    video_source=0,
    output_dir="./output",
    output_filename="video.mp4",
    class_names=None,
    show_window=True,
    resize_fx=0.6,
    resize_fy=0.6,
):
    if class_names is None:
        class_names = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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
    cap = cv2.VideoCapture(int(video_source) if video_source.isdigit() else video_source)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

    
    # Frames it takes to remove the tracker if the object is missing
    MAX_MISSING_FRAMES = 4

    tracking_objects = {}
    track_id = 0

    while cap.isOpened():
        center_points_current_frame = []

        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, stream=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                center_points_current_frame.append((cx, cy))

                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                class_name = class_names[cls] if cls < len(class_names) else f'class_{cls}'
                label = f'{class_name} {conf}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                text_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + text_size[0], y1 - text_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 0], -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], 1, cv2.LINE_AA)

        # Match current points to existing tracked objects
        new_tracking_objects = {}

        for pt in center_points_current_frame:
            matched = False
            for obj_id, (prev_pt, miss_count) in tracking_objects.items():
                dist = math.hypot(prev_pt[0] - pt[0], prev_pt[1] - pt[1])
                if dist < 50:
                    new_tracking_objects[obj_id] = (pt, 0)
                    matched = True
                    break

            if not matched:
                new_tracking_objects[track_id] = (pt, 0)
                track_id += 1

        # Increment missing frame count for unmatched objects
        for obj_id in list(tracking_objects.keys()):
            if obj_id not in new_tracking_objects:
                prev_pt, miss_count = tracking_objects[obj_id]
                miss_count += 1
                if miss_count < MAX_MISSING_FRAMES:
                    new_tracking_objects[obj_id] = (prev_pt, miss_count)
                

        tracking_objects = new_tracking_objects

        # Draw active tracked objects
        for object_id, (pt, _) in tracking_objects.items():
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 165, 255), 2)

        output.write(frame)

        if show_window:
            resized_frame = cv2.resize(frame, (0, 0), fx=resize_fx, fy=resize_fy)
            cv2.imshow('Frame', resized_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    output.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO Object Tracking")
    parser.add_argument('--video_source', type=str, default="0", help='Path to video file or camera index (default: 0)')
    parser.add_argument('--output_filename', type=str, default="tracked.mp4", help='Output video filename (default: tracked.mp4)')
    args = parser.parse_args()

    # detect_objects_on_video(video_source=args.video_source, output_filename=args.output_filename)
    input = "./assets/video1.mp4"
    output = "tracked.mp4"
    detect_objects_on_video(video_source=input, output_filename=output)