# YOLOv8 Object Detection & Naive Tracking with OpenCV

This project enables **real-time object detection and simple multi-object tracking** on video streams using [YOLOv8](https://github.com/ultralytics/ultralytics) from Ultralytics, alongside **OpenCV** for video processing and visualization.

> ⚠️ **Disclaimer:** The tracking method here is a basic implementation using **Euclidean distance** between object centers across frames. It is for educational purposes only and not suitable for production. For advanced tracking, consider solutions like `DeepSORT` or `ByteTrack`.

---

## 📦 Features

- Real-time object detection via `YOLOv8` (`ultralytics`)
- Naive object tracking using center-point matching
- Automatic assignment of object IDs
- IDs are removed after a set number of missed frames
- Annotated video output
- Adjustable display window size

---

## ✅ Requirements

Install dependencies:

```bash
pip install ultralytics opencv-python
```

You’ll need:

* Python ≥ 3.8
* A webcam or video file for testing

---

## How it Works

1. YOLOv8 detects objects in each frame.
2. The center of each bounding box is calculated.
3. If a detected object is close (distance < 30 pixels) to a previously tracked object, it keeps the same ID.
4. Otherwise, a new ID is assigned.
5. If an object isn’t detected for `MAX_MISSING_FRAMES` (default: 20), its ID is removed.

---

## Usage

### Run the script:

```bash
python detect_and_track.py --video_source ./assets/video1.mp4 --output_filename tracked.mp4
```

### Arguments:

| Argument            | Description                                               | Default       |
| ------------------- | --------------------------------------------------------- | ------------- |
| `--video_source`    | Path to video file or webcam index (e.g., `0` for webcam) | `0`           |
| `--output_filename` | Output video filename saved in `./output/`                | `tracked.mp4` |

---

## Function Overview

```python
def detect_objects_on_video(
    model_path="yolov8n.pt",
    video_source=0,
    output_dir="./output",
    output_filename="video.mp4",
    class_names=None,
    show_window=True,
    resize_fx=0.6,
    resize_fy=0.6,
)
```

### Inputs:

* `model_path`: Path to YOLOv8 model (`yolov8n.pt`, `yolov8s.pt`, etc.)
* `video_source`: Camera index or video file path
* `output_dir`: Directory for saving annotated video
* `output_filename`: Name of the output file
* `class_names`: Optional list of COCO class names
* `show_window`: Show real-time display window
* `resize_fx/fy`: Resize factors for display window

### Output:

* Annotated video saved to `output_dir/output_filename`
* Optional real-time display of detection and tracking

---

## Examples

Run with webcam:

```bash
python detect_and_track.py --video_source 0
```

Run with video file:

```bash
python detect_and_track.py --video_source ./assets/video1.mp4 --output_filename demo_output.mp4
```

---

## 📌 Notes

* The default model is `yolov8n.pt` for speed; larger models offer better accuracy.
* This is a **naive tracker** and may not perform well in crowded or fast-moving scenarios.
* There are demo data to use in `assets` folder.