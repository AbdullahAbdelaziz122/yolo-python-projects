# Basketball Trajectory Prediction with YOLO and Kalman Filter

This project uses a YOLO model and a Kalman Filter to detect and predict the trajectory of a basketball in video frames.

## Features

- Detects basketballs in video using a trained YOLO model.
- Tracks and predicts the ball's trajectory using a Kalman Filter.
- Visualizes detection and prediction on video frames.
- Outputs processed video with overlays.

## Requirements

- Python 3.x
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- OpenCV (`cv2`)
- Custom `KalmanFilter` implementation

## Setup

1. Clone the repository.
2. Place your YOLO model weights in `model_weights/best.pt`.
3. Add your input video to `videos/demo1.mp4`.
4. Ensure `kalmanfilter.py` is present in the project directory.

Install dependencies:
```bash
pip install ultralytics opencv-python numpy
```

## Usage

Run the script:
```bash
python main.py
```

- Output video will be saved to `output/output.mp4`.
- Press `q` to exit the video window.

## File Structure

```
├── model_weights/
│   └── best.pt
├── videos/
│   └── demo1.mp4
├── output/
│   └── output.mp4
├── kalmanfilter.py
└── main.py
```

## Notes

- The script only detects basketballs (`CLASS_NAMES = ["Basketball"]`).
- Modify `CLASS_NAMES` for other object classes if needed.
- There are demo data to try the project located in folders `image` and `videos`