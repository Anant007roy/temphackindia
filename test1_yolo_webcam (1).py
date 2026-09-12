"""
Step 3: Get YOLO running on the live webcam feed.

What this script does:
1. Checks that your GPU (CUDA) is actually being used.
2. Opens your laptop webcam.
3. Runs a pretrained YOLO model on each frame in real time.
4. Draws bounding boxes + class labels + confidence on screen.
5. Shows a live FPS counter so you know how fast it's actually running.

This is intentionally simple: no distance estimation, no zones, no
filtering yet. The only goal right now is "prove detection works
in real time on my hardware." Later steps build on top of this file.

Run it with:
    pip install -r requirements.txt
    python step3_yolo_webcam.py

Press 'q' to quit the window.
"""

import time
import cv2
import torch
from ultralytics import YOLO


def check_gpu():
    """Print GPU status. If this says CPU, detection will be much slower,
    and every later step (depth estimation especially) will struggle to
    run in real time."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[OK] CUDA available. Using GPU: {gpu_name}")
        return "cuda"
    else:
        print("[WARNING] CUDA not available. Running on CPU — this will be slow.")
        print("          Check your PyTorch install matches your GPU/driver version.")
        return "cpu"


def main(camera_index: int = 0, model_name: str = "yolov8n.pt", conf_threshold: float = 0.35):
    device = check_gpu()

    # yolov8n.pt is the smallest/fastest checkpoint — good for real-time testing.
    # If your GPU has plenty of headroom, try "yolov8s.pt" for better accuracy.
    print(f"[INFO] Loading model: {model_name}")
    model = YOLO(model_name)
    model.to(device)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Try a different index (0, 1, 2...) if you have multiple cameras."
        )

    print("[INFO] Starting detection. Press 'q' to quit.")

    prev_time = time.time()
    fps_smoothed = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        # Run detection on this frame. conf sets the minimum confidence
        # to keep a detection — lower catches more (higher recall) but
        # also more false positives. For this safety-critical use case
        # we intentionally lean toward catching more.
        results = model.predict(frame, conf=conf_threshold, device=device, verbose=False)
        result = results[0]

        annotated_frame = result.plot()  # draws boxes + labels automatically

        # FPS calculation (smoothed so the number doesn't jitter wildly)
        current_time = time.time()
        instant_fps = 1.0 / max(current_time - prev_time, 1e-6)
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * instant_fps if fps_smoothed > 0 else instant_fps
        prev_time = current_time

        cv2.putText(
            annotated_frame,
            f"FPS: {fps_smoothed:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLO Webcam Test - press 'q' to quit", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
