"""
Step 5: Fuse detection boxes with depth values (Option B).

Strategy:
- Run YOLO on every frame (fast, ~30 FPS) so new objects are caught immediately.
- Run depth estimation only every DEPTH_UPDATE_INTERVAL frames (slow, ~15 FPS),
  and reuse ("cache") the last computed depth map on the frames in between.
- For each YOLO box, crop the matching region of the current depth map and
  take the MEDIAN value inside that box as the object's distance signal.
  Median (not mean) avoids background pixels at the box edges skewing the
  result if the box isn't a perfectly tight fit around the object.

Important: this still uses the relative-depth checkpoint from step 4, so
the numbers shown are NOT meters yet — they're relative depth scores where
(depending on the model) higher generally means closer. Once this fusion
logic is confirmed working, the next step is swapping in a metric-tuned
depth checkpoint so these become real distances in meters.

Run it with:
    pip install -r requirements.txt
    python step5_fusion.py

Press 'q' to quit the window.
"""

import time
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

# How often (in frames) to recompute depth. 3 means: compute depth on
# frame 0, 3, 6, 9... and reuse the last result on frames in between.
DEPTH_UPDATE_INTERVAL = 3

DETECTION_CONF_THRESHOLD = 0.35


def check_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[OK] CUDA available. Using GPU: {gpu_name}")
        return "cuda", 0
    else:
        print("[WARNING] CUDA not available. Running on CPU — this will be slow.")
        return "cpu", -1


def get_depth_map(depth_pipe, frame):
    """Run the depth model on a frame and return a depth array resized
    to match the frame's dimensions."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    result = depth_pipe(pil_image)
    depth_array = np.array(result["depth"]).astype(np.float32)
    depth_resized = cv2.resize(depth_array, (frame.shape[1], frame.shape[0]))
    return depth_resized


def colorize_depth(depth_array: np.ndarray) -> np.ndarray:
    depth_norm = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)


def main(
    camera_index: int = 0,
    yolo_model_name: str = "yolov8n.pt",
    depth_model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
):
    device_str, device_idx = check_gpu()

    print(f"[INFO] Loading YOLO model: {yolo_model_name}")
    yolo_model = YOLO(yolo_model_name)
    yolo_model.to(device_str)

    print(f"[INFO] Loading depth model: {depth_model_name}")
    depth_pipe = pipeline(task="depth-estimation", model=depth_model_name, device=device_idx)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    print("[INFO] Starting fused detection + depth. Press 'q' to quit.")

    prev_time = time.time()
    fps_smoothed = 0.0
    frame_count = 0
    cached_depth_map = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        # --- Detection: every frame ---
        results = yolo_model.predict(frame, conf=DETECTION_CONF_THRESHOLD, device=device_str, verbose=False)
        result = results[0]

        # --- Depth: only every Nth frame, otherwise reuse cached map ---
        if frame_count % DEPTH_UPDATE_INTERVAL == 0 or cached_depth_map is None:
            cached_depth_map = get_depth_map(depth_pipe, frame)
        depth_map = cached_depth_map

        annotated_frame = frame.copy()

        # Pull ALL box data off the GPU in one batch transfer instead of
        # one .cpu() call per box inside the loop. With several objects
        # detected, per-box transfers cause repeated GPU-CPU sync stalls,
        # which is what was causing the lag you noticed with more objects.
        boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
        boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
        boxes_conf = result.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), cls_id, conf in zip(boxes_xyxy, boxes_cls, boxes_conf):
            class_name = yolo_model.names[int(cls_id)]

            # Clip box coordinates to frame bounds just in case.
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, depth_map.shape[1] - 1), min(y2, depth_map.shape[0] - 1)

            depth_crop = depth_map[y1:y2, x1:x2]
            median_depth = float(np.median(depth_crop)) if depth_crop.size > 0 else -1.0

            # Draw box + label with class, confidence, and relative depth score.
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f} | depth: {median_depth:.1f}"
            cv2.putText(
                annotated_frame, label, (x1, max(y1 - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )

        # FPS
        current_time = time.time()
        instant_fps = 1.0 / max(current_time - prev_time, 1e-6)
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * instant_fps if fps_smoothed > 0 else instant_fps
        prev_time = current_time
        cv2.putText(
            annotated_frame, f"FPS: {fps_smoothed:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
        )

        depth_display = colorize_depth(depth_map)
        combined = np.hstack((annotated_frame, depth_display))
        cv2.imshow("Left: Detections+Depth | Right: Depth Map - press 'q' to quit", combined)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
