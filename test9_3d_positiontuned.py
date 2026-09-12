"""
Step 6: Real 3D position per object.

Combines:
- YOLO detection (every frame)
- Metric depth estimation (every Nth frame, cached - same as step 5, but
  now using the METRIC checkpoint so values are in real meters)
- Camera calibration (from calibrate_camera.py, if you've run it -
  otherwise falls back to an approximate estimate so you can keep testing)

For each detected object, this computes:
- distance_m: how far away it is, in meters
- x_offset_m: how far left(-)/right(+) of the camera's center it is, in meters
- y_offset_m: how far up(-)/down(+) of the camera's center it is, in meters

This uses the pinhole camera model:
    X = (pixel_x - cx) * depth / fx
    Y = (pixel_y - cy) * depth / fy
    Z = depth
where (cx, cy) is the optical center and fx, fy are focal lengths (all
in pixels), coming from calibration.

Run it with:
    python step6_3d_position.py

Press 'q' to quit the window.
"""

import time
import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO
from ble_output import BLEHazardSender
from latest_frame_reader import LatestFrameReader

DEPTH_UPDATE_INTERVAL = 3
DETECTION_CONF_THRESHOLD = 0.35
CALIBRATION_FILE = "camera_calibration.npz"
INFERENCE_IMAGE_SIZE = 480  # smaller = faster, especially for NMS when many objects are present
MAX_DETECTIONS_PER_MODEL = 20  # caps post-processing cost when a scene has lots of objects

# Distance (in meters) below which a box is drawn in red as a proximity
# warning. Tune this based on real testing - closer to the person's actual
# walking reaction time matters more than a "nice" round number.
DANGER_DISTANCE_M = 1.0
WARNING_DISTANCE_M = 2.0  # yellow zone between warning and danger


def check_gpu():
    if torch.cuda.is_available():
        print(f"[OK] CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        # Lets cuDNN pick the fastest algorithm for your specific input sizes
        # after a brief warm-up, instead of a safe-but-slower default every time.
        torch.backends.cudnn.benchmark = True
        return "cuda", 0
    print("[WARNING] CUDA not available. Running on CPU.")
    return "cpu", -1


def warm_up_models(base_model, finetuned_model, depth_pipe, frame_width, frame_height, device_str):
    """Run one dummy inference through every model before the real loop
    starts. First-time inference often triggers one-off CUDA kernel
    compilation / memory allocation that can look like a random stutter
    later if it happens to coincide with your first busy multi-object frame."""
    print("[INFO] Warming up models (avoids a stall on the first busy frame)...")
    dummy_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    base_model.predict(dummy_frame, device=device_str, verbose=False, imgsz=INFERENCE_IMAGE_SIZE)
    finetuned_model.predict(dummy_frame, device=device_str, verbose=False, imgsz=INFERENCE_IMAGE_SIZE)
    get_depth_map(depth_pipe, dummy_frame)
    print("[OK] Warm-up complete.")


def load_camera_intrinsics(frame_width: int, frame_height: int):
    """Load fx, fy, cx, cy from calibration file if it exists, otherwise
    fall back to a rough estimate assuming a ~70 degree horizontal FOV
    (typical for laptop webcams). The fallback is good enough to keep
    testing with, but run calibrate_camera.py for accurate real numbers."""
    if os.path.exists(CALIBRATION_FILE):
        data = np.load(CALIBRATION_FILE)
        camera_matrix = data["camera_matrix"]
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        print(f"[OK] Loaded calibration: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        return fx, fy, cx, cy
    else:
        assumed_hfov_degrees = 70.0
        fx = frame_width / (2 * np.tan(np.radians(assumed_hfov_degrees / 2)))
        fy = fx  # assume square pixels
        cx, cy = frame_width / 2, frame_height / 2
        print(f"[WARNING] No calibration file found - using APPROXIMATE intrinsics.")
        print(f"          fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        print(f"          Run calibrate_camera.py for accurate values.")
        return fx, fy, cx, cy


def get_depth_map(depth_pipe, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    result = depth_pipe(pil_image)
    depth_array = np.array(result["depth"]).astype(np.float32)  # already in meters for metric checkpoints
    return cv2.resize(depth_array, (frame.shape[1], frame.shape[0]))


def colorize_depth(depth_array: np.ndarray) -> np.ndarray:
    """Normalize the depth map to 0-255 and apply a color map so distance
    changes are visible at a glance - warmer colors = closer, cooler = farther."""
    depth_norm = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)


# Classes from the base COCO model that overlap with the fine-tuned indoor
# model's classes (using different but equivalent names). We skip these from
# the base model to avoid drawing two overlapping boxes for the same object -
# the fine-tuned model's version is used for these instead.
BASE_MODEL_EXCLUDE_CLASSES = {"chair", "couch", "bed", "dining table", "toilet", "sink"}


def get_merged_detections(base_model, finetuned_model, frame, device_str):
    """Run both models on the same frame and merge their detections into
    one list of (x1, y1, x2, y2, class_name, confidence) tuples."""
    detections = []

    base_results = base_model.predict(
        frame, conf=DETECTION_CONF_THRESHOLD, device=device_str, verbose=False,
        imgsz=INFERENCE_IMAGE_SIZE, max_det=MAX_DETECTIONS_PER_MODEL,
    )[0]
    for (x1, y1, x2, y2), cls_id, conf in zip(
        base_results.boxes.xyxy.cpu().numpy().astype(int),
        base_results.boxes.cls.cpu().numpy().astype(int),
        base_results.boxes.conf.cpu().numpy(),
    ):
        class_name = base_model.names[int(cls_id)]
        if class_name in BASE_MODEL_EXCLUDE_CLASSES:
            continue
        detections.append((x1, y1, x2, y2, class_name, float(conf)))

    finetuned_results = finetuned_model.predict(
        frame, conf=DETECTION_CONF_THRESHOLD, device=device_str, verbose=False,
        imgsz=INFERENCE_IMAGE_SIZE, max_det=MAX_DETECTIONS_PER_MODEL,
    )[0]
    for (x1, y1, x2, y2), cls_id, conf in zip(
        finetuned_results.boxes.xyxy.cpu().numpy().astype(int),
        finetuned_results.boxes.cls.cpu().numpy().astype(int),
        finetuned_results.boxes.conf.cpu().numpy(),
    ):
        class_name = finetuned_model.names[int(cls_id)]
        detections.append((x1, y1, x2, y2, class_name, float(conf)))

    return detections


def get_box_color(distance_m: float):
    """Red when dangerously close, yellow as a warning zone, green when
    at a safe distance. Colors are in BGR (OpenCV's order), not RGB."""
    if distance_m < DANGER_DISTANCE_M:
        return (0, 0, 255)  # red
    elif distance_m < WARNING_DISTANCE_M:
        return (0, 255, 255)  # yellow
    else:
        return (0, 255, 0)  # green


def pixel_depth_to_3d(px, py, depth_m, fx, fy, cx, cy):
    """Pinhole camera model: convert a pixel position + depth into a
    real-world 3D offset (meters) relative to the camera."""
    x = (px - cx) * depth_m / fx
    y = (py - cy) * depth_m / fy
    z = depth_m
    return x, y, z


def describe_position(x_offset_m, y_offset_m, distance_m):
    """Turn raw 3D numbers into a human-readable direction, useful for
    debugging and eventually for deciding vibration zone/intensity."""
    if x_offset_m < -0.3:
        horizontal = "LEFT"
    elif x_offset_m > 0.3:
        horizontal = "RIGHT"
    else:
        horizontal = "CENTER"

    if y_offset_m < -0.3:
        vertical = "HIGH"
    elif y_offset_m > 0.3:
        vertical = "LOW"
    else:
        vertical = "MID"

    return f"{horizontal}-{vertical}"


def main(
    camera_source="http://192.168.1.5:8080/video",  # <-- update to your IP Webcam address + /video
    base_model_name: str = "yolov8n.pt",  # COCO classes - person, chair, table, couch, etc.
    finetuned_model_name: str = "runs/detect/obstacle_finetune/run1-7/weights/best.pt",  # stairs, door-frame, etc.
    depth_model_name: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
):
    device_str, device_idx = check_gpu()

    print(f"[INFO] Loading base COCO model: {base_model_name}")
    base_model = YOLO(base_model_name)
    base_model.to(device_str)

    print(f"[INFO] Loading fine-tuned indoor model: {finetuned_model_name}")
    finetuned_model = YOLO(finetuned_model_name)
    finetuned_model.to(device_str)

    print(f"[INFO] Loading metric depth model: {depth_model_name}")
    depth_pipe = pipeline(task="depth-estimation", model=depth_model_name, device=device_idx)

    cap = LatestFrameReader(camera_source)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fx, fy, cx, cy = load_camera_intrinsics(frame_width, frame_height)

    if device_str == "cuda":
        warm_up_models(base_model, finetuned_model, depth_pipe, frame_width or 640, frame_height or 480, device_str)

    print("[INFO] Connecting to ESP32 over BLE (running in background)...")
    ble_sender = BLEHazardSender(device_name="ObstacleAlert-ESP32")
    ble_sender.start()

    print("[INFO] Starting 3D position tracking. Press 'q' to quit.")

    prev_time = time.time()
    fps_smoothed = 0.0
    frame_count = 0
    cached_depth_map = None

    while True:
        ret, frame = cap.read()
        if not ret:
            # No frame available yet (e.g. still connecting to the stream) -
            # unlike a normal video file, this doesn't mean the stream ended.
            time.sleep(0.01)
            continue

        merged_detections = get_merged_detections(base_model, finetuned_model, frame, device_str)

        if frame_count % DEPTH_UPDATE_INTERVAL == 0 or cached_depth_map is None:
            cached_depth_map = get_depth_map(depth_pipe, frame)
        depth_map = cached_depth_map

        annotated_frame = frame.copy()

        # Track the single nearest hazard this frame - that's what gets sent
        # to the wrist device. The person doesn't need every object, just
        # "what's the closest thing I should worry about right now."
        nearest_distance_m = None
        nearest_zone = None

        for x1, y1, x2, y2, class_name, conf in merged_detections:

            x1c, y1c = max(x1, 0), max(y1, 0)
            x2c, y2c = min(x2, depth_map.shape[1] - 1), min(y2, depth_map.shape[0] - 1)
            depth_crop = depth_map[y1c:y2c, x1c:x2c]
            if depth_crop.size == 0:
                continue
            distance_m = float(np.median(depth_crop))

            center_px, center_py = (x1 + x2) / 2, (y1 + y2) / 2
            x_off, y_off, z_off = pixel_depth_to_3d(center_px, center_py, distance_m, fx, fy, cx, cy)
            zone = describe_position(x_off, y_off, distance_m)

            if nearest_distance_m is None or distance_m < nearest_distance_m:
                nearest_distance_m = distance_m
                nearest_zone = zone

            box_color = get_box_color(distance_m)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            label1 = f"{class_name} {conf:.2f}"
            label2 = f"{distance_m:.2f}m | {zone} (x={x_off:+.2f}m,y={y_off:+.2f}m)"
            cv2.putText(annotated_frame, label1, (x1, max(y1 - 25, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            cv2.putText(annotated_frame, label2, (x1, max(y1 - 5, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

        if nearest_distance_m is not None:
            ble_sender.send_hazard(distance_cm=nearest_distance_m * 100, zone=nearest_zone)

        current_time = time.time()
        instant_fps = 1.0 / max(current_time - prev_time, 1e-6)
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * instant_fps if fps_smoothed > 0 else instant_fps
        prev_time = current_time
        cv2.putText(annotated_frame, f"FPS: {fps_smoothed:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("Left: Detections+Depth | Right: Depth Heatmap - press 'q' to quit",
                   np.hstack((annotated_frame, colorize_depth(depth_map))))

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    ble_sender.stop()


if __name__ == "__main__":
    main()
