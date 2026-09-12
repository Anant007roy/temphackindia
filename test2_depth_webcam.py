"""
Step 4: Get depth estimation running on the same feed (standalone test).

What this script does:
1. Loads Depth Anything V2 (small checkpoint) via Hugging Face transformers.
2. Opens your laptop webcam.
3. Runs depth estimation on each frame.
4. Displays a colorized depth map side-by-side with the raw feed, plus FPS.

Note: this checkpoint outputs RELATIVE depth (closer vs farther), not
metric meters yet. That's fine for this step — we're just confirming
the model runs fast enough on your GPU. Once this works, we'll swap
in a metric-tuned checkpoint for step 5 (fusing with YOLO boxes),
since real-world distance is what you need for vibration intensity.

Run it with:
    pip install -r requirements.txt
    python step4_depth_webcam.py

Press 'q' to quit the window.
"""

import time
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


def check_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[OK] CUDA available. Using GPU: {gpu_name}")
        return 0  # device index for transformers pipeline (0 = first GPU)
    else:
        print("[WARNING] CUDA not available. Running on CPU — this will be slow.")
        return -1  # -1 means CPU for transformers pipeline


def colorize_depth(depth_array: np.ndarray) -> np.ndarray:
    """Normalize a raw depth map to 0-255 and apply a color map so it's
    visually readable. Closer objects will appear warmer (red/yellow),
    farther objects cooler (blue), depending on the colormap used."""
    depth_norm = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    return depth_colored


def main(camera_index: int = 0, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"):
    device = check_gpu()

    print(f"[INFO] Loading depth model: {model_name}")
    depth_pipe = pipeline(task="depth-estimation", model=model_name, device=device)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    print("[INFO] Starting depth estimation. Press 'q' to quit.")

    prev_time = time.time()
    fps_smoothed = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        # transformers pipeline expects a PIL RGB image, OpenCV gives BGR.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        result = depth_pipe(pil_image)
        depth_array = np.array(result["depth"])  # relative depth map, same size as input (roughly)

        # Resize depth map back to match the original frame size for display.
        depth_resized = cv2.resize(depth_array, (frame.shape[1], frame.shape[0]))
        depth_colored = colorize_depth(depth_resized)

        # FPS calculation
        current_time = time.time()
        instant_fps = 1.0 / max(current_time - prev_time, 1e-6)
        fps_smoothed = 0.9 * fps_smoothed + 0.1 * instant_fps if fps_smoothed > 0 else instant_fps
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps_smoothed:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        # Show raw feed and depth map side by side.
        combined = np.hstack((frame, depth_colored))
        cv2.imshow("Left: Raw Feed | Right: Depth Map - press 'q' to quit", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
