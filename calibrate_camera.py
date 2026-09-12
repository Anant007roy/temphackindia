"""
Camera calibration (one-time setup, do this before step 6).

Why this matters: converting a pixel position + a depth value into a real
3D point (meters left/right, meters up/down, meters away) requires knowing
your specific camera's focal length and optical center. Every camera is
slightly different (even between iPhone and this laptop webcam), so this
is a one-time calibration per camera, not something we can hardcode.

What you need:
1. Print a checkerboard pattern. Easiest: search "opencv checkerboard
   9x6 pattern pdf" and print one (any standard 9x6 internal-corner
   checkerboard works, e.g. from https://github.com/opencv/opencv/blob/4.x/doc/pattern.png).
   Tape it to something flat and rigid (a book, clipboard).
2. Run this script. It opens your webcam. Hold the checkerboard in view
   from different angles/distances/positions (tilt it, move it around the
   frame - corners, center, close, far) and press SPACE to capture each
   time the pattern is detected (it'll highlight the corners in green
   when found). Capture 15-20 good shots.
3. Press 'q' when done. It'll compute and save calibration to
   camera_calibration.npz.

Run it with:
    python calibrate_camera.py
"""

import cv2
import numpy as np
from latest_frame_reader import LatestFrameReader

# Number of INTERNAL corners on your checkerboard (not squares - corners
# where black meets white). A standard 9x6 pattern has 9x6 internal corners.
CHECKERBOARD = (9, 6)
MIN_CAPTURES_RECOMMENDED = 15


def main(camera_source="http://192.168.4.97:8080/video"):#-- update to your IP Webcam address + /video, or use 0 for a plain webcam
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare real-world object points for the checkerboard (0,0,0), (1,0,0), ...
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane

    cap = LatestFrameReader(camera_source)

    print(f"[INFO] Show the checkerboard to the camera. Press SPACE to capture, 'q' to finish.")
    print(f"[INFO] Aim for at least {MIN_CAPTURES_RECOMMENDED} captures from varied angles/positions.")

    gray_shape = None
    capture_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            import time
            time.sleep(0.01)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        display = frame.copy()
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners_refined, found)

        cv2.putText(display, f"Captures: {capture_count}/{MIN_CAPTURES_RECOMMENDED}+",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "SPACE = capture | q = finish",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Camera Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            obj_points.append(objp)
            img_points.append(corners_refined)
            capture_count += 1
            print(f"[INFO] Captured {capture_count}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if capture_count < 5:
        print("[ERROR] Not enough captures to calibrate reliably. Run again and capture more.")
        return

    print("[INFO] Computing calibration...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray_shape, None, None
    )

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    print(f"[OK] Calibration complete. Reprojection error: {ret:.4f} (lower is better, <1.0 is good)")
    print(f"     Focal length: fx={fx:.2f}, fy={fy:.2f}")
    print(f"     Optical center: cx={cx:.2f}, cy={cy:.2f}")

    np.savez("camera_calibration.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print("[OK] Saved to camera_calibration.npz - step6_3d_position.py will load this automatically.")


if __name__ == "__main__":
    main()
