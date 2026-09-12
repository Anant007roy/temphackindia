"""
Threaded video reader that eliminates lag buildup from network streams.

The problem: a normal cv2.VideoCapture().read() loop processes frames in
strict order. If your processing (YOLO + depth) is slower than the phone's
frame rate, frames queue up faster than you can process them. You end up
always processing OLD frames from the backlog - which is exactly what
"laggy" video feels like, and it gets progressively worse over time.

The fix: run frame-grabbing on its own thread that does nothing but
continuously read the latest frame and store it, overwriting whatever was
there before. Your main processing loop then always grabs whatever the
newest available frame is, and simply skips/drops any frames it didn't
have time to process. You trade "process every frame" for "always process
the most current one" - which is the right trade-off for a real-time
safety system, where an old, stale frame is worse than a dropped one.
"""

import threading
import cv2


class LatestFrameReader:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        self._lock = threading.Lock()
        self._latest_frame = None
        self._running = True

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                continue  # transient network hiccup - just try again
            with self._lock:
                self._latest_frame = frame  # always overwrite, never queue

    def read(self):
        """Returns (True, frame) using whatever the newest frame is, or
        (False, None) if nothing has arrived yet."""
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def get(self, prop_id):
        return self.cap.get(prop_id)

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()
