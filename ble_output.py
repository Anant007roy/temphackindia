"""
BLE client that connects to the ESP32 (esp32_ble_receiver.ino) and sends
hazard data (distance + zone) to it.

Why this is a separate background thread: BLE communication in Python
(via the 'bleak' library) is async (uses asyncio), but your main video
loop in step6_3d_position.py is a normal blocking loop (grab frame,
process, repeat). Mixing the two directly would stall your video loop
every time it waits on a BLE operation. Instead, this runs its own
asyncio event loop on a background thread, keeps a persistent BLE
connection open, and exposes one simple thread-safe function -
send_hazard() - that your main loop can call freely without any waiting.

Install the dependency first:
    pip install bleak

Usage from your main script:
    from ble_output import BLEHazardSender

    sender = BLEHazardSender(device_name="ObstacleAlert-ESP32")
    sender.start()  # connects in the background

    # ... inside your video loop, whenever you have a nearest hazard:
    sender.send_hazard(distance_cm=85, zone="LEFT-MID")

    # when your program exits:
    sender.stop()
"""

import asyncio
import threading
import time

from bleak import BleakClient, BleakScanner

# Must match the UUIDs in esp32_ble_receiver.ino exactly.
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# Don't send BLE messages faster than this, even if the video loop runs
# faster - BLE isn't meant for high-frequency streaming, and the vibration
# motor's physical response doesn't need updates faster than this anyway.
MIN_SEND_INTERVAL_SECONDS = 0.1  # 10 messages/second max


class BLEHazardSender:
    def __init__(self, device_name: str = "ObstacleAlert-ESP32"):
        self.device_name = device_name
        self._loop = None
        self._thread = None
        self._client = None
        self._connected = threading.Event()
        self._latest_message = None
        self._lock = threading.Lock()
        self._stop_flag = False

    def start(self):
        """Starts the background thread that manages the BLE connection."""
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()

    def send_hazard(self, distance_cm: float, zone: str):
        """Call this from your main loop. Non-blocking - just updates
        what the background thread should send next."""
        with self._lock:
            self._latest_message = f"{int(distance_cm)},{zone}"

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def stop(self):
        self._stop_flag = True
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def _run_event_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_and_send())

    async def _connect_and_send(self):
        print(f"[BLE] Scanning for '{self.device_name}'...")
        device = await BleakScanner.find_device_by_name(self.device_name, timeout=15.0)
        if device is None:
            print(f"[BLE ERROR] Could not find a device named '{self.device_name}'. "
                  "Make sure the ESP32 is powered on and advertising.")
            return

        async with BleakClient(device) as client:
            self._client = client
            self._connected.set()
            print(f"[BLE OK] Connected to {self.device_name}.")

            last_sent_message = None
            last_sent_time = 0.0

            while not self._stop_flag:
                with self._lock:
                    message = self._latest_message

                now = time.time()
                if (
                    message is not None
                    and message != last_sent_message
                    and (now - last_sent_time) >= MIN_SEND_INTERVAL_SECONDS
                ):
                    try:
                        await client.write_gatt_char(CHARACTERISTIC_UUID, message.encode("utf-8"))
                        last_sent_message = message
                        last_sent_time = now
                    except Exception as e:
                        print(f"[BLE ERROR] Failed to send: {e}")

                await asyncio.sleep(0.02)

            self._connected.clear()
            print("[BLE] Connection closed.")


if __name__ == "__main__":
    # Quick standalone test: sends a fake hazard message every second.
    sender = BLEHazardSender()
    sender.start()

    print("[TEST] Waiting for connection...")
    while not sender.is_connected():
        time.sleep(0.5)

    print("[TEST] Sending test messages. Press Ctrl+C to stop.")
    try:
        distances = [150, 100, 60, 30, 90]
        i = 0
        while True:
            sender.send_hazard(distance_cm=distances[i % len(distances)], zone="CENTER-MID")
            i += 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        sender.stop()
