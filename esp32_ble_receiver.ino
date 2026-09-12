/*
  ESP32 BLE receiver for obstacle hazard data.

  This sets up the ESP32 as a BLE "peripheral" (server) that the laptop
  connects to. Every time the laptop sees a hazard, it writes a small
  text message to a BLE characteristic, and this code receives it here.

  Message format sent from Python (as plain text, comma-separated):
      "<distance_cm>,<zone>"
  Example: "85,LEFT-MID" means an obstacle 85cm away, to the left, mid-height.

  What you need to fill in yourself: the actual vibration motor wiring
  and PWM logic marked TODO below - this only demonstrates receiving
  the data and parsing it. Wire a vibration motor (via a transistor,
  don't drive it directly off a GPIO pin) to an ESP32 PWM-capable pin.

  Library needed (install via Arduino IDE Library Manager):
      "ESP32 BLE Arduino" (by Neil Kolban / h2zero, usually pre-bundled
      with the ESP32 board package)
*/

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// These UUIDs must match EXACTLY on both the ESP32 and Python sides.
// They're just randomly generated identifiers - you don't need to change
// them, just keep them consistent between this file and the Python script.
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// TODO: set this to whatever GPIO pin your vibration motor driver is on.
const int VIBRATION_PIN = 25;

BLECharacteristic *pHazardCharacteristic;
bool deviceConnected = false;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *pServer) {
    deviceConnected = true;
    Serial.println("[BLE] Laptop connected.");
  }
  void onDisconnect(BLEServer *pServer) {
    deviceConnected = false;
    Serial.println("[BLE] Laptop disconnected. Restarting advertising...");
    pServer->getAdvertising()->start();  // so it can reconnect automatically
  }
};

class HazardCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *pCharacteristic) {
    std::string value = pCharacteristic->getValue();
    if (value.length() == 0) return;

    String message = String(value.c_str());
    Serial.print("[BLE] Received: ");
    Serial.println(message);

    // Parse "distance_cm,zone"
    int commaIndex = message.indexOf(',');
    if (commaIndex == -1) return;

    int distanceCm = message.substring(0, commaIndex).toInt();
    String zone = message.substring(commaIndex + 1);

    // TODO: replace this with real vibration intensity logic.
    // Simple example: closer objects = stronger vibration.
    // Map distance (e.g. 0-200cm) to a PWM duty cycle (0-255), inverted
    // so smaller distance = higher intensity.
    int intensity = constrain(map(distanceCm, 0, 200, 255, 0), 0, 255);

    // PWM output: newer ESP32 Arduino core versions (3.x+) support analogWrite()
    // directly. If yours doesn't compile with analogWrite, your board package
    // is on the older LEDC-based API - replace the line below with:
    //   ledcWrite(0, intensity);   // channel 0, set up once in setup() via:
    //   ledcSetup(0, 5000, 8); ledcAttachPin(VIBRATION_PIN, 0);
    analogWrite(VIBRATION_PIN, intensity);

    Serial.print("  -> distance_cm=");
    Serial.print(distanceCm);
    Serial.print(" zone=");
    Serial.print(zone);
    Serial.print(" intensity=");
    Serial.println(intensity);
  }
};

void setup() {
  Serial.begin(115200);
  pinMode(VIBRATION_PIN, OUTPUT);

  BLEDevice::init("ObstacleAlert-ESP32");  // this name will show up when scanning from Python
  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  pHazardCharacteristic = pService->createCharacteristic(
      CHARACTERISTIC_UUID,
      BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR
  );
  pHazardCharacteristic->setCallbacks(new HazardCallbacks());

  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->start();

  Serial.println("[BLE] Advertising as 'ObstacleAlert-ESP32'. Waiting for laptop to connect...");
}

void loop() {
  // Nothing needed here - all the work happens in the BLE callbacks above.
  delay(20);
}
