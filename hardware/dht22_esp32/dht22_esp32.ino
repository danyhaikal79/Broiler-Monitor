/*
 * DHT-22 -> ESP32 -> USB Serial   (for Broiler Feed Monitor dashboard)
 *
 * Streams one JSON line every 2 seconds over USB serial, e.g.:
 *     {"temp":25.4,"humidity":68.2}
 * On a failed read:
 *     {"temp":null,"humidity":null,"error":"read_failed"}
 *
 * The dashboard opens the COM port, reads the latest line, and uses temp+humidity
 * for the THI calculation.
 *
 * ---- Libraries (install via Arduino IDE Library Manager) ----
 *   - "DHT sensor library" by Adafruit
 *   - "Adafruit Unified Sensor"  (dependency)
 *   Board: ESP32 (install "esp32 by Espressif Systems" in Boards Manager)
 *
 * ---- Wiring (see hardware/dht22_esp32/WIRING.md) ----
 *   DHT-22 module (3-pin):  + -> 3V3,  OUT -> GPIO4,  - -> GND
 *   Bare DHT-22 (4-pin)  :  VCC -> 3V3, DATA -> GPIO4 (+10k pull-up to 3V3), GND -> GND
 */

#include "DHT.h"

#define DHTPIN  4        // GPIO4 (change if you wire DATA to another pin)
#define DHTTYPE DHT22    // AM2302 / DHT-22

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);   // must match the dashboard's baud setting
  delay(200);
  dht.begin();
}

void loop() {
  float h = dht.readHumidity();
  float t = dht.readTemperature();   // Celsius

  if (isnan(h) || isnan(t)) {
    Serial.println("{\"temp\":null,\"humidity\":null,\"error\":\"read_failed\"}");
  } else {
    Serial.print("{\"temp\":");
    Serial.print(t, 1);
    Serial.print(",\"humidity\":");
    Serial.print(h, 1);
    Serial.println("}");
  }

  delay(2000);   // DHT-22 max sample rate is 0.5 Hz -> 2s minimum
}
