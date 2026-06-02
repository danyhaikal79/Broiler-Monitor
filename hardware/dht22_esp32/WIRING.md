# DHT-22 ↔ ESP32 wiring + setup

## Wiring

### If you have a 3-pin DHT-22 *module* (small PCB, has a pull-up resistor built in)
| DHT-22 module pin | ESP32 pin |
|---|---|
| **+** (VCC) | **3V3** |
| **OUT** (DATA / S) | **GPIO4** |
| **−** (GND) | **GND** |

### If you have a bare 4-pin DHT-22 sensor
Pins left→right with the grille facing you (1 = leftmost):
| DHT-22 pin | Connects to |
|---|---|
| 1 — VCC | ESP32 **3V3** |
| 2 — DATA | ESP32 **GPIO4** + a **10 kΩ resistor** from DATA to 3V3 (pull-up) |
| 3 — NC | (not connected) |
| 4 — GND | ESP32 **GND** |

> Use **3V3**, not 5V — the ESP32's GPIO logic is 3.3 V. The DHT-22 runs fine at 3.3 V.
> Change `DHTPIN` in the sketch if you use a pin other than GPIO4.

```
   ESP32                         DHT-22 (module)
 ┌────────┐                     ┌──────────┐
 │   3V3  ├─────────────────────┤ +        │
 │  GPIO4 ├─────────────────────┤ OUT      │
 │   GND  ├─────────────────────┤ -        │
 └────────┘                     └──────────┘
```

## Flash the firmware (Arduino IDE)

1. **Install ESP32 board support**: File → Preferences → Additional Boards URLs →
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   then Tools → Board → Boards Manager → install **esp32**.
2. **Install libraries**: Tools → Manage Libraries → install
   **"DHT sensor library"** (Adafruit) and **"Adafruit Unified Sensor"**.
3. Open `hardware/dht22_esp32/dht22_esp32.ino`.
4. Tools → Board → your ESP32 model (e.g. "ESP32 Dev Module").
5. Tools → Port → the COM port that appears when the ESP32 is plugged in (note it — e.g. `COM5`).
6. Click **Upload**.

## Verify it works

Open **Tools → Serial Monitor**, set baud to **115200**. You should see, every 2 s:
```
{"temp":25.4,"humidity":68.2}
```
If you see `read_failed` repeatedly → check wiring (DATA pin, pull-up, 3V3/GND).

## ⚠️ Important for the dashboard

The serial port can only be opened by **one** program at a time. **Close the Arduino
Serial Monitor before using "Read sensor" in the dashboard**, or the dashboard won't
be able to open the port.
