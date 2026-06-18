"""
DHT-22 (via ESP32 over USB serial) reader + shared environment-input UI.

The ESP32 streams JSON lines like {"temp":25.4,"humidity":68.2} every ~2s.
We open the port on demand (button press), grab the latest valid line, close.
This avoids holding the port open across Streamlit reruns.
"""

from __future__ import annotations

import json
import threading
import time

import streamlit as st


def list_ports() -> list[str]:
    try:
        from serial.tools import list_ports as lp
        return [p.device for p in lp.comports()]
    except Exception:
        return []


# Persistent serial connections, keyed by (port, baud). Module-level so they
# survive Streamlit reruns (the script re-executes but this module is imported
# once). Keeping the port OPEN avoids the Windows reopen race + the ESP32 reset
# that happens on every fresh open.
_SERIAL_CACHE: dict = {}

# Serialize access to the shared cached handle. In the current architecture the
# WORKER (a separate process) owns the live serial; the dashboard reads serial only
# in Upload mode (sidebar_env_inputs), when the worker's camera/serial are paused —
# so there's no real in-process contention now. pyserial handles still aren't
# thread-safe, so the RLock is kept as a defensive guard against any future
# concurrent caller. RLock (re-entrant) so read_latest's error path can call
# release_port() while still holding it.
_SERIAL_LOCK = threading.RLock()


def _get_serial(port: str, baud: int):
    import serial

    key = (port, baud)
    ser = _SERIAL_CACHE.get(key)
    if ser is not None and getattr(ser, "is_open", False):
        return ser

    # Drop any stale handle for this port (different baud, or closed)
    for k, s in list(_SERIAL_CACHE.items()):
        if k[0] == port:
            try:
                s.close()
            except Exception:
                pass
            _SERIAL_CACHE.pop(k, None)

    # Open with a few retries — covers the brief window where Windows / a just-closed
    # handle hasn't released the port yet.
    last_err = None
    for _ in range(4):
        try:
            ser = serial.Serial(port, baud, timeout=1)
            _SERIAL_CACHE[key] = ser
            time.sleep(0.2)  # let the ESP32 settle after the open-reset
            return ser
        except Exception as e:
            last_err = e
            time.sleep(0.4)
    raise last_err


def release_port(port: str | None = None):
    """Close and forget cached connections (all, or just one port)."""
    with _SERIAL_LOCK:
        for k, s in list(_SERIAL_CACHE.items()):
            if port is None or k[0] == port:
                try:
                    s.close()
                except Exception:
                    pass
                _SERIAL_CACHE.pop(k, None)


def read_latest(port: str, baud: int = 115200, window_sec: float = 3.0):
    """
    Read the most recent valid {temp,humidity} JSON line from the (persistent)
    serial connection. Returns (temp, humidity) or (None, error_message).
    """
    try:
        import serial  # noqa: F401
    except ImportError:
        return None, "pyserial not installed (pip install pyserial)"

    with _SERIAL_LOCK:    # one reader at a time on the shared handle
        try:
            ser = _get_serial(port, baud)
        except Exception as e:
            return None, f"cannot open {port}: {e} (close Arduino IDE / other serial apps)"

        try:
            ser.reset_input_buffer()
            end = time.time() + window_sec
            while time.time() < end:
                raw = ser.readline().decode(errors="ignore").strip()
                if not raw:
                    continue
                try:
                    d = json.loads(raw)
                except json.JSONDecodeError:
                    continue   # skip partial first line / non-JSON
                if d.get("temp") is not None and d.get("humidity") is not None:
                    return (float(d["temp"]), float(d["humidity"])), None   # first fresh line wins
            return None, "no valid reading (check wiring / baud)"
        except Exception as e:
            # Connection went bad -> drop it so the next call reopens cleanly
            release_port(port)
            return None, f"read error: {e}"


def sidebar_env_inputs(prefix: str):
    """
    Environment block: DHT-22 sensor (USB) with a LIVE auto-refresh mode + manual
    fallback. Returns (temperature_c, humidity_pct, age_days).

    - Live mode ON : a fragment re-reads the sensor every 5s, shows read-only live
      values, and refreshes the whole app so the THI/feed math stays current.
      (YOLO is cached by image, so the refresh is cheap — models don't re-run.)
    - Live mode OFF: a 'Read once' button + manual editable inputs (demos / no sensor).
    """
    tkey, hkey = f"{prefix}_temp", f"{prefix}_hum"
    okkey = f"{prefix}_sensor_ok"
    st.session_state.setdefault(tkey, 25.0)
    st.session_state.setdefault(hkey, 70.0)

    st.sidebar.markdown("### 🌡️ Environment")

    with st.sidebar.expander("📡 DHT-22 sensor (USB)", expanded=True):
        ports = list_ports()
        port = st.selectbox("Serial port", ports or ["(no ports found)"], key=f"{prefix}_port",
                            help="COM port the ESP32 enumerates as. Close the Arduino Serial "
                                 "Monitor first — only one app can hold the port.")
        baud = st.selectbox("Baud rate", [115200, 9600], index=0, key=f"{prefix}_baud")
        live = st.checkbox("🔴 Live mode (auto every 5s)", key=f"{prefix}_live",
                           help="Continuously read the sensor and refresh the whole dashboard.")

        if live:
            # Full-page auto-refresh every 5s. The whole script reruns together, so
            # temp/humidity/THI/feed all update consistently. YOLO is cached by image
            # so the models DON'T re-run on each tick — only the sensor + cheap math.
            try:
                from streamlit_autorefresh import st_autorefresh
                st_autorefresh(interval=5000, key=f"{prefix}_autorefresh")
            except ImportError:
                st.caption("⚠️ pip install streamlit-autorefresh for live mode")
            # Read inline (returns on the first fresh line — usually <1s)
            reading, err = read_latest(port, int(baud), window_sec=2.5) if ports else (None, "no ports")
            if reading:
                st.session_state[tkey] = round(reading[0], 1)
                st.session_state[hkey] = round(reading[1], 1)
                st.session_state[okkey] = True
                st.markdown(f"**🔴 LIVE** · {st.session_state[tkey]} °C · {st.session_state[hkey]} % RH")
            else:
                st.session_state[okkey] = False
                st.caption(f"⚠️ {err} — retrying…")
        else:
            if st.button("📡 Read once", key=f"{prefix}_read", use_container_width=True):
                reading, err = read_latest(port, int(baud)) if ports else (None, "no serial ports")
                if reading:
                    st.session_state[tkey] = round(reading[0], 1)
                    st.session_state[hkey] = round(reading[1], 1)
                    st.session_state[okkey] = True
                    st.rerun()
                else:
                    st.session_state[okkey] = False
                    st.warning(f"⚠️ {err}")
            if st.session_state.get(okkey):
                st.caption(f"✅ last reading: {st.session_state[tkey]} °C · {st.session_state[hkey]} % RH")
            if st.button("🔌 Release port", key=f"{prefix}_release",
                         help="Free the COM port so you can use it in the Arduino IDE again."):
                release_port()
                st.session_state[okkey] = False
                st.toast("Serial port released.")

    live = st.session_state.get(f"{prefix}_live", False)
    if live:
        # read-only display (no +/- steppers) — value driven by the sensor
        st.sidebar.metric("Temperature", f"{st.session_state[tkey]} °C")
        st.sidebar.metric("Humidity", f"{st.session_state[hkey]} % RH")
        temp, hum = float(st.session_state[tkey]), float(st.session_state[hkey])
    else:
        temp = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, step=0.5, key=tkey)
        hum = st.sidebar.number_input("Humidity (% RH)", 5.0, 99.0, step=1.0, key=hkey)
    age = st.sidebar.number_input("Chicken age (days)", 0, 70, 21, 1, key=f"{prefix}_age")
    return temp, hum, age
