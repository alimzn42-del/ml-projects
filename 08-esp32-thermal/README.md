# 08 — ESP32 thermal logging, round two (work in progress)

**Type:** own hardware data — the successor to the `heat_log.csv` session in project 01.
**Hardware:** ESP32 logging two temperature/humidity sensors (inside/outside) over serial.
**Firmware:** lives in the PlatformIO project `Documents/PlatformIO/Projects/esp_sensor`.
**Status:** log recorded and analysis started in `notebook.ipynb` — not yet written up.

## The data

`data/thermal_log.csv` — recorded 2026-09-03, ~2 s sample interval:

| Column | Meaning |
|---|---|
| `timestamp` | wall-clock time (ISO) |
| `ms` | milliseconds since boot — the drift-free time base (lesson from project 01) |
| `t_in`, `h_in` | temperature / humidity, inner sensor |
| `t_out`, `h_out` | temperature / humidity, outer sensor |

Like `heat_log.csv` in project 01, this is original sensor data that exists nowhere
else — it is the one file in this folder worth committing.

This README becomes the full write-up when the project is finished, following the
pattern of projects 01–04.
