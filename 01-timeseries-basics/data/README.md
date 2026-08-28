# Data — 01 Timeseries Basics

This project uses two datasets. One is downloaded at runtime, the other is committed here.

## 1. `daily-min-temperatures` — downloaded, not committed

Daily minimum temperature, Melbourne, 1981–1990. 3650 rows × 2 columns.

Notebook `01-cleaning-and-features.ipynb` reads it straight from the URL, so
there is nothing to do before running it:

```python
df = pd.read_csv(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
)
```

- **Source:** [jbrownlee/Datasets](https://github.com/jbrownlee/Datasets) — mirror of the
  Australian Bureau of Meteorology series.
- **Direct link:** <https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv>
- **Size:** ~65 KB.

To keep a local copy instead:

```bash
curl -o daily-min-temperatures.csv \
  https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv
```

## 2. `heat_log.csv` — committed

**This is original data, recorded for this project.** It is the only dataset in
the whole repository that is committed to git.

- **Hardware:** ESP32 DevKit + DS18B20 temperature sensor.
- **Capture:** `../src/serial_logger.py`, reading the serial port at ~1 Hz.
- **Contents:** 1491 rows over roughly 25 minutes, columns `timestamp` and `temp_c`.
- **Sequence:** ~5 min at rest → heating by hand (29 → 35.75 °C) → natural cooling → rest.
- **Size:** 45 KB.

It is committed deliberately: it is small, it is not redistributable from anywhere
else, and without it `02-sensor-log-analysis.ipynb` cannot be reproduced at all.

Note that the file is raw on purpose — the four defects documented in the project
README (boot burst, impossible values, a valid reading in the wrong place, sample-rate
drift) are all still in it. The notebook is the thing that finds them.

To record a fresh log from your own rig:

```bash
python ../src/serial_logger.py      # edit PORT at the top of the file first
```
