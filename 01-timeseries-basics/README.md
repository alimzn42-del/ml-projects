# 01 — Time-Series Fundamentals

Before any model can be trusted, the series feeding it has to be. This project builds that habit on two datasets deliberately chosen to be opposites: ten years of Melbourne daily minimum temperature (3650 rows, clean, public) to learn gap handling, resampling, residuals and window features — then 25 minutes of my own ESP32 + DS18B20 sensor log to meet the same problems in their raw form. It exists to build the diagnostic reflexes everything downstream depends on, not to predict anything.

**Headline result:** on my own hardware, measurement noise came out at **±0.09 °C** and sample-rate drift at **43 seconds over 1476 readings (≈3%)** — the first sets the floor beneath which no alarm threshold is meaningful, the second turned out to be a one-line firmware bug.

**Notebooks:** [`01-cleaning-and-features.ipynb`](01-cleaning-and-features.ipynb) (Melbourne) · [`02-sensor-log-analysis.ipynb`](02-sensor-log-analysis.ipynb) (ESP32 log)

---

## Day 2 — Reading and cleaning temporal data

**Data.** `daily-min-temperatures` — daily minimum temperature, Melbourne, 1981–1990. 3650 rows × 2 columns. Source: jbrownlee/Datasets.

### Checks

| Check | Tool | Result |
|---|---|---|
| Types | `dtypes` | `Date` was text → `to_datetime` |
| Axis regularity | `diff().value_counts()` | two two-day gaps |
| Gap location | boolean mask + indexing | 1984-12-31 and 1988-12-31 |
| Duplication | `duplicated().sum()` | 0 |
| Explicit missing | `isna().sum()` | 0 |
| Plausibility | `describe()` | `min = 0.0` — suspicious |
| Visual detection | `.plot()` | the zeros all sit in winter troughs |

### Conclusions

1. **The two gaps are systematic, not random** — both are the last day of a leap year. Likely cause: a leap-year handling bug at export, not a sensor fault. Consequence: do not interpolate them before understanding the cause.
2. **The zeros are real readings** — 4–5 points, all in winter troughs, not one in summer. Had they been a missing-value code they would not have respected the season.

### Concepts

- Order is itself information — shuffling rows destroys the signal, unlike an ordinary table.
- The type governs what can be asked — a date stored as text blocks every temporal check.
- An empty row ≠ a row that does not exist — `isna()` catches only the first.
- The plot is a diagnostic instrument, not decoration — the position of a value on the axis is what settled the question of the zeros.

**Output:** `reports/16_8.png`

---

## Day 3 — Resampling and temporal gaps

### Execution

| Step | Tool | Result |
|---|---|---|
| Temporal index | `set_index("Date")` | `DatetimeIndex` — `freq=None` |
| Partial indexing | `df.loc["1985"]` | 365 rows — a full year from one string |
| Absence detection | `asfreq("D")` | 3650 → 3652, two explicit holes |
| Marking the intervention | `is_imputed` | a boolean column, added *before* filling |
| Filling | `interpolate(method="time")` | 1984-12-31 = 14.85, between 13.3 and 16.4 |
| Seasonal reference | `groupby(index.month).agg` | 12 rows — mean and std per month |
| Actual values | `resample("ME").mean()` | 120 rows |
| Standard score | `(actual - mean) / std` | range between −0.8 and +0.9 |

### Result

No anomalous month across ten years. Random fluctuation around zero, no trend and no clustering — as expected from a stable weather station.

### Concepts

- **Anomaly is contextual** — 4 °C in July is normal, in January it is an anomaly. The global mean is useless in the presence of a seasonal signal.
- **Two levels of detection** — a point anomalous within its month, and a month anomalous within its history. The first catches the instantaneous jump, the second the slow drift (predictive maintenance).
- **The standard score unifies units** — temperature, pressure and current all become "how many deviations from normal", so they can be compared and combined into a single health index.
- **`resample` ≠ `groupby(month)`** — the first preserves the time axis (how it changed over time), the second breaks it (what is normal for each season).
- **Downsampling ≠ upsampling** — the first is summarisation by choice (mean/max/sum according to physical meaning), the second invents unmeasured data.
- **Interpolation is an assumption, not a measurement** — a straight line where the physics is an exponential curve. Acceptable for a short gap, a lie for a long one.
- **Mean vs median** — the mean is contaminated by the very anomaly you are looking for; the median is more robust in fault-laden data.
- **Order of operations:** types → gaps → anomalies → summarisation. Every aggregation hides whatever you did not deal with before it.

### Two rules

1. Turn the silent problem into a visible one before treating it (`asfreq` before `interpolate`).
2. An imputed value is not a measurement — record it as such, and exclude it at evaluation.

### Technical notes

- Notebook cells share one memory and visual order ≠ execution order. Before saving: Restart Kernel, then Run All.
- matplotlib does not support right-to-left text — axis labels are in English.

---

## Day 4 — Plotting time series and visual anomaly detection

### Execution

| Tool | Instruction | What it revealed |
|---|---|---|
| Moving average | `rolling(7).mean()` | the trend beneath the noise |
| Residuals | `Temp - rolling(7, center=True).mean()` | oscillation around zero, no seasonality |
| Monthly boxplot | `boxplot(column, by="month")` | the full distribution per month |

### Results

- **The residual band has constant width** across ten years → system behaviour has not changed.
- **The residuals are asymmetric:** the positive side reaches +11, the negative barely −6.
- **The boxplot explained why:** the outliers are all above, in the hot months, and not one below in the cold ones. Minimum temperature has a natural floor and no comparable ceiling.
- **Volatility is not seasonally constant:** March's box is longer than July's — March is less stable, not merely warmer. This appears neither in the mean nor in the line plot.

### Four levels of context — not parallel tools

| Level | Comparison | Reveals |
|---|---|---|
| Seasonal (boxplot) | value against its calendar month | seasonal anomaly |
| Local (residuals) | value against its immediate neighbours | instantaneous jump |
| Behavioural (`rolling().std()`) | dispersion against its own history | slow degradation |
| Reference (z-score) | the month against all its counterparts | deviation of a whole period |

Each level catches a fault the others do not → industrial monitoring runs them together.

### Concepts

- **`rolling` ≠ `resample`** — the first smooths and keeps 3650 points, the second summarises to 120. The first is for seeing, the second for summarising.
- **`center=True`** has no lag — valid for historical analysis, **impossible live**, because it needs the future. Live operation: look backwards only.
- **Residuals = detrending** — the moving average carries everything slow (season + trend); subtracting it leaves the fast alone, so the anomaly surfaces.
- **The boxplot = automatic contextual anomaly detection** — its threshold is computed from the box length itself, so each group is measured with its own ruler.
- **The rectangle = the middle half** (Q1→Q3), not the range of values. The whisker = the furthest plausible value. Every value outside it is drawn, even a single one.
- **An outlier ≠ an error** — the tool says "statistically distant"; the decision comes from understanding the system. Deleting outliers here = deleting the event you are trying to detect.
- **Window size follows the time constant** — much shorter picks up noise, much longer erases the response. With τ = 150 s: 2–5 minutes. In practice several windows are computed together, not one.
- **The function follows the physical meaning** — `mean` for temperature, `max` for pressure if the danger is in the peak, `sum` for energy, `std` for stability.

### Why any of this matters

1. **Setting the alarm threshold** — measured from the dispersion of the healthy system, not invented.
2. **Detecting degradation before failure** — a weakening fan does not change the mean, it changes the dispersion. Watching the mean alone leaves you blind until the day it fails.
3. **Model inputs** — the reading says where you are, the mean says where you came from, the deviation says how stable you are. The reading alone is not enough.

### Tool not yet implemented

`rolling(30).std()` plotted over time — constancy was inferred visually from the width of the residual band, not measured directly.

---

## Day 5 — Extracting features from a time window

Feature table `feat` over a 7-day window: `mean` · `std` · `max` · `min` · `range` (max−min) · `rms`

```python
feat["rms"] = df_full["Temp"].rolling(w).apply(lambda x: (x**2).mean()**0.5)
```

### The central idea

An instantaneous reading is a point without a history. 25 °C rising ≠ falling ≠ steady ≠ oscillating — all of them are 25, and all are different states of the system. Features turn the window into a description: where it is, how stable, what it reached.

**The reading says where you are; the features say where you came from and how stable you are.**

### The features and what they measure

| Feature | Measures | Note |
|---|---|---|
| `mean` | position of the centre | |
| `std` | general dispersion | **the most important one for predictive maintenance** — a weakening fan does not change the mean, it changes the dispersion |
| `range` | extent of the extremes | fastest and easiest to interpret, but a single outlier jumps it |
| `rms` | signal power regardless of direction | |

### RMS — a corrected understanding

- **RMS ≥ mean always**, and they are equal only for a perfectly constant signal.
- The difference between them is not an error to be reduced — **the difference itself is the information** (the amount of oscillation).
- An alternating signal: mean zero, RMS not zero — hence the squaring.
- **With temperature it is of little use** (positive, closely spaced values) — verified in practice. Its place is vibration and current. Vibration RMS is the standard indicator of bearing health.

### Two observations from the output

1. **Features are strongly autocorrelated in time** — two adjacent windows share 6 of 7 values, so the values change slowly from row to row. → feature rows are not independent → a random train/test split is outright cheating.
2. **High sensitivity to a single point** — 1990-12-31: one low value left the window and `std` dropped from 1.77 to 0.98, the range from 5.7 to 2.8. → useful for fast detection, but noisy → compute several windows of different lengths together.

### When is a feature "good"? — there is no absolute value

In order of strength:

1. **The system's own baseline** — run it healthy, record, compute. The result is the definition of normal *for this system*. You compare it against itself, not against others. Threshold at 3 deviations.
2. **Industrial standards** — they exist for vibration and noise (ISO), rarely for anything else.
3. **The process requirement** — 70 ± 1 °C: the limit comes from the requirement, not from statistics.

**And the trend matters more than the number:** a high but steady value may be normal; a value rising slowly over weeks is the degradation — even while it is still under the threshold.

### Technical note

`apply` drops back into Python for every window — far slower than `mean`/`std`, which are written in C. Imperceptible at 3652 rows, perceptible at millions.

### Open

The first 6 rows of `feat` are empty (the window needs 7 values) — must be handled before training.

---

## Day 6 — Logging real readings to CSV and analysing them

**Hardware:** ESP32 DevKit + DS18B20 · logged over the serial port
**File:** [`data/heat_log.csv`](data/heat_log.csv) — 1491 rows / 25 minutes
**Logger:** [`src/serial_logger.py`](src/serial_logger.py)

**The sequence:** ~5 min at rest → heating by hand (29 → 35.75 °C) → natural cooling → rest

### The four defects found

| # | Problem | How it was caught | Decision |
|---|---|---|---|
| 1 | Boot burst: 15 readings in 15 ms | time difference = 0 | dropped as a unit |
| 2 | Impossible values: 13232 · 3028 · 1184 · 2 | physical limit | dropped |
| 3 | **46.0 — a valid value in the wrong place** | temporal context, not the value | dropped with the burst |
| 4 | Rate drift: 1.03 s instead of 1.00 s | `diff().describe()` | to be fixed in the firmware |

### The measured numbers

- **Measurement noise:** residual deviation ≈ 0.09 °C → no alarm threshold can be tighter than this
- **Sensor step:** 0.25 °C (DS18B20 at 10-bit resolution)
- **Total drift:** 43 seconds over 1476 readings (≈3%) — matching the calculation

### Rules

- **A syntactic check ≠ a physical check** — the filter rejected text and accepted 13232. Every sensor has a known range; a value outside it is rejected at logging time, not after.
- **A valid value in the wrong place** (46) is caught by no physical limit — it is caught by temporal context alone.
- **When you do not know the bounds of the value, you know the bounds of time** — a reading that arrived at an impossible time is rejected whatever its value. And the rate limit comes from the sensor, not from preference (DS18B20: conversion time can reach 750 ms).
- **Do not assume the interval is what you asked for** — measure the actual interval from the file. The first check in any recording.
- **Over-smoothing does not show up as an error** — a 120-second window produced a cleaner curve that was further from the truth: lag plus a clipped peak. Smoothness is not evidence of correctness.
- **Residuals reveal fast change, not high value** — the 35.75 peak did not appear in them because it arrived slowly; the moment of touch did, because it was fast.
- **Physics gives the maximum rate of change** — from the time constant you derive the fastest possible change per second. Anything beyond it is not temperature, it is a measurement fault. An anomaly detector with no learning and no statistics — derived from the model, so it cannot be contaminated by the data.
- **An exceedance is investigated, not corrected** — the physically impossible is deleted; the possible-but-distant is flagged and examined. Deleting it automatically = deleting the event you are trying to detect.
- **The temperature did not stabilise — the reading stabilised.** The residuals go to exactly zero when the real change becomes smaller than the sensor's step.

### Quantisation noise

The residuals are a regular sawtooth, not random: the sensor jumps in a fixed step while the moving average is continuous between them. The source is the sensor's limited resolution, not electrical noise.

**The practical difference:** random noise falls with averaging; quantisation noise does not — it is a physical limit on reading precision.

### Log file design (for the heater — deferred by agreement)

```
timestamp · ms_since_boot · temp_ds18b20 · temp_lm35
temp_ambient · humidity · heater_cmd · fan_state
```

Sample rate: one second (deliberately faster than needed — `resample` can go down, it cannot invent). **Protocol:** rest → heat to steady state → full cooldown. The cooldown gives the time constant cleanly, without the heating influence.

### Notes for control (outside the AI track)

- **Fixed `millis` scheduling** — compute the next reading instant by adding the interval to the stored instant, not to the current time. The difference is one line; the effect was 43 seconds. It immunises you against changes in read time from resolution, cable length, or any line added later.
- Print the number alone on the port — descriptive text breaks parsing.
- A relay is unsuitable for a PID loop (mechanical life) — a MOSFET or SSR is required.
- LM35 on the ESP32 ADC: the signal sits in the first tenth of the range → poor resolution.

### Open

- Measure the sensor's time constant from the cooldown curve (63% of the distance)
- The heater: the most important missing piece — with no measured input there is no modelling and no prediction
- A unified inspection function (that displays and does not decide)
