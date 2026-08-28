# Progress log — integrated control engineer plan

The day-by-day thread. Each entry records the task, what was actually done, and where the
full write-up lives. The detailed documentation — decisions, results and lessons — sits in
the project READMEs; this file is the chronology and the rules that outlived any single
project.

---

## 2026-08-16 — Phase 1 / Week 1 / Day 2

**AI task:** Pandas — reading and cleaning temporal data
**Hours:** planned 1.5 / actual ___

### Environment (one-off)

- Jupyter Notebook inside `.venv` via VS Code
- pandas 3.0.5 + matplotlib

Read `daily-min-temperatures` (Melbourne, 1981–1990, 3650 rows). Seven checks run on it;
two of them changed what happened next: two two-day gaps that both land on the last day of
a leap year, and a `min` of 0.0 that turned out to be real winter readings rather than a
missing-value code.

**Output:** `reports/16_8.png`
→ Full write-up: [01-timeseries-basics — Day 2](01-timeseries-basics/README.md#day-2--reading-and-cleaning-temporal-data)

---

## 2026-08-17 — Phase 1 / Week 1 / Day 3

**AI task:** Resampling and handling temporal gaps
**Hours:** planned 1.5 / actual ___

Made the gaps explicit with `asfreq` before filling them, marked the intervention in an
`is_imputed` column, then built a seasonal reference and scored each month against it. No
anomalous month across ten years — as expected from a stable weather station.

→ Full write-up: [01-timeseries-basics — Day 3](01-timeseries-basics/README.md#day-3--resampling-and-temporal-gaps)

---

## 2026-08-18 — Phase 1 / Week 1 / Day 4

**AI task:** Plotting time series + visual anomaly detection
**Hours:** planned 1.5 / actual ___

Rolling mean, residuals, and a monthly boxplot. The residual band holds constant width
over ten years, but the residuals are asymmetric (+11 against −6) — the boxplot explained
why, and also showed March to be less stable than July, which appears in neither the mean
nor the line plot.

This is the day the four levels of context got separated: seasonal, local, behavioural,
reference — each catching a fault the others do not.

→ Full write-up: [01-timeseries-basics — Day 4](01-timeseries-basics/README.md#day-4--plotting-time-series-and-visual-anomaly-detection)

---

## 2026-08-19 — Phase 1 / Week 1 / Day 5

**AI task:** Extracting features from a time window (mean, std, RMS, peak)
**Hours:** planned 1.5 / actual ___

Built a feature table over a 7-day window. Two findings that mattered later: the feature
rows are strongly autocorrelated (adjacent windows share 6 of 7 values), which rules out a
random train/test split; and RMS is near-useless for temperature, its place being vibration
and current.

→ Full write-up: [01-timeseries-basics — Day 5](01-timeseries-basics/README.md#day-5--extracting-features-from-a-time-window)

---

## 2026-08-20 — Phase 1 / Week 1 / Day 6 (practical)

**AI task:** Storing readings in CSV and analysing them
**Hardware:** ESP32 DevKit + DS18B20 · logged over the serial port
**File:** `01-timeseries-basics/data/heat_log.csv` — 1491 rows / 25 minutes

First contact with my own data rather than a curated dataset. Four defects found, one of
which — a valid reading of 46.0 sitting in the wrong place — no physical limit can catch,
only temporal context. Measured the noise floor (0.09 °C), the sensor step (0.25 °C) and a
3% sample-rate drift traced to a one-line scheduling bug.

**Deferred by agreement:** the heater log file design. **Still open:** measuring the
sensor time constant from the cooldown curve; the heater itself, which is the missing piece
that blocks any modelling.

→ Full write-up: [01-timeseries-basics — Day 6](01-timeseries-basics/README.md#day-6--logging-real-readings-to-csv-and-analysing-them)

---

## 2026-08-21 — Phase 1 / Week 2

**Task for 22/8 brought forward a day:** regression and evaluation metrics
**Plus:** start of the NASA Turbofan project

Two parts. First, California Housing as a vehicle for the regression metrics themselves —
MAE against RMSE, what the gap between them diagnoses, and why the baseline comes before
the model. Second, the start of NASA Turbofan FD001: reading a headerless whitespace-
separated file, constructing the RUL target that is not in it, dropping seven dead sensors,
capping the target at 125, and splitting by engine rather than by row.

**Prediction recorded before running:** will R² be higher or lower than California?

→ Full write-up: [02-nasa-turbofan — Parts 1 and 2](02-nasa-turbofan/README.md#part-1--regression-and-evaluation-metrics-california-housing)

---

## 2026-08-22 — Phase 1 / Week 2

**Completed today from the plan:** 23/8 (classification) · 24/8 (splitting) · 25/8 (leakage) · 26/8 (pipeline)
**Project:** NASA Turbofan FD001 — complete

The important day. Four experiments all landed at R² ≈ 0.59, and the stability of that
number across two models and different feature sets was the signal that the ceiling was not
in either — it was in the evaluation. We had been testing on engines whose stories run
through to failure, so the last row of every one has the answer zero. Switching to one
point per engine against the supplied `RUL_FD001` answers took R² from 0.59 to 0.81 without
touching the model or the features.

Three hypotheses were raised and all three fell to measurement in about five minutes each:
window features, noise, and the capping.

**Next:** 27/8 — pinning the experiments down: seeds, versions, requirements.

→ Full write-up: [02-nasa-turbofan — Parts 3 to 7](02-nasa-turbofan/README.md#part-3--regression-a-diagnostic-journey)

---

## 2026-08-24 — AI4I 2020 Predictive Maintenance

**Type:** supervised classification · tabular data with independent rows
**Size:** 10000 rows · target `Machine failure` at **3.4%**

The first project where the row is independent and none of the time-series tooling applies.
Its lesson is the mirror of the NASA one: a model reporting 98.9% accuracy and zero false
alarms was missing a third of the failures, and the misses were concentrated almost entirely
in a single failure type. Proving that type unpredictable — 94% of machines exceeding the
wear threshold never failed — turned the deliverable from a model into a hardware
recommendation.

→ Full write-up: [03-ai4i-maintenance](03-ai4i-maintenance/README.md)

---

## Retail customer value (undated in the original log)

**Type:** composite — unsupervised (clustering) + supervised (classification)

Two iterations. The first, on the cleaned single-year retail export, built RFM features
over a 9-month observation window and predicted churn over the following 3 months. The
second widened to the full two-year Online Retail II with an 18/6-month split and asked not
only whether a customer returns but how much they spend.

The first project where the raw row is not the unit of analysis — a row is an invoice line
and the question is about customers, so the table has to be rebuilt before any model sees it.

→ Full write-up: [04-retail-clv](04-retail-clv/README.md)

---

# Rules that outlived their project

Collected across the whole track. Each was learned once, in one project, and then applied
in the next.

### On measurement and evaluation

- **The baseline first, before the model.** Without a baseline any number looks good. And
  the baseline changes with the problem: for time series "the value stays as it is", for
  classification "the most common class".
- **The aggregate metric hides and misleads.** Dissect the error by the region that matters
  to you. It would have led to the wrong decision in both project 02 and project 03.
- **Read the number in its unit.** MAE = 0.53 means $53,000 on a house price. The number
  does not judge itself.
- **The metric you optimise on determines the model you get.** `recall` alone produced a
  model that over-warns; `f1` balanced it.
- **When a result is stable across different models and different features, the ceiling is
  in neither.** Look at what you are measuring.
- **Evaluation before the model.**

### On hypotheses

- **A hypothesis is tested, not believed.** Six were raised across projects 02 and 03; all
  six were settled by a number in about five minutes each.
- **Two cases do not make a conclusion.** A pairwise comparison generates a hypothesis;
  correlation over the full sample judges it.
- **Do not drop a feature based on an expectation** — train with it and without it and
  compare.

### On splitting and leakage

- **The governing question for the split type: what will be new at run time?**
- **The unit of independence is not always the row** — it was the engine in project 02, the
  customer in project 04.
- **Three sets, not two.** Every time you choose based on a set, you leak information about
  it into your decision. The test set opens once.
- **The single-row test:** if one row reached you at run time, could you process it? Yes →
  sound. No, you need the rest of the rows → leakage.
- **Cause or consequence?** A cause or circumstance is a legitimate feature. A consequence
  of the predicted event is leakage.
- **Leakage throws no error and appears as no defect.** Delight at a high number is the only
  symptom.
- **A time-based split is required when the goal is predicting the future from past
  behaviour.**

### On data and physics

- **An outlier is not an error.** Inspect it physically, not statistically. Deleting
  outliers is often deleting the event you are trying to detect.
- **A syntactic check is not a physical check.** Every sensor has a known range; a value
  outside it is rejected at logging time.
- **When you do not know the bounds of the value, you know the bounds of time.**
- **A clean `info()` does not mean clean information** — a full, numeric, constant column
  looks healthy and is useless.
- **The type governs what can be asked.** A date stored as text blocks every temporal check.
- **An imputed value is not a measurement.** Record it as such and exclude it at evaluation.
- **Do not assume the interval is what you asked for.** Measure it from the file. The first
  check in any recording.
- **Order is itself information** in a time series; shuffling destroys the signal.

### On models and features

- **The simpler one often generalises better.** Linear beat Random Forest twice in project
  02; Logistic Regression beat it in project 04.
- **A feature derived from the failure mechanism is more powerful than any parameter
  tuning** — it explains itself and it transfers.
- **Do not replace — add.** Keep the raw quantities alongside the derived ones.
- **Feature engineering does not guarantee an improvement.** Test it; do not assume it.
- **The model gives a probability. Turning it into a decision is your job**, and 0.5 is the
  library's default, not a scientific rule.
- **The tools follow the structure of the data.** Time-series tooling does not transfer to
  a table of independent rows.
- **The code is the record of the processing.** Save the raw and the final; regenerate the
  intermediate.
- **Save what cannot easily be reproduced:** the trained model and the experiment log.

### On engineering output

- **Diagnose before you treat.** Every improvement in project 03 came from an inspection,
  not from experimentation.
- **Sometimes the recommendation is not "a better model" — it is "an additional sensor".**
  When the information is not in the data, the limit is in the instrumentation, not in the
  model or its parameters.
