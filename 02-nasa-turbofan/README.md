# 02 — NASA Turbofan: Remaining Useful Life

Given 21 sensor channels from 100 simulated turbofan engines run to failure, predict how many cycles each engine has left. The RUL target does not exist in the files — it has to be constructed from the data's own structure, and the split has to be by engine rather than by row, because two consecutive cycles of the same engine are nearly identical. This project exists to learn regression evaluation properly, and it delivered that lesson the hard way: the model was fine for four experiments in a row while the *measurement* of it was broken.

**Headline result:** **R² 0.806, MAE 14.10, RMSE 17.65** on unseen engines — from plain `LinearRegression`. Published deep-learning models on FD001 under the same evaluation reach RMSE ≈ 13–16. Neither the model nor the features changed to get there; only what was being measured did, taking R² from 0.59 to 0.81.

**Notebooks:** [`01-regression-metrics-california.ipynb`](01-regression-metrics-california.ipynb) (metrics warm-up) · [`02-turbofan-rul.ipynb`](02-turbofan-rul.ipynb) (the project)

---

## Part 1 — Regression and evaluation metrics (California Housing)

20640 rows · 8 features · `LinearRegression` · random split (rows are independent)

| Metric | Train | Test |
|---|---|---|
| MAE | 0.5286 | 0.5332 |
| RMSE | 0.7197 | 0.7456 |
| R² | 0.6126 | 0.5758 |

**Single-feature baseline (MedInc):** R² = 0.4589 · MAE = 0.6299

### Regression concepts

- **There is no "correct"** — the question is not "how many did I get right" but "how far off was I".
- **The mean of the raw errors is zero** — positive cancels negative. Hence MAE (absolute value) or RMSE (squaring). The same logic as RMS.
- **RMSE punishes the large error severely** (the squaring); MAE treats all errors equally. The choice is a question about the system: is a rare large error worse than a constant small one? In a heating station — yes → RMSE.
- **The gap between them is a diagnosis:** RMSE ≫ MAE means a few large errors are dragging it → the model is good on average and fails in specific cases. (Here 0.74 against 0.53.)
- **R² compares against a reference** — 1 is perfect, 0 is the level of predicting the mean, negative is worse than that.
- **Read the number in its unit:** MAE = 0.53 means **$53,000** of error on a house price. The number does not judge itself — you judge it with domain knowledge.
- **The baseline first, before the model.** Without a baseline any number looks good. And the baseline changes: for time series "the value stays as it is", for classification "the most common class".
- **The time-series trap:** predicting temperature one second ahead gives R² ≈ 1 — a deception. The model learned to copy the input. The correct baseline is "no change", not the mean.
- **Linear regression rarely overfits** (too simple to memorise) — confirmed here: the train/test difference is only 3.5%. Its opposite problem is underfitting.
- **Scaling does not affect linear regression** — it is required for distance-based models, networks, and anything with regularisation.
- **Do not drop a feature based on an expectation** — train with it and without it and compare. Metrics are the decision instrument; without them any change is a guess.

---

## Part 2 — NASA Turbofan (FD001): setup

`train_FD001.txt` · 100 engines · ~20600 rows · 26 columns
Structure: unit · cycle · 3 settings · 21 sensors

**Reading the file:** `sep=r"\s+"` + `header=None` — no header, and irregular whitespace separators.

**Building the target (RUL) — it is not in the file:**

```python
life = df.groupby("unit")["cycle"].transform("max")
df["RUL"] = life - df["cycle"]
```

`transform("max")` returns something the length of the table (one value per row according to its engine), unlike `max()`, which returns only 100 values.

**Dropping 7 dead sensors** (standard deviation zero, no information): s1, s5, s6, s10, s16, s18, s19 → 14 sensors remain

**Capping the target at 125:** `df["RUL"].clip(upper=125)`

**Splitting by groups:** 80 engines for training / 20 for testing — not rows.

### Project concepts

- **A row is a moment in time, not an entity** — the engine is not repeated; there is one engine and many moments. The data shows the *shape* of degradation, and that is what the model learns.
- **The target is built, not read** — the file contains readings only; RUL is derived from the structure of the data (the last cycle = the moment of failure).
- **Make the target the closest thing to what you measure** — the sensors measure the current degradation state, which corresponds to what remains, not to what has passed. Predicting total life would need a history you do not have (how long the engine lived before recording began).
- **Capping does not delete data — it deletes a question with no answer.** Before roughly cycle 120 the readings of a healthy engine are nearly identical while RUL differs greatly; asking the model to distinguish between them is teaching it noise.
- **Group splitting, not random and not merely temporal** — the unit of independence is the engine, not the row. Cycles 100 and 101 of the same engine are nearly identical; separating them across train and test = the model has seen the answer.
- **The question that governs the split type:** what will be new at run time? Here: **an engine you have never seen** — so the test must simulate that.
- **`test_FD001` is truncated before failure on purpose** — to prevent computing the answer by subtraction. The answers live in `RUL_FD001` alone.
- **Excluded features, for different reasons:** `RUL` is the target · `unit` is an arbitrary identifier with no physical meaning · **`cycle` is a feature that short-circuits** — the model learns average lifetimes from it and stops learning the sensor signals. And engines fail anywhere between 128 and 362 cycles, so the counter does not know the state of any particular engine. If the counter were enough, we would not need sensors.
- **A small standard deviation is not a defect** — a mean of 1400 with a deviation of 0.5 reflects the nature of degradation: slow, slight change. The signal is small relative to the value → the problem is hard, not the data bad. This is also why scaling here is not optional.
- **A clean `info()` does not mean clean information** — a column that is full, numeric and constant looks healthy and is useless.

---

## Part 3 — Regression: a diagnostic journey

**First result (evaluated on all rows):**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear | 32.26 | 43.13 | 0.5959 |
| Forest | 29.38 | 43.22 | 0.5942 |

**Dissecting the error by remaining-life band:**

| Remaining life | Mean error | Deviation |
|---|---|---|
| 0–25 (most critical) | −2.2 | 26.2 |
| 25–50 | +25.9 | 20.5 |
| 50–75 | +31.7 | 23.0 |
| 75–100 | +26.1 | 23.8 |
| 100–125 | +12.5 | 24.6 |

- **It overestimates in every band except the first** → it tells a degrading engine it is better than it is = a false negative, the more dangerous kind.
- **The shape is the systematic error of a straight line on a curved phenomenon:** it lifts the middle and drops at the extremes.
- **The first band: a mean of −2.2 looks excellent, and the deviation is 26.** The range there is only 25 cycles → **the error is the size of the entire range.** The mean hid this because positive cancelled negative.

> **One metric is not enough — dissect the error by the region that matters to you.**

### Three hypotheses tested, all falsified by measurement

**1. Window features (mean/std/delta over 20 cycles) improved nothing.** The result got slightly worse. `fillna(0)` on the `delta` columns contaminated ~1600 training rows with a value that resembles nothing natural (values are around 1400, and the deltas are small fractions).

**2. Noise is not the cause — measuring SNR:**

```python
signal = d.tail(20).mean() - d.head(20).mean()          # the shift across life
noise  = (d - d.rolling(11, center=True).mean()).std()  # the residuals
```

11 sensors with a ratio > 4 (s11: 7.5 · s4: 6.8). The signal is 4–7 times clearer than the noise. **The signal is readable, and noise is ruled out.**

The data is simulated with C-MAPSS and the noise was added deliberately — its level cannot be inferred from the reputation of the organisation that produced it.

**Side discovery:** `set1/set2/set3` are constant (FD001 has a single operating condition) — three dead columns that were inside the model and had not been dropped.

**3. Capping wasted no information:**

- A pairwise comparison (shortest engine at 128 cycles against longest at 362) over the first 20 cycles: `s14` gave a difference of 4.56 × noise → it looked as though the information was there.
- **Correlation across all one hundred engines refuted it:** the strongest correlation was only 0.27, and `s14` itself was 0.11.
- **Early state does not predict lifetime.** There is no "engine that was poor from the factory" readable by the sensors → the capping is correct.

> **Rule: two cases do not make a conclusion.** A pairwise comparison generates a hypothesis; correlation over the full sample judges it.

---

## Part 4 — The real defect: the evaluation, not the model

**The mistake:** we split `train_FD001` 80/20, and the test engines have their complete stories through to failure. So the last row of every engine = the moment of death = **the answer is zero for all of them.**

`MAE = 2.94` for the forest looked like an achievement and was a broken number: an exam whose every answer is zero measures recognition of a dead engine, not prediction. (And the linear model was 40 cycles off on a finished engine — its weakness is obvious, but that is not the issue.)

**And the solution was designed in advance:** `test_FD001` is truncated at different points deliberately, and its answers are in `RUL_FD001` — and we ignored it.

**Correct evaluation — one point per engine (the last available reading):**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear** | **14.10** | **17.65** | **0.806** |
| Forest | 13.66 | 19.19 | 0.7707 |

The best published deep models on FD001 under the same evaluation: RMSE ≈ 13–16. **A linear regression comes close to them — because the features are right.**

> **Neither the model nor the features changed — what we measured on changed.**
> R² went from 0.59 to 0.81. Evaluating on all rows asks 200 questions about one engine, most of them from stages with no signal, so the inevitable error dominates the average. Reality asks once: **where is the engine now?**

**And the linear model beat the forest on RMSE** — the forest memorises the patterns of the training engines and a new engine falls outside that memorisation. **The simpler one generalised better.**

---

## Part 5 — The pipeline

A function `build(path, W, cap, with_rul)` gathers every preparation step: read and name → drop 10 dead columns → build RUL and cap it → window features (mean/std/delta within each engine) → fill the holes.

**It is applied literally to both files.** The only difference is `with_rul=False` for the test set, because it is truncated, so computing RUL from it gives false zeros.

**Without the function:** cleaning the training set by hand and then the test set by hand will inevitably differ — a different window or a forgotten sensor — so the model trains on one shape and is tested on another, and the fault is presumed to be in the model.

**Mandatory check before training:**

```python
list(X_tr.columns) == list(X_te.columns)   # must be True
```

Equality of count is not enough — `sklearn` reads by position, not by name, so a different order means one sensor is read in the place of another **silently**.

**And `sort_values("unit")`** before comparing against `RUL_FD001` — that file is ordered by engine number, and the mistake here yields random numbers that look plausible.

---

## Part 6 — Classification

Same data, different question: **will it fail within 30 cycles?**

```python
y_cls = (y < 30).astype(int)
```

**The price:** you lose the distinction between 29 cycles and 3 cycles — two practically different situations. You buy predictive accuracy at the cost of decision detail.

**The balance:** 85.5% healthy · 14.5% close to failure.
→ **an "always healthy" baseline is 85.5% accurate** and worthless (it detected not a single fault).

**`RandomForestClassifier` at the default 0.5 threshold:**

```
        pred 0   pred 1
true 0      74        1
true 1       8       17
```

Accuracy 91% (> the baseline, so the model learned something real).
**But the matrix revealed: it caught 17 of 25 — it missed a third of the faults.** Recall 0.68 · precision 0.94.

**The balance is inverted:** careful about false alarms, tolerant of the missed fault — the opposite of what a maintenance system wants. **The cause is structural:** the positive class is rare, and the model leans towards the majority because that minimises its total error.

**The remedy — the threshold is your decision, not the model's:**

| Threshold | Caught | Missed | False alarms | Recall | Precision |
|---|---|---|---|---|---|
| 0.5 | 17/25 | 8 | 1 | 0.68 | 0.94 |
| 0.4 | 20/25 | 5 | 1 | 0.80 | 0.95 |
| 0.3 | 21/25 | 4 | 3 | 0.84 | 0.88 |
| **0.2** | **24/25** | **1** | **5** | **0.96** | **0.83** |
| 0.15 | 24/25 | 1 | 6 | 0.96 | 0.80 |

**The choice: 0.2** — seven additional faults caught against four additional false alarms. And 0.15 pays a sixth false alarm for nothing in return → 0.2 is the stopping point.

> **Overall accuracy at 0.2 is lower than at 0.5.** Had you chosen by accuracy you would have chosen the setting that misses a third of the faults. **The aggregate metric would have led you to the wrong decision** — the same lesson as in the regression, in another form.

**The model gives a probability. Turning it into a decision is not the model's job — it is yours.** And 0.5 is not a scientific rule, it is the library's default. The threshold comes from the cost of the two errors.

**Recorded reservation:** we chose the threshold on the test set itself — a mild leak through repeated selection. The correct approach: choose on a validation set, and measure once, finally, on the test set.

---

## Part 7 — Splitting and leakage

**Three sets, not two:**

- **Train** — the model learns from it.
- **Validation** — you try models and settings on it and choose.
- **Test** — opened **once**, after the choice has settled.

Every time you choose based on a set, you have leaked information about it into your decision. Try five models and pick the best → you have picked what suits that particular set. **Every number you saw during selection is slightly contaminated.**

**Split by groups, not by rows** — the unit of independence is the engine. And the governing question: **what will be new at run time?** Here, an engine you have not seen.

**Training is not truncated, evaluation is:**

- Training wants the largest number of examples — an engine with a life of 200 gives 200 samples.
- Evaluation wants to simulate reality — one point per engine, with a varied answer.

**Processing order:** split → non-learning processing on both sides → learned processing fitted on train and applied to both → train the model.

**The distinguishing criterion — does the step look at more than one row?**

| Step | Needs several rows? | When |
|---|---|---|
| Drop a column · convert a type | no | safe before the split |
| A feature within a single group | yes, but inside the group | safe (our case) |
| Scaling · mean imputation · feature selection · derived limits | yes, across all data | **after the split, from the training set** |

> **The single-row test:** if one row reached you at run time, could you process it?
> Yes → sound. No, you need the rest of the rows → **leakage.**
> (In real operation one engine reaches you — there is no "test-set mean" to compute.)

**And no intermediate files are saved** — the split is in the code and the result is in memory. A file is saved at two points: the raw, and the final if needed. **The code is the record of the processing.**

---

## Conclusions

1. **Evaluation before the model.** Four experiments gave R² ≈ 0.59 — the stability of the result across two models and different features was saying that the ceiling was not in them. It was in the metric.
2. **The aggregate metric hides and misleads** — in the regression (band dissection) and in the classification (accuracy vs the matrix). In both cases it would have led to a wrong decision.
3. **The simpler one generalised better** — the linear model beat the forest on RMSE, twice.
4. **A hypothesis is tested, not believed** — noise, capping and features: three plausible hypotheses, all falsified by measurement in five minutes each.
