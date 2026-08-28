# 03 — AI4I 2020: Machine Failure Classification

10000 rows of milling-machine telemetry — temperatures, rotational speed, torque, tool wear — with a machine failure in 3.4% of them. Predict the failure. Unlike project 02 the rows are independent, with no time axis and no grouping key, so none of the rolling-window tooling carries over; the leverage here comes from features derived from the physics of each failure mechanism instead. This project exists to learn what to do when the aggregate metric says you are doing well and you are not.

**Headline result:** **recall 0.838, precision 0.934, F1 0.884** on a test set never touched during selection. More usefully: of the four failure modes with a mechanism the sensors can actually see, **58 of 58 were caught — zero missed.** Almost every remaining miss is tool wear failure, which the project then proves is not predictable from these sensors at all.

**Notebook:** [`notebook.ipynb`](notebook.ipynb)

---

## Structural difference from NASA

| | NASA Turbofan | AI4I |
|---|---|---|
| Structure | time series grouped by engine | every row independent |
| Split | by groups (the engine) | **random + `stratify`** |
| Features | time windows within each engine | no windows — no temporal neighbour |

**The rule:** inspection is always mandatory, but **its tools follow the structure of the data**. Time-series tools (rolling · residuals · trend) do not transfer to a table of independent rows.

---

## Leakage — caught by a criterion, not by memorisation

**Dropped columns:**

- `UDI` · `Product ID` — two arbitrary identifiers. (The second contains `Type` in its first character plus a sequential number.)
- `TWF · HDF · PWF · OSF · RNF` — **the failure types.** The target is computed from them, so handing them to the model = handing it the answer at 100% accuracy without reading a single sensor.

> **The governing criterion: is this column a cause or a consequence?**
> A cause or a circumstance → a legitimate feature. A consequence of the predicted event → leakage.

**And the danger of leakage is that it throws no error and does not appear as a defect** — it gives you a perfect result that you are delighted by. **Delight at a high number is the only symptom.**

**An important distinction:** using `TWF` and its siblings **for diagnosis** is entirely legitimate (and we did). What is forbidden is feeding them to the model. **Understanding the data may use anything; feeding the model is restricted to what is available at run time.**

---

## Encoding — the first encounter with a categorical column

`Type` (L/M/H — material quality) is text, and the model needs numbers.

**Why not 0/1/2:** it imposes **an order and a distance**. A linear model would assume that L→M equals M→H in effect, and that H is twice M — a mathematical relationship that does not exist.

**Two reservations:**

- The order here **does actually exist** (graded quality), so numeric encoding is not entirely wrong. The remaining problem is the distance: is L→M = M→H? You do not know.
- **Tree models are unaffected by this anyway** — they split on thresholds and assume no distance. The problem belongs primarily to linear models.

**The manual implementation — its essence is one line:**

```python
if fit:
    TYPES = sorted(d["Type"].unique())   # stored from training only
for t in TYPES:
    d[f"Type_{t}"] = (d["Type"] == t).astype(int)
```

> **We did not separate the encoding — we unified it.** The difference is in **who decides**: training decides and test follows. Had each side read its own categories and `H` were absent from the test set, it would have produced two columns instead of three — and the shape would differ.

And this is literally what `OneHotEncoder` does internally. There is no magic in it.

---

## `stratify` — why it appeared now

The positive class is only 3.4% (~340 failures out of 10000). Without a stratified split the ratio may skew between the two sides, so the model trains on one reality and is tested on another — and the difference comes from the split, not from the model.

**The effect intensifies the rarer the class.** At 50/50 the balance is nearly automatic; at 3.4% the fluctuation is noticeable.

**Three-way split:** 6000 / 2000 / 2000 · ratios 0.0338 and 0.034. And the test set was not touched until the final judgement.

---

## Features derived from physics — the most powerful tool in the project

| Feature | Formula | Corresponds to |
|---|---|---|
| `temp_diff` | process temperature − ambient temperature | the cooling rate |
| `power` | torque × speed | power |
| `strain` | wear × torque | cumulative strain |

**A linear model cannot build a difference or a product on its own** — it only sums with weights. So you give it what it cannot form.

**And do not replace — add.** We kept both the ambient and process temperatures alongside the difference: the same difference (5 degrees) means something different in a cold ambient and in a hot one.

**The effect of `strain` in isolation (one line changed):** `OSF` went **from 5 missed to zero**. And no other type changed at all.

> **The feature fixed only what belongs to it** — clean behaviour confirming that the improvement is caused by what you think it is and not by coincidence. **A feature derived from the failure mechanism is more powerful than any parameter tuning**, it explains itself, and it travels with you to another problem.

---

## Reading outliers — physics, not statistics

**The minimum torque of 3.8 against a first quartile at 33.2** looked like a measurement error. The check: power = torque × angular speed, and a very low torque means power below the minimum limit → **these are genuine `PWF` cases, not an error.** Deleting them as outliers = deleting the case you are supposed to detect. (The same lesson as Melbourne.)

**`Tool wear = 0` is not a stray value** — it is a counter that starts at zero by definition, and means a new tool. **The criterion is not "far from the first quartile" but the physical meaning.**

**A logical error I made and corrected:** inferring from a single feature about **a composite target**. `Machine failure` combines five different mechanisms, so no single feature explains them all. (A new tool does not fail through wear, but it does fail through `PWF` or `OSF` immediately.)

**And the unit:** temperature in Kelvin, not "multiplied by a thousand". 310 K = 37 °C. The absolute unit is an engineering standard because ratios in it have physical meaning.

---

## Dissecting the failure — the most important step in the project

**First result (RandomForest, threshold 0.5):**

```
[[1932    0]
 [  23   45]]
recall 0.662 · precision 1.000 · accuracy 0.989
```

**Zero false alarms is not an achievement — it is a symptom.** The model is extremely conservative: it does not warn unless it is completely certain, so it loses the borderline cases. And 98.9% accuracy against a **baseline of 96.6%** ("never a failure") — only 2.3 points.

**The dissection by failure type (missed against caught):**

```
     missed  caught
TWF     14       1     ← an almost total blind spot
HDF      3      13
PWF      0      24     ← complete, thanks to the power feature
OSF      5      15
RNF      0       0
```

> **The failure is not general — it is concentrated in one type.** And that appears in no aggregate metric.

**And why `PWF` is complete and `TWF` almost nonexistent:** `PWF`/`HDF`/`OSF` are derived from **relationships between the current readings** — present in the row. `TWF` occurs above a threshold of wear, but the sensors do not see the state of the tool edge — they see the counter only. **The signal is inherently weaker.**

---

## Proving that `TWF` is not predictable

**A first hypothesis, falsified:** does `RNF` (the random one) explain the misses?
→ **one** failure out of 339 (0.3%). It does not explain 23 misses. **So the misses have a physical, learnable cause** — unlike NASA.

**And the decisive check:**

```
TWF    : count 46 · min 198 · mean 216 · max 253
healthy: count 9661 · mean 107 · max 246
healthy above 198: 768 of 9661
```

**94% of those that exceeded the threshold did not fail.** High wear **warns of danger and does not determine occurrence**. And the model is right: warning at every wear > 198 means 46 hits against 768 false alarms.

**And three attempted remedies, all of which bought `TWF` at an exorbitant price:**

| Attempt | Result |
|---|---|
| Lowering the threshold to 0.2 | +9 failures · +23 false alarms |
| `class_weight="balanced"` | `TWF` from 1 to 2 only |
| `GridSearch` + `scoring="recall"` | `TWF` 12/15 · **35 false alarms** |

**And the last one exposes the mechanism:** the model did not learn a separation — it learned to **warn at every high wear value**. Exactly what the numbers had predicted.

> **The information is not in the data. The limit is not in the model and not in the parameters — it is in the sensors.**

**And the trade-off is harsher here than in NASA** (3.4% against 14.5%): every lowering of the threshold pulls in many healthy rows before it pulls in a single failure.

---

## The winning model — and why it won

`HistGradientBoostingClassifier` via `GridSearchCV`:

```python
{'class_weight': 'balanced', 'learning_rate': 0.1,
 'max_iter': 400, 'max_leaf_nodes': 15, 'min_samples_leaf': 10}
```

**`scoring="f1"`, not `"recall"`** — recall alone led to a model that over-warns (35 false alarms). F1 balances recall against precision.

> **The metric you optimise on determines the model you get.**

**Why boosting beat the forest:** it builds trees **in sequence**, each correcting its predecessor's error, so it automatically concentrates on the difficult cases and picks up `HDF` (an interaction between two variables) without our giving it the feature explicitly. The forest builds **independent** trees that vote, so it leans towards the dominant pattern and neglects the minority.

**Final result — on the test set (untouched by any decision):**

```
[[1928    4]
 [  11   57]]
recall 0.838 · precision 0.934 · f1 0.884

     missed  caught
TWF      9       1
HDF      0      29
PWF      0      13
OSF      0      16
```

**Every type with a mechanism readable by the sensors: zero missed (58/58).** And almost all of the misses are `TWF` — the type proven to be beyond the capacity of the data.

**And the test result came out better than the validation** (0.884 against 0.796) — rare, and it means the selection did not overfit the validation set.

---

## The engineering output

1. **A predictive model** for `HDF` · `PWF` · `OSF` — near-perfect performance.
2. **A preventive maintenance rule** for `TWF`: replace the tool at a time limit, not a model. (And this is what factories actually do: preventive for tools, predictive for the rest of the failures.)
3. **A hardware recommendation:** a vibration sensor or spindle current measurement — tool breakage shows up in them before it shows up in anything measured now.

> **The recommendation is not "a better model" — it is "an additional sensor".** And this is an output a data analyst does not arrive at; it is arrived at by someone who understands failure mechanisms.

---

## Rules extracted

- **Diagnose before you treat.** Every improvement today came from an inspection, not from experimentation: `strain` from the `OSF` mechanism · the `TWF` threshold from the numbers before spending time on it.
- **A hypothesis is tested, not believed** — `RNF` and the wear threshold: both settled by a number in five minutes.
- **The aggregate metric hides and misleads** — 98.9% accuracy while a third of the failures go missed.
- **Inspect the outlier physically, not statistically** — a torque of 3.8 is a failure, and a wear of 0 is normal.
- **Save what cannot easily be reproduced:** the trained model and the experiment log. Processed data is regenerated in one line — and the code is the record of the processing.

```python
import joblib
joblib.dump(gs2.best_estimator_, "../models/ai4i_hgb.joblib")
```

---

## Note on method

We split **after** the processing, so the encoding learned from the 80% that includes the validation set — a minor indulgence. The more rigorous approach is for it to learn from the 60% alone.
