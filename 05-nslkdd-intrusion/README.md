# NSL-KDD — Network Intrusion Detection

**Date:** 2026-08-29/30 · **Type:** supervised + unsupervised together
**Data:** NSL-KDD · `KDDTrain+` 125,973 rows · `KDDTest+` 22,544 rows · 43 columns

*(الأصل العربي محفوظ في [README.ar.md](README.ar.md))*

---

## Why this dataset

Chosen to serve two specific goals untouched by the previous projects:
**PCA** (41 correlated features — dimensionality reduction as a necessity, not an exercise)
and **anomaly detection without labels**, with a genuine justification: **nobody has ever
seen the new attack.**

**The objection that killed the bearing project before it does not apply here:**
there the load was constant and the failure obvious in RMS — a perfect environment for an
`if` on a threshold, where a detector adds nothing. Here **no threshold detects an
intrusion, and no ISO standard exists for one.**

> **The bearing lesson stands:** an engineer who builds a model where a threshold
> suffices is wasting their time. The first question about any detection project:
> **does an `if` suffice?**

---

## Structure of the data

**41 features in four groups:**
- **About the connection:** duration, bytes sent/received, failed login attempts.
- **Three categorical:** `protocol_type` (3) · `service` (70) · `flag` (11).
- **About the last two seconds:** number of connections to the same host/service and
  their error rates → catches **denial of service**.
- **About the last 100 connections:** the same rates over a longer window → catches the
  **slow scan** that evades the two-second window.

> **The dataset's author built time-window features** — exactly what I built on 19/8 and
> in the Retail project. **An instantaneous reading is not enough; context comes from a
> window.** The same idea in every domain.

**Distribution:** 53.5% normal · 46.5% attack → **the "all normal" baseline = 53.5%**,
and accuracy is a valid metric here (unlike AI4I, where the baseline was 96.6% and
accuracy lied).

**22 attack types — `neptune` alone is 41,214 (70% of the attacks).**

---

## Encoding — a decision by measurement, not intuition

`service` has 70 values. The six most common cover 90k rows; 36k rows are spread over
64 services.

**Worrying about 70 columns is legitimate for two reasons:** many services appear ten
times (the model memorises them rather than learns them), and with PCA coming, columns
that are mostly zeros inject noise.

**But grouping might erase a signal** — some attacks target rare services specifically.
**So inspect before deciding:**

```
Rare services (< 500 rows): 38 services · 9,082 rows
Their combined attack rate: 0.929
The top eight of them: 1.000 attack — without exception
```

> **Rarity itself is the signal.** `efs` · `systat` · `link` are services nobody uses —
> whoever connects to them is scanning ports. **Grouping does not erase the signal, it
> strengthens it:** one column at 93% attack is clearer than 38 scattered columns.

**The decision: two columns, not one.**
```python
d["svc_freq"]  = d[2].map(freq_map)                     # frequency — carries the rarity
d["svc_group"] = d[2].where(d[2].isin(top10), "rare")   # then one-hot
```
**No harmful redundancy** — each carries information the other does not: which service,
and how rare it is.

**41 columns → 64** (instead of 123 had all seventy been one-hot encoded).

### The four encoding methods

| Method | Idea | Limit |
|---|---|---|
| One-Hot | one column per category | safe, no assumptions — but 70 columns |
| Ordinal | one number per category | **imposes a false ordering** — except with trees |
| Frequency | the count | one column, **but two categories with the same count merge** |
| Target | rate of the target within the category | the strongest and the most dangerous — **outright leakage** unless computed from the training set |

**A misconception corrected:** "frequency encoding makes the model care about high
values and neglect low ones" — false. **A tree splits on thresholds**, so it learns
`freq < 500 → attack`; a low value is the other end of the same column, not a weaker
value. **What is affected by magnitude is linear models and distance-based methods.**

**Another objection that is valid:** frequency encoding **ties you to trees** — with PCA
and distance-based detectors the number becomes misleading (`http` is eighty times
larger than `efs`, meaninglessly).

**A technical mistake learned:** mixed column names (numbers + strings) are rejected by
`sklearn`. `d.columns = d.columns.astype(str)` — **and it was placed inside the
function, not outside it.**
> **Any fix you apply by hand outside the function, you will forget when the test file
> arrives. Put it at the source.**

---

## PCA — a measurement that ended in rejection

**Scaling first, mandatorily:** PCA looks for the largest variance; unscaled, it picks
the bytes column (in the thousands) and ignores the rates (0–1).

**A discovery from the correlation map:** column 19 (`num_outbound_cmds`) is **a fully
white row** — constant, zero standard deviation, division by it gives `NaN`. **No
information; dropped.** (The same as the seven dead sensors in NASA.)

**The result:**
```
 2 components → 25.4%      10 → 56.2%      20 → 74.5%      30 → 89.6%
for 95%: 36 components out of 63
```

**Verdict: PCA is not suitable here.** Three pieces of evidence:
1. **63 → 36 is a weak reduction.** A successful one compresses 63 to 10–15.
2. **The first two components carry only 25%** — no dominant direction. (With truly
   correlated data the first alone reaches 40–50%.)
3. **The curve is near-linear** (every 10 components ≈ 15%) — **no elbow, no natural
   cut-off point.**

> **Your columns genuinely carry different information, not duplicated information.
> There is nothing in them to compress.**

**And the unacceptable price — a correction of understanding:** PCA **does not drop
columns, it replaces all of them.** A component is not a column but an equation:
`0.31×src_bytes − 0.18×duration + ...` — **nameless and uninterpretable.**

In a security problem:
- Without PCA: "alarm because 500 connections in two seconds, all failed" → **actionable.**
- With PCA: "alarm because component seven = −3.4" → **meaningless to everyone.**

> **When the price is unacceptable regardless of the gain, the measurement is a luxury,
> not a decision.** (The same logic that rejected PCA in the Retail clustering: the
> deliverable is understanding, and clarity is part of the quality.)

**A fourth strike specific to anomaly detection:** PCA keeps the largest variance —
**and anomalies are rare, and the rare has small variance.** It may throw the direction
that distinguishes the attack into the discarded components.

**Reduction remains possible by selection rather than transformation** — original
columns, with their names.

---

## Model 1: the binary classifier (supervised) — and the lying number

```
[[13462     7]        accuracy 0.9994
 [    9 11717]]       recall 0.9992 · precision 0.9994
```

**99.94% — a near-perfect result on a real security problem is a warning sign, not an
achievement.**

**The reason:** attack types are distributed randomly between training and validation,
**so the model has seen every type and memorised its fingerprint.** `neptune` alone is
41k rows with a blunt fingerprint (zero bytes, flag `S0`).

> **The model did not learn "what an attack is" — it learned "what these twenty-two look
> like."** The same pattern as `MAE = 2.94` in NASA: a number that looked like an
> achievement and was broken.

---

## Model 2: the detector (unsupervised) — the central idea

**It trains on `normal` alone — 53,874 rows — and never sees a single attack.**

**`IsolationForest` — its mechanism:** it splits the data with random lines, repeatedly,
and counts how many lines were needed to isolate each point on its own. **An outlying
point is isolated by one or two lines; a central one needs ten.** **Fast isolation =
alone = anomalous.** (It is tree-based → needs no scaling and is insensitive to
dimensionality — a good fit after rejecting PCA.)

**The guard analogy:**
The classifier is a guard who memorised photos of fifty thieves — **he catches them, and
the fifty-first walks past.**
The detector is a guard who knows the residents' faces — **he stops everyone who is not
one of them, old or new.**

**`contamination` — a sensitivity knob, not a truth:** the detector produces a
continuous score, and this rate sets where the threshold falls. In theory zero (the
training data is all `normal`), but real `normal` contains legitimate oddities.

| c | caught /11,726 | missed | false | recall | precision |
|---|---|---|---|---|---|
| 0.01 | 9,337 | 2,389 | 132 | 0.796 | 0.986 |
| 0.05 | 10,341 | 1,385 | 619 | 0.882 | 0.944 |
| 0.10 | 10,937 | 789 | 1,363 | 0.933 | 0.889 |
| **0.15** | **11,266** | **460** | **2,024** | **0.961** | **0.848** |
| 0.30 | 11,634 | 92 | 4,041 | 0.992 | 0.742 |

**0.15 was chosen** — a missed attack is worse than a false alarm.
(From 0.10 to 0.30: 697 more attacks caught at a cost of 2,678 false alarms — four for
each one.)

**A reservation on record — alarm fatigue:** flagging 15% of normal connections is
operationally a big number. **The cure is tiers, not a binary decision:** low score →
ignore · medium → log · high → alert the operator. **`score_samples` provides the
continuous score.**

**And the limit of every anomaly detector:** **it measures rarity, not danger.**
A huge nightly backup = rare and legitimate → false alarm.
A slow infiltration that mimics a user = common-looking and dangerous → walks through.
**Both will inevitably happen; the remedy is operational, not algorithmic.**
(The same lesson as kurtosis in the bearings: 6.2 on a healthy bearing from hour one.)

---

## The final test — `KDDTest+`

**Opened once. 22,544 rows · 56.9% attack · 17 new attack types (3,750 rows).**

| Model | recall | precision | false | **new types** |
|---|---|---|---|---|
| Classifier | 0.648 | 0.968 | 271 | **1,288 / 3,750 (34%)** |
| Detector | 0.783 | 0.917 | 912 | **2,925 / 3,750 (78%)** |
| **Union** | **0.812** | 0.918 | 932 | — |

**Accuracy:** classifier 0.7875 · detector 0.8357 · **union 0.8517**
(the "everything is an attack" baseline = 0.569)

**The classifier collapsed from 99.9% to 64.8% and caught only a third of the new.**
**The detector held at 78.3% and caught 78% of the new — having never seen an attack in
its life.**

> **The classifier is 3.8 points better on the known, and 44 points worse on the new.**

**The union beats both at the cost of only twenty extra false alarms** — because they
err on different cases, each covers the other's blind spot.

### Confirmation on `KDDTest-21`
(A subset of the same file: the rows most traditional classifiers got wrong.)

| | `KDDTest+` | `KDDTest-21` | drop |
|---|---|---|---|
| Classifier | 0.648 | 0.534 | −11.4 |
| Detector | 0.783 | 0.712 | −7.1 |
| Union | 0.812 | 0.751 | −6.1 |

**The ranking held → the result is a real property, not a split accident.**
**And the harder the data, the wider the gap in the detector's favour** (the classifier
dropped more).

---

## The gap — and what it revealed

```
Binary classifier:      train 0.9998 | validation 0.9994 | test 0.7875
Multiclass classifier:  train 0.9981 | validation 0.9967
Detector (alarm rate):  train 0.150  | validation 0.527  | test 0.486
```

**The train/validation gap = 0.0004 — practically zero.**
**By the traditional measure: no overfitting whatsoever.**

**The collapse came on the test set alone — 21 points.**

> **The problem is not memorising examples — it is that the distribution itself changed**
> (17 types that did not exist in training). **Its name is distribution shift.**
> **The traditional gap measurement would have said "excellent model" and deceived you.**

**The detector is consistent:** it alarms on 48.6% while the attack rate is 56.9% —
**it did not collapse because it had memorised nothing to lose.**

**A rule about rejection:** a 21-point gap is grounds for flatly rejecting **the claim,
not the model.** The classifier still catches 64.8% at 0.968 precision; **the question
is not "accept it or reject it" but: where does it work, where does it fail, and what
covers its failure?**

---

## Model 3: the multiclass classifier — and abstention

**It trains on the attacks alone** (the exact opposite of the detector) — because it
runs **after** the binary stage.

**Grouping the types:** the nine most common cover 99.7%, leaving `others` = 376 rows.
(**The five most common cover only 92%** — that would have thrown `nmap`, `back` and
`teardrop` into a catch-all despite being large and classifiable. **The same logic as
grouping `service`: group what is actually rare.**)

**On validation: 99.7%.** The same trap — `neptune` alone is 70% of the evaluation.
**The one honest row: `others` at 0.773** — the non-homogeneous class.

**On `KDDTest+`:**
```
overall label accuracy:   0.5664
on known types:           0.7953
on new types:             0.0117   ← 73 rows correct out of 3,750
```

**What it called the new types:** `neptune` 1,474 · `warezclient` 770 · `satan` 674 …
**and `others` — the correct class — it chose only 44 times.**

> **A confident wrong answer is worse than "I don't know."** The cause is structural:
> the model is **forced** to pick one of the ten classes, so it sticks every stranger
> onto the nearest familiar face.

### The cure: a confidence threshold
```python
p = np.where(proba.max(axis=1) >= t, prediction, "UNKNOWN")
```
| threshold | known: correct | known: abstained | **new: abstained** |
|---|---|---|---|
| 0.0 | 0.795 | 0.000 | 0.000 |
| 0.5 | 0.792 | 0.022 | 0.072 |
| 0.7 | 0.788 | 0.049 | 0.303 |
| **0.9** | **0.786** | **0.056** | **0.565** |

**With a single added threshold: 57% of the new became "unknown" instead of a false
label — at a cost of 0.9 points on the known.**

> **The model had the signal all along (lower confidence on the new) and we were not
> using it.** **Abstaining is a correct decision, not a failure.**

---

## The integrated system

```
Union (classifier ∪ detector)   →  attack or not
        ↓ if attack
Multiclass at threshold 0.9     →  the type's name, or UNKNOWN
```

**On `KDDTest+`:**
```
normal       11,190
neptune       5,473
UNKNOWN       2,174
satan         1,080
warezclient     739
...
```

**Dissecting `UNKNOWN` — the verdict on the whole idea:**

| | inside `UNKNOWN` | in the whole file |
|---|---|---|
| attack rate | **73.2%** | 56.9% |
| share of new types | **61.7%** | 16.6% |

**The concentration of new types is ~4× random selection.**

> **`UNKNOWN` is not a mess bin — it is the strongest signal in the system.**
> **The classifier's inability to name something is the most important alarm: a
> candidate for an unknown attack.**

**The operational deliverable:** named types → a known automated response.
`UNKNOWN` → **human investigation of 2,174 cases out of 22,544 (a tenth of the volume),
three quarters of them real attacks, containing most of the new types.**

---

## On comparison with the literature — a reservation on record

The union reached **85.2% accuracy**; commonly circulated numbers on `KDDTest+` sit
between 75 and 82%. **But a claim of superiority is unjustified:**

- Papers differ in **what they measure** — which file, which metric, which split protocol.
- The 75–82 range is an approximation from memory, **not a systematic survey**.
- **Not a single paper on this dataset was read.**

**One methodological difference in our favour:** most of them train on the full data;
**the detector here was deprived of 60k examples and still came close.**

**A more important difference:** most papers publish **the aggregate number only** —
which hides the classifier's collapse on new types (34% against 78%). **The question
measured here is sharper than theirs.**

> **Before any claim: read three or four recent papers and verify the protocol
> matches.** A claim without a literature review harms the reputation more than it
> helps it. **This is submitted as a learning project with documented results, not as a
> paper claiming superiority.**

---

## Clustering — checking the data's own structure

`KMeans` with k=10 on the attacks, **with no labels at all**, then compared against the
true types.

**Purity (share of the dominant type in each cluster): between 0.556 and 1.000,
averaging ~0.85.**

**So clustering discovered real structure without seeing a single name. But the picture
is finer than that:**

- **`neptune` split into three clusters** (16,339 · 5,520 · 11,112) — the algorithm sees
  three patterns where the experts see one. (Plausible: denial of service is executed
  against varied services and ports.)
- **`satan` split into two.**
- **Two clusters are genuine mixtures:** `satan` with `teardrop` · `back` with
  `warezclient`.
- **`others` scattered across everything** — natural, it is a non-homogeneous bin to
  begin with.

> **The experts' labels and the data's structure do not match exactly.** Some types are
> broader than their single name suggests; some resemble each other despite different
> names. **The deliverable of unsupervised learning: it shows you your data's
> structure, not your taxonomy.**

---

## Summary

**Technically:** a three-layer system that catches 81.2% of attacks on `KDDTest+`
(85.2% accuracy), **including entirely new types**, and escalates 10% of the volume for
human investigation with a concentration of new types four times better than random.

**Methodologically — five lessons:**

1. **A near-perfect number calls for inspection, not celebration.** 99.94% collapsed to
   64.8%, and the cause of the collapse was known before it was measured.

2. **The right test is what the model has not seen.** A validation split cut randomly
   from the same file measures memorisation, not generalisation.

3. **The traditional gap measurement does not detect distribution shift.** 0.0004
   between train and validation, and 21 points on the test. **Two metrics for two
   different risks.**

4. **A tool is rejected by measurement or by price.** PCA was measured (63→36) and
   rejected on price (losing interpretability in a security problem). **Measurement is
   a luxury when the price is unacceptable to begin with.**

5. **Abstention is a correct decision.** A model forced to answer sticks the stranger
   onto the nearest familiar face; **one confidence threshold turned 57% of the
   confident errors into a useful signal.**

---

## Pending
- Tuning the detector's parameters (never tuned at all)
- Two alarm tiers via `score_samples` instead of the binary decision
- Reading 3–4 papers on NSL-KDD and verifying the comparison protocol
- **Cleaning the duplicated `NSL-KDD/nsl-kdd` folder before pushing to Git**

---

# 2026-08-31 — Autoencoder for anomaly detection

**Task 31/8 from the plan** · continuation of the NSL-KDD project
**The first neural network of the track** — originally scheduled for 19/9, brought
forward by two weeks.

---

## The neural network — the concept

**Start from linear regression:** `y = w₁x₁ + w₂x₂ + b` — multiply by weights and sum;
training is finding the weights. **Its structure is fixed: a straight line.**
(Which is why it failed on NASA — degradation is curved and a line cannot represent it.)

**The network adds two things:**

**1. Successive layers** — each one's output is the next one's input. Each layer builds
a higher representation than the one before, instead of a single transformation.

**2. Non-linearity — the key.** After each layer, a simple function breaks the
straightness (`relu`: negatives become zero).

> **Without the non-linearity, ten linear layers equal one layer mathematically** — it
> stays a straight line no matter how many you stack. **The non-linearity is what
> allows representing curves and interactions.**

**Training:** feed it an input → it produces a prediction → measure the error →
**nudge all the weights slightly in the direction that reduces it** → repeat.

**There is no magic in it:** weights, multiply, sum, and a function that breaks
straightness. **The difference from trees:** a tree splits on thresholds; a network
builds graded representations.

---

## Autoencoder — the idea

```python
ae.fit(Zn, Zn)      # the output is the input itself
```

**You ask the network to reproduce what you gave it.** It looks absurd — **except for
the constraint:**

```
63 columns  →  16 numbers  →  63 columns
                ↑ the bottleneck
```

**The middle layer is smaller than the input, so it cannot copy.**
(Were it 63, it would copy and be done, learning nothing.)

**The narrowness forces it to choose: what deserves keeping?**
**So it learns the essential structure and discards the details.**

**The link to detection:**
Train it on `normal` alone → it masters compressing and rebuilding it.
Then an attack arrives — a pattern unlike anything it has seen → **it compresses it with
the rules of the normal and fails to rebuild it.**

> **Reconstruction error is the anomaly score.**
> Small → looks like normal. Large → does not.

**The idea is very close to PCA** — compression to fewer dimensions, then
reconstruction. **The difference: PCA is linear and the network is not** — so it
captures curved relations PCA was blind to yesterday.

---

## Design decisions

**1. Scaling is mandatory.** The error is measured as a squared difference; a column in
the thousands (bytes) generates a huge error while a 0–1 column (rates) contributes
nothing. **The total error becomes the error of one column.** And the scaler learns
from normal alone, like the network.

**2. A linear, unbounded output layer.** The data is scaled (positive and negative
values around zero). **An activation that clamps the output to [0, 1] makes
reconstructing negatives impossible — so everything looks anomalous.** A common
mistake, discovered late.

**3. Why `63→32→16` rather than `63→16` directly:**
The direct jump is **a single linear transformation** plus an activation — barely more
than PCA with a slight bend. The gradual version: the first layer builds an
intermediate representation (grouping similar columns), **and the second compresses
those concepts, not the raw columns.** **Depth is the source of a network's power:
graded representations, not one transformation.**

**On what basis are layers added — no rule settles it:**
- **The complexity of the relation, not the size of the data.** Tabular → one or two
  layers. Images and audio → dozens, because the representation is hierarchical by
  nature (edges → shapes → objects).
- **Data size limits depth** — every weight needs examples. 54k rows are enough for a
  small network.
- **The practical criterion: start with one, measure, add, measure. Stop when the
  improvement stops or the train/test gap widens.**

> **Architecture choice is fundamentally empirical. Experience shortens the
> experimentation; it does not eliminate it.**

---

## An environment obstacle

`tensorflow` refused to install: **Python 3.14 unsupported** (its builds lag new Python
releases by months).
**The fix: a second environment `.venv313` alongside the current one, untouched.**
```
winget install Python.Python.3.13
py -3.13 -m venv .venv313
pip install tensorflow pandas matplotlib scikit-learn ipykernel
```
**The collision will recur** — CNN in month 4, LSTM in month 11. **Solving it once
beats working around it.**

**The methodological lesson:** a friend was asked which version works on his machine —
a fair question. **The sharper move: the official `tensorflow` page states exactly what
is supported.** Read from the source instead of experimenting — the same rule as
reading the `Readme` before the data.

**In practice `MLPRegressor` from `sklearn` was used** — one line, keeping the focus on
the concept rather than on assembling layers by hand.

---

## Results

**The metric is not the reconstruction error nor the ratio between them — it is the
confusion matrix at a threshold.** **And the threshold is derived from normal alone:**
```python
thr = np.quantile(e_train_normal, q)     # the exact analogue of contamination
```
`q=0.90` → 10% of normal exceeds it · `q=0.99` → only 1%.

| Architecture | normal error | attack error | ratio |
|---|---|---|---|
| `(16,)` · 300 | 0.4614 | 32.76 | **71×** |
| `(32,16,32)` · 300 | 0.1327 | 5.61 | **42×** |

**The ratio suggested the shallow one was better. Measurement said otherwise:**

| Threshold | shallow 16 | | deep 32-16-32 | |
|---|---|---|---|---|
| | recall | false | recall | false |
| q=0.90 | 0.959 | 1,362 | 0.949 | 1,326 |
| q=0.95 | 0.871 | 680 | **0.910** | 698 |
| **q=0.99** | **0.504** | 148 | **0.800** | 151 |

> **At the strict threshold: 0.504 against 0.800 — thirty points, at practically the
> same number of false alarms.** And that is what matters operationally.
> **The ratio (71 vs 42) was a misleading indicator — the metric that maps to the
> decision settled it.** The same lesson as Retail: dropping features raised AUC and
> lowered revenue.

**`(32,16,32)` at `q=0.95` was chosen — on validation, before opening the test set.**

---

## The final test — `KDDTest+`

```
Autoencoder     | recall 0.780 | precision 0.920 | false 869 | new 2808/3750
IsolationForest | recall 0.783 | precision 0.917 | false 912 | new 2925/3750
Supervised clf  | recall 0.648 | precision 0.968 | false 271 | new 1288/3750
```

**The two detectors are practically identical — a difference of three per thousand.**

> **This is a result, not a meaningless tie:** two radically different tools — a
> tree-based one measuring **ease of isolation**, and a network measuring **difficulty
> of reconstruction** — arrived at the same number. **The ceiling is not in the tool
> but in what the data carries about "normal."**
> The recurring lesson again: **when diverse tools converge, the limit is in the
> problem, not the model.**

**The saner choice: `IsolationForest`** — the same performance, no scaling, and no
second Python environment. **The network's only advantage is 43 fewer false alarms —
marginal.**

---

## A methodological note

A first draft displayed all three thresholds on the test set — **which opens the door
to late selection, i.e. leakage through repeated choosing.**
**Corrected: the threshold was chosen on validation, and the test was measured with it
alone, once.**

## Pending
- Trying deeper/narrower architectures (`(64,32,8,32,64)`) and measuring the effect of a
  tighter bottleneck
- `tensorflow` is now available in `.venv313` — needed for architectures `sklearn`
  cannot express
