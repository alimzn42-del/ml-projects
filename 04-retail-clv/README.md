# 04 — Retail Customer Value

Two years of invoice lines from a UK online retailer, and the first project where the raw row is not the unit of analysis: a row is one line on one invoice, and the question is about customers, so the table has to be rebuilt by `groupby` before any model can see it. The split is time-based rather than random — features come from an observation window, the target from a strictly later one — because the question is "predict the future from past behaviour", and a random split would let the model see across that boundary. It combines unsupervised segmentation with supervised prediction in one project.

**Headline result:** the customer-value model reaches **AUC 0.809**, and targeting the **top 10% of customers by predicted value captures 48.2% of the actual revenue** of the following period. The earlier churn iteration documented below settled on a Decision Tree at **F1 0.6531**, having cut Random Forest's train/test gap from 0.367 to 0.0123.

**Notebook:** [`notebook.ipynb`](notebook.ipynb) · **Model:** [`models/clv_retail.joblib`](models/clv_retail.joblib)

---

## Two iterations, two datasets

This folder documents two passes at the same domain:

| | Section A — RFM & churn | Section B — customer value |
|---|---|---|
| Data | Cleaned UCI Online Sale, Dec 2010 – Dec 2011, 531,282 rows | Online Retail II, Dec 2009 – Dec 2011 |
| Window | 9 months features / 3 months target | 18 months features / 6 months target |
| Target | did the customer come back (churn) | will they return, **and how much will they spend** |
| Result | F1 0.6531 (Decision Tree) | AUC 0.809 (HistGradientBoosting) |
| In this repo | documentation only | documentation **and** `notebook.ipynb` |

Section A is the full written record carried over from the project journal. Section B is the later work that `notebook.ipynb` actually contains; its numbers come from the notebook's own outputs and from `models/clv_retail_meta.json`, not from the journal. See [`data/README.md`](data/README.md) for both dataset sources.

---

# Section A — RFM segmentation and churn prediction

## Overview

**Type:** a composite project — unsupervised learning (clustering) + supervised learning (classification)
**Dataset:** Online Retail (UCI Machine Learning Repository) — Dr Daqing Chen, London South Bank University
**Raw data size:** 531,282 transaction rows (invoice lines), period December 2010 – December 2011

### The goal

- **Commercially:** identify the different customer segments (VIP, ordinary, at risk of leaving) and predict in advance who is liable to leave, so the company can intervene before it is too late.
- **Technically:** combine unsupervised learning (K-Means for segmentation) with supervised learning (classification to predict churn) in one integrated project.

## Part one: RFM analysis and segmentation (unsupervised)

### 1) Basic cleaning (before any split or statistical processing)

- Dropping rows with no known customer (`CustomerID == 0`): 531,282 → 397,924 rows (25.1% removed).
- Converting `InvoiceDate` from text to `datetime` — necessary for any later temporal computation.
- Dropping `UnitPrice == 0` rows (only 40 rows, 0.01% — negligible effect): 397,924 → 397,884 rows.
- Adding a `TotalPrice = Quantity × UnitPrice` column (basic feature engineering for computing Monetary).

### 2) The time-based split — instead of a random `train_test_split`

**The most important decision in the entire project:** since the goal is "predict the future from past behaviour", the data was split by date rather than randomly:

- **Observation period** (~9 months, up to 2011-09-09): to compute RFM (features).
- **Prediction period** (~3 months, up to 2011-12-09): to determine whether the customer came back and bought (target/churn).

**Why this matters:** a random split would have caused a temporal leak (the model "sees" information from the future while the features are being built), which is more dangerous than the other kinds of data leakage encountered in earlier projects.

**Seasonal note:** the prediction period covers the holiday season (September–December), which may raise the "repeat purchase" rate in a way unrelated to habitual behaviour — a constraint that must be stated when interpreting the results.

### 3) Building RFM (`groupby('CustomerID')` over the observation period only)

| Measure | Method | Note |
|---|---|---|
| Recency | days since the **last** purchase up to the snapshot date | measures the recency of current activity, not the start of the customer relationship |
| Frequency | `InvoiceNo.nunique()` (unique invoices, **not** row count) | a row is a line on an invoice, not a shopping visit; using `.count()` would have counted a single multi-product invoice as several visits, wrongly |
| Monetary | `TotalPrice.sum()` | total spend across all the customer's invoices in the observation period |

### 4) Processing before clustering

- **`np.log1p()`** on R, F, M: correcting the severe skewness of the distribution — the enormous difference between median ($566), mean ($1617) and max ($178,302) would have let the extreme values dominate the distance computation in K-Means.
- **`StandardScaler`** after the log (not before): unifying the scales of three naturally different dimensions (days, counts, dollars).

### 5) Choosing K via the elbow method

K from 2 to 10 was tried and the inertia observed; the largest "break" in the rate of improvement appeared after K=4 (the improvement from K=4→5 was less than half the improvement from K=3→4), so **K=4** was chosen as a balance between mathematical accuracy and commercial interpretability.

### 6) Segmentation results (K=4)

| Cluster | Count | Recency | Frequency | Monetary | Classification |
|---|---|---|---|---|---|
| 1 | 470 (14%) | 16.4 days | 12.2 times | $6,842.5 | 🏆 active VIPs |
| 3 | 484 (14%) | 15.4 days | 2.2 times | $644.6 | 🌱 new / promising |
| 0 | 978 (29%) | 82.7 days | 3.4 times | $1,534.7 | 😐 ordinary, stable |
| 2 | 1433 (43%) | 156 days | 1.2 times | $287.7 | ⚠️ at risk of leaving / dormant |

**Commercial note:** 43% of customers (the largest segment) are effectively dormant; and the VIP segment (only 14%) spends roughly 24 times the average of the dormant segment — a direct application of the Pareto rule (80/20).

## Part two: building the churn target and predicting it (supervised)

### 7) Building the Churn column

Using the previously isolated prediction period: `Churn = 1` if the CustomerID does not appear among the customers of the prediction period, and `0` if it does.

```python
active_customers = prediction_df['CustomerID'].unique()
rfm['Churn'] = (~rfm.index.isin(active_customers)).astype(int)
```

**Churn distribution:** 57.06% not churned (0) / 42.94% churned (1) — a reasonably balanced ratio compared to the severe imbalance problems met earlier.

**Why is there no leakage risk here despite using the prediction period?** Because the target alone is built from the prediction period (which is required by definition), while all the features (R, F, M) are built exclusively from the observation period that precedes it in time.

### 8) A random `train_test_split` (not temporal) at this stage

After the final RFM table was built, every row became an independent customer with fixed values (time has been "swallowed" inside the numbers themselves, as Recency). So using `train_test_split(..., stratify=y)` randomly is correct here, unlike at the RFM/churn construction stage.

### 9) Additional feature engineering

- **`AvgOrderValue = Monetary / Frequency`**: distinguishes a "frequent customer with small purchases" from a "rare customer with huge purchases" despite their Monetary being equal.
- **`IsUK`**: simplifying the `Country` column (textual, unordered) to binary, because of the overwhelming dominance of Britain (90.04% of customers) and the scattering of the remaining countries over very small samples (fewer than tens each) — one-hot on every country would have added more noise than useful signal.

**The result:** the two additions did not improve performance appreciably (F1 before: 0.6296, after: 0.6309) — most likely because AvgOrderValue is mathematically derived from columns that already exist, and IsUK itself suffers from imbalance (90%/10%).

### 10) Model comparison (before tuning)

| Model | Accuracy | Train Acc | Test Acc | Gap | Test F1 (churned) |
|---|---|---|---|---|---|
| Logistic Regression | 0.6731 | 0.676 | 0.673 | **0.003** | 0.6296 |
| Decision Tree | 0.6701 | 0.715 | 0.652 | 0.062 | 0.6176 |
| Random Forest | 0.6285 | **1.000** | 0.633 | **0.367** | 0.5690 |
| KNN | 0.6270 | 0.761 | 0.623 | 0.138 | 0.5544 |

**An important discovery:** the "strongest" model (Random Forest) was the worst, because of blatant overfitting (Train Acc = 100%). With few features (4–6) and a medium data size (~2700 training customers), a complex model tends to memorise the data instead of learning a general rule. Logistic Regression (the simplest) was the most stable, because its mathematical simplicity prevents it from "memorising" in the first place.

### 11) Hyperparameter tuning (GridSearchCV, scoring='f1', cv=5)

The goal here was not only to improve performance but to **reduce the gap** between train and test — that is, to treat overfitting directly.

| Model | Best parameters | Train Acc | Test Acc | Gap | Test F1 (churned) |
|---|---|---|---|---|---|
| **Decision Tree** | max_depth=4, min_samples_leaf=5 | 0.6965 | 0.6701 | 0.0264 | **0.6531** 🏆 |
| KNN | n_neighbors=25 | 0.6902 | 0.6850 | 0.0052 | 0.6319 |
| Random Forest | max_depth=3, min_samples_leaf=10, n_estimators=50 | 0.6839 | 0.6716 | **0.0123** | 0.6298 |
| Logistic Regression | C=10 | 0.6768 | 0.6716 | 0.0052 | 0.6298 |

**A decisive result:** the Random Forest gap fell from **0.367 to 0.0123** (roughly a 30-fold improvement) after `max_depth` was constrained. Train accuracy dropped from 100% to 68% — and this is a **healthy** indicator, reflecting the model ceasing to "memorise" and beginning genuinely to generalise.

**The final adopted model:** Decision Tree (max_depth=4, min_samples_leaf=5) — the best F1 with an acceptably small gap, and with the additional advantage of being directly interpretable visually (drawing the tree).

## The most important lessons learned

1. **A raw data row ≠ the required unit of analysis.** The first real challenge in the project was converting "row = invoice line" into "row = one customer" via `groupby` — a challenge that had not appeared in earlier projects whose data came ready as row = sample.
2. **A time-based split is necessary when the goal is a future prediction from past behaviour**, unlike the general rule of random splitting used in traditional classification/regression problems.
3. **A pipeline is not always the first step.** In projects that require restructuring the data (aggregating transactions into customers, building RFM) before it becomes "ML-ready", those steps must be built manually in the correct order; the pipeline enters only at the final stage, after the data structuring is complete.
4. **Severe skewness ruins distance-based clustering** — a log transformation before scaling (and not the reverse) is necessary when there are extreme values, as with financial quantities.
5. **The strongest model is not always the best.** With few features and a nearly linear relationship to the problem, Random Forest (very flexible) at default settings was the worst because of severe overfitting (gap = 0.367), while simple Logistic Regression was the most stable.
6. **Hyperparameter tuning is a control on the bias–variance balance, not merely a way to improve one number.** The tuning here did not raise performance much so much as it "rescued" Random Forest from total collapse in generalisation (reducing the gap 30-fold).
7. **Feature engineering does not guarantee an improvement.** Adding AvgOrderValue and IsUK seemed logical in theory, but the actual improvement was almost nil — an important lesson that every new feature must be tested experimentally, not assumed successful on theoretical grounds alone.
8. **Handling categorical columns depends on their actual distribution, not on a fixed rule.** Full one-hot would have suited a relatively balanced distribution, but here (90% dominance by one category) it required simplification to a binary instead.

## Tools and libraries used

`pandas`, `numpy`, `scikit-learn` (KMeans, StandardScaler, train_test_split, Pipeline, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, GridSearchCV, classification_report), `matplotlib`

---

# Section B — Customer value on Online Retail II

> Sourced from `notebook.ipynb` and `models/clv_retail_meta.json`, not from the project journal.

The later iteration widens both the data and the question. Instead of asking only *will this customer come back*, it asks *will they come back and how much will they spend*, and multiplies the two into an expected value that a marketing budget can actually be spent against.

**Window.** 18 months of history for the features, the following 6 months for the target, cut at a single date (`CUT`) computed from the first invoice. Same principle as Section A, longer arms.

**Cleaning specific to this data.** Online Retail II carries structure the cleaned export did not: cancellations are marked by a leading `C` on the invoice number and appear as negative quantities, and ~243k rows have no `Customer ID` at all. Rows without a customer are dropped as out of scope; negative prices turn out to be accounting entries rather than sales; and customers whose returns exceed their gross purchases are removed, since a return ratio above 1 describes a bookkeeping artefact rather than a buyer.

**Features.** 18 built from the invoice history — counts, totals, averages, recency, tenure, average gap between invoices, purchase rate, two trailing 100-day windows (`amt_w1`/`amt_w2`, `cnt_w1`/`cnt_w2`) and the trends between them, plus the return columns. Later additions: product-variety features (`n_sku`, `sku_per_inv`, `repeat_sku`) and a "return shock" feature comparing invoice behaviour in the 100 days before and after a customer's first return.

**Two models, multiplied.**

- A `HistGradientBoostingClassifier` for *will they return*, tuned by `GridSearchCV` on `roc_auc`.
- A `HistGradientBoostingRegressor` for *how much*, fitted **only on the customers who did return** in the training set, then predicted across the whole test set.
- Expected value = probability × predicted amount. Training the regressor only on returners is the point: asking it to fit the zeros of customers who never came back would teach it to predict zero.

**Results.**

| Metric | Value |
|---|---|
| AUC (return classifier) | **0.809** |
| Customers modelled | 4911 |
| Revenue captured by targeting the top 5% | 33.6% |
| Revenue captured by targeting the top 10% | 48.2% |
| Revenue captured by targeting the top 20% | 63.0% |

The coverage numbers are the ones that matter commercially: contacting one customer in ten, chosen by the model, reaches nearly half the revenue of the next six months. The notebook plots this as a cumulative gain curve against the random-targeting diagonal.

**Permutation importance** on the tuned classifier was used to cut the feature set to 13, confirming `recency`, the trailing-window amounts and the gap/rate features as the load-bearing ones.

**Segmentation.** The RFM + K-Means work reappears at the end of the notebook, now on `recency`/`n_inv`/`total` from this feature table, with the same `log1p`-then-scale ordering that Section A established.

**What is saved.** `models/clv_retail.joblib` holds the fitted classifier, the fitted regressor, the feature list and the cut date together, so a prediction can be reproduced without re-running the notebook. `models/clv_retail_meta.json` holds the headline metrics above.
