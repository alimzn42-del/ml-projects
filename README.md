# ml-projects

A two-week self-study track combining control engineering and machine learning, kept as a
worked record rather than a product. Each project is a complete piece of work — data
inspection, a decision that turned out to be wrong, the measurement that caught it, and the
result — and the running journal in [`progress.md`](progress.md) is the day-by-day thread
that connects them.

The recurring theme, visible in every project below, is that **the aggregate metric hides
and misleads**. In project 02 four experiments plateaued at R² ≈ 0.59 because the
evaluation was broken, not the model. In project 03 an accuracy of 98.9% concealed a third
of the failures going undetected. Both were caught by dissecting the error rather than
reading its average.

## The projects

| # | Project | Topic | Dataset | Technique | Result |
|---|---|---|---|---|---|
| 01 | [Time-series fundamentals](01-timeseries-basics/) | Cleaning, gaps, resampling, window features | Melbourne daily min temperature (3650 rows) + own ESP32/DS18B20 log (1491 rows) | `asfreq` · `interpolate` · `rolling` · residuals · z-score | Measurement noise **±0.09 °C**, sample-rate drift **43 s / 1476 readings** |
| 02 | [NASA Turbofan RUL](02-nasa-turbofan/) | Regression — remaining useful life | C-MAPSS FD001, 100 engines, ~20600 rows · California Housing (20640 rows) | Target construction · group split · `LinearRegression` vs `RandomForest` | **R² 0.806 · MAE 14.10 · RMSE 17.65** on unseen engines — linear, near published deep models |
| 03 | [AI4I 2020 maintenance](03-ai4i-maintenance/) | Classification — machine failure at 3.4% positives | AI4I 2020, 10000 rows | Physics-derived features · threshold tuning · `HistGradientBoosting` | **recall 0.838 · precision 0.934 · F1 0.884**; **58/58** caught on sensor-visible failure modes |
| 04 | [Retail customer value](04-retail-clv/) | Clustering + two-stage prediction | Online Retail II, ~1M invoice lines | Time-based split · RFM · K-Means · classifier × regressor | **AUC 0.809**; top 10% of customers captures **48.2%** of next-period revenue |

## In progress — not yet published

| # | Project | Topic | Status |
|---|---|---|---|
| 05 | [NSL-KDD intrusion detection](05-nslkdd-intrusion/) | Supervised + unsupervised · PCA · anomaly detection without labels | Full report written (Arabic), notebook done — pending translation/polish |
| 06 | [Retailrocket](06-retailrocket/) | Implicit-feedback e-commerce events | Exploration underway |
| 07 | [Neural networks](07-neural-networks/) | First PyTorch — CNN on CIFAR-10, grid search | Working notebooks; Jena Climate queued |
| 08 | [ESP32 thermal log](08-esp32-thermal/) | Own two-sensor hardware data, round two | Log recorded, analysis started |

## Running it

```bash
git clone https://github.com/alimzn42-del/ml-projects.git
cd ml-projects

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter lab
```

Package versions in `requirements.txt` are the ones the notebooks were actually executed
with — pandas 3.0.5, scikit-learn 1.9.0, numpy 2.5.2.

## Getting the data

**No dataset is committed except `01-timeseries-basics/data/heat_log.csv`**, which is
original data recorded from my own sensor rig and exists nowhere else. Everything else is
public and downloadable, and each project has a `data/README.md` giving the source, a
direct link, and the commands to fetch and place the files:

- [`01-timeseries-basics/data/README.md`](01-timeseries-basics/data/README.md)
- [`02-nasa-turbofan/data/README.md`](02-nasa-turbofan/data/README.md)
- [`03-ai4i-maintenance/data/README.md`](03-ai4i-maintenance/data/README.md)
- [`04-retail-clv/data/README.md`](04-retail-clv/data/README.md)

Project 01 needs no download to run — one notebook reads its data straight from a URL and
the other reads the committed sensor log. Projects 02–04 need their files fetched first.

Notebook outputs are kept in the committed files, so every plot and result renders on
GitHub without running anything.

## Layout

```
01-timeseries-basics/     two notebooks · committed sensor data · logger script · figures
02-nasa-turbofan/         two notebooks — regression metrics warm-up, then the RUL project
03-ai4i-maintenance/      one notebook
04-retail-clv/            one notebook · the saved model
progress.md               the daily journal
```
