# Data — 02 NASA Turbofan

Two datasets. Neither is committed.

## 1. California Housing — no download needed

`01-regression-metrics-california.ipynb` pulls it from scikit-learn directly:

```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
```

20640 rows × 8 features, target `MedHouseVal` in units of $100,000.
scikit-learn caches it under `~/scikit_learn_data/` on first call. Origin: the
StatLib repository, derived from the 1990 U.S. census.

## 2. NASA C-MAPSS Turbofan Engine Degradation — download required

`02-turbofan-rul.ipynb` expects the **FD001** files in this directory:

```
02-nasa-turbofan/data/
  train_FD001.txt      ~3.5 MB    100 engines run to failure
  test_FD001.txt       ~2.2 MB    100 engines, cut off before failure
  RUL_FD001.txt        429 B      the answers for test_FD001
```

- **Source:** NASA Prognostics Center of Excellence (PCoE) data repository.
- **Direct link:** <https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip>
- **Size:** ~12 MB zipped, all four subsets FD001–FD004.

```bash
cd 02-nasa-turbofan/data
curl -L -o cmapss.zip \
  "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
unzip cmapss.zip
# the archive nests a second zip; unpack that too, then move the FD001 files here
unzip -j "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip" \
  train_FD001.txt test_FD001.txt RUL_FD001.txt -d .
rm -rf cmapss.zip "6. Turbofan Engine Degradation Simulation Data Set"
```

**File format.** No header, whitespace-separated, irregular spacing — hence
`sep=r"\s+", header=None`. 26 columns: `unit`, `cycle`, 3 operating settings,
21 sensors. There is no RUL column; the notebook constructs it.

**Do not compute RUL from `test_FD001.txt` by subtraction.** That file is truncated
before failure on purpose, so its last cycle is not the failure moment. The answers
live in `RUL_FD001.txt` alone, ordered by unit number.

The dataset is public domain (U.S. Government work) and citable as:
A. Saxena and K. Goebel (2008), "Turbofan Engine Degradation Simulation Data Set",
NASA Ames Prognostics Data Repository.
