# Data — 03 AI4I 2020 Predictive Maintenance

Not committed. One file, ~522 KB.

`notebook.ipynb` expects:

```
03-ai4i-maintenance/data/ai4i2020.csv
```

- **Source:** UCI Machine Learning Repository, dataset 601 — *AI4I 2020 Predictive
  Maintenance Dataset*, donated by Stephan Matzka (HTW Berlin).
- **Page:** <https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>
- **Direct link:** <https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip>
- **Licence:** CC BY 4.0.

```bash
cd 03-ai4i-maintenance/data
curl -L -o ai4i.zip \
  "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
unzip ai4i.zip && rm ai4i.zip
```

## Contents

10000 rows, one row per product, rows independent of each other — there is no time
axis and no grouping key, which is why none of the rolling-window tooling from
project 01 applies here.

| Column | Note |
|---|---|
| `UDI`, `Product ID` | arbitrary identifiers — dropped |
| `Type` | L / M / H product quality — the one categorical column |
| `Air temperature [K]`, `Process temperature [K]` | Kelvin, not Celsius. 310 K = 37 °C |
| `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]` | process measurements |
| `Machine failure` | the target — positive in 3.4% of rows |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | the five failure modes |

**The five failure-mode columns must not be fed to the model.** `Machine failure` is
computed from them, so handing them over is handing over the answer — 100% accuracy
without reading a single sensor. They are used in the notebook for diagnosis only,
which is legitimate: understanding the data may use anything, feeding the model is
restricted to what exists at inference time.
