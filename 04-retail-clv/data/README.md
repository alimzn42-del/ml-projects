# Data — 04 Retail Customer Value

Not committed. ~95 MB uncompressed — far too large for git, which keeps blobs in
history forever even after a later deletion.

`notebook.ipynb` expects:

```
04-retail-clv/data/online_retail_II.csv
```

- **Source:** UCI Machine Learning Repository, dataset 502 — *Online Retail II*,
  donated by Dr Daqing Chen, London South Bank University.
- **Page:** <https://archive.ics.uci.edu/dataset/502/online+retail+ii>
- **Direct link:** <https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip>
- **Licence:** CC BY 4.0.

```bash
cd 04-retail-clv/data
curl -L -o retail.zip \
  "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
unzip retail.zip && rm retail.zip
# ships as .xlsx with two sheets; the notebook reads a single CSV:
python -c "import pandas as pd; \
pd.concat(pd.read_excel('online_retail_II.xlsx', sheet_name=None), ignore_index=True) \
  .to_csv('online_retail_II.csv', index=False)"
```

## Contents

Transaction lines from a UK online giftware retailer, December 2009 – December 2011.
One row is **one invoice line, not one customer** — reshaping it to one row per
customer via `groupby` is the first real step of the project.

| Column | Note |
|---|---|
| `Invoice` | invoice number; a leading `C` marks a cancellation |
| `StockCode`, `Description` | product |
| `Quantity` | negative on cancellations |
| `InvoiceDate` | parsed to datetime — every split in this project is time-based |
| `Price` | unit price |
| `Customer ID` | **missing on ~243k rows**, which are dropped as out of scope |
| `Country` | ~90% United Kingdom |

## Two versions of this dataset

The project README documents an earlier iteration built on the *cleaned* single-year
export of the same retailer (Dec 2010 – Dec 2011, 531,282 rows), which is a different
file:

<https://raw.githubusercontent.com/eaintkyawthmu/UCI_Online_Retail_Dataset_Cleaned_Version/master/Cleaned_UCI_Online_Sale_Dataset.csv>

`notebook.ipynb` in this folder is the later work and uses the full two-year
**Online Retail II** above. Both are documented in the project README; only the
second is reproducible from the notebook.

## Trained model

`../models/clv_retail.joblib` is committed (612 KB) — it holds the fitted classifier,
the fitted regressor, the feature list and the cut-off date. It is the one artifact
here that cannot be regenerated in a single line, so it is kept rather than rebuilt.
