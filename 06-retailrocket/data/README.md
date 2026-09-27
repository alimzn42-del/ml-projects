# Data — 06 Retailrocket

`Retailrocket.zip` is kept here (not committed). `notebook.ipynb` expects:

```
06-retailrocket/data/events.csv
```

```powershell
cd 06-retailrocket/data
Expand-Archive Retailrocket.zip -DestinationPath .
```

- **Source:** Retailrocket recommender system dataset, Kaggle.
- **Page:** <https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset>
- **Contents:** `events.csv` (~2.76M rows: timestamp · visitorid · event · itemid ·
  transactionid) · `item_properties_part1/2.csv` (~900 MB, hashed item attributes) ·
  `category_tree.csv`.
- Only `events.csv` is read so far; the item-properties files are large and hashed —
  extract them only if needed.
