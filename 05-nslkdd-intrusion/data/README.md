# Data — 05 NSL-KDD

`NSL-KDD.zip` is kept here (not committed — see `.gitignore`). `notebook.ipynb` expects
the files extracted flat into this folder:

```
05-nslkdd-intrusion/data/KDDTrain+.txt
05-nslkdd-intrusion/data/KDDTest+.txt
05-nslkdd-intrusion/data/KDDTest-21.txt
```

```powershell
cd 05-nslkdd-intrusion/data
Expand-Archive NSL-KDD.zip -DestinationPath .
```

- **Source:** NSL-KDD — the cleaned revision of KDD Cup 1999, University of New Brunswick / CIC.
- **Page:** <https://www.unb.ca/cic/datasets/nsl.html>
- **Size:** `KDDTrain+` 125,973 rows · `KDDTest+` 22,544 rows · 43 columns
  (41 features + label + difficulty). No header row in the `.txt` files.
- The `.arff` copies inside the zip are the same data for Weka; the notebook uses the `.txt` files only.
