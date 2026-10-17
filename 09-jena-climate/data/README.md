# Data — 09 Jena Climate

Not committed. `Jena Climate.ipynb` expects:

```
09-jena-climate/data/jena_climate_2009_2016.csv
```

- **Source:** Max Planck Institute for Biogeochemistry weather station, Jena, Germany —
  10-minute readings, 2009–2016. Distributed with the Keras tutorials.
- **Direct link:** <https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip>
- The notebook's first cell downloads and extracts it automatically if the CSV is missing
  (`jena_climate.zip` is the archive it fetched).

Beware: the file has **hidden gaps** (missing stretches of time, and -9999 sentinel values in the
wind columns). `jena_pipeline.py` cuts the data into segments at every gap > 30 minutes and never
fills across a gap — any new data must go through the same pipeline.
