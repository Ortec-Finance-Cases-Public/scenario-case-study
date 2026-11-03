# Data

Copy the files you received into this folder:

```
data/
├── train.parquet
├── validation.parquet
├── test.parquet
├── variables.csv
└── DATASET.md       (the dataset's own README, renamed so it doesn't replace this file; optional)
```

The folder is git-ignored: do not commit the data.

To use another location, set the environment variable `CASESTUDY_DATA_DIR` or pass `data_dir=...`
to the loading functions.
