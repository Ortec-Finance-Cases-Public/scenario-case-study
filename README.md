# Synthetic macro-financial scenarios: case study starter

Starter code for the case study. It loads the scenario data, gives you tools to explore it, and shows
the full workflow from a forecast to a valid submission, using a deliberately naive baseline.

> **Draft.** The forecast horizon, the number of samples, the scoring and the deadlines are still
> subject to confirmation by the instructor. Items marked *DRAFT* in the code may change.

## The data

Synthetic monthly histories of macroeconomic and financial variables for the United States, the euro
area, Japan and global markets: activity, prices, labour markets, interest rates, equities, real
estate, FX, commodities, credit and volatility. 50 variables in total, 44 monthly and 6 quarterly.
Dates are fictional.

| Split | Histories | Months per history | Future months |
|---|---|---|---|
| `train` | 100 | 360 | – |
| `validation` | 80 | 360 | 24, held back |
| `test` | 120 | 360 | 24, held back |

Each split is a long table: `history_id`, `period` (`YYYY-MM`), `variable`, `value`.

- **Monthly variables** have a value every month.
- **Quarterly variables** only have values at quarter-end months (March, June, September, December).
- **Units:** values are in the units given in `variables.csv`. Most are levels; equity and REIT
  variables are monthly returns in %.
- **Histories are independent of each other:** each one is a separate synthetic 30-year history.

## The task (DRAFT)

For every history in `validation` and `test`, forecast the **next 24 months of all 50 variables** as
a probabilistic forecast: **100 joint sample paths** per history. Forecasts are scored with the
CRPS (per variable and month, scaled to be comparable across variables) and the energy score (all
variables and months jointly); lower is better. See [`scoring.py`](src/casestudy/scoring.py).

## Getting started

Requires Python 3.10 or later.

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then:

1. Copy the data files into [`data/`](data/README.md).
2. Check the installation: `python -m pytest` (the tests don't need the data).
3. Open the notebooks: `jupyter lab` and start with `notebooks/01_explore_data.ipynb`.

The environment contains only what the starter code needs (numpy, pandas, pyarrow, matplotlib,
JupyterLab, pytest). Add whatever your own approach requires to `pyproject.toml` or
`requirements.txt`.

## Repository layout

```
├── data/                      put the data files here (git-ignored)
├── notebooks/
│   ├── 01_explore_data.ipynb        variables, histories, plots, summary statistics
│   └── 02_forecast_and_score.ipynb  local backtest, scoring, writing a submission
├── scripts/
│   ├── backtest_baseline.py         score the baseline on held-out months of training histories
│   ├── make_baseline_submission.py  baseline submission for validation or test
│   └── check_submission.py          check a submission file before handing it in
├── src/casestudy/
│   ├── data.py          loading: load_split, load_variables, history_wide, to_array
│   ├── transforms.py    diff, log_diff, pct_change, year_on_year, cumulate_returns
│   ├── backtest.py      split_history: hide the last months of training histories
│   ├── submission.py    submission format, from_array, validate
│   ├── scoring.py       CRPS, energy score, score()
│   ├── baseline.py      naive reference forecast (joint bootstrap of past changes)
│   └── plotting.py      plot_history, plot_forecast
└── tests/
```

## Using the library

```python
from casestudy import load_split, load_variables, history_ids, history_wide, to_array

variables = load_variables()                 # one row per variable: frequency, units, region, ...
train = load_split("train")                  # long table
h = history_wide(train, history_ids(train)[0])   # one history: 360 months x 50 variables
arr, ids, periods, names = to_array(train)   # numpy array: (histories, months, variables)
```

Local evaluation on the training split:

```python
from casestudy.backtest import split_history
from casestudy.baseline import bootstrap_forecast
from casestudy.scoring import score

observed, future = split_history(train, horizon=24)       # hide the last 24 months
forecast = bootstrap_forecast(observed, variables)         # replace with your method
result = score(forecast, future, observed, variables)
result["mean_scaled_crps"], result["mean_energy_score"], result["by_variable"]
```

## Submission format (DRAFT)

One parquet file per split with columns `history_id, period, variable, sample, value`:

- **Cells:** every history of the split, every month of the 24-month horizon for monthly variables,
  and quarter-end months only for quarterly variables.
- **Samples:** `sample = 0 … 99`. Rows sharing `(history_id, sample)` form one joint path across all
  variables and months.
- **Values:** in the same units as the data, all finite.

`casestudy.submission.from_array` builds this table from an array of shape
(histories, samples, months, variables). Before handing in:

```bash
python scripts/check_submission.py outputs/my_test_submission.parquet --split test
```

## Reference results

The naive baseline (`python scripts/backtest_baseline.py --histories 20`) scores a mean scaled CRPS
of about **2.8** on 20 training histories with the last 24 months held out. It grows with the horizon,
from about 1.0 at one month to 4.2 at 24 months.

## Notes

- Seeds: the baseline and the helpers take a `seed` argument; set it to make results reproducible.
- A submission with 100 samples has about 9–13 million rows (40–60 MB as parquet).
