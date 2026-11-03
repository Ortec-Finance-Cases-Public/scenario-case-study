"""Submission format.

A submission is a set of simulated future paths for every history of a split:

    history_id   period    variable    sample   value
    h01e346a4    2131-01   US_IP_NSA   0        104.2

* ``period``: every month of the forecast horizon for monthly variables, quarter-end months only for
  quarterly variables (the same pattern as in the data).
* ``sample``: 0 .. n_samples - 1. Rows with the same ``(history_id, sample)`` form one joint scenario
  across all variables and months, so samples should be *paths*, not independent draws per cell.
* ``value``: in the same units as the data (levels, or period returns in % for return variables).

Save as parquet: ``submission.to_parquet("outputs/<name>.parquet", index=False)``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 24  # months  (DRAFT: to be confirmed by the instructor)
N_SAMPLES = 100  # (DRAFT: to be confirmed by the instructor)
COLUMNS = ["history_id", "period", "variable", "sample", "value"]


def forecast_periods(last_period: pd.Period, horizon: int = HORIZON) -> pd.PeriodIndex:
    return pd.period_range(last_period + 1, periods=horizon, freq="M")


def expected_cells(history_ids: list[str], last_period: pd.Period, variables: pd.DataFrame,
                   horizon: int = HORIZON) -> pd.DataFrame:
    """All (history_id, period, variable) cells a submission must cover."""
    periods = forecast_periods(last_period, horizon)
    rows = []
    for var, freq in variables["frequency"].items():
        p = periods if freq == "M" else periods[periods.month % 3 == 0]
        rows.append(pd.DataFrame({"period": p, "variable": var}))
    cells = pd.concat(rows, ignore_index=True)
    return pd.concat([cells.assign(history_id=h) for h in history_ids], ignore_index=True)[
        ["history_id", "period", "variable"]]


def from_array(samples: np.ndarray, history_ids: list[str], periods: pd.PeriodIndex, variables: list[str],
               frequencies: dict[str, str]) -> pd.DataFrame:
    """Build a submission from an array (histories x samples x months x variables).

    Values of quarterly variables outside quarter-end months are dropped.
    """
    h, s, t, v = samples.shape
    if (h, t, v) != (len(history_ids), len(periods), len(variables)):
        raise ValueError("array shape does not match history_ids / periods / variables")
    df = pd.DataFrame({
        "history_id": np.repeat(np.asarray(history_ids), s * t * v),
        "sample": np.tile(np.repeat(np.arange(s), t * v), h),
        "period": np.tile(np.repeat(np.asarray(periods), v), h * s),
        "variable": np.tile(np.asarray(variables), h * s * t),
        "value": samples.reshape(-1).astype(np.float32),
    })
    quarterly = df["variable"].map(frequencies).eq("Q").to_numpy()
    month = pd.PeriodIndex(df["period"], freq="M").month
    df = df[~quarterly | (month % 3 == 0)]
    return df[COLUMNS].reset_index(drop=True)


def validate(submission: pd.DataFrame, history_ids: list[str], last_period: pd.Period, variables: pd.DataFrame,
             horizon: int = HORIZON) -> int:
    """Check the format; raises ValueError with a description of the first problem. Returns n_samples."""
    missing = set(COLUMNS) - set(submission.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    keys = ["history_id", "period", "variable"]
    sub = submission[keys + ["sample", "value"]].copy()
    for k in keys:
        sub[k] = sub[k].astype(str)
    if not np.isfinite(sub["value"].to_numpy(dtype=float)).all():
        raise ValueError("values must be finite (no NaN or inf)")
    if sub.duplicated(keys + ["sample"]).any():
        raise ValueError("duplicate (history_id, period, variable, sample) rows")
    n = sub.groupby(keys, sort=False).size()
    if n.nunique() != 1:
        raise ValueError("every cell must have the same number of samples")
    cells = expected_cells(history_ids, last_period, variables, horizon)
    cells["period"] = cells["period"].astype(str)
    got = n.index.to_frame(index=False)
    merged = cells.merge(got, how="outer", indicator=True)
    if (merged["_merge"] != "both").any():
        bad = merged[merged["_merge"] != "both"].head(5)
        raise ValueError(f"cells missing or unexpected, e.g.\n{bad}")
    return int(n.iloc[0])
