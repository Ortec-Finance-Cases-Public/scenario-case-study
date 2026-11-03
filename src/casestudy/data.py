"""Loading the scenario data.

Each split is a long table with one row per observation:

    history_id   period    variable     value
    h01e346a4    2101-01   US_IP_NSA    103.7

* ``history_id``: one synthetic history (independent of all other histories).
* ``period``: month, ``YYYY-MM``. Dates are fictional.
* Monthly variables have a value every month; quarterly variables only at quarter-end months
  (March, June, September, December).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CASESTUDY_DATA_DIR", REPO_ROOT / "data"))
SPLITS = ("train", "validation", "test")


def _data_dir(data_dir: str | Path | None) -> Path:
    path = Path(data_dir) if data_dir is not None else DATA_DIR
    if not path.exists():
        raise FileNotFoundError(f"Data folder {path} not found; see data/README.md")
    return path


def load_split(split: str, data_dir: str | Path | None = None) -> pd.DataFrame:
    """Long table of one split (``train``, ``validation`` or ``test``)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    path = _data_dir(data_dir) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; see data/README.md")
    df = pd.read_parquet(path)
    df["period"] = pd.PeriodIndex(df["period"].astype(str), freq="M")
    df["history_id"] = df["history_id"].astype(str)
    df["variable"] = df["variable"].astype(str)
    return df


def load_variables(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Variable descriptions, indexed by variable name."""
    return pd.read_csv(_data_dir(data_dir) / "variables.csv").set_index("variable")


def monthly_variables(variables: pd.DataFrame) -> list[str]:
    return variables.index[variables["frequency"] == "M"].tolist()


def quarterly_variables(variables: pd.DataFrame) -> list[str]:
    return variables.index[variables["frequency"] == "Q"].tolist()


def history_ids(df: pd.DataFrame) -> list[str]:
    return sorted(df["history_id"].unique())


def history_wide(df: pd.DataFrame, history_id: str, variables: list[str] | None = None) -> pd.DataFrame:
    """One history as a (months x variables) table; quarterly variables are NaN outside quarter-end months."""
    h = df[df["history_id"] == history_id]
    if h.empty:
        raise KeyError(f"history {history_id!r} not found")
    wide = h.pivot(index="period", columns="variable", values="value").sort_index()
    wide = wide.reindex(pd.period_range(wide.index.min(), wide.index.max(), freq="M"))
    wide.index.name, wide.columns.name = "period", None
    return wide[variables] if variables is not None else wide


def to_array(df: pd.DataFrame, variables: list[str] | None = None) -> tuple[np.ndarray, list[str], pd.PeriodIndex, list[str]]:
    """All histories as an array (histories x months x variables).

    Returns ``(array, history_ids, periods, variables)``. Missing values (quarterly variables outside
    quarter-end months) are NaN. Assumes all histories cover the same months.
    """
    ids = history_ids(df)
    variables = variables or sorted(df["variable"].unique())
    periods = pd.period_range(df["period"].min(), df["period"].max(), freq="M")
    wide = df.pivot_table(index=["history_id", "period"], columns="variable", values="value")
    full = pd.MultiIndex.from_product([ids, periods], names=["history_id", "period"])
    wide = wide.reindex(index=full, columns=variables)
    return wide.to_numpy().reshape(len(ids), len(periods), len(variables)), ids, periods, variables
