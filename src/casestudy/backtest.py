"""Local evaluation on the training split: hide the last months of each history and forecast them.

The validation and test splits come without their future values. To compare approaches yourself,
cut the training histories: the first part is the input, the last ``horizon`` months are the target.
"""

from __future__ import annotations

import pandas as pd

from .submission import HORIZON


def split_history(df: pd.DataFrame, horizon: int = HORIZON) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a long table into (observed, future) at ``horizon`` months before the end of each history."""
    last = df.groupby("history_id")["period"].transform("max")
    cut = last - horizon
    is_past = df["period"] <= cut
    return df[is_past].reset_index(drop=True), df[~is_past].reset_index(drop=True)


def sample_histories(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Random subset of ``n`` histories (for quick experiments)."""
    ids = pd.Series(df["history_id"].unique()).sample(n=n, random_state=seed)
    return df[df["history_id"].isin(ids)].reset_index(drop=True)
