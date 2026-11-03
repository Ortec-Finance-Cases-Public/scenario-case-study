"""Common transformations of a wide (months x variables) history.

All functions accept a DataFrame from ``history_wide`` and return a DataFrame of the same shape.
Quarterly variables are handled at quarterly frequency (changes between consecutive quarter-ends).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _per_column(wide: pd.DataFrame, fn) -> pd.DataFrame:
    out = {}
    for col in wide:
        s = wide[col].dropna()
        out[col] = fn(s).reindex(wide.index)
    return pd.DataFrame(out, index=wide.index)


def diff(wide: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Change between consecutive observations (months, or quarters for quarterly variables)."""
    return _per_column(wide, lambda s: s.diff(periods))


def log_diff(wide: pd.DataFrame, periods: int = 1, scale: float = 100.0) -> pd.DataFrame:
    """``scale`` x change in log level; NaN where the level is not positive."""
    return _per_column(wide, lambda s: scale * np.log(s.where(s > 0)).diff(periods))


def pct_change(wide: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Percentage change between consecutive observations."""
    return _per_column(wide, lambda s: 100 * s.pct_change(periods, fill_method=None))


def year_on_year(wide: pd.DataFrame) -> pd.DataFrame:
    """Percentage change over 12 months (4 quarters for quarterly variables)."""
    return _per_column(wide, lambda s: 100 * (s / s.shift(12 if _monthly(s) else 4) - 1))


def cumulate_returns(returns: pd.DataFrame | pd.Series, start: float = 100.0):
    """Turn period returns in % into an index level starting at ``start``."""
    return start * (1 + returns / 100).cumprod()


def _monthly(s: pd.Series) -> bool:
    return len(s) > 1 and (s.index[1] - s.index[0]).n == 1
