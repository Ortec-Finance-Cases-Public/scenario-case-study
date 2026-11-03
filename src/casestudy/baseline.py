"""A deliberately simple reference forecast: joint bootstrap of past changes.

For every history and sample path:

* monthly variables: each future month re-uses the changes of one randomly drawn past month, the same
  month for all monthly variables (so contemporaneous co-movement is kept);
* quarterly variables: the same with randomly drawn past quarters;
* changes are differences for rates, spreads and other series measured in % or percentage points,
  log-changes for other always-positive series, and the returns themselves for return variables;
  paths start from the last observed value.

It ignores everything else about the data. Use it as a reference point and a template for producing a
submission in the right format.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .submission import HORIZON, N_SAMPLES, forecast_periods


def _kind(series: pd.Series, units: str, value_type: str) -> str:
    if "return" in value_type:
        return "return"
    if "%" in units or "pp" in units.split() or not (series > 0).all():
        return "diff"  # rates, spreads and other percentage-unit series
    return "log"


def _changes(s: pd.Series, kind: str) -> np.ndarray:
    if kind == "return":
        return s.to_numpy()
    if kind == "log":
        return np.diff(np.log(s.to_numpy()))
    return np.diff(s.to_numpy())


def _paths(last: float, steps: np.ndarray, kind: str) -> np.ndarray:
    if kind == "return":
        return steps
    if kind == "log":
        return last * np.exp(np.cumsum(steps, axis=-1))
    return last + np.cumsum(steps, axis=-1)


def bootstrap_forecast(history: pd.DataFrame, variables: pd.DataFrame, horizon: int = HORIZON,
                       n_samples: int = N_SAMPLES, lookback: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Submission-format forecast for every history in ``history`` (long format)."""
    rng = np.random.default_rng(seed)
    frames = []
    for hid, h in history.groupby("history_id", sort=True):
        wide = h.pivot(index="period", columns="variable", values="value").sort_index()
        if lookback:
            wide = wide.iloc[-lookback:]
        periods = forecast_periods(wide.index[-1], horizon)
        q_periods = periods[periods.month % 3 == 0]
        out = {}
        for freq, fut in (("M", periods), ("Q", q_periods)):
            cols = [v for v in wide if variables.loc[v, "frequency"] == freq]
            if not cols or len(fut) == 0:
                continue
            obs = wide[cols].dropna(how="all")
            kinds = {c: _kind(obs[c], str(variables.loc[c].get("units", "")), variables.loc[c, "value_type"])
                     for c in cols}
            ch = {c: _changes(obs[c], kinds[c]) for c in cols}
            n_hist = min(len(v) for v in ch.values())
            draw = rng.integers(0, n_hist, size=(n_samples, len(fut)))  # shared across variables
            for c in cols:
                steps = ch[c][-n_hist:][draw]
                out[c] = (fut, _paths(obs[c].iloc[-1], steps, kinds[c]))
        for c, (fut, vals) in out.items():
            frames.append(pd.DataFrame({
                "history_id": hid,
                "period": np.tile(fut.astype(str), n_samples),
                "variable": c,
                "sample": np.repeat(np.arange(n_samples), len(fut)),
                "value": vals.reshape(-1).astype(np.float32),
            }))
    return pd.concat(frames, ignore_index=True)
