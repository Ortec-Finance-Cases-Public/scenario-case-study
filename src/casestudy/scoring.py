"""Scoring of probabilistic forecasts given as sample paths (DRAFT: final weighting to be confirmed).

* **CRPS** (continuous ranked probability score) per forecast cell, estimated from the samples:
  ``CRPS = E|X - y| - 0.5 E|X - X'|``. Lower is better; it rewards calibration and sharpness.
* **Scaled CRPS**: CRPS divided by a per-(history, variable) scale computed from the observed history,
  so variables with different units can be averaged.
* **Energy score**: multivariate generalisation of the CRPS on the joint vector of all scaled cells of
  a history, ``ES = E||X - y|| - 0.5 E||X - X'||``. It also rewards getting the dependence between
  variables and horizons right.

Scales: mean absolute change between consecutive observations (months, or quarters for quarterly
variables) in the observed history; for return variables, mean absolute deviation of the returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def crps_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS for each row of ``samples`` (cells x n_samples) against ``y`` (cells,)."""
    x = np.sort(np.asarray(samples, dtype=float), axis=1)
    y = np.asarray(y, dtype=float)
    m = x.shape[1]
    term1 = np.mean(np.abs(x - y[:, None]), axis=1)
    weights = 2 * np.arange(1, m + 1) - m - 1
    term2 = (x @ weights) / (m * m)  # = 0.5 * E|X - X'|
    return term1 - term2


def energy_score(samples: np.ndarray, y: np.ndarray) -> float:
    """Energy score of joint samples (n_samples x dim) against the observation (dim,)."""
    x = np.asarray(samples, dtype=float)
    term1 = np.mean(np.linalg.norm(x - y[None, :], axis=1))
    sq = np.sum(x * x, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0.0)
    term2 = 0.5 * np.mean(np.sqrt(d2))
    return float(term1 - term2)


def scales(history: pd.DataFrame, variables: pd.DataFrame) -> pd.DataFrame:
    """Per-(history_id, variable) scale from the observed (long-format) history."""
    df = history.sort_values(["history_id", "variable", "period"])
    returns = variables.index[variables["value_type"].str.contains("return")]
    is_ret = df["variable"].isin(returns)
    g = df.groupby(["history_id", "variable"])["value"]
    level_scale = g.diff().abs().groupby([df["history_id"], df["variable"]]).mean()
    ret_scale = g.transform(lambda s: (s - s.mean()).abs()).groupby([df["history_id"], df["variable"]]).mean()
    out = level_scale.where(~level_scale.index.get_level_values("variable").isin(returns), ret_scale)
    return out.rename("scale").reset_index()


def _sample_matrix(submission: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    sub = submission.sort_values(["history_id", "variable", "period", "sample"])
    keys = sub[["history_id", "variable", "period"]].drop_duplicates().reset_index(drop=True)
    n = len(sub) // len(keys)
    return keys, sub["value"].to_numpy(dtype=float).reshape(len(keys), n)


def score(submission: pd.DataFrame, truth: pd.DataFrame, history: pd.DataFrame,
          variables: pd.DataFrame, joint: bool = True) -> dict[str, pd.DataFrame | float]:
    """Score a submission against the realised future values.

    Returns a dict with ``cells`` (CRPS per cell), ``by_variable``, ``by_horizon``, ``by_history``
    and the headline numbers ``mean_scaled_crps`` and (if ``joint``) ``mean_energy_score``.
    """
    sub = submission.copy()
    truth = truth.copy()
    hist_last = history.groupby("history_id")["period"].max().astype(str)
    for d in (sub, truth):
        d["period"] = d["period"].astype(str)
    keys, x = _sample_matrix(sub)
    cells = keys.merge(truth, on=["history_id", "variable", "period"], how="left")
    if cells["value"].isna().any():
        raise ValueError("truth is missing for some submitted cells")
    cells = cells.merge(scales(history, variables), on=["history_id", "variable"], how="left")
    cells["scale"] = cells["scale"].where(cells["scale"] > 0)
    y = cells["value"].to_numpy(dtype=float)
    cells["crps"] = crps_samples(x, y)
    cells["scaled_crps"] = cells["crps"] / cells["scale"]
    p = pd.PeriodIndex(cells["period"], freq="M")
    last = pd.PeriodIndex(cells["history_id"].map(hist_last), freq="M")
    cells["horizon"] = (p.year - last.year) * 12 + (p.month - last.month)

    out = {
        "cells": cells.drop(columns="value"),
        "by_variable": cells.groupby("variable")["scaled_crps"].mean().sort_values().to_frame(),
        "by_horizon": cells.groupby("horizon")["scaled_crps"].mean().to_frame(),
        "by_history": cells.groupby("history_id")["scaled_crps"].mean().to_frame(),
        "mean_scaled_crps": float(cells["scaled_crps"].mean()),
    }
    if joint:
        es = {}
        scale = cells["scale"].to_numpy()
        ok = np.isfinite(scale)
        z, yz = x / np.where(ok, scale, 1.0)[:, None], y / np.where(ok, scale, 1.0)
        for h, idx in cells.groupby("history_id").indices.items():
            idx = idx[ok[idx]]  # cells with a usable scale only
            es[h] = energy_score(z[idx].T, yz[idx])
        out["by_history"]["energy_score"] = pd.Series(es)
        out["mean_energy_score"] = float(np.mean(list(es.values())))
    return out
