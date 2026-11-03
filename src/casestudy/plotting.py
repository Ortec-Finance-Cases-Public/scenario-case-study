"""Plotting helpers for a single history and for forecasts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_history(wide: pd.DataFrame, variables: list[str], ncols: int = 3, title: str | None = None):
    """Small multiples of selected variables of one history (``history_wide`` output)."""
    nrows = int(np.ceil(len(variables) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 2.6 * nrows), squeeze=False)
    for ax, var in zip(axes.ravel(), variables):
        s = wide[var].dropna()
        ax.plot(s.index.to_timestamp(), s.to_numpy(), lw=0.9)
        ax.set_title(var, fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(variables):]:
        ax.set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_forecast(history_wide: pd.DataFrame, forecast: pd.DataFrame, variable: str, truth: pd.DataFrame | None = None,
                  months_shown: int = 72, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95), ax=None):
    """Fan chart of a forecast (submission rows of one history) for one variable."""
    ax = ax or plt.gca()
    h = history_wide[variable].dropna().iloc[-months_shown:]
    ax.plot(h.index.to_timestamp(), h.to_numpy(), color="black", lw=1)
    f = forecast[forecast["variable"] == variable]
    q = f.groupby("period")["value"].quantile(list(quantiles)).unstack()
    x = pd.PeriodIndex(q.index.astype(str), freq="M").to_timestamp()
    ax.fill_between(x, q[quantiles[0]], q[quantiles[-1]], alpha=0.2, color="tab:blue")
    ax.fill_between(x, q[quantiles[1]], q[quantiles[-2]], alpha=0.4, color="tab:blue")
    ax.plot(x, q[0.5], color="tab:blue", lw=1)
    if truth is not None:
        t = truth[truth["variable"] == variable].sort_values("period")
        ax.plot(pd.PeriodIndex(t["period"].astype(str), freq="M").to_timestamp(), t["value"], color="tab:red", lw=1)
    ax.set_title(variable, fontsize=9)
    ax.grid(alpha=0.3)
    return ax
