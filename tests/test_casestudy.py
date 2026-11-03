import numpy as np
import pandas as pd
import pytest

from casestudy.backtest import split_history
from casestudy.baseline import bootstrap_forecast
from casestudy.data import history_wide, to_array
from casestudy.scoring import crps_samples, energy_score, score
from casestudy.submission import expected_cells, from_array, validate
from casestudy.transforms import cumulate_returns, log_diff, year_on_year

VARIABLES = pd.DataFrame(
    {"frequency": ["M", "M", "Q"], "value_type": ["level", "period return, %", "level"]},
    index=pd.Index(["LEVEL", "RET", "QLEV"], name="variable"),
)


def toy_data(n_hist=3, months=48, seed=0):
    rng = np.random.default_rng(seed)
    periods = pd.period_range("2101-01", periods=months, freq="M")
    rows = []
    for h in range(n_hist):
        hid = f"h{h:03d}"
        level = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, months)))
        ret = rng.normal(0.5, 4, months)
        q = 50 + np.cumsum(rng.normal(0, 1, months))
        for i, p in enumerate(periods):
            rows += [(hid, p, "LEVEL", level[i]), (hid, p, "RET", ret[i])]
            if p.month % 3 == 0:
                rows.append((hid, p, "QLEV", q[i]))
    return pd.DataFrame(rows, columns=["history_id", "period", "variable", "value"])


def test_history_wide_and_array():
    df = toy_data()
    w = history_wide(df, "h001")
    assert w.shape == (48, 3) and w["QLEV"].notna().sum() == 16
    arr, ids, periods, variables = to_array(df, ["LEVEL", "QLEV"])
    assert arr.shape == (3, 48, 2) and ids == ["h000", "h001", "h002"]
    assert np.isnan(arr[0, 0, 1]) and not np.isnan(arr[0, 2, 1])


def test_transforms():
    w = history_wide(toy_data(), "h000")
    ld = log_diff(w)
    assert ld["QLEV"].notna().sum() == 15  # quarterly changes only at quarter ends
    assert np.isclose(ld["LEVEL"].iloc[1], 100 * np.log(w["LEVEL"].iloc[1] / w["LEVEL"].iloc[0]))
    assert year_on_year(w)["LEVEL"].notna().sum() == 36
    assert cumulate_returns(pd.Series([10.0, -10.0])).iloc[-1] == pytest.approx(99.0)


def test_crps_matches_definition_and_point_forecast():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 200))
    y = rng.normal(size=5)
    brute = np.mean(np.abs(x - y[:, None]), axis=1) - 0.5 * np.mean(
        np.abs(x[:, :, None] - x[:, None, :]), axis=(1, 2))
    assert np.allclose(crps_samples(x, y), brute)
    assert crps_samples(np.full((1, 10), 2.0), np.array([5.0]))[0] == pytest.approx(3.0)


def test_energy_score_reduces_to_crps_in_one_dimension():
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    assert energy_score(x[:, None], np.array([0.3])) == pytest.approx(crps_samples(x[None, :], np.array([0.3]))[0])


def test_split_baseline_validate_and_score():
    df = toy_data()
    observed, future = split_history(df, horizon=12)
    assert observed.period.max() == pd.Period("2103-12", "M") and future.period.min() == pd.Period("2104-01", "M")
    sub = bootstrap_forecast(observed, VARIABLES, horizon=12, n_samples=20, seed=0)
    n = validate(sub, sorted(df.history_id.unique()), observed.period.max(), VARIABLES, horizon=12)
    assert n == 20
    res = score(sub, future, observed, VARIABLES)
    assert set(res["by_variable"].index) == {"LEVEL", "RET", "QLEV"}
    assert np.isfinite(res["mean_scaled_crps"]) and np.isfinite(res["mean_energy_score"])
    assert res["by_horizon"].index.min() == 1 and res["by_horizon"].index.max() == 12


def test_from_array_drops_quarterly_off_months_and_validate_catches_errors():
    periods = pd.period_range("2131-01", periods=6, freq="M")
    arr = np.zeros((2, 4, 6, 3))
    sub = from_array(arr, ["a", "b"], periods, ["LEVEL", "RET", "QLEV"], VARIABLES["frequency"].to_dict())
    assert len(sub) == 2 * 4 * (6 + 6 + 2)
    assert validate(sub, ["a", "b"], pd.Period("2130-12", "M"), VARIABLES, horizon=6) == 4
    with pytest.raises(ValueError):
        validate(sub.iloc[1:], ["a", "b"], pd.Period("2130-12", "M"), VARIABLES, horizon=6)
    bad = sub.copy()
    bad.loc[0, "value"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate(bad, ["a", "b"], pd.Period("2130-12", "M"), VARIABLES, horizon=6)
    cells = expected_cells(["a"], pd.Period("2130-12", "M"), VARIABLES, horizon=6)
    assert len(cells) == 6 + 6 + 2
