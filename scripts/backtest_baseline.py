"""Evaluate the baseline on the training split by hiding the last months of each history.

    python scripts/backtest_baseline.py --histories 20
"""

from __future__ import annotations

import argparse

from casestudy import load_split, load_variables
from casestudy.backtest import sample_histories, split_history
from casestudy.baseline import bootstrap_forecast
from casestudy.scoring import score


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--histories", type=int, default=20, help="number of training histories to use (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df, variables = load_split("train"), load_variables()
    if args.histories:
        df = sample_histories(df, args.histories, seed=args.seed)
    observed, future = split_history(df)
    forecast = bootstrap_forecast(observed, variables, seed=args.seed)
    result = score(forecast, future, observed, variables)
    print(f"mean scaled CRPS: {result['mean_scaled_crps']:.3f}   mean energy score: {result['mean_energy_score']:.2f}")
    print("\nScaled CRPS by horizon (months):")
    print(result["by_horizon"].T.round(2).to_string())
    print("\nScaled CRPS by variable (best 5 / worst 5):")
    bv = result["by_variable"]
    print(bv.head(5).round(2).to_string(), "\n...\n", bv.tail(5).round(2).to_string())


if __name__ == "__main__":
    main()
