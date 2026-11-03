"""Create a reference (naive bootstrap) submission for a split.

    python scripts/make_baseline_submission.py --split validation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from casestudy import history_ids, load_split, load_variables
from casestudy.baseline import bootstrap_forecast
from casestudy.submission import HORIZON, N_SAMPLES, validate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--samples", type=int, default=N_SAMPLES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    df, variables = load_split(args.split), load_variables()
    sub = bootstrap_forecast(df, variables, n_samples=args.samples, seed=args.seed)
    validate(sub, history_ids(df), df["period"].max(), variables, HORIZON)
    out = args.out or Path("outputs") / f"baseline_{args.split}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(out, index=False)
    print(f"{len(sub):,} rows ({sub.history_id.nunique()} histories x {args.samples} samples) -> {out}")


if __name__ == "__main__":
    main()
