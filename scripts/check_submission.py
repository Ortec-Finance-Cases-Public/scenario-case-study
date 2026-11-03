"""Check that a submission file has the required format before handing it in.

    python scripts/check_submission.py outputs/my_submission.parquet --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from casestudy import history_ids, load_split, load_variables
from casestudy.submission import HORIZON, validate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("submission", type=Path)
    p.add_argument("--split", default="test", choices=["validation", "test"])
    args = p.parse_args()

    df, variables = load_split(args.split), load_variables()
    sub = pd.read_parquet(args.submission)
    n = validate(sub, history_ids(df), df["period"].max(), variables, HORIZON)
    print(f"OK: {sub.history_id.nunique()} histories, {n} samples per cell, {len(sub):,} rows")


if __name__ == "__main__":
    main()
