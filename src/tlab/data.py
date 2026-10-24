"""Load complete interaction histories of sampled Yambda users.

Yambda timestamps are integer 5-second bins; never interpret as Unix seconds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BINS_PER_DAY = 24 * 60 * 60 // 5
POSITIVE_PLAYED_RATIO_PCT = 50


def is_positive_listen(played_ratio_pct: pd.Series) -> pd.Series:
    """Official Yambda Listen+ convention: strictly more than 50% listened."""
    return played_ratio_pct > POSITIVE_PLAYED_RATIO_PCT


def load_yambda(path: str | Path, users: int = 100, seed: int = 42) -> pd.DataFrame:
    """Deterministically sample user IDs, then fetch *all* their event rows."""
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Yambda parquet not found: {path}")

    # One compact column (~200 MB for Yambda-50M), not a full-data dataframe.
    uid_array = pq.read_table(path, columns=["uid"]).column("uid").to_numpy()
    unique_ids = np.unique(uid_array)
    rng = np.random.default_rng(seed)
    if users > 0 and users < len(unique_ids):
        chosen = np.sort(rng.choice(unique_ids, size=users, replace=False))
    else:
        chosen = unique_ids
    del uid_array, unique_ids

    cols = ["uid", "item_id", "timestamp", "played_ratio_pct"]
    table = ds.dataset(str(path), format="parquet").to_table(
        columns=cols, filter=ds.field("uid").isin(chosen.tolist())
    )
    out = table.to_pandas()
    out["played_ratio_pct"] = out["played_ratio_pct"].fillna(0).astype("float32")
    if out.empty:
        raise ValueError("No rows loaded for selected users.")
    return out.sort_values(["uid", "timestamp", "item_id"], kind="mergesort").reset_index(drop=True)


def synthetic_events(n_users: int = 20, days: int = 80, seed: int = 42) -> pd.DataFrame:
    """Deterministic non-research fixture for local smoke runs and unit tests."""
    rng = np.random.default_rng(seed)
    rows = []
    for uid in range(n_users):
        favorites = rng.choice(np.arange(50), size=6, replace=False)
        for day in range(days):
            for _ in range(8):
                item = int(rng.choice(favorites)) if rng.random() < 0.75 else int(rng.integers(50))
                t = day * BINS_PER_DAY + int(rng.integers(0, BINS_PER_DAY))
                ratio = int(rng.integers(51, 101)) if rng.random() < 0.82 else 5
                rows.append((uid, item, t, ratio))
    return pd.DataFrame(rows, columns=["uid", "item_id", "timestamp", "played_ratio_pct"]).sort_values(
        ["uid", "timestamp", "item_id"], kind="mergesort"
    ).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Download Yambda-50M listens.parquet to local data/.")
    p.add_argument("action", choices=["download"])
    p.add_argument("--output", type=Path, default=Path("data"))
    args = p.parse_args()
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="yandex/yambda",
        repo_type="dataset",
        filename="flat/50m/listens.parquet",
        local_dir=str(args.output),
    )
    print(f"Downloaded to {path}")


if __name__ == "__main__":
    main()
