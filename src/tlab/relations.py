"""Optional time-causal indirect user-item features using public Yambda relations.

Artist/album mapping is static metadata. Listening aggregates strictly precede
the per-snapshot cutoff. Tracks can have multiple artists; we sum affinities
across all distinct mapped entities (we do not arbitrarily select one artist).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from .data import BINS_PER_DAY

RELATION_NAMES = ("artist", "album")
MANUAL_RELATION_DAYS = (7, 30)
AUTO_RELATION_DAYS = (1, 3, 14, 60)
MANUAL_RELATION_COLUMNS = [
    f"user_{relation}_count_{days}d"
    for relation in RELATION_NAMES for days in MANUAL_RELATION_DAYS
]
AUTO_RELATION_COLUMNS = [
    f"user_{relation}_count_{days}d"
    for relation in RELATION_NAMES for days in AUTO_RELATION_DAYS
]


def load_mappings(data_directory: str | Path) -> dict[str, pd.DataFrame]:
    """Load the two standalone static parquet lookup tables, without user events."""
    root = Path(data_directory)
    result: dict[str, pd.DataFrame] = {}
    for name in RELATION_NAMES:
        path = root / f"{name}_item_mapping.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing metadata file: {path}")
        relation_id = f"{name}_id"
        frame = pd.read_parquet(path, columns=["item_id", relation_id])
        if frame[["item_id", relation_id]].isna().any().any():
            raise ValueError(f"Null IDs in metadata: {path}")
        result[name] = frame.drop_duplicates(["item_id", relation_id]).reset_index(drop=True)
    return result


def add_relation_features(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    cutoff: int,
    mappings: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Returns a new frame, preserving row order, labels and candidate pairs.

    A mapping with multiple artists for one track increases the count by the
    number of historic listens to each artist. Only listening history before
    the cutoff contributes. No target events can alter these features.
    """
    out = frame.copy()
    cand = out[["uid", "item_id"]].reset_index(drop=True).copy()
    cand["_row_id"] = cand.index.to_numpy()
    history = events.loc[
        events["timestamp"] < cutoff, ["uid", "item_id", "timestamp"]
    ]
    for name in RELATION_NAMES:
        mapping = mappings.get(name)
        if mapping is None:
            raise ValueError(f"Missing {name} mapping; supply both lookup tables")
        relation_id = f"{name}_id"
        required = {"item_id", relation_id}
        if not required.issubset(mapping.columns):
            raise ValueError(f"Mapping '{name}' requires columns {sorted(required)}")
        # For one track/artist pair, count its historical listens only once.
        mapping = mapping[["item_id", relation_id]].drop_duplicates()
        cand_linked = cand.merge(mapping, on="item_id", how="left", sort=False)
        historic_linked = history.merge(mapping, on="item_id", how="inner", sort=False)
        for days in sorted(set(MANUAL_RELATION_DAYS + AUTO_RELATION_DAYS)):
            col = f"user_{name}_count_{days}d"
            recent = historic_linked.loc[
                historic_linked["timestamp"] >= cutoff - days * BINS_PER_DAY
            ]
            counts = recent.groupby(["uid", relation_id]).size().rename(col)
            count_per_link = cand_linked.join(counts, on=["uid", relation_id])
            count_per_candidate = count_per_link.groupby("_row_id", sort=False)[col].sum()
            out[col] = cand["_row_id"].map(count_per_candidate).fillna(0).astype("float32").to_numpy()
    return out
