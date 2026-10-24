"""Top-K recommender metrics; IDCG denominator includes unretrieved positives."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import is_positive_listen


def ranking_metrics(
    pairs: pd.DataFrame,
    scores: np.ndarray,
    *,
    positive_counts: pd.Series,
    k: int = 10,
) -> dict[str, float]:
    """Macro-average over users having >=1 future positive; no future-gifted candidates."""
    if len(scores) != len(pairs):
        raise ValueError("score count mismatch")
    ranked = pairs[["uid", "item_id", "label"]].copy()
    ranked["score"] = scores
    ranked = ranked.sort_values(
        ["uid", "score", "item_id"], ascending=[True, False, True], kind="mergesort"
    )
    hits = ranked.groupby("uid", sort=False).head(k)
    hit_count = hits.groupby("uid")["label"].sum()
    hit_ranks = hits.assign(rank=hits.groupby("uid").cumcount() + 1)
    hit_ranks["gain"] = hit_ranks["label"] / np.log2(hit_ranks["rank"] + 1)
    dcg = hit_ranks.groupby("uid")["gain"].sum()

    ndcgs, recalls = [], []
    discounts = 1.0 / np.log2(np.arange(k) + 2)
    for uid, total in positive_counts.items():
        if total <= 0:
            continue
        ideal = discounts[:min(k, int(total))].sum()
        ndcgs.append(float(dcg.get(uid, 0)) / ideal)
        recalls.append(float(hit_count.get(uid, 0)) / float(total))
    if not ndcgs:
        raise ValueError("No users with future positive interactions")
    return {
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"recall@{k}": float(np.mean(recalls)),
        "evaluated_users": len(ndcgs),
    }


def future_positive_counts(events: pd.DataFrame, cutoff: int, bins_per_day: int) -> pd.Series:
    positives = events.loc[
        (events["timestamp"] >= cutoff) & (events["timestamp"] < cutoff + bins_per_day)
        & is_positive_listen(events["played_ratio_pct"]), ["uid", "item_id"]
    ].drop_duplicates()
    return positives.groupby("uid").size()
