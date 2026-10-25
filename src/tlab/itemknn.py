"""Compact causal ItemKNN collaborative baseline for fixed candidate ranking.

User-item co-occurrences are estimated from a bounded window of past listens.
A per-user cap keeps memory predictable on Yambda-50M cohort experiments.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt

import numpy as np
import pandas as pd

from .data import BINS_PER_DAY


def itemknn_scores(
    events: pd.DataFrame,
    cutoff: int,
    candidates: pd.DataFrame,
    *,
    lookback_days: int = 30,
    max_items_per_user: int = 25,
    recent_seeds: int = 5,
    max_neighbors: int = 80,
) -> np.ndarray:
    """One past-only score per user/item candidate; no future labels or impressions.

    Build a binary user-item ItemKNN cosine graph over each user's 25 most
    recent distinct items from the last 30 days. Score candidates using
    neighboring tracks of the five latest distinct listening seeds.
    """
    if lookback_days < 1 or max_items_per_user < 2 or recent_seeds < 1:
        raise ValueError("Invalid ItemKNN configuration")
    if candidates.empty:
        return np.empty(0, dtype=np.float32)

    history = events.loc[
        (events["timestamp"] < cutoff)
        & (events["timestamp"] >= cutoff - lookback_days * BINS_PER_DAY),
        ["uid", "item_id", "timestamp"],
    ]
    history = (
        history.sort_values(
            ["uid", "timestamp", "item_id"],
            ascending=[True, False, True], kind="mergesort",
        )
        .drop_duplicates(["uid", "item_id"])
        .groupby("uid", sort=False).head(max_items_per_user)
    )
    item_lists = history.groupby("uid", sort=False)["item_id"].apply(list).to_dict()

    item_user_counts: Counter[int] = Counter()
    pair_user_counts: Counter[tuple[int, int]] = Counter()
    for items in item_lists.values():
        items = [int(x) for x in items]
        item_user_counts.update(items)
        for i in items:
            pair_user_counts.update((i, j) for j in items if j != i)

    neighbors: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (i, j), common_users in pair_user_counts.items():
        # Cosine over binary user-item matrix plus cooccurrence shrinkage.
        similarity = (
            common_users / sqrt(item_user_counts[i] * item_user_counts[j])
            * common_users / (common_users + 5)
        )
        neighbors[i].append((j, float(similarity)))
    for i, links in neighbors.items():
        links.sort(key=lambda p: (-p[1], p[0]))
        neighbors[i] = links[:max_neighbors]

    rows = candidates[["uid", "item_id"]]
    candidate_items = rows.groupby("uid", sort=False)["item_id"].apply(list).to_dict()
    scores_lookup: dict[tuple[int, int], float] = {}
    for uid, cand_items in candidate_items.items():
        user_scores: dict[int, float] = defaultdict(float)
        for rank, seed in enumerate(item_lists.get(uid, [])[:recent_seeds]):
            for neighbor, similarity in neighbors.get(int(seed), []):
                user_scores[neighbor] += similarity / (rank + 1)
        for item in cand_items:
            scores_lookup[(int(uid), int(item))] = user_scores.get(int(item), 0.0)
    return np.array([
        scores_lookup[(int(uid), int(item))]
        for uid, item in rows.itertuples(index=False, name=None)
    ], dtype=np.float32)
