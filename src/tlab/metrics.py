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


def binary_diagnostics(
    labels: pd.Series | np.ndarray,
    scores: np.ndarray,
    *,
    probabilities: bool,
) -> dict[str, float | int | None]:
    """Candidate-conditional ROC-AUC and (when probabilities) binary LogLoss.

    Unplayed candidates are only *unobserved*, not logged non-click impressions.
    These diagnostics must not be interpreted as exposure-calibrated CTR.
    """
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(scores, dtype=np.float64)
    if len(y) != len(p) or not len(y):
        raise ValueError("Mismatched or empty labels/scores")
    if not np.isfinite(p).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("Non-finite scores or non-binary labels")
    if probabilities and ((p < 0).any() or (p > 1).any()):
        raise ValueError("Probability predictions must lie in [0, 1]")

    positives = int(y.sum())
    negatives = len(y) - positives
    auc = None
    if positives and negatives:
        # Mann-Whitney AUC with mean-rank correction for tied scores.
        order = np.argsort(p, kind="mergesort")
        sorted_scores = p[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
        ends = np.r_[starts[1:], len(y)]
        positive_cumsum = np.r_[0.0, np.cumsum(y[order], dtype=float)]
        group_positives = positive_cumsum[ends] - positive_cumsum[starts]
        mean_ranks = (starts + ends + 1) / 2
        rank_sum = np.dot(group_positives, mean_ranks)
        auc = float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))
    loss = None
    if probabilities:
        clipped = np.clip(p, 1e-15, 1 - 1e-15)
        loss = float(np.mean(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped)))
    return {
        "candidate_roc_auc": auc,
        "candidate_logloss": loss,
        "candidate_positive_pairs": positives,
        "candidate_negative_unobserved_pairs": negatives,
    }
