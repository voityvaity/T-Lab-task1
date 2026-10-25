"""Paired user-level bootstrap uncertainty for the held-out recommender split."""
from __future__ import annotations

import numpy as np
import pandas as pd


def user_ranking_metrics(
    pairs: pd.DataFrame, scores: np.ndarray, positive_counts: pd.Series, k: int = 10
) -> pd.DataFrame:
    """Per-user NDCG/Recall, counting all future positives in the denominator."""
    if len(pairs) != len(scores):
        raise ValueError("scores/pairs length mismatch")
    if not np.isfinite(scores).all():
        raise ValueError("Ranking scores must be finite")
    target = positive_counts.loc[positive_counts > 0].sort_index()
    if target.empty:
        raise ValueError("No users with positive test interactions")

    ranked = pairs[["uid", "item_id", "label"]].copy()
    ranked["score"] = np.asarray(scores)
    ranked = ranked.sort_values(
        ["uid", "score", "item_id"], ascending=[True, False, True], kind="mergesort"
    )
    top = ranked.groupby("uid", sort=False).head(k).copy()
    top["rank"] = top.groupby("uid", sort=False).cumcount() + 1
    top["gain"] = top["label"] / np.log2(top["rank"] + 1)
    dcg = top.groupby("uid")["gain"].sum().reindex(target.index, fill_value=0.0)
    hits = top.groupby("uid")["label"].sum().reindex(target.index, fill_value=0.0)
    discounts = 1.0 / np.log2(np.arange(k) + 2)
    ideal = np.array([discounts[:min(k, int(n))].sum() for n in target], dtype=float)
    return pd.DataFrame({
        "uid": target.index.to_numpy(),
        "ndcg": dcg.to_numpy(dtype=float) / ideal,
        "recall": hits.to_numpy(dtype=float) / target.to_numpy(dtype=float),
    })


def paired_bootstrap(
    pairs: pd.DataFrame,
    scores_by_model: dict[str, np.ndarray],
    positive_counts: pd.Series,
    *,
    k: int = 10,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """95% percentile CIs and paired model differences; no re-training or tuning.

    Resample users with replacement; same indices for every model to preserve
    covariance. This quantifies only user-sampling uncertainty, not time/cohort
    shifts or retrieval bias. Intended for descriptive final-test analysis.
    """
    if n_resamples < 100:
        raise ValueError("n_resamples should be at least 100")
    tables = {
        name: user_ranking_metrics(pairs, scores, positive_counts, k=k)
        for name, scores in scores_by_model.items()
    }
    first = next(iter(tables.values()))
    users = first["uid"].to_numpy()
    if not all(np.array_equal(users, table["uid"].to_numpy()) for table in tables.values()):
        raise ValueError("Models were evaluated on different user sets")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(users), size=(n_resamples, len(users)))

    output = {
        "evaluated_users": int(len(users)),
        "seed": int(seed),
        "resamples": int(n_resamples),
        "note": "User-bootstrap 95% percentile intervals; excludes cohort/time uncertainty",
        "models": {},
        "paired_deltas": {},
    }
    def summarise(values: np.ndarray) -> dict[str, float | list[float]]:
        sampled = values[sample_indices].mean(axis=1)
        return {
            "estimate": float(values.mean()),
            "ci95": [float(v) for v in np.quantile(sampled, [0.025, 0.975])],
        }

    for name, table in tables.items():
        output["models"][name] = {
            "ndcg@10": summarise(table["ndcg"].to_numpy(dtype=float)),
            "recall@10": summarise(table["recall"].to_numpy(dtype=float)),
        }

    if "manual" in tables:
        base = tables["manual"]
        for name in ("auto", "manual_plus_auto"):
            if name not in tables:
                continue
            candidate = tables[name]
            output["paired_deltas"][f"{name}_minus_manual"] = {
                "ndcg@10": summarise(
                    candidate["ndcg"].to_numpy(dtype=float) - base["ndcg"].to_numpy(dtype=float)
                ),
                "recall@10": summarise(
                    candidate["recall"].to_numpy(dtype=float) - base["recall"].to_numpy(dtype=float)
                ),
            }
    return output
