"""Leakage-safe candidate retrieval, feature generation, and labels."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd

from .data import BINS_PER_DAY, is_positive_listen

Scope = Literal["user", "item", "pair"]
_KEYS: dict[Scope, list[str]] = {"user": ["uid"], "item": ["item_id"], "pair": ["uid", "item_id"]}


@dataclass(frozen=True)
class FeatureSpec:
    scope: Scope
    kind: Literal["count", "decay"]
    days: int

    @property
    def name(self) -> str:
        return f"{self.scope}_{self.kind}_{self.days}d"


MANUAL_SPECS = tuple(
    FeatureSpec(scope, "count", days)
    for scope in ("user", "item", "pair")
    for days in (1, 7, 30, 60)
)
AUTO_SPECS = tuple(
    [FeatureSpec(scope, "count", days) for scope in ("user", "item", "pair") for days in (1, 3, 7, 10, 14, 30, 60)]
    + [FeatureSpec(scope, "decay", days) for scope in ("user", "item", "pair") for days in (2, 5, 7, 10, 21, 30)]
)
MANUAL_COLUMNS = [spec.name for spec in MANUAL_SPECS] + [
    "user_recency_days", "item_recency_days", "pair_recency_days",
    "user_unique_items_30",
    "user_mean_played_30", "item_mean_played_30", "pair_mean_played_30",
    "user_trend_7_over_30", "item_trend_7_over_30", "pair_trend_7_over_30",
    "user_popband_30",
]
AUTO_COLUMNS = [spec.name for spec in AUTO_SPECS]


def global_cutoffs(
    events: pd.DataFrame,
    snapshots: int = 4,
    anchor_timestamp: int | None = None,
    test_lag_days: int = 0,
) -> tuple[list[int], int, int]:
    """Disjoint one-day outcomes with a cohort-independent full-source anchor.

    A lag of 21 moves the entire train/validation/test timeline back by 21
    days, not just the test labels. No training label can touch its validation
    or test window. History computations still use timestamps < each cutoff.
    """
    if events.empty:
        raise ValueError("Empty events")
    if snapshots < 1:
        raise ValueError("snapshots must be positive")
    if test_lag_days < 0 or int(test_lag_days) != test_lag_days:
        raise ValueError("test_lag_days must be a nonnegative integer")
    sampled_max = int(events["timestamp"].max())
    if anchor_timestamp is None:
        anchor_timestamp = sampled_max
    if anchor_timestamp < sampled_max:
        raise ValueError("Global anchor cannot precede any sampled event")
    test = int(anchor_timestamp) - (1 + int(test_lag_days)) * BINS_PER_DAY
    valid = test - 2 * BINS_PER_DAY
    train = [valid - (3 * i) * BINS_PER_DAY for i in range(snapshots, 0, -1)]
    if min(train) - 60 * BINS_PER_DAY < int(events["timestamp"].min()):
        raise ValueError("Insufficient event history for 60-day windows at first cutoff")
    assert all(x + BINS_PER_DAY < y for x, y in zip(train, train[1:] + [valid, test]))
    return train, valid, test


def past(events: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """All and only observations available strictly before inference time."""
    return events.loc[events["timestamp"] < cutoff]


def retrieve_candidates(
    events: pd.DataFrame, cutoff: int, all_users: np.ndarray, candidate_limit: int = 200
) -> pd.DataFrame:
    """Future-blind mix of popularity, repeats and 14-day within-session transitions.

    Transitions are computed solely from history: a source track and its next
    played track for the same user within one hour. This retrieval mechanism
    is deliberately shared unchanged across all ranking models.
    """
    if candidate_limit < 3:
        raise ValueError("candidate_limit must be >= 3")
    history = past(events, cutoff)
    pop_budget = candidate_limit // 3
    recent_budget = candidate_limit // 3
    transition_budget = candidate_limit - pop_budget - recent_budget
    global_items_all = (
        history.groupby("item_id").size().sort_values(ascending=False, kind="mergesort")
        .index[:candidate_limit].tolist()
    )
    global_items = global_items_all[:pop_budget]
    recent = (
        history.sort_values(["uid", "timestamp", "item_id"], ascending=[True, False, True], kind="mergesort")
        .drop_duplicates(["uid", "item_id"])
        .groupby("uid", sort=False).head(max(recent_budget, 3))
        .groupby("uid")["item_id"].apply(list).to_dict()
    )

    # One-hour temporal transitions, counted only in the past 14 days.
    tail = history.loc[
        history["timestamp"] >= cutoff - 14 * BINS_PER_DAY,
        ["uid", "item_id", "timestamp"],
    ].sort_values(["uid", "timestamp", "item_id"], kind="mergesort").copy()
    tail["next_item"] = tail.groupby("uid", sort=False)["item_id"].shift(-1)
    tail["next_time"] = tail.groupby("uid", sort=False)["timestamp"].shift(-1)
    transitions = tail.loc[
        tail["next_item"].notna()
        & (tail["next_item"] != tail["item_id"])
        & ((tail["next_time"] - tail["timestamp"]) > 0)
        & ((tail["next_time"] - tail["timestamp"]) <= 720),
        ["item_id", "next_item"],
    ]
    top_transition = (
        transitions.groupby(["item_id", "next_item"], sort=False).size()
        .rename("count").reset_index()
        .sort_values(["item_id", "count", "next_item"], ascending=[True, False, True], kind="mergesort")
        .groupby("item_id", sort=False).head(10)
        .groupby("item_id")["next_item"].apply(list).to_dict()
    )

    rows = []
    for uid in np.sort(all_users):
        user_recent = recent.get(uid, [])
        suggestions = []
        for seed in user_recent[:3]:
            suggestions.extend(top_transition.get(seed, []))
            if len(suggestions) >= transition_budget * 3:
                break
        # Insertion order matters: reserve budget per retrieval source.
        ordered = list(dict.fromkeys(
            global_items[:pop_budget]
            + user_recent[:recent_budget]
            + suggestions[:transition_budget]
            # Recover capacity lost to duplicated sources, using past popularity.
            + global_items_all
        ))
        rows.extend((int(uid), int(item)) for item in ordered[:candidate_limit])
    return pd.DataFrame(rows, columns=["uid", "item_id"])



def add_features(
    events: pd.DataFrame, candidates: pd.DataFrame, cutoff: int,
    specs: tuple[FeatureSpec, ...] = MANUAL_SPECS,
    manual_extras: bool = True,
) -> pd.DataFrame:
    """Compute point-in-time features. Future rows never enter an aggregate."""
    history = past(events, cutoff)
    out = candidates[["uid", "item_id"]].copy().reset_index(drop=True)
    for spec in specs:
        keys = _KEYS[spec.scope]
        lo = cutoff - spec.days * BINS_PER_DAY
        if spec.kind == "count":
            selection = history.loc[history["timestamp"] >= lo]
            series = selection.groupby(keys, sort=False).size().rename(spec.name)
        else:
            # A 180-day tail bounds memory; even at tau=21d older per-event weights are < 0.0002.
            selection = history.loc[history["timestamp"] >= cutoff - 180 * BINS_PER_DAY, keys + ["timestamp"]].copy()
            ages = (cutoff - selection["timestamp"].to_numpy(dtype=np.float64)) / BINS_PER_DAY
            selection["weight"] = np.exp(-ages / spec.days)
            series = selection.groupby(keys, sort=False)["weight"].sum().rename(spec.name)
        out = out.join(series, on=keys)
        out[spec.name] = out[spec.name].fillna(0.0).astype("float32")

    if manual_extras:
        for scope in ("user", "item", "pair"):
            keys = _KEYS[scope]
            last = history.groupby(keys, sort=False)["timestamp"].max().rename("_last")
            out = out.join(last, on=keys)
            out[f"{scope}_recency_days"] = (
                (cutoff - out["_last"]) / BINS_PER_DAY
            ).fillna(9999.0).astype("float32")
            out = out.drop(columns=["_last"])

        recent = history.loc[
            history["timestamp"] >= cutoff - 30 * BINS_PER_DAY,
            ["uid", "item_id", "played_ratio_pct"],
        ].copy()
        unique = recent.groupby("uid")["item_id"].nunique().rename("user_unique_items_30")
        out = out.join(unique, on="uid")
        out["user_unique_items_30"] = out["user_unique_items_30"].fillna(0).astype("float32")

        for scope in ("user", "item", "pair"):
            keys = _KEYS[scope]
            name = f"{scope}_mean_played_30"
            avg = recent.groupby(keys)["played_ratio_pct"].mean().rename(name)
            out = out.join(avg, on=keys)
            out[name] = out[name].fillna(0.0).astype("float32")
            # Compare recent activity to a longer-term reference rate.
            out[f"{scope}_trend_7_over_30"] = (
                out[f"{scope}_count_7d"] /
                (out[f"{scope}_count_30d"] + 1.0)
            ).astype("float32")

        # Indirect user/item link: preference for *past popularity band* of the candidate.
        pop = recent.groupby("item_id").size()
        limits = [1, 3, 10, 30]
        out["pop_band"] = np.digitize(out["item_id"].map(pop).fillna(0).to_numpy(), limits)
        recent["pop_band"] = np.digitize(recent["item_id"].map(pop).fillna(0).to_numpy(), limits)
        band_affinity = recent.groupby(["uid", "pop_band"]).size().rename("user_popband_30")
        out = out.join(band_affinity, on=["uid", "pop_band"])
        out["user_popband_30"] = out["user_popband_30"].fillna(0).astype("float32")
        out = out.drop(columns=["pop_band"])
    return out


def make_snapshot(
    events: pd.DataFrame, cutoff: int, all_users: np.ndarray, candidate_limit: int = 200
) -> tuple[pd.DataFrame, dict[str, int]]:
    start = perf_counter()
    cand = retrieve_candidates(events, cutoff, all_users, candidate_limit=candidate_limit)
    retrieval_time = perf_counter() - start
    start = perf_counter()
    manual = add_features(events, cand, cutoff, MANUAL_SPECS, manual_extras=True)
    manual_time = perf_counter() - start
    start = perf_counter()
    auto = add_features(events, cand, cutoff, AUTO_SPECS, manual_extras=False)
    auto_time = perf_counter() - start
    # Some automatically generated count windows are identical to manual ones.
    # Keep exactly one copy; pandas duplicate columns break CatBoost feature names.
    new_auto_columns = [name for name in AUTO_COLUMNS if name not in manual.columns]
    feature_frame = pd.concat([manual, auto[new_auto_columns]], axis=1)
    if not feature_frame.columns.is_unique:
        raise ValueError("Feature generator emitted duplicate column names")
    future = events.loc[
        (events["timestamp"] >= cutoff)
        & (events["timestamp"] < cutoff + BINS_PER_DAY)
        & is_positive_listen(events["played_ratio_pct"]),
        ["uid", "item_id"],
    ].drop_duplicates()
    labels = future.assign(label=1)
    feature_frame = feature_frame.merge(labels, on=["uid", "item_id"], how="left", sort=False)
    feature_frame["label"] = feature_frame["label"].fillna(0).astype("int8")
    return feature_frame, {
        "cutoff": int(cutoff),
        "candidate_pairs": len(cand),
        "future_positive_pairs": len(future),
        "retrieved_positive_pairs": int(feature_frame["label"].sum()),
        "active_users": int(future["uid"].nunique()),
        "timings_seconds": {
            "candidate_retrieval": round(retrieval_time, 6),
            "manual_features": round(manual_time, 6),
            "auto_features": round(auto_time, 6),
            "relation_features": 0.0,
        },
    }
