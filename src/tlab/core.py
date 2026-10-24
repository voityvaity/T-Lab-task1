"""Leakage-safe candidate retrieval, feature generation, and labels."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .data import BINS_PER_DAY

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
    for days in (7, 30)
)
AUTO_SPECS = tuple(
    [FeatureSpec(scope, "count", days) for scope in ("user", "item", "pair") for days in (1, 3, 14, 60)]
    + [FeatureSpec(scope, "decay", days) for scope in ("user", "item", "pair") for days in (2, 7, 21)]
)
MANUAL_COLUMNS = [spec.name for spec in MANUAL_SPECS] + ["pair_recency_days", "user_popband_30"]
AUTO_COLUMNS = [spec.name for spec in AUTO_SPECS]


def global_cutoffs(events: pd.DataFrame, snapshots: int = 4) -> tuple[list[int], int, int]:
    """Disjoint 1-day labels: train cutoffs 3 days apart, validation, final test."""
    if events.empty:
        raise ValueError("Empty events")
    test = int(events["timestamp"].max()) - BINS_PER_DAY
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
    """Deterministic future-blind popularity + recent-repeat candidate pool."""
    if candidate_limit < 2:
        raise ValueError("candidate_limit must be >= 2")
    history = past(events, cutoff)
    pop_budget = candidate_limit // 2
    recent_budget = candidate_limit - pop_budget
    global_items = (
        history.groupby("item_id").size().sort_values(ascending=False, kind="mergesort")
        .index[:pop_budget].tolist()
    )
    recent = (
        history.sort_values(["uid", "timestamp", "item_id"], ascending=[True, False, True], kind="mergesort")
        .drop_duplicates(["uid", "item_id"])
        .groupby("uid", sort=False).head(recent_budget)
        .groupby("uid")["item_id"].apply(list).to_dict()
    )
    rows = []
    for uid in np.sort(all_users):
        items = dict.fromkeys(global_items + recent.get(uid, []))
        rows.extend((int(uid), int(item)) for item in list(items)[:candidate_limit])
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
            # A 60-day tail bounds runtime; any discarded contribution is < exp(-60/tau).
            selection = history.loc[history["timestamp"] >= cutoff - 60 * BINS_PER_DAY, keys + ["timestamp"]].copy()
            ages = (cutoff - selection["timestamp"].to_numpy(dtype=np.float64)) / BINS_PER_DAY
            selection["weight"] = np.exp(-ages / spec.days)
            series = selection.groupby(keys, sort=False)["weight"].sum().rename(spec.name)
        out = out.join(series, on=keys)
        out[spec.name] = out[spec.name].fillna(0.0).astype("float32")

    if manual_extras:
        last = history.groupby(["uid", "item_id"], sort=False)["timestamp"].max()
        last.name = "_last"
        out = out.join(last, on=["uid", "item_id"])
        out["pair_recency_days"] = (
            (cutoff - out["_last"]) / BINS_PER_DAY
        ).fillna(9999.0).astype("float32")
        out = out.drop(columns=["_last"])

        # Indirect user/item link: preference for *past popularity band* of the candidate.
        recent = history.loc[history["timestamp"] >= cutoff - 30 * BINS_PER_DAY, ["uid", "item_id"]].copy()
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
    cand = retrieve_candidates(events, cutoff, all_users, candidate_limit=candidate_limit)
    manual = add_features(events, cand, cutoff, MANUAL_SPECS, manual_extras=True)
    auto = add_features(events, cand, cutoff, AUTO_SPECS, manual_extras=False)
    feature_frame = pd.concat([manual, auto[AUTO_COLUMNS]], axis=1)
    future = events.loc[
        (events["timestamp"] >= cutoff)
        & (events["timestamp"] < cutoff + BINS_PER_DAY)
        & (events["played_ratio_pct"] >= 30),
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
    }
