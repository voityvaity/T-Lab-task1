import numpy as np
import pandas as pd
import pytest

from tlab.core import AUTO_COLUMNS, MANUAL_COLUMNS, add_features, global_cutoffs, make_snapshot, retrieve_candidates
from tlab.data import BINS_PER_DAY, synthetic_events
from tlab.metrics import ranking_metrics


def test_global_splits_are_disjoint():
    events = synthetic_events(5, days=80)
    train, valid, test = global_cutoffs(events, snapshots=4)
    for start, next_start in zip(train, train[1:] + [valid, test]):
        assert start + BINS_PER_DAY < next_start
    assert valid + BINS_PER_DAY < test


def test_future_cannot_change_candidates_features_or_labels():
    events = synthetic_events(5, days=80)
    cut = 70 * BINS_PER_DAY
    users = events["uid"].unique()
    candidates = retrieve_candidates(events, cut, users, candidate_limit=30)
    past_only = events.loc[events.timestamp < cut].copy()
    np.testing.assert_array_equal(
        candidates.values, retrieve_candidates(past_only, cut, users, candidate_limit=30).values
    )
    for specs, manual in ((None, True), (None, False)):
        from tlab.core import AUTO_SPECS, MANUAL_SPECS
        chosen_specs = MANUAL_SPECS if manual else AUTO_SPECS
        before = add_features(events, candidates, cut, chosen_specs, manual_extras=manual)
        after = add_features(past_only, candidates, cut, chosen_specs, manual_extras=manual)
        pd.testing.assert_frame_equal(before, after, check_dtype=True)


def test_label_uses_one_day_and_not_current_history():
    ev = pd.DataFrame([
        (1, 1, 0, 100),
        (1, 2, 10, 100),
        (1, 1, BINS_PER_DAY, 100),
        (1, 2, BINS_PER_DAY+2, 5),
        (1, 3, 2*BINS_PER_DAY, 100),
    ], columns=["uid", "item_id", "timestamp", "played_ratio_pct"])
    result, detail = make_snapshot(ev, BINS_PER_DAY, np.array([1]), candidate_limit=10)
    # Before cutoff both tracks 1 and 2 may be candidates, but only 1 is a future positive.
    labels = dict(zip(result.item_id, result.label))
    assert labels[1] == 1
    assert labels[2] == 0
    assert detail["future_positive_pairs"] == 1
    assert all(col in result.columns for col in MANUAL_COLUMNS + AUTO_COLUMNS)


def test_recall_counts_unretrieved_future_items():
    pairs = pd.DataFrame([(7, 2, 1), (7, 3, 0)], columns=["uid", "item_id", "label"])
    score = ranking_metrics(pairs, np.array([0.9, 0.1]), positive_counts=pd.Series({7: 2}), k=1)
    assert score["recall@1"] == pytest.approx(0.5)
    assert score["ndcg@1"] == pytest.approx(1.0)


def test_late_rows_do_not_change_past_aggregates():
    event = pd.DataFrame([(1, 1, 4, 50), (1, 1, 9, 50), (1, 1, 11, 50)], columns=[
        "uid", "item_id", "timestamp", "played_ratio_pct"
    ])
    cand = pd.DataFrame([(1, 1)], columns=["uid", "item_id"])
    from tlab.core import FeatureSpec
    f = add_features(event, cand, cutoff=10, specs=(FeatureSpec("pair", "count", 1),), manual_extras=False)
    assert f["pair_count_1d"].iloc[0] == 2.0


def test_end_to_end_synthetic_smoke(tmp_path):
    """Small deterministic run: generates full report without accessing real Yambda."""
    from tlab.experiment import run

    events = synthetic_events(6, days=80, seed=21)
    result = run(
        events,
        dataset_mode="synthetic",
        output=tmp_path,
        seed=21,
        iterations=5,
        snapshots=1,
        candidate_limit=24,
        threads=2,
    )
    assert all(name in result["models"] for name in ("popularity", "manual", "auto", "manual_plus_auto"))
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "feature_selection.json").exists()
    assert (tmp_path / "REPORT.md").exists()
    assert 0 <= result["models"]["manual"]["test"]["ndcg@10"] <= 1


def test_yambda_listen_plus_threshold_is_strict():
    """The dataset authors define Listen+ as more than half of a track."""
    from tlab.data import is_positive_listen
    assert list(is_positive_listen(pd.Series([0, 30, 50, 51, 100, 190]))) == [
        False, False, False, True, True, True
    ]


def test_label_ignores_events_outside_horizon_and_50_percent():
    event = pd.DataFrame(
        [
            (7, 5, 1, 80),
            (7, 6, 2, 80),
            (7, 5, BINS_PER_DAY, 50),
            (7, 6, BINS_PER_DAY + 1, 51),
            (7, 5, 2 * BINS_PER_DAY, 100),
        ],
        columns=["uid", "item_id", "timestamp", "played_ratio_pct"],
    )
    frame, info = make_snapshot(event, BINS_PER_DAY, np.array([7]), candidate_limit=10)
    assert dict(zip(frame.item_id, frame.label)) == {5: 0, 6: 1}
    assert info["future_positive_pairs"] == 1
    assert info["retrieved_positive_pairs"] == 1


def test_snapshot_is_invariant_to_any_observations_after_target_window():
    events = synthetic_events(5, days=80)
    cutoff = 70 * BINS_PER_DAY
    user_ids = np.sort(events.uid.unique())
    frame, details = make_snapshot(events, cutoff, user_ids, candidate_limit=24)
    new_future = pd.DataFrame(
        [(int(user_ids[0]), 99999, cutoff + BINS_PER_DAY + 17, 150)],
        columns=["uid", "item_id", "timestamp", "played_ratio_pct"],
    )
    augmented = pd.concat([events, new_future], ignore_index=True)
    frame2, details2 = make_snapshot(augmented, cutoff, user_ids, candidate_limit=24)
    pd.testing.assert_frame_equal(frame, frame2)
    # CPU wall clock values naturally differ and are not model outputs.
    for key in details:
        if key != "timings_seconds":
            assert details[key] == details2[key]
    assert set(details["timings_seconds"]) == set(details2["timings_seconds"])


def test_relations_count_all_artists_without_future_leakage():
    from tlab.relations import add_relation_features
    cutoff = 10 * BINS_PER_DAY
    mapping = {
        "artist": pd.DataFrame(
            [(10, 7), (20, 8), (5, 7), (5, 8), (6, 9), (5, 7)],
            columns=["item_id", "artist_id"],
        ),
        "album": pd.DataFrame(
            [(10, 3), (20, 4), (5, 3), (6, 5)],
            columns=["item_id", "album_id"],
        ),
    }
    events = pd.DataFrame(
        [(1, 10, cutoff - 1, 90), (1, 20, cutoff - 2, 99),
         (1, 10, cutoff + 1, 100)],
        columns=["uid", "item_id", "timestamp", "played_ratio_pct"],
    )
    base = pd.DataFrame([(1, 5), (1, 6), (2, 5)],
                        columns=["uid", "item_id"])
    before = add_relation_features(base, events, cutoff, mapping)
    assert before["user_artist_count_7d"].tolist() == [2.0, 0.0, 0.0]
    assert before["user_album_count_7d"].tolist() == [1.0, 0.0, 0.0]
    past_only = events.loc[events.timestamp < cutoff]
    after = add_relation_features(base, past_only, cutoff, mapping)
    pd.testing.assert_frame_equal(before, after)


def test_synthetic_run_with_relations(tmp_path):
    from tlab.experiment import run
    events = synthetic_events(6, days=80, seed=21)
    mappings = {
        "artist": pd.DataFrame({"item_id": np.arange(50), "artist_id": np.arange(50) % 5}),
        "album": pd.DataFrame({"item_id": np.arange(50), "album_id": np.arange(50) // 5}),
    }
    result = run(
        events, dataset_mode="synthetic", output=tmp_path, seed=21,
        iterations=5, snapshots=1, candidate_limit=24, threads=2,
        mappings=mappings,
    )
    assert result["using_artist_album_metadata"] is True
    assert "user_artist_count_30d" in result["models"]["manual"]["features"]
    import json
    selection = json.loads((tmp_path / "feature_selection.json").read_text(encoding="utf-8"))
    assert "user_album_count_14d" in {
        feature["name"] for feature in selection["importance_train_only"]
    }
    assert 0 <= result["models"]["auto"]["test"]["recall@10"] <= 1


def test_real_parquet_sampler_keeps_full_histories_and_fixed_seed(tmp_path):
    """Exercise the same Arrow filter used for the downloaded dataset."""
    from tlab.data import load_yambda
    source = pd.DataFrame([
        (u, item, t, ratio)
        for u in range(1, 5)
        for item, t, ratio in [(10, 1, 90), (11, 99, 75), (12, 3000, 100)]
    ], columns=["uid", "item_id", "timestamp", "played_ratio_pct"])
    path = tmp_path / "listens.parquet"
    source.to_parquet(path, index=False, engine="pyarrow")
    selected = load_yambda(path, users=2, seed=42)
    assert selected["uid"].nunique() == 2
    assert len(selected) == 6  # three rows per user, without history truncation
    assert set(selected["timestamp"]) == {1, 99, 3000}
    pd.testing.assert_frame_equal(selected, load_yambda(path, users=2, seed=42))


def test_real_parquet_loader_rejects_missing_columns(tmp_path):
    from tlab.data import load_yambda
    path = tmp_path / "broken.parquet"
    pd.DataFrame({"uid": [1, 1], "item_id": [10, 11]}).to_parquet(path)
    with pytest.raises(ValueError, match="schema missing columns"):
        load_yambda(path, users=1, seed=42)


def test_feature_columns_unique_when_manual_and_auto_overlap():
    """Regression: overlapping auto/manual count windows must not create duplicate columns."""
    from tlab.core import MANUAL_COLUMNS, AUTO_COLUMNS
    events = synthetic_events(4, days=80, seed=10)
    cutoff = 70 * BINS_PER_DAY
    result, _ = make_snapshot(events, cutoff, np.sort(events.uid.unique()), candidate_limit=30)
    assert result.columns.is_unique
    for column in set(MANUAL_COLUMNS + AUTO_COLUMNS):
        assert list(result.columns).count(column) == 1
    assert len(set(MANUAL_COLUMNS).intersection(AUTO_COLUMNS)) == 12


def test_filtered_relation_joins_equal_simple_reference_for_all_windows():
    """Join optimization must never change a manual or auto relation feature."""
    from tlab.relations import (
        add_relation_features, MANUAL_RELATION_DAYS, AUTO_RELATION_DAYS,
    )
    cut = 100 * BINS_PER_DAY
    raw = [
        (uid, item, cut - days * BINS_PER_DAY, 80)
        for uid in (1, 2)
        for item in (10, 11, 12, 99)
        for days in (0, 1, 3, 7, 14, 30, 60, 61, 90)
    ]
    raw.extend([(1, 11, cut + 1, 100), (2, 12, cut + 7, 100)])
    events = pd.DataFrame(raw, columns=["uid", "item_id", "timestamp", "played_ratio_pct"])
    mappings = {
        "artist": pd.DataFrame(
            [(10, 7), (11, 7), (11, 8), (12, 8), (12, 9), (99, 55),
             (11, 8), (20, 33)],
            columns=["item_id", "artist_id"],
        ),
        "album": pd.DataFrame(
            [(10, 101), (11, 101), (12, 102), (99, 103), (20, 104)],
            columns=["item_id", "album_id"],
        ),
    }
    cand = pd.DataFrame(
        [(1, 10), (1, 11), (1, 12), (2, 11), (2, 20), (3, 12)],
        columns=["uid", "item_id"],
    )
    actual = add_relation_features(cand, events, cut, mappings)
    assert actual[["uid", "item_id"]].equals(cand)
    for name, raw_map in mappings.items():
        identifier = f"{name}_id"
        dedup = raw_map.drop_duplicates(["item_id", identifier])
        linked = events.merge(dedup, on="item_id", how="inner")
        for days in sorted(set(MANUAL_RELATION_DAYS + AUTO_RELATION_DAYS)):
            recent = linked.loc[
                (linked.timestamp < cut)
                & (linked.timestamp >= cut - days * BINS_PER_DAY)
            ]
            column = f"user_{name}_count_{days}d"
            expected = []
            for row in cand.itertuples(index=False):
                identifiers = dedup.loc[dedup.item_id == row.item_id, identifier].unique()
                value = len(recent.loc[
                    (recent.uid == row.uid)
                    & recent[identifier].isin(identifiers)
                ])
                expected.append(float(value))
            assert actual[column].tolist() == expected, column


def test_candidate_pool_backfills_unique_items_with_historical_popularity():
    """Pop+repeat source overlap must not silently waste a candidate budget."""
    events = pd.DataFrame(
        [
            (user, item, day * BINS_PER_DAY + item, 80)
            for user in range(3)
            for item in range(60)
            for day in (1, 3)
        ],
        columns=["uid", "item_id", "timestamp", "played_ratio_pct"],
    )
    cutoff = 5 * BINS_PER_DAY
    pool = retrieve_candidates(events, cutoff, np.array([0, 1, 2]), candidate_limit=30)
    assert pool.groupby("uid").size().to_dict() == {0: 30, 1: 30, 2: 30}
    assert not pool.duplicated(["uid", "item_id"]).any()
    historical_only = events.loc[events.timestamp < cutoff]
    pd.testing.assert_frame_equal(
        pool, retrieve_candidates(historical_only, cutoff, np.array([0, 1, 2]), candidate_limit=30)
    )


def test_per_user_bootstrap_matches_existing_global_ranking_metrics():
    """Uncertainty code must not silently change NDCG/Recall definitions."""
    from tlab.uncertainty import paired_bootstrap, user_ranking_metrics
    pairs = pd.DataFrame(
        [(1, 10, 1), (1, 11, 0), (1, 12, 1),
         (2, 10, 0), (2, 20, 1), (2, 21, 0),
         (3, 12, 0), (3, 13, 1)],
        columns=["uid", "item_id", "label"],
    )
    positives = pd.Series({1: 3, 2: 2, 3: 1})
    baseline = np.array([0.9, 0.2, 0.5, 0.4, 0.3, 0.1, 0.0, 0.7])
    alternative = baseline[::-1].copy()
    raw = ranking_metrics(pairs, baseline, positive_counts=positives, k=2)
    per_user = user_ranking_metrics(pairs, baseline, positives, k=2)
    assert per_user["ndcg"].mean() == pytest.approx(raw["ndcg@2"])
    assert per_user["recall"].mean() == pytest.approx(raw["recall@2"])
    bootstrap = paired_bootstrap(
        pairs, {"manual": baseline, "auto": alternative}, positives,
        k=10, n_resamples=200, seed=42,
    )
    paired = bootstrap["paired_deltas"]["auto_minus_manual"]["ndcg@10"]
    assert np.isfinite(paired["estimate"])
    assert paired["ci95"][0] <= paired["ci95"][1]
    assert bootstrap == paired_bootstrap(
        pairs, {"manual": baseline, "auto": alternative}, positives,
        k=10, n_resamples=200, seed=42,
    )


def test_global_time_anchor_is_independent_of_selected_users(tmp_path):
    from tlab.data import load_yambda, global_last_timestamp

    parquet = tmp_path / "listens.parquet"
    source = pd.DataFrame([
        (1, 10, 0, 80), (1, 11, 100, 80),
        (2, 10, 0, 80), (2, 12, 90, 80),
        (3, 13, 0, 80), (3, 14, 100 * BINS_PER_DAY + 500, 80),
    ], columns=["uid", "item_id", "timestamp", "played_ratio_pct"])
    source.to_parquet(parquet, engine="pyarrow", index=False)
    full_max = 100 * BINS_PER_DAY + 500
    assert global_last_timestamp(parquet) == full_max
    sample = load_yambda(parquet, users=1, seed=42)
    assert sample.attrs["full_dataset_max_timestamp"] == full_max
    _, valid, test = global_cutoffs(sample, snapshots=1, anchor_timestamp=full_max)
    assert test == full_max - BINS_PER_DAY
    assert valid == test - 2 * BINS_PER_DAY


def test_binary_diagnostics_auc_ties_and_logloss_reference():
    from tlab.metrics import binary_diagnostics

    y = np.array([0, 1, 0, 1], dtype=np.int8)
    s = np.array([0.1, 0.9, 0.5, 0.5])
    got = binary_diagnostics(y, s, probabilities=True)
    # Positive .9 beats both negatives; positive .5 beats .1 and ties .5.
    assert got["candidate_roc_auc"] == pytest.approx(0.875)
    reference_loss = -np.mean(
        y * np.log(s) + (1 - y) * np.log1p(-s)
    )
    assert got["candidate_logloss"] == pytest.approx(reference_loss)
    assert got["candidate_positive_pairs"] == 2
    raw = binary_diagnostics(y, np.array([1, 5, 5, 9]), probabilities=False)
    assert raw["candidate_logloss"] is None
    assert raw["candidate_roc_auc"] == pytest.approx(0.875)
    all_positive = binary_diagnostics(np.ones(3), np.array([0.5, 0.6, 0.7]), probabilities=True)
    assert all_positive["candidate_roc_auc"] is None


def test_itemknn_similarity_is_history_only_and_nontrivial():
    from tlab.itemknn import itemknn_scores
    cutoff = 70 * BINS_PER_DAY
    # Different co-listening users support a genuine item co-occurrence.
    rows = [
        (1, 1, cutoff - 100, 70), (1, 2, cutoff - 99, 90),
        (2, 1, cutoff - 100, 70), (2, 2, cutoff - 98, 90),
        (3, 1, cutoff - 100, 70), (3, 3, cutoff - 98, 90),
        (4, 1, cutoff - 100, 70), (4, 3, cutoff - 99, 90),
        (5, 1, cutoff - 1, 80),
    ]
    events = pd.DataFrame(
        rows, columns=["uid", "item_id", "timestamp", "played_ratio_pct"]
    )
    candidates = pd.DataFrame(
        [(5, 2), (5, 3), (5, 99)], columns=["uid", "item_id"]
    )
    scores = itemknn_scores(events, cutoff, candidates)
    assert scores[0] > 0 and scores[1] > 0
    assert scores[2] == 0.0
    future = pd.DataFrame(
        [(3, 2, cutoff + 100, 100)],
        columns=events.columns,
    )
    augmented = pd.concat([events, future], ignore_index=True)
    np.testing.assert_array_equal(scores, itemknn_scores(augmented, cutoff, candidates))


def test_itemknn_is_included_in_shared_snapshot_baselines(tmp_path):
    from tlab.experiment import run
    events = synthetic_events(6, days=80, seed=21)
    summary = run(
        events, dataset_mode="synthetic", output=tmp_path,
        seed=21, iterations=5, snapshots=1, candidate_limit=24, threads=2,
    )
    assert "itemknn" in summary["models"]
    assert "itemknn_score" in summary["models"]["manual"]["features"]
    assert summary["feature_cost_seconds"]["itemknn_features"] >= 0
