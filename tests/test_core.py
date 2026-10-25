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
    """Regression: the overlapping 1d and 60d counts previously broke CatBoost."""
    from tlab.core import MANUAL_COLUMNS, AUTO_COLUMNS
    events = synthetic_events(4, days=80, seed=10)
    cutoff = 70 * BINS_PER_DAY
    result, _ = make_snapshot(events, cutoff, np.sort(events.uid.unique()), candidate_limit=30)
    assert result.columns.is_unique
    for column in set(MANUAL_COLUMNS + AUTO_COLUMNS):
        assert list(result.columns).count(column) == 1
    assert len(set(MANUAL_COLUMNS).intersection(AUTO_COLUMNS)) == 6
