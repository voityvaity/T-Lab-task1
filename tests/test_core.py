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
