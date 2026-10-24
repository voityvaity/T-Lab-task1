# T-Lab task 1 — Automated Temporal Features for Recommendations

Research prototype for the T-Bank ML selection task (October 2026).

**Question:** can a constrained automatic search over time windows and exponentially decayed interaction aggregates match or improve a hand-engineered CatBoost ranking baseline under an identical, leakage-safe offline protocol?

**Status:** implementation / reproducibility scaffold. No scores from the real Yambda dataset are claimed until experiments have actually run.

## Research design

- **Dataset:** [Yandex Yambda-50M](https://huggingface.co/datasets/yandex/yambda), flat `listens.parquet`. Its event `timestamp` is expressed in **5-second bins**, not Unix seconds. No artist/genre metadata are assumed.
- **Task:** recommend tracks at a global cutoff time; positive means a qualifying listen (`played_ratio_pct > 50`) occurs in the **next day**. A missing listen is *unobserved*, not a confirmed negative.
- **Candidate generation:** fixed, deterministic union of popular historical tracks, user's recently listened tracks and historical one-hour transitions (counted only in the past 14 days), derived only from data earlier than each cutoff. This is intentionally a relatively simple retrieval baseline; candidate coverage is explicitly measured.
- **Hand-made features:** activity/popularity and repeated user–item interaction counts in 1/7/30/60-day windows; user/item/pair recency; 30-day track diversity, listen-completion averages, 7-to-30-day trends, and a user–item-popularity-stratum affinity (23 manual features).
- **Auto features:** programmatic cross-product of `{user,item,user-item} × {1,3,14,60-day counts or 2,7,21-day exponential decays}`. Rank candidates by CatBoost training-only feature importance; choose the number of retained features on validation, not test.
- **Comparisons:** historical popularity, hand-engineered CatBoost, automatically selected features CatBoost, and hand-engineered + automatically selected features CatBoost. All share the exact same temporal candidate pool.
- **Evaluation:** global cutoff with disjoint one-day label periods, same candidates across models at each cutoff, NDCG@10/Recall@10 over users with future positives (denominator counts **all** positive tracks, including unretrieved ones), candidate coverage, CPU training times and feature computation times.
- **Time safety:** every history aggregate is computed from events with `timestamp < cutoff`. Validation determines feature count/model selection. Test is only scored once after that selection.

### Caveats

- This is **not** the official Yambda benchmark or a directly comparable leaderboard score: the objective, horizon, candidate pool, and training snapshots differ.
- The data do not include logged impressions, so absence of a listen cannot be treated as a true exposure-negative; measured quality may be biased.
- The popularity-based candidates cap achievable Recall@10; always inspect candidate coverage.
- Randomly sampling *users* preserves their full personal histories but estimates aggregate item popularity over that cohort, not the full service.
- Feature importance selection is a straightforward automatic-selection baseline, not a claim of novel AutoML optimization or an AI agent.
- This baseline intentionally does not model artists/genres or deep sequential architecture because those aren't in the selected Yambda event file.

## Quick start

Requires Python 3.11+; CPU is sufficient for a 100-user smoke experiment.

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m tlab.experiment --demo --users 20 --iterations 60 --output results/demo
```

The demo uses **synthetic** generated interactions, solely to check execution. Its numbers are **not** the research result.

For genuine research, fetch the public dataset (large download):

```bash
python -m tlab.data download --output data --with-metadata
python -m tlab.experiment --data data/flat/50m/listens.parquet --users 100 --iterations 200 --metadata-dir data --output results/yambda-100
python -m tlab.experiment --data data/flat/50m/listens.parquet --users 1000 --iterations 350 --metadata-dir data --output results/yambda-1000
```

With sufficient compute, repeat at 10,000 users (no sampling): `--users 0`. **Do not** extrapolate small-sample metrics to the full population.

Each experiment writes `metrics.json`, `feature_selection.json`, `run_config.json`, and `REPORT.md` in its output directory (excluded from Git).

## Experiment checklist

- [x] Research references documented
- [x] Point-in-time features, disjoint future labels, shared candidate pool
- [x] Train-only importance and validation-only selection
- [x] Synthetic regression tests for future leakage, metric denominators and end-to-end execution (defined; run status must be verified)
- [ ] Reproduce on real 100-user Yambda cohort
- [ ] Reproduce on real 1,000-user cohort
- [ ] Run full 10,000-user benchmark if hardware permits
- [ ] Confidence intervals and stability across cohort seeds
- [ ] Compare ItemKNN / ALS candidate retrieval and add model ablations

## Sources

See [research/RELATED_WORK.md](research/RELATED_WORK.md). Sources are cited with URLs; academic claims and published model scores must not be mistaken for measurements from this repository.

## Notes on reproducibility

Fixed user sampling seed and event ordering, bounded dependency ranges in `pyproject.toml`, version metadata in `run_config.json`, and snapshot cutoffs in `metrics.json`. Results require an explicit run on real data. `data/` and `results/` are gitignored.

**GitHub Actions:** `.github/workflows/manual-tests.yml` uses only `workflow_dispatch`. It never runs on pushes or pull requests, to avoid spending CI minutes automatically. Click **Actions → Manual tests → Run workflow** when you want to execute it.


## Optional artist / album relations

The Yambda dataset includes static `artist_item_mapping.parquet` and `album_item_mapping.parquet`, with `(artist_id, item_id)` and `(album_id, item_id)` columns. With `--metadata-dir data` the baseline adds **4 indirect relation features** (7/30-day counts across user–artist and user–album), while the generator adds **8** (1/3/14/60-day counts across those relations). Mapping is potentially one-to-many and all distinct artist/item links are retained; counts are summed over mapped entities. Aggregation always uses history strictly earlier than the snapshot cutoff.

Without `--metadata-dir`, previous listening-only experiments still work. **Do not compare scores from different configurations** as if they used identical features.

These features use extra memory: history is joined to static relations for each cutoff. On a large sample, measure peak RSS and consider a pre-aggregated or streaming implementation.
