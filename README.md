# T-Lab task 1 — Automated Temporal Features for Recommendations

Research prototype for the T-Bank ML selection task (October 2026).

**Question:** can a constrained automatic search over time windows and exponentially decayed interaction aggregates match or improve a hand-engineered CatBoost ranking baseline under an identical, leakage-safe offline protocol?

**Status:** implementation / reproducibility scaffold. No scores from the real Yambda dataset are claimed until experiments have actually run.

## Research design

- **Dataset:** [Yandex Yambda-50M](https://huggingface.co/datasets/yandex/yambda), flat `listens.parquet`. Its event `timestamp` is expressed in **5-second bins**, not Unix seconds. Optional track–artist and track–album mappings are supported; no genre information is assumed.
- **Task:** recommend tracks at a global cutoff time; positive means a qualifying listen (`played_ratio_pct > 50`) occurs in the **next day**. A missing listen is *unobserved*, not a confirmed negative.
- **Candidate generation:** fixed, deterministic union of popular historical tracks, user's recently listened tracks and historical one-hour transitions (counted only in the past 14 days), derived only from data earlier than each cutoff. This is intentionally a relatively simple retrieval baseline; candidate coverage is explicitly measured.
- **Hand-made features:** activity/popularity and repeated user–item interaction counts in 1/7/30/60-day windows; user/item/pair recency; 30-day track diversity, listen-completion averages, 7-to-30-day trends, and a user–item-popularity-stratum affinity (23 manual features).
- **Auto features:** programmatic cross-product of `{user,item,user-item}` with count windows `{1,3,7,10,14,30,60}` and exponential decay scales `{2,5,7,10,21,30}` days. Rank candidates by training-only CatBoost feature importance and choose the retained subset size `{4,8,12,20,32,all}` on validation (not test).
- **Comparisons:** historical popularity, hand-engineered CatBoost, automatically selected CatBoost, manual + automatic CatBoost, count-only CatBoost and decay-only CatBoost. All share the exact same temporal candidate pool.
- **Evaluation:** global cutoff with disjoint one-day label periods, same candidates across models at each cutoff, NDCG@10/Recall@10 over users with future positives (denominator counts **all** positive tracks, including unretrieved ones), candidate coverage, CPU training times and feature computation times.
- **Time safety:** every history aggregate is computed from events with `timestamp < cutoff`. Validation determines feature count/model selection. Test is only scored once after that selection.

### Caveats

- This is **not** the official Yambda benchmark or a directly comparable leaderboard score: the objective, horizon, candidate pool, and training snapshots differ.
- The data do not include logged impressions, so absence of a listen cannot be treated as a true exposure-negative; measured quality may be biased.
- The popularity-based candidates cap achievable Recall@10; always inspect candidate coverage.
- Randomly sampling *users* preserves their full personal histories but estimates aggregate item popularity over that cohort, not the full service.
- Feature importance selection is a straightforward automatic-selection baseline, not a claim of novel AutoML optimization or an AI agent.
- The basic listens-only baseline omits artist/album relations; the optional metadata mode adds them. Deep sequential architectures and audio embeddings are not implemented.

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
- [x] User-level bootstrap interval implementation and paired comparisons (to be verified in the next CI run)
- [ ] Stability across cohort seeds and time horizons
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


## 2026-10-24 audit amendments

- The default positive is now `played_ratio_pct > 50` in accordance with Yambda's Listen+ convention; this **changes the objective** compared to any prior >=30% results.
- Test ranking scores are now computed only after all validation-based feature selection has finished.
- Fixed-window baseline uses 1/7/30/60-day history, consistent with the minimum history check.
- The runtime report now breaks feature computation down into retrieval, manual, generated and relation-feature stages.
- Optional artist/album mappings increase the baseline to **27 manual features** and the automatic bank to **47 generated features** (39 time-window/decay features and 8 artist/album counts).
- **Important:** This repository does not yet contain completed empirical results on real Yambda, confidence intervals, a tuned strong collaborative-filtering baseline, or a scalable benchmark of the full 10,000-user dataset. Do not submit synthetic results as research evidence.
- See [research/AUDIT_2026-10-24.md](research/AUDIT_2026-10-24.md) for verified fixes and remaining work.


## 2026-10-25 preflight (do this before the first real run)

The Yambda download is pinned to revision `e8b04458db77e2fb30a24e65f78923fa48b7d297`; the run output records this revision and a SHA-256 fingerprint of the sorted user IDs so experiments can be traced back to the exact sample. The source file is downloaded from Yandex's official dataset, with no locally stored credentials.

1. Open [Actions → Manual Yambda research](https://github.com/voityvaity/T-Lab-task1/actions/workflows/manual-yambda-research.yml), choose **100 users, 120 iterations**, and click **Run workflow**. It checks `pytest` and executes a synthetic end-to-end smoke **before** downloading Yambda. Both gate checks must succeed to proceed with the real dataset. No push trigger exists.
2. Wait for downloading Yambda, feature engineering and CatBoost training to finish. The workflow uploads the measured research report only after reaching the experiment.
3. Download the `yambda-research-report` artifact from the completed run, inspect `REPORT.md`, `metrics.json`, `feature_selection.json` and `run_config.json`. Check candidate coverage and nonzero train positives before interpreting NDCG@10.
4. **Only if this succeeds:** repeat at 1,000 users. This is computationally demanding; the pandas joins may exceed the RAM or timeout of GitHub-hosted runners. If so, use a higher-memory machine or optimize the joins rather than presenting an incomplete run as a finding.

**Still missing before final submission:** a stronger collaborative baseline (ItemKNN or ALS), evaluation with multiple seeds/days, and scaling results. One measured 100-user run exists on an **earlier** SHA; the updated candidate protocol and expanded feature search require a new run. No claim of complete validation is made merely by merging these source fixes.


## Second-audit changes after the first 100-user run (25 October 2026)

The first fully successful Yambda run is [GitHub Actions #40174763195](https://github.com/voityvaity/T-Lab-task1/actions/runs/40174763195) at SHA `a8a59d2c95140e09375630c43919e5b1f73305bd`. Its *test* NDCG@10 for manually constructed features was 0.21970 vs 0.22178 for auto features, but **only 14 test users were positive**; validation instead favored manual 0.10953 vs auto 0.09833. This is not evidence of a statistically reliable win.

To address problems observed in this run we now:
- backfill the 200-candidate retrieval list after merging sources and dropping duplicates; this changes the experimental protocol and prevents direct comparisons with the first run,
- calculate artist/album aggregate histories for only the 60-day maximum window and only relevant candidate-related entities, with a brute-force correctness test,
- test both count-only and decay-only ablations with the same data and CatBoost training settings,
- expand the generated feature space to include e.g. 10-day counts and 10-day exponential scales, and choose subset size on validation,
- report per-user 95% bootstrap confidence intervals and paired model differences, plus stage computation times,
- log peak resident memory in the manual research workflow.

See [POST_RUN_AUDIT_2026-10-25.md](research/POST_RUN_AUDIT_2026-10-25.md) for measured first-run evidence. **The new code has not yet been validated in CI.** Its first full run is gated by `pytest` and a synthetic end-to-end test before downloading Yambda.

### Recommended next execution

Go to [Manual Yambda research](https://github.com/voityvaity/T-Lab-task1/actions/workflows/manual-yambda-research.yml), select branch `main`, `users=1000`, `iterations=120`. This is a larger experiment on a random cohort with *complete histories*, **not** the full population of 10,000 users. If it passes, inspect `yambda-research-report` (metrics, bootstrap CIs, feature selection, ablation results) and GitHub logs for the maximum resident memory. Do not compare against the previous 100-user metrics directly because the candidate protocol has changed.

The separate baseline comparison with ALS/ItemKNN remains a major limitation and must be addressed before positioning this as a strong industrial-quality recommender.
