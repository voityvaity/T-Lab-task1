# Post-first-run audit — 2026-10-25

## Evidence already gathered

Verified successful GitHub Actions run [40174763195](https://github.com/voityvaity/T-Lab-task1/actions/runs/40174763195), at source SHA [a8a59d2](https://github.com/voityvaity/T-Lab-task1/commit/a8a59d2c95140e09375630c43919e5b1f73305bd), **100 sampled users**, seed 42, 120 CatBoost iterations. The CI log confirms **14 pytest tests passed**, synthetic smoke passed, Yambda and relation mappings downloaded, real training finished, and artifact [12592006833](https://github.com/voityvaity/T-Lab-task1/actions/runs/40174763195/artifacts/12592006833) uploaded.

### Real test rankings (exploratory only)

| Model | Validation NDCG@10 | Test NDCG@10 | Test Recall@10 |
|---|---:|---:|---:|
| Popularity | 0.02461 | 0.03902 | 0.03345 |
| Manual CatBoost | **0.10953** | 0.21970 | 0.24405 |
| Auto features CatBoost | 0.09833 | **0.22178** | 0.23328 |
| Manual + auto CatBoost | 0.09855 | 0.19807 | **0.24915** |

- Only **14** sampled users had positive test interactions and **27** on validation. A 0.0021 difference in test NDCG does **not** establish superiority.
- Test candidate coverage **30/116 = 25.86%**; validation **56/318 = 17.61%**. Retrieval is the bottleneck and score ceiling.
- Candidate pairs averaged **12,874 / 100 ≈ 129** per user, despite a 200-item budget. Duplicated retrieval sources left many unused slots.
- Aggregate computation **48.51 sec** (all six snapshots): relation features **46.01 sec**, versus manual **0.60 sec**, generated **0.60 sec** and retrieval **1.26 sec**. The relation join is the main cost.
- All 29 automatically generated features were selected on validation; this is a *feature bank selection baseline*, not evidence of an optimal temporal feature search.
- Reports and metrics are from the **old SHA**; results from a new SHA must be identified separately, not mixed.

## Changes made before the next run

1. **Candidate recall/capacity:** After combining popularity, recent repeats, and 14-day transitions, backfill de-duplicated spaces with additional historical popular tracks, up to requested limit. Same candidate pool for every model within the new run. It changes the retrieval protocol versus run 40174763195, so **its metrics are not directly comparable** with the old 100-user numbers.
2. **Relation compute:** Restrict event-to-artist/album joins to the longest actual relation window (60 days) and to entity IDs relevant to the currently scored candidates. This preserves all relevant features; a unit test checks each window against a naïve full-history join on edge cases.
3. **Bootstrap:** Report 95% user-resampling percentile confidence intervals for each model and paired NDCG/Recall differences. This describes only user-level uncertainty; it does not solve time or dataset-selection bias.
4. **Ablations:** Train additional equal-candidate, equal-split CatBoost models on *count-only* generated features and *decay-only* generated features. All final-test scoring occurs after validation-based feature selection. Added metric-consistency and reproducibility tests.

## Reproducibility limitations

- The first measured run proves the pipeline executed correctly on the previous SHA, but **does not prove the new changes pass**. Always let the new manual workflow's `pytest` + synthetic gate finish before it downloads real data.
- No ALS/ItemKNN collaborative retrieval baseline implemented yet. Without it, the manual CatBoost should not be described as state of the art. Official Yambda examples: https://huggingface.co/datasets/yandex/yambda/blob/main/benchmarks/README.md
- Data lacks true impression negatives; observed versus unobserved is *not* rejection probability.
- `is_organic`, likes/dislikes and audio embeddings are provided by Yambda but not yet modeled; they are valuable extension points, not claims of current coverage.
- Test is one global day; user-bootstrap CIs do **not** address variation between days or random cohort samples.
- The computation is pandas-heavy; full 10,000-user scaling has not been demonstrated. Measure max resident memory for the next experiment.
- Source paper and dataset: https://arxiv.org/abs/2505.22238 and https://huggingface.co/datasets/yandex/yambda
- Temporal feature cutoff methods are not novel (Featuretools): https://docs.featuretools.com/en/stable/getting_started/handling_time.html
- Multi-window and decay-based recsys features are known prior art (RecSys Challenge 2025): https://doi.org/10.1145/3758126.3758138

## Next run acceptance

Run `.github/workflows/manual-yambda-research.yml` with **1,000 full-history users**, **120 iterations**, branch **main**. The workflow gates with `pytest` and synthetic end-to-end first. Inspect the real artifact for (1) candidate coverage, (2) 6-way model comparison, (3) bootstrap delta CIs, (4) selected time features, (5) relation compute time and peak memory.

If the run exceeds memory/time limits, do not keep re-running blindly: optimize feature computation and resume from saved checkpoints after addressing the bottleneck. Any report must cite the exact SHA and seed of its results.
