# Pre-run source audit — 2026-10-25

This memo tracks the **exact implementation and claims** checked before real Yambda experiments. The results repository currently has **no GitHub Actions runs**, so do not claim the full test suite or any real-dataset model has passed.

## Primary-source comparison

- Official Yambda README and event schema: https://huggingface.co/datasets/yandex/yambda (50M: 10,000 users; 46,467,212 listens; `uid, item_id, timestamp, played_ratio_pct, is_organic`).
- Listen+ target is strictly `played_ratio_pct > 50` per dataset authors, not >= 30 or >= 50.
- Five-second timestamp bins (not Unix seconds), album/artist mapping tables at the repo root.
- Official reproducibility baselines ALS, BPR and SASRec: https://huggingface.co/datasets/yandex/yambda/blob/main/benchmarks/README.md
- Featuretools/Deep Feature Synthesis is existing work: https://docs.featuretools.com/en/latest/
- Multi-window and exp-decay features are existing RecSys work, not novel: https://doi.org/10.1145/3758126.3758131
- Offline recommendation metric methodology is protocol-dependent: https://doi.org/10.1145/3545796
- Our experimental task is a separate **next-day Listen+ candidate reranking protocol**, not the official Yambda benchmark.

## Fixes after the 2026-10-24 review

1. Corrected a flaky future-data-invariance test: `make_snapshot` records nondeterministic wall-clock timings, so its regression test must compare stable data/diagnostics separately from timing values.
2. Pinned Yambda download to revision `e8b04458db77e2fb30a24e65f78923fa48b7d297` for stable inputs and added required-Parquet-schema validation.
3. Recorded the source revision and SHA256 fingerprint of **sorted user IDs** in output. Actual sampled ID lists are not committed.
4. Raised actionable errors if training candidates have a single class, or validation/test have no future qualifying listens, rather than producing misleading metrics.
5. Added real-Parquet (synthetic fixture file) tests checking deterministic random user selection and **all** events for selected user IDs, plus schema failure.
6. Added `pytest -q` as the **first gate** of the opt-in manual Yambda Actions workflow. No automatic Action triggers.
7. Updated README with step-by-step preflight and metadata-source consistency.

## Verification status

- **Verified by GitHub connector:** source files, workflow definitions and committed revisions exist on main.
- **Isolated local pandas edge check passed:** duplicate artist mappings, future-blind artist-affinity aggregation and empty transition dictionary.
- **Not verified:** a full `pytest` run on the exact GitHub checkout, real Yambda-50M download, RAM/timeout adequacy of 1000 users, numeric NDCG/Recall, statistically supported improvement.
- The remote GitHub Actions history was **zero runs** when checked for this audit. The upcoming first manual run supplies the missing evidence.

## Acceptance criteria for first execution

1. Manual tests run: all unit tests and synthetic end-to-end complete successfully, with an artifact.
2. Manual Yambda run: start with 100 users and 120 CatBoost iterations. It reruns unit tests before data download.
3. Examine: `REPORT.md`, `metrics.json`, `feature_selection.json`, `run_config.json`; verify no training single-class failure and assess `retrieved_positive_pairs/future_positive_pairs`.
4. If first real-data run succeeds, repeat with 1,000 complete user histories, watching Actions memory and runtime.
5. Before submission: at least a competitive item-based collaborative baseline/ALS; count-only vs decay-only vs relations ablations; user-level uncertainty; multiple cohort seeds; clear failure analysis.

## Known non-blocking but important limitations

- This is a constrained generated-feature selector, **not a new optimal AutoML feature-search algorithm**.
- Two-stage candidate generation lacks learned ALS/ItemKNN retrieval; ranking is capped by candidate recall.
- Music listening is implicit feedback; unobserved does not mean unexposed dislikes.
- Pandas feature computation performs repeated groupby and artist/album joins, potentially exceeding 7GB on standard GitHub-hosted runners for 1,000 users.
- One global final-day test has temporal variance; backtesting across other days/seeds and bootstrap confidence intervals are not yet implemented.
- `train_seconds` is wall-clock training time, not pure CPU time.
- No real-data metric values should appear in submitted conclusions until confirmed by the actual run.
