# Independent audit: source, experiments and implementation (2026-10-25)

## Scope and evidence

This audit rechecks the original automated temporal feature-search proposal against
[official Yambda](https://huggingface.co/datasets/yandex/yambda/blob/main/README.md),
[their baseline guide](https://huggingface.co/datasets/yandex/yambda/blob/main/benchmarks/README.md),
[Featuretools' cutoff-time semantics](https://docs.featuretools.com/en/stable/getting_started/handling_time.html),
[Time to Split (RecSys 2025)](https://doi.org/10.1145/3705328.3748164), and
[RecSys 2025 sampling-bias study](https://doi.org/10.1145/3705328.3748086).

Prior **successful** GitHub Actions run [40178980085](https://github.com/voityvaity/T-Lab-task1/actions/runs/40178980085)
was at commit `b05429449738380d0ad5750307b142ed85914a01` on the
**1,000-user cohort**, 120 CatBoost iterations. At this *old* commit 17 tests passed.
Manual test NDCG@10=0.15969, auto=0.16695, decay-only=0.16763.
Paired 95% bootstrap CI for (auto − manual) was [-0.00763, +0.02578],
so superiority was **not established**.

The old protocol retrieved 606/2878 test future positives = 21.06% coverage.
That is a retrieval-stage recall ceiling: stronger reranking alone cannot fix it.
The old implementation consumed roughly 2.25 GiB peak RAM during the experiment.

**Historical note:** when this earlier audit was written, CI had not validated
its code changes. Subsequent runs [40181074981](https://github.com/voityvaity/T-Lab-task1/actions/runs/40181074981)
and [40182238147](https://github.com/voityvaity/T-Lab-task1/actions/runs/40182238147)
did validate the implementation at SHA `b14c04b` / documentation-only SHA
`239d8ac`. Later reporting/bootstrap improvements require their own test run.

## Corrections made in this audit

1. **Cross-cohort time consistency (important).** Evaluation cutoffs were based
   on the maximum timestamp *among randomly selected users*, not the entire
   source. Different user samples therefore evaluated at slightly different
   global times. New loader obtains dataset-wide max timestamp from Parquet
   row-group statistics (streaming fallback), stores it on the loaded frame,
   and anchors cutoffs identically across cohorts. Added regression test.
2. **Missing task metrics (important).** Added tie-corrected ROC-AUC and
   probabilistic LogLoss for fixed candidate labels, alongside NDCG and
   Recall. Popularity and ItemKNN are uncalibrated scorers; their LogLoss is
   `null`. AUC/LogLoss are explicitly **candidate-conditional diagnostics**,
   not exposure-corrected CTR measurement. Added hand-computable tie test.
3. **Post-selection test integrity (important).** A single headline model is
   chosen by **validation NDCG@10 before any test scoring**; the remaining
   test comparisons are descriptive exploratory evidence. All per-model
   comparisons are still on exactly the same candidate pool and cutoffs.
4. **Independent collaborative benchmark.** Added a bounded, causal ItemKNN
   baseline from binary user-item co-occurrence of each user's last 25
   distinct tracks in the previous 30 days, ranking a shared candidate
   pool from the latest 5 seeds. Added its score to the manual CatBoost
   baseline. The ItemKNN model is **not** yet an independent candidate
   retrieval engine, so it does not address the pool's recall ceiling.
5. **Regression coverage.** Added tests for source-wide time anchor,
   ItemKNN score's invariance to future appended events, candidate
   scoring alignment, and AUC/LogLoss ties/edge cases.

## Confirmed existing safeguards

- All per-snapshot history features and ItemKNN use `timestamp < cutoff`.
- Candidate generation is the same for all models within a snapshot.
- Training/validation/test one-day label windows are disjoint and
  training model selection only uses training/validation.
- Automated-feature pilot importance computed on training only; subset
  size chosen on validation only.
- Manual vs automatic model fitting parameters are held fixed.
- Output records source dataset revision and sampled-cohort fingerprint.
- All GitHub Actions workflows are manually triggered, with test and
  synthetic gates ahead of downloading a large dataset.

## Major remaining limitations and priorities

- **P0: No validated current HEAD.** Run manual tests and a real 100-user
  integration smoke after these changes. Inspect logs, not just the check mark.
- **P0: No robust full-catalog retrieval.** Candidate coverage previously
  ~21%; add and fairly ablate *ItemKNN retrieval* or ALS/BPR candidates,
  then hold the candidate pool fixed for each ranker comparison.
- **P1: Original scientific hypothesis too weakly isolated.** The
  hand-made 1/7/30/60-day features overlap the automated feature grid,
  and proposed automatic search uses preset grids/feature importance rather
  than autonomous formula discovery. Keep precise language about prior art.
- **P1: Missing event types.** This version uses listens and static artist/
  album mappings, but not `is_organic`, likes/dislikes or audio embeddings
  available in Yambda. Do not claim multi-event feature synthesis.
- **P1: Not enough independent holdout repetitions.** One global last-day
  test and one random cohort seed cannot establish robust improvement.
  Bootstrap intervals measure within-day user sampling only.
- **P1: Missing cost-normalized Pareto and feature ablations by entity
  type.** Report feature computation and model selection costs including
  all trials; compare full 10k-user behavior if feasible.
- **P2: No checkpoint/retry strategy for 10,000-user runs.** Bounded ItemKNN
  graphs and repeated pandas groupbys still need resource profiling.

## Next safe execution

1. Run `.github/workflows/manual-tests.yml` on HEAD, verify pytest
   and synthetic full pipeline. The current HEAD is different from the
   previously successful 1,000-user experiment.
2. Run `.github/workflows/manual-yambda-research.yml` on HEAD for
   **100 users, 120 iterations**, to verify new global cutoff, ItemKNN,
   ROC-AUC/LogLoss and report serialization.
3. If this is successful, run **1,000 users, 120 iterations**, save artifact
   and read the **validation-selected** model plus paired CIs.
4. Add ItemKNN/ALS retrieval and independent holdout days before treating
   any gain as a scientific/competition claim.

Keep real measured results tied to the exact Git SHA, user-cohort hash,
source data revision, candidate generation protocol and time anchor.
