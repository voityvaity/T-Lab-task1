# Preregistered offline protocol

Defined 2026-10-24, **before any measured Yambda run**.

## Scientific question

Can automatic generation and validation-based selection of time-aware aggregate features improve a substantial hand-built CatBoost reference without changing candidates, split, or information access?

## Dataset and processing

- Yambda-50M flat listens file only; the original source provides five-second discretized timestamps and listened percentages.
- Fix random seed and sample complete histories of 100 users for smoke, 1,000 for primary, and optionally all 10,000 for scaling.
- No filter that removes earlier history for sampled users. The positive target definition follows the original Yambda Listen+ convention (played_ratio_pct > 50); all prior listening events may contribute to history features.
- Distinct future user–item pairs in each one-day horizon are positives. Unobserved candidates are not observed dislikes or exposed negatives.

## Temporal evaluation

Use global max timestamp M measured in five-second bins:
- Test cutoff = M − 1 day
- Validation cutoff = M − 3 days
- Four training cutoffs = validation cutoff − 12, 9, 6, 3 days, respectively
- Each snapshot's label horizon is exactly one day [cutoff, cutoff + day)
- Thus labels of adjacent snapshots are separated in time; all event features require timestamp < that snapshot's cutoff.
- **Data at validation time may be available in test inference history**, just as in real time, but never in model fitting or automatic feature choice.

These cutoffs are not identical to Yambda's official benchmark, and result numbers must not be compared directly to the official published baselines.

## Retrieval consistency

For each snapshot, candidate items are chosen by the same time-causal rule for all rankers:
- 1/3 global historical popularity
- 1/3 user's recent distinct items
- 1/3 transitions within past 14 days; transitions require neighboring listens for the same user within one hour

Pool length may be smaller than the requested limit when retrieved items overlap. Candidate retrieval itself is a major cap on attainable ranking metrics and a possible failure point.

## Experiments and fair comparison

1. Historical-popularity ranking baseline, as-of cutoff.
2. CatBoost classifier with **23 manually fixed** aggregate / recency / completion / diversity / trend / indirect-stratum features.
3. CatBoost over generated 1-, 3-, 14-, 60-day counts and 2-, 7-, 21-day decays over user, item and user-item groups. Feature importance estimated **only on training**. The best number among 4, 8, 12, all is chosen on validation.
4. CatBoost with manual + selected automatic features.

All CatBoost models use the same optimizer settings and training snapshots. The auto search performs more total fits; runtime must include *all* selection trials.

NDCG@10 and Recall@10 are computed per user with at least one future positive event. Ideal relevance for each user includes positive tracks that were **not retrieved**: don't inflate offline recall by evaluating only accessible positives.

Track test retrieval coverage across all future positives separately. Record user count, random seed, cutoffs, dependency versions, feature computation time, model training time, and top generated features.

## Candidate ablations / negative results

After collecting the first valid run, potentially compare count-only versus decay-only feature banks, then evaluate bootstrap intervals over test users. Also compare candidate retrieval with/without transitions and add ItemKNN/ALS retrieval if coverage is weak. Those follow-ons are not yet implemented and must not be presented as tested.

## Known threats to validity

- Exposure bias: not interacting with a track does not prove it was shown or rejected.
- Subsampling users changes catalog popularity and item overlap; results need stability across sizes/seeds.
- Temporal distribution shift: fixed last-day testing may be atypical.
- Tuning after inspecting the test scores invalidates a clean final estimate; use new untouched test horizon when revising hypotheses.
- Restricting candidates to historical/repeated/transition tracks hurts cold-item discovery. Include coverage in every report.
- CPU cost of recomputing many group-bys can dominate model fit; do not claim the method is production-ready.

## Source and provenance

This protocol is inspired by Featuretools Deep Feature Synthesis, the Yambda benchmark, RecSys 2025 Time to Split, and published two-stage recommender approaches. Full primary-source links: [RELATED_WORK.md](RELATED_WORK.md).


## Relation-feature audit amendment, 2026-10-24

The original baseline omitted Yambda's static track–artist and track–album lookup tables. They exist and are available separately. The implementation now supports optional `--metadata-dir data`:

- Manual additional columns: 7/30-day user–artist and user–album listen counts (4).
- Auto additional candidates: 1/3/14/60-day user–artist and user–album listen counts (8).
- Items with multiple artists retain all distinct relation pairs. Users' aggregate artist affinity is summed across the mapped artists of a candidate item; document that this may reward large collaborations.
- Those relations are static; **aggregate interaction histories are still restricted to timestamp < cutoff**.
- The optional metadata mode must be turned on or off consistently for all competing models in one run.

**Newly discovered limitation:** Feature generation repeatedly joins event histories against relation tables at every cutoff. This may use substantial peak memory and must be profiled before claiming scalable performance. Also, 27 manual signals are stronger than 23, but still not equivalent to a top industrial ranking system with collaborative retrieval/latent factor features.

A proper research result still requires measured full-dataset or large-cohort NDCG and recall, computation costs, and uncertainty estimates.


## Post-first-run revision (25 October 2026; new experimental protocol)

The exploratory run [40174763195](https://github.com/voityvaity/T-Lab-task1/actions/runs/40174763195) at commit `a8a59d2` revealed low candidate retrieval coverage and only 14 test-positive users in the 100-user sample. Its test results are descriptive only; the auto-versus-manual NDCG difference is too small for a confident conclusion.

The *new* research protocol changes:
- Retain fixed historical popularity, repeat and 14-day transition sources but backfill de-duplicated candidate slots up to the 200-item limit with historically popular tracks. Retrieval still uses only data strictly before each cutoff. Because the candidate pool changes, new NDCG/Recall scores **must not be presented as directly comparable** with the old 100-user run.
- Use efficient 60-day bounded event joins filtered to artists/albums attached to candidate items; results are checked against a naïve full-history reference for every relevant time window.
- Expand automatic feature candidates to `{1,3,7,10,14,30,60}`-day counts and `{2,5,7,10,21,30}`-day exponentially decayed histories over user/item/user-item scopes (39), plus 8 relation-based user–artist/user–album count windows (47 generated columns in metadata mode). Manual feature set remains at 27.
- Rank the 47 generated columns using train-only importance, then choose best `{4,8,12,20,32,47}` based on *validation-only* NDCG@10.
- Independently train additional count-only and decay-only CatBoost ablations using exactly the same candidates/splits/parameters; test these only after selection.
- Report a **paired 1,000-resample user bootstrap** for the final test rankings and auto/hybrid-minus-manual differences. This quantifies only which users were sampled, not the uncertainty caused by a different future day or candidate set; do not claim significance with 14 positive users.
- Capture peak resident memory during real experiments using `/usr/bin/time -v` and per-stage computation times.

**Research integrity:** The 100-user test was examined before this revision, so the revised experiment is a new exploratory protocol. For a final unbiased estimate, reserve an untouched evaluation horizon or independent data and do not choose further configurations by looking at its test values. A strong ItemKNN/ALS collaborative baseline, alternative seed cohorts and multiple time-split evaluations still need to be implemented for a high-quality submission.

Primary sources:
- Yambda dataset: https://huggingface.co/datasets/yandex/yambda
- Featuretools point-in-time cutoff correctness: https://docs.featuretools.com/en/stable/getting_started/handling_time.html
- Time to Split (RecSys 2025): https://doi.org/10.1145/3705328.3748164
- Known time-decay recommender prior art: https://doi.org/10.1145/3758126.3758138
