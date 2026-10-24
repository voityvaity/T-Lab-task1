# Preregistered offline protocol

Defined 2026-10-24, **before any measured Yambda run**.

## Scientific question

Can automatic generation and validation-based selection of time-aware aggregate features improve a substantial hand-built CatBoost reference without changing candidates, split, or information access?

## Dataset and processing

- Yambda-50M flat listens file only; the original source provides five-second discretized timestamps and listened percentages.
- Fix random seed and sample complete histories of 100 users for smoke, 1,000 for primary, and optionally all 10,000 for scaling.
- No filter that removes earlier history for sampled users. Only the positive target definition requires played_ratio_pct > 50; all prior listening events may contribute to history features.
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
