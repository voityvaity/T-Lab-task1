# Related work and design decisions

Checked 2026-10-24. These are **external published methods**, not experiments we claim to have reproduced.

1. **Featuretools / Deep Feature Synthesis**. Automatic feature construction across time-indexed relational tables. Important prior art: feature generation itself is not novel. https://docs.featuretools.com/en/latest/ and https://docs.featuretools.com/en/stable/getting_started/afe.html

2. **Yandex Yambda**. Official multi-scale recommendation dataset with temporal ordering and baseline implementations (ALS, BPR, SASRec). This project uses only the Yambda-50M listens event file; it does not claim to reproduce the official baseline methodology. Dataset: https://huggingface.co/datasets/yandex/yambda and baseline guide: https://huggingface.co/datasets/yandex/yambda/blob/main/benchmarks/README.md . Paper: https://arxiv.org/abs/2505.22238

3. **Gusak et al., Time to Split (ACM RecSys 2025)**. Shows why global temporal evaluation design and target selection can change apparent model ranking. https://arxiv.org/abs/2507.16289 ; code: https://github.com/monkey0head/time-to-split

4. **Le, Liu & Medlar, Don't Get Ahead of Yourself (ACM RecSys 2025)**. Studies train/test look-ahead biases in sequential recommender evaluation. Published abstract: https://researchportal.helsinki.fi/en/publications/dont-get-ahead-of-yourself-a-critical-study-on-data-leakage-in-of/

5. **RecSys challenge winning approaches**. Boosted trees remain a competitive ranking component; see the official challenge winners catalogue. https://recsys.acm.org/challenges/

6. **RecBole**. General purpose framework for recommendation model baselines, useful for follow-on SASRec/ItemKNN benchmarking. https://recbole.io/docs/

## Distinction from related work

Our intentionally modest research hypothesis: *do programmatically generated and validation-selected recency features improve a strong fixed-window CatBoost baseline at equal candidate exposure and cutoff times?* We measure retrieval coverage and feature computation cost as well as rank metrics.

We do **not** claim a published leaderboard score, a novel deep architecture, or a finished empirical conclusion. We will record both improvements and negative outcomes after running the actual data.

## Important limitations

- Negative candidates were not shown to users in a controlled experiment; absence of listen is not a verified dislike.
- Yambda-50M flat listen records have user/item/time/listen fields; this file does **not** have artist or genre fields.
- Sampling only certain users preserves their histories, but changes estimated population popularity and item overlap. Document the seed and selected count.
- This repo uses a different protocol from the official Yambda Global Temporal Split; numbers are not directly comparable.

7. **Rekko Challenge, Tinkoff/T-Bank team (2019)**. A competition solution combining item signals, collaborative scores and a second-stage gradient boosted ranker; useful prior art for two-stage retrieval/ranking and stronger ablations. Article with source-code references: https://habr.com/ru/companies/tbank/articles/454818/ . This is *not* a solution to the 2026 selection assignment.

8. **RecSys Challenge 2024, FeatureSalad**. Public example of CatBoost/LightGBM ranking and stacking; substantial prior art for ranker ensembles. https://github.com/remaplab/recsys-challenge-2024-ekstrabladet . Different task and evaluation; scores are not portable.


## Additional primary-source verification (2026-10-24)

9. **Yambda original schema and Listen+ definition:** Official Yambda dataset explicitly provides static `artist_item_mapping.parquet` (`artist_id,item_id`) and `album_item_mapping.parquet` (`album_id,item_id`) at its repository root, in addition to listening events. "Listen+" means **strictly more than 50% listened**, not 30%. We corrected the threshold and implemented optional point-in-time user–artist/user–album features. Source: https://huggingface.co/datasets/yandex/yambda/blob/main/README.md and the primary paper https://arxiv.org/abs/2505.22238 .

10. **RecSys Challenge 2025, 4th place:** *Blending Sequential Embeddings, Graphs, and Engineered Features.* Hundreds of manual feature aggregations, several time windows, and exponentially weighted history were already used competitively. Our simple exponential decay is **prior art, not an original invention**. The contribution to test is whether restricted, validation-selected automatically generated temporal features can compete with strong manual counterparts at measured cost. https://doi.org/10.1145/3758126.3758131

11. **RecSys Challenge 2025, Triple-Feature Transformer:** Prior successful use of 1/7/30/60/90-day windows and decay for user behaviour. https://doi.org/10.1145/3758126.3758138

12. **Apple Fortress (July 2026):** Temporal prediction instability can justify measuring feature stability across different time intervals and pruning volatile features. This is a *possible next experiment*, not something already implemented. https://machinelearning.apple.com/research/fortress-stabilizing-search-recommendations
