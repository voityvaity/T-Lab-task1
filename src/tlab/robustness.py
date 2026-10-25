"""Run the same fixed recommendation protocol at several separated time cutoffs.

This is a *temporal robustness* analysis, not a new untouched final-test
leaderboard. It shares a full-history user cohort and model design across days,
but each cutoff retrains only on events before that prediction time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import BINS_PER_DAY, load_yambda, synthetic_events
from .experiment import run
from .relations import load_mappings


def evaluate_time_lags(
    events,
    *,
    output: Path,
    dataset_mode: str,
    seed: int = 42,
    lags: tuple[int, ...] = (0, 21, 42),
    iterations: int = 120,
    snapshots: int = 4,
    candidate_limit: int = 200,
    threads: int = 4,
    mappings=None,
    anchor_timestamp: int | None = None,
) -> dict:
    if not lags or any(x < 0 or int(x) != x for x in lags) or len(lags) != len(set(lags)):
        raise ValueError("Time lags must be unique nonnegative whole days")
    # Non-overlapping day-long targets, including intervals between the
    # earliest train/validation/test snapshots in each experiment.
    if any(abs(a - b) <= 1 for i, a in enumerate(lags) for b in lags[i + 1:]):
        raise ValueError("Test horizons must be at least two days apart")

    output.mkdir(parents=True, exist_ok=True)
    evaluations = []
    cohort_fingerprint = None
    for lag in sorted(lags):
        result = run(
            events, dataset_mode=dataset_mode, output=output / f"lag-{lag}",
            seed=seed, iterations=iterations, snapshots=snapshots,
            candidate_limit=candidate_limit, threads=threads, mappings=mappings,
            anchor_timestamp=anchor_timestamp, test_lag_days=lag,
        )
        if cohort_fingerprint is None:
            cohort_fingerprint = result["cohort_sha256"]
        elif cohort_fingerprint != result["cohort_sha256"]:
            raise RuntimeError("Cohort changed between time windows")
        summary = result["snapshot_details"]["test"]
        positive_pairs = summary["future_positive_pairs"]
        selected = result["model_selected_by_validation"]
        models = result["models"]
        paired = result["bootstrap_uncertainty"]["paired_deltas"]
        contrast = paired.get(f"{selected}_minus_manual")
        evaluations.append({
            "test_lag_days": lag,
            "test_cutoff": result["cutoffs"]["test"],
            "validation_cutoff": result["cutoffs"]["validation"],
            "cohort_sha256": cohort_fingerprint,
            "evaluated_positive_users": models["manual"]["test"]["evaluated_users"],
            "future_positive_pairs": positive_pairs,
            "retrieved_positive_pairs": summary["retrieved_positive_pairs"],
            "candidate_coverage": summary["retrieved_positive_pairs"] / max(1, positive_pairs),
            "selected_by_validation": selected,
            "selected_test_ndcg@10": models[selected]["test"]["ndcg@10"],
            "manual_test_ndcg@10": models["manual"]["test"]["ndcg@10"],
            "auto_test_ndcg@10": models["auto"]["test"]["ndcg@10"],
            "decay_test_ndcg@10": models["auto_decays_only"]["test"]["ndcg@10"],
            "auto_plus_itemknn_test_ndcg@10": models["auto_plus_itemknn"]["test"]["ndcg@10"],
            "manual_without_itemknn_test_ndcg@10": models["manual_without_itemknn"]["test"]["ndcg@10"],
            "selected_vs_manual_ci95": (
                contrast["ndcg@10"]["ci95"] if contrast is not None else None
            ),
            "manual_vs_without_itemknn_ci95": paired[
                "manual_minus_manual_without_itemknn"
            ]["ndcg@10"]["ci95"],
            "auto_plus_itemknn_vs_auto_ci95": paired[
                "auto_plus_itemknn_minus_auto"
            ]["ndcg@10"]["ci95"],
            "feature_seconds": result["feature_seconds"],
        })

    overview = {
        "dataset_mode": dataset_mode,
        "seed": seed,
        "cohort_sha256": cohort_fingerprint,
        "users": len(events["uid"].unique()),
        "iterations": iterations,
        "candidate_limit": candidate_limit,
        "snapshots": snapshots,
        "lags": list(sorted(lags)),
        "same_users_across_days": True,
        "note": "Exploratory temporal robustness. The main lag=0 horizon was previously inspected; "
                "the same users are reused across dates. No fresh independent dataset or "
                "causal/online A/B test is claimed.",
        "evaluations": evaluations,
    }
    (output / "temporal_robustness.json").write_text(
        json.dumps(overview, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows = [
        "# Temporal robustness and ItemKNN matched-input ablations",
        "",
        "**Exploratory only.** Multiple separated dates, same sampled full-history users,",
        "same source data, candidate generator, training code and CatBoost settings.",
        "The main lag=0 test day was already inspected in earlier experiments,",
        "so do not treat this report as a new untouched confirmation of superiority.",
        "",
        f"- Cohort: {len(events['uid'].unique())} full-history users; seed {seed}",
        f"- Cohort fingerprint: \`{cohort_fingerprint}\`",
        f"- Relative test lags: {list(sorted(lags))} days (Yambda 5-second bins)",
        f"- CatBoost iterations: {iterations}; candidate limit: {candidate_limit}",
        "",
        "## Models: primary seven-way validation-selected result",
        "",
        "| Lag (days) | Test cutoff | Positive users | Coverage | Validation winner | Selected test NDCG@10 | Manual test | Auto test | Decay-only test |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for r in evaluations:
        rows.append(
            f"| {r['test_lag_days']} | {r['test_cutoff']} | "
            f"{r['evaluated_positive_users']} | {r['candidate_coverage']:.2%} | "
            f"{r['selected_by_validation']} | {r['selected_test_ndcg@10']:.5f} | "
            f"{r['manual_test_ndcg@10']:.5f} | {r['auto_test_ndcg@10']:.5f} | "
            f"{r['decay_test_ndcg@10']:.5f} |"
        )
    rows += [
        "",
        "## Fixed-feature-set ItemKNN controls",
        "",
        "Manual with and without the ItemKNN score uses **the same 27 non-ItemKNN",
        "manual features**; auto with and without ItemKNN uses **the same",
        "validation-selected generated feature subset**. All train/validation/",
        "test data, candidate pairs and classifier hyperparameters are identical",
        "within each lag. The two extra models are explanatory, and cannot",
        "retroactively win the main seven-way validation selection.",
        "",
        "| Lag | Manual without ItemKNN | Manual with ItemKNN | Auto without ItemKNN | Auto with ItemKNN |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in evaluations:
        rows.append(
            f"| {r['test_lag_days']} | {r['manual_without_itemknn_test_ndcg@10']:.5f} "
            f"| {r['manual_test_ndcg@10']:.5f} | {r['auto_test_ndcg@10']:.5f} "
            f"| {r['auto_plus_itemknn_test_ndcg@10']:.5f} |"
        )
    rows += [
        "",
        "The per-lag REPORT.md files include *paired* user-bootstrap intervals",
        "for differences with/without ItemKNN and versus manual. Such intervals",
        "only quantify within-day user sampling, not variability over time.",
        "",
        "## Reproduction and constraints",
        "",
        "See run_config.json, metrics.json, feature_selection.json, REPORT.md",
        "in each lag-N subdirectory. Never choose models from their test scores.",
        "Validation is carried out independently per date using strictly",
        "earlier historical events. Test targets do not overlap.",
        "Missing listening is not an observed rejection; no exposure-based CTR.",
        "Yambda sub-samples preserve **all events** of chosen users.",
        "Because the same user sample and past events recur across dates,",
        "these observations are temporally separated, not independent users.",
    ]
    (output / "TEMPORAL_REPORT.md").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )
    return overview


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run temporal CatBoost study at separated cutoffs.")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lags", default="0,21,42", help="Comma-separated earlier test days")
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--snapshots", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/temporal"))
    args = parser.parse_args()
    if args.demo == (args.data is not None):
        parser.error("Supply exactly one of --demo or --data PATH")
    try:
        lags = tuple(int(x.strip()) for x in args.lags.split(","))
    except ValueError:
        parser.error("--lags must contain comma-separated integers")
    if args.iterations < 1 or args.snapshots < 1:
        parser.error("Iterations and snapshots must be positive")
    if args.demo:
        # Long fixture allows full 60-day histories before shifted cutoffs.
        days = max(80, 80 + max(lags, default=0))
        events = synthetic_events(
            n_users=max(4, args.users), days=days, seed=args.seed
        )
    else:
        events = load_yambda(args.data, users=args.users, seed=args.seed)
    mappings = load_mappings(args.metadata_dir) if args.metadata_dir else None
    summary = evaluate_time_lags(
        events, dataset_mode="synthetic" if args.demo else "yambda-50m",
        output=args.output, seed=args.seed, lags=lags, mappings=mappings,
        anchor_timestamp=events.attrs.get("full_dataset_max_timestamp"),
        iterations=args.iterations, snapshots=args.snapshots,
        candidate_limit=args.candidates, threads=args.threads,
    )
    print("Temporal robustness report:", args.output / "TEMPORAL_REPORT.md")
    for r in summary["evaluations"]:
        print(
            f"lag {r['test_lag_days']}d: validation winner "
            f"{r['selected_by_validation']}, test NDCG@10 "
            f"{r['selected_test_ndcg@10']:.5f}, candidate coverage "
            f"{r['candidate_coverage']:.2%}"
        )


if __name__ == "__main__":
    main()
