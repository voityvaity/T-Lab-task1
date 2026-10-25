"""End-to-end temporal feature experiment. Run: python -m tlab.experiment --help"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd

from .core import AUTO_COLUMNS, MANUAL_COLUMNS, global_cutoffs, make_snapshot
from .data import BINS_PER_DAY, YAMBDA_REVISION, load_yambda, synthetic_events
from .metrics import future_positive_counts, ranking_metrics
from .uncertainty import paired_bootstrap
from .relations import (
    AUTO_RELATION_COLUMNS, MANUAL_RELATION_COLUMNS,
    add_relation_features, load_mappings,
)


def train_model(frame: pd.DataFrame, columns: list[str], *, iterations: int, seed: int, threads: int):
    model = catboost.CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.06,
        loss_function="Logloss",
        random_seed=seed,
        thread_count=threads,
        verbose=False,
        allow_writing_files=False,
    )
    if frame["label"].nunique() < 2:
        raise ValueError("Train labels need both positive and unobserved candidates; increase user/sample size")
    started = time.perf_counter()
    model.fit(frame[columns], frame["label"])
    return model, time.perf_counter() - started


def _evaluate(model, frame: pd.DataFrame, columns: list[str], counts: pd.Series) -> dict:
    scores = model.predict_proba(frame[columns])[:, 1]
    return ranking_metrics(frame, scores, positive_counts=counts)


def _report(metrics: dict, selector: dict) -> str:
    lines = [
        "# Experiment report",
        "",
        "**Measured output of this run, not a published leaderboard result.**",
        "",
        f"- Dataset mode: {metrics['dataset_mode']}",
        f"- Sampled users: {metrics['users']}",
        f"- Seed: {metrics['seed']}",
        f"- Cutoffs (Yambda 5-second bins): {metrics['cutoffs']}",
        "- Positive: strictly >50% played in one day after the cutoff (Yambda Listen+).",
        "- Unobserved candidates are not proven dislikes or impressions.",
        "",
        "## Candidate coverage",
        "",
        "| Split | Future positive pairs | Retrieved positive pairs | Coverage |",
        "|---|---:|---:|---:|",
    ]
    for name in ("validation", "test"):
        s = metrics["snapshot_details"][name]
        cov = s["retrieved_positive_pairs"] / max(1, s["future_positive_pairs"])
        lines.append(f"| {name} | {s['future_positive_pairs']} | {s['retrieved_positive_pairs']} | {cov:.4f} |")
    lines += [
        "",
        "## Model comparison",
        "",
        "| Model | Validation NDCG@10 | Test NDCG@10 | Test Recall@10 | Fit seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in metrics["models"].items():
        lines.append(
            f"| {name} | {result['validation']['ndcg@10']:.5f} | "
            f"{result['test']['ndcg@10']:.5f} | {result['test']['recall@10']:.5f} | "
            f"{result['train_seconds']:.1f} |"
        )
    lines += [
        "",
        "## User-level bootstrap uncertainty (held-out test)",
        "",
        "These 95% intervals resample the evaluated test users; they do not measure",
        "uncertainty from different future days, cohorts, retrieval strategies or model tuning.",
        "",
        "| Model | Test NDCG@10 | 95% CI |",
        "|---|---:|---:|",
    ]
    for name, model_summary in metrics["bootstrap_uncertainty"]["models"].items():
        stat = model_summary["ndcg@10"]
        lo, hi = stat["ci95"]
        lines.append(f"| {name} | {stat['estimate']:.5f} | [{lo:.5f}, {hi:.5f}] |")
    lines += ["", "| Paired comparison | NDCG@10 difference | 95% CI |", "|---|---:|---:|"]
    for name, model_summary in metrics["bootstrap_uncertainty"]["paired_deltas"].items():
        stat = model_summary["ndcg@10"]
        lo, hi = stat["ci95"]
        lines.append(f"| {name} | {stat['estimate']:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    lines += [
        "",
        "## Automatic feature selection",
        "",
        f"Selected {len(selector['selected_features'])} features via validation NDCG@10.",
        "",
        "Selected columns: " + ", ".join(selector["selected_features"]),
        "",
        "Selection decisions were made **only** using training importance and validation scores.",
        "The test split was not used to search feature subsets.",
        "",
        "## Limitations and next actions",
        "",
        "- Candidate generation uses popularity, repeats and historical transitions; it is not exhaustive retrieval.",
        "- No impression logs: results cannot be interpreted as calibrated CTR.",
        "- Compare to stronger candidate retrieval, try alternative seeds and cohort sizes.",
        "- Observe both feature creation and model training costs.",
        "",
        f"Total feature computation seconds: {metrics['feature_seconds']:.1f}",
        "",
        "## Computation costs by stage",
        "",
        "| Stage | Aggregate seconds across snapshots |",
        "|---|---:|",
        *[
            f"| {name} | {elapsed:.3f} |"
            for name, elapsed in metrics["feature_cost_seconds"].items()
        ],
    ]
    return "\n".join(lines) + "\n"


def run(
    events: pd.DataFrame, *,
    dataset_mode: str, output: Path,
    seed: int = 42, iterations: int = 200,
    snapshots: int = 4, candidate_limit: int = 200, threads: int = 4,
    mappings: dict[str, pd.DataFrame] | None = None,
) -> dict:
    if events.empty:
        raise ValueError("Events are empty")
    users = np.sort(events["uid"].unique())
    train_cuts, valid_cut, test_cut = global_cutoffs(events, snapshots=snapshots)
    cohort_sha256 = hashlib.sha256(users.astype("<i8", copy=False).tobytes()).hexdigest()
    manual_columns = MANUAL_COLUMNS + (MANUAL_RELATION_COLUMNS if mappings is not None else [])
    auto_columns = AUTO_COLUMNS + (AUTO_RELATION_COLUMNS if mappings is not None else [])

    def snapshot(cutoff: int):
        frame, info = make_snapshot(events, cutoff, users, candidate_limit=candidate_limit)
        if mappings is not None:
            start_relations = time.perf_counter()
            frame = add_relation_features(frame, events, cutoff, mappings)
            info["timings_seconds"]["relation_features"] = round(
                time.perf_counter() - start_relations, 6
            )
        return frame, info

    details = {}
    train_parts = []
    feature_secs = 0.0
    for i, cut in enumerate(train_cuts):
        start = time.perf_counter()
        frame, detail = snapshot(cut)
        feature_secs += time.perf_counter() - start
        train_parts.append(frame)
        details[f"train_{i}"] = detail
    train = pd.concat(train_parts, ignore_index=True)
    del train_parts
    if train["label"].nunique() < 2:
        raise ValueError(
            "The sampled training candidates contain only one label class. "
            "Increase --users or candidate coverage; do not report this as model quality."
        )
    start = time.perf_counter()
    valid, details["validation"] = snapshot(valid_cut)
    feature_secs += time.perf_counter() - start
    start = time.perf_counter()
    test, details["test"] = snapshot(test_cut)
    feature_secs += time.perf_counter() - start

    val_counts = future_positive_counts(events, valid_cut, BINS_PER_DAY)
    test_counts = future_positive_counts(events, test_cut, BINS_PER_DAY)
    if val_counts.empty or test_counts.empty:
        raise ValueError(
            "Validation/test contains no qualifying future listens; "
            "increase the sampled user count or revise the held-out horizon."
        )

    # A non-personalized yet realistic historical popularity baseline.
    pop_score_val = valid["item_count_7d"].to_numpy(dtype=float)
    pop_score_test = test["item_count_7d"].to_numpy(dtype=float)
    models = {
        "popularity": {
            "validation": ranking_metrics(valid, pop_score_val, positive_counts=val_counts),
            "test": None,
            "train_seconds": 0.0,
            "features": ["item_count_7d"],
        }
    }
    manual, elapsed = train_model(train, manual_columns, iterations=iterations, seed=seed, threads=threads)
    models["manual"] = {
        "validation": _evaluate(manual, valid, manual_columns, val_counts),
        "test": None, "train_seconds": elapsed, "features": manual_columns,
    }

    # Step 1: rank the generated features by train-only CatBoost importance.
    pilot, pilot_seconds = train_model(train, auto_columns, iterations=iterations, seed=seed, threads=threads)
    importances = sorted(
        zip(auto_columns, pilot.get_feature_importance(), strict=True),
        key=lambda pair: (-float(pair[1]), pair[0]),
    )
    ordered = [x[0] for x in importances]
    # Step 2: choose feature *count* using validation, never test.
    sizes = sorted(set(min(x, len(ordered)) for x in (4, 8, 12, len(ordered))))
    trials = []
    selected_cols = None
    selected_model = None
    trial_training_seconds = 0.0
    best_score = -1.0
    for size in sizes:
        columns = ordered[:size]
        model, fit_secs = train_model(train, columns, iterations=iterations, seed=seed, threads=threads)
        trial_training_seconds += fit_secs
        score = _evaluate(model, valid, columns, val_counts)["ndcg@10"]
        trials.append({"features": size, "validation_ndcg@10": score, "fit_seconds": fit_secs})
        if score > best_score:
            best_score = score
            selected_cols = columns
            selected_model = model
    assert selected_cols is not None and selected_model is not None
    models["auto"] = {
        "validation": _evaluate(selected_model, valid, selected_cols, val_counts),
        "test": None, "train_seconds": pilot_seconds + trial_training_seconds,
        "features": selected_cols,
    }
    hybrid_columns = list(dict.fromkeys(manual_columns + selected_cols))
    hybrid, hybrid_secs = train_model(
        train, hybrid_columns, iterations=iterations, seed=seed, threads=threads
    )
    models["manual_plus_auto"] = {
        "validation": _evaluate(hybrid, valid, hybrid_columns, val_counts),
        "test": None, "train_seconds": hybrid_secs, "features": hybrid_columns,
    }

    # Preregistered ablations: what do fixed-window counts and temporal decays
    # contribute independently? Never choose hyperparameters from the test.
    count_only_columns = [name for name in auto_columns if "_count_" in name]
    decay_only_columns = [name for name in auto_columns if "_decay_" in name]
    count_model, count_seconds = train_model(
        train, count_only_columns, iterations=iterations, seed=seed, threads=threads
    )
    decay_model, decay_seconds = train_model(
        train, decay_only_columns, iterations=iterations, seed=seed, threads=threads
    )
    models["auto_counts_only"] = {
        "validation": _evaluate(count_model, valid, count_only_columns, val_counts),
        "test": None, "train_seconds": count_seconds, "features": count_only_columns,
    }
    models["auto_decays_only"] = {
        "validation": _evaluate(decay_model, valid, decay_only_columns, val_counts),
        "test": None, "train_seconds": decay_seconds, "features": decay_only_columns,
    }

    # All selection is now complete; access held-out test metrics only here.
    models["popularity"]["test"] = ranking_metrics(test, pop_score_test, positive_counts=test_counts)
    models["manual"]["test"] = _evaluate(manual, test, manual_columns, test_counts)
    models["auto"]["test"] = _evaluate(selected_model, test, selected_cols, test_counts)
    models["manual_plus_auto"]["test"] = _evaluate(hybrid, test, hybrid_columns, test_counts)
    models["auto_counts_only"]["test"] = _evaluate(
        count_model, test, count_only_columns, test_counts
    )
    models["auto_decays_only"]["test"] = _evaluate(
        decay_model, test, decay_only_columns, test_counts
    )

    # Final-test analysis only: the bootstrap never selects features or models.
    confidence = paired_bootstrap(
        test,
        {
            "popularity": pop_score_test,
            "manual": manual.predict_proba(test[manual_columns])[:, 1],
            "auto": selected_model.predict_proba(test[selected_cols])[:, 1],
            "manual_plus_auto": hybrid.predict_proba(test[hybrid_columns])[:, 1],
            "auto_counts_only": count_model.predict_proba(test[count_only_columns])[:, 1],
            "auto_decays_only": decay_model.predict_proba(test[decay_only_columns])[:, 1],
        },
        test_counts,
        seed=seed,
    )

    selection = {
        "importance_train_only": [{"name": name, "importance": float(imp)} for name, imp in importances],
        "trials_validation_only": trials,
        "selected_features": selected_cols,
        "selection_metric": "validation_ndcg@10",
    }
    result = {
        "dataset_mode": dataset_mode,
        "using_artist_album_metadata": mappings is not None,
        "seed": seed,
        "users": len(users),
        "cohort_sha256": cohort_sha256,
        "yambda_revision": YAMBDA_REVISION if dataset_mode == "yambda-50m" else None,
        "cutoffs": {"train": train_cuts, "validation": valid_cut, "test": test_cut},
        "snapshot_details": details,
        "feature_seconds": round(feature_secs, 3),
        "bootstrap_uncertainty": confidence,
        "feature_cost_seconds": {
            name: round(sum(x["timings_seconds"][name] for x in details.values()), 4)
            for name in ("candidate_retrieval", "manual_features", "auto_features", "relation_features")
        },
        "models": models,
    }
    config = {
        "dataset_mode": dataset_mode, "seed": seed, "users": len(users),
        "cohort_sha256": cohort_sha256,
        "yambda_revision": YAMBDA_REVISION if dataset_mode == "yambda-50m" else None,
        "iterations": iterations, "snapshots": snapshots, "candidate_limit": candidate_limit,
        "threads": threads, "using_artist_album_metadata": mappings is not None,
        "python": platform.python_version(),
        "pandas": pd.__version__, "numpy": np.__version__, "catboost": catboost.__version__,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "feature_selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "REPORT.md").write_text(_report(result, selection), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Temporal feature study on synthetic or real Yambda data.")
    p.add_argument("--data", type=Path, default=None, help="Yambda flat/50m/listens.parquet")
    p.add_argument("--demo", action="store_true", help="Synthetic smoke run, not research")
    p.add_argument("--users", type=int, default=100, help="Sampled users, 0 means all (real dataset)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--snapshots", type=int, default=4)
    p.add_argument("--candidates", type=int, default=200)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("results/run"))
    p.add_argument("--metadata-dir", type=Path, default=None,
                   help="Optional directory containing artist_item_mapping.parquet and album_item_mapping.parquet")
    args = p.parse_args()
    if args.demo == (args.data is not None):
        p.error("Supply exactly one of --data PATH or --demo")
    if args.snapshots < 1 or args.iterations < 1:
        p.error("--snapshots and --iterations must be positive")
    events = synthetic_events(n_users=max(4, args.users), seed=args.seed) if args.demo else load_yambda(
        args.data, users=args.users, seed=args.seed
    )
    mappings = load_mappings(args.metadata_dir) if args.metadata_dir is not None else None
    results = run(
        events, mappings=mappings, dataset_mode="synthetic" if args.demo else "yambda-50m",
        output=args.output, seed=args.seed, snapshots=args.snapshots,
        candidate_limit=args.candidates, iterations=args.iterations, threads=args.threads,
    )
    print(f"Wrote measured run outputs to {args.output}")
    for name, metrics in results["models"].items():
        print(f"{name}: test NDCG@10={metrics['test']['ndcg@10']:.5f}, "
              f"Recall@10={metrics['test']['recall@10']:.5f}")


if __name__ == "__main__":
    main()
