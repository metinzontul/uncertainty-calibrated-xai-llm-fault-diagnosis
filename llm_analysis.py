from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
from llm import pearson_correlation, compute_ers, CONDITION_LABELS, CONDITIONS, DiagnosisPayload
from stat_utils import (
    calibration_metrics,
    cluster_bootstrap_ci,
    cohens_d,
    cohens_h,
    format_ci,
    format_p,
    odds_ratio,
    paired_permutation_test,
    risk_coverage_curve,
    selective_risk_at_coverage,
)
from paths import OUT_DIR

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
DEFAULT_RESULTS = str(OUT_DIR / "llm_revision" / "llm_final_consolidated" / "final_results.csv")
OUTPUT_DIR = str(OUT_DIR / "figures")
METRICS = [
    ("top1_faithfulness", "Evidence Adherence@1"),
    ("faithfulness", "Recall faithfulness"),
    ("rank_correlation", "Rank correlation"),
    ("hallucination_rate", "Hallucination rate"),
]


def add_tables_arguments(parser):
    parser.add_argument("--results", type=str, default=DEFAULT_RESULTS)
    parser.add_argument("--combine", nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--no-figure", action="store_true")


def load_results(args):
    if args.combine:
        paths = []
        for pattern in args.combine:
            paths.extend(sorted(glob.glob(pattern)))
        if not paths:
            raise FileNotFoundError(f"No files matched: {args.combine}")
        frames = [pd.read_csv(p) for p in paths]
        print(f"Combined {len(paths)} result files.")
        return pd.concat(frames, ignore_index=True)
    if not os.path.exists(args.results):
        raise FileNotFoundError(
            f"Results not found at '{args.results}'.\nRun the experiment first, e.g.:\n  python llm_experiment.py run --models gpt5_mini qwen35_9b"
        )
    return pd.read_csv(args.results)


def _format_mean_sd(series):
    clean = series.dropna()
    if clean.empty:
        return "n/a"
    sd = clean.std()
    return f"{clean.mean():.3f} ± {(0.0 if pd.isna(sd) else sd):.3f}"


def report_condition_ablation(df):
    print("\n" + "=" * 78)
    print("TABLE A - EXPLANATION QUALITY BY GROUNDING CONDITION")
    print("=" * 78)
    answered = df[df["abstained"] == 0]
    rows = []
    for condition in CONDITIONS:
        subset = answered[answered["condition"] == condition]
        if subset.empty:
            continue
        all_subset = df[df["condition"] == condition]
        rows.append(
            {
                "Condition": CONDITION_LABELS[condition],
                "n": len(subset),
                "Faithfulness": _format_mean_sd(subset["faithfulness"]),
                "Hallucination": _format_mean_sd(subset["hallucination_rate"]),
                "Rank corr.": _format_mean_sd(subset["rank_correlation"]),
                "Fault consist.": _format_mean_sd(subset["fault_consistency"]),
                "Abstained": f"{int(all_subset['abstained'].sum())}/{len(all_subset)}",
            }
        )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    return table


def report_model_comparison(df):
    print("\n" + "=" * 78)
    print("TABLE B - EXPLANATION QUALITY BY MODEL")
    print("=" * 78)
    answered = df[df["abstained"] == 0]
    rows = []
    for model_name, subset in answered.groupby("model_name"):
        rows.append(
            {
                "Model": model_name,
                "n": len(subset),
                "Faithfulness": _format_mean_sd(subset["faithfulness"]),
                "Hallucination": _format_mean_sd(subset["hallucination_rate"]),
                "Fault consist.": _format_mean_sd(subset["fault_consistency"]),
                "Mean latency (s)": f"{subset['latency_s'].mean():.2f}",
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        print("No answered responses to summarise.")
        return table
    table = table.sort_values("Model")
    print(table.to_string(index=False))
    print("\n--- Model x Condition (faithfulness mean) ---")
    pivot = answered.pivot_table(
        index="model_name", columns="condition", values="faithfulness", aggfunc="mean"
    )
    ordered = [c for c in CONDITIONS if c in pivot.columns]
    print(pivot[ordered].round(3).to_string())
    return table


def report_degradation_sweep(df):
    if "degradation" not in df.columns:
        return None
    settings = df.groupby(["degradation", "degradation_level"]).size()
    if len(settings) < 2:
        print("\nTable C skipped: results contain a single degradation setting.")
        return None
    print("\n" + "=" * 78)
    print("TABLE C - DEGRADATION SWEEP: CALIBRATION -> EXPLANATION RELIABILITY")
    print("=" * 78)
    answered = df[df["abstained"] == 0]
    rows = []
    for (deg, level), subset in answered.groupby(["degradation", "degradation_level"]):
        rows.append(
            {
                "Degradation": deg,
                "Level": level,
                "n": len(subset),
                "Data quality": f"{subset['data_quality'].mean():.3f}",
                "Cal. confidence": f"{subset['calibrated_confidence'].mean():.3f}",
                "Pred. accuracy": f"{subset['prediction_correct'].mean():.3f}",
                "ERS": f"{subset['ers'].mean():.3f}",
                "Faithfulness": f"{subset['faithfulness'].mean():.3f}",
                "Hallucination": f"{subset['hallucination_rate'].mean():.3f}",
            }
        )
    table = pd.DataFrame(rows).sort_values(["Degradation", "Level"])
    print(table.to_string(index=False))
    return table


def report_ers_validity(df):
    print("\n" + "=" * 78)
    print("TABLE D - DOES ERS PREDICT EXPLANATION FAITHFULNESS?")
    print("=" * 78)
    answered = df[(df["abstained"] == 0) & df["ers"].notna()]
    if len(answered) < 3:
        print("Not enough answered cases to correlate ERS with faithfulness.")
        return
    overall = pearson_correlation(answered["ers"].tolist(), answered["faithfulness"].tolist())
    print(f"Overall Pearson r (ERS vs faithfulness): {overall:+.4f}  (n={len(answered)})")
    print("\nPer condition:")
    for condition in CONDITIONS:
        subset = answered[answered["condition"] == condition]
        if len(subset) < 3:
            continue
        r = pearson_correlation(subset["ers"].tolist(), subset["faithfulness"].tolist())
        print(f"  {CONDITION_LABELS[condition]:<55} r = {r:+.4f} (n={len(subset)})")
    print("\n--- Prediction reliability vs explanation reliability ---")
    conf_vs_faith = pearson_correlation(
        answered["calibrated_confidence"].tolist(), answered["faithfulness"].tolist()
    )
    print(
        f"Calibrated confidence vs faithfulness: r = {conf_vs_faith:+.4f}\n  (A weak correlation supports the study's core argument: a confident prediction does not imply a trustworthy explanation.)"
    )
    correct = answered[answered["prediction_correct"] == 1]
    wrong = answered[answered["prediction_correct"] == 0]
    if not correct.empty and (not wrong.empty):
        print(
            f"\nFaithfulness when the prediction was correct  : {correct['faithfulness'].mean():.3f} (n={len(correct)})"
        )
        print(
            f"Faithfulness when the prediction was incorrect: {wrong['faithfulness'].mean():.3f} (n={len(wrong)})"
        )


def plot_condition_figure(df, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("default")
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "figure.autolayout": True,
        }
    )
    answered = df[df["abstained"] == 0]
    conditions = [c for c in CONDITIONS if c in answered["condition"].unique()]
    if not conditions:
        print("Condition figure skipped: no answered responses.")
        return
    metric_keys = [k for k, _ in METRICS]
    metric_labels = [label for _, label in METRICS]
    means = np.zeros((len(conditions), len(metric_keys)))
    errs = np.zeros_like(means)
    for i, condition in enumerate(conditions):
        subset = answered[answered["condition"] == condition]
        for j, key in enumerate(metric_keys):
            values = subset[key].dropna()
            means[i, j] = values.mean() if not values.empty else 0.0
            sd = values.std() if len(values) > 1 else 0.0
            errs[i, j] = 0.0 if pd.isna(sd) else sd
    x = np.arange(len(metric_keys))
    width = 0.8 / len(conditions)
    colors = ["#7f8c8d", "#3498db", "#2ecc71"]
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, condition in enumerate(conditions):
        offset = (i - (len(conditions) - 1) / 2) * width
        ax.bar(
            x + offset,
            means[i],
            width,
            yerr=errs[i],
            capsize=5,
            label=CONDITION_LABELS[condition],
            color=colors[i % len(colors)],
            edgecolor="black",
            alpha=0.88,
            error_kw=dict(lw=1.5, capthick=1.5),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Explanation Alignment by Grounding Condition", fontweight="bold", pad=20)
    ax.legend(loc="upper right", fontsize=10, frameon=True)
    ax.axhline(0, color="black", linewidth=0.8)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "Figure4_Explanation_Alignment.png")
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"\nCondition figure saved at 600 DPI: {path}")


def tables_command(args):
    try:
        df = load_results(args)
    except FileNotFoundError as exc:
        print(f"{exc}")
        sys.exit(1)
    synthetic = (
        df["model"].astype(str).str.contains("mock").any()
        or df["case_id"].astype(str).str.startswith("SYNTHETIC").any()
    )
    print(
        f"Loaded {len(df)} records ({df['model_name'].nunique()} models, {df['condition'].nunique()} conditions)."
    )
    if synthetic:
        print("\nThese records are SYNTHETIC (mock backend and/or mock cases).")
    error_text = df["error"].fillna("").astype(str).str.strip()
    failed = error_text.str.len() > 0
    if failed.any():
        print(f"{int(failed.sum())} generations failed; excluded from scoring.")
        df = df[~failed]
    report_condition_ablation(df)
    report_model_comparison(df)
    report_degradation_sweep(df)
    report_ers_validity(df)
    if not args.no_figure:
        plot_condition_figure(df, args.output_dir)
    if synthetic:
        print(
            "\n[REMINDER] Synthetic run - re-run with real models and real TEP cases before using any of these numbers."
        )


OUTPUT = os.path.join(OUT_DIR, "statistical_results.json")


def load_result_files(pattern):
    files = [f for f in glob.glob(os.path.join(OUT_DIR, pattern)) if "SYNTHETIC" not in f]
    if not files:
        return pd.DataFrame()
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return frame[frame["error"].fillna("").astype(str).str.strip().str.len() == 0]


def load_json(path):
    full = os.path.join(OUT_DIR, path)
    if not os.path.exists(full):
        return {}
    with open(full, "r", encoding="utf-8") as fh:
        return json.load(fh)


def analyse_grounding(dataset, frame):
    answered = frame[frame.abstained == 0]
    output = {"conditions": {}, "comparisons": {}}
    for condition in ("plain", "xai_grounded", "xai_uncertainty_grounded"):
        subset = answered[answered.condition == condition]
        if subset.empty:
            continue
        entry = {}
        for metric in ("top1_faithfulness", "faithfulness", "rank_correlation"):
            point, low, high = cluster_bootstrap_ci(subset, lambda d, m=metric: float(d[m].mean()))
            entry[metric] = {"point": point, "ci_low": low, "ci_high": high}
        entry["n"] = len(subset)
        output["conditions"][condition] = entry
    plain = answered[answered.condition == "plain"]
    grounded = answered[answered.condition == "xai_grounded"]
    if not plain.empty and (not grounded.empty):
        p1 = float(grounded.top1_faithfulness.mean())
        p0 = float(plain.top1_faithfulness.mean())
        both = answered[answered.condition.isin(["plain", "xai_grounded"])]
        delta, delta_low, delta_high = cluster_bootstrap_ci(
            both,
            lambda d: float(
                d[d.condition == "xai_grounded"].top1_faithfulness.mean()
                - d[d.condition == "plain"].top1_faithfulness.mean()
            ),
        )
        _, p_value = paired_permutation_test(both, "top1_faithfulness", "condition", "xai_grounded", "plain")
        n_cases = both.case_id.nunique()
        output["comparisons"]["plain_vs_grounded"] = {
            "plain": p0,
            "grounded": p1,
            "delta": delta,
            "delta_ci_low": delta_low,
            "delta_ci_high": delta_high,
            "p_value": p_value,
            "cohens_h": cohens_h(p1, p0),
            "odds_ratio": odds_ratio(p1, p0, n=len(grounded)),
            "n_cases": n_cases,
            "n_rows": len(both),
        }
    output["per_model"] = {}
    for model in sorted(answered.model_name.unique()):
        subset = answered[answered.model_name == model]
        both = subset[subset.condition.isin(["plain", "xai_grounded"])]
        if both.empty:
            continue
        p0 = float(both[both.condition == "plain"].top1_faithfulness.mean())
        p1 = float(both[both.condition == "xai_grounded"].top1_faithfulness.mean())
        delta, low, high = cluster_bootstrap_ci(
            both,
            lambda d: float(
                d[d.condition == "xai_grounded"].top1_faithfulness.mean()
                - d[d.condition == "plain"].top1_faithfulness.mean()
            ),
        )
        _, p_value = paired_permutation_test(both, "top1_faithfulness", "condition", "xai_grounded", "plain")
        output["per_model"][model] = {
            "plain": p0,
            "grounded": p1,
            "delta": delta,
            "delta_ci_low": low,
            "delta_ci_high": high,
            "p_value": p_value,
            "cohens_h": cohens_h(p1, p0),
        }
    return output


def analyse_calibration():
    sources = [
        ("TEP", "Clean", os.path.join(OUT_DIR, "calibration_outputs.npz")),
        ("TEP", "0 dB", os.path.join(OUT_DIR, "final_runs/tep_snr/calib_snr0.npz")),
        ("TEP", "+5 dB", os.path.join(OUT_DIR, "final_runs/tep_snr/calib_snr5.npz")),
        ("TEP", "+10 dB", os.path.join(OUT_DIR, "final_runs/tep_snr/calib_snr10.npz")),
        ("CWRU", "Clean", os.path.join(OUT_DIR, "results", "cwru", "calibration_outputs.npz")),
        (
            "CWRU",
            "0 dB",
            os.path.join(OUT_DIR, "results", "cwru", "noise_white_0dB", "calibration_outputs.npz"),
        ),
        (
            "CWRU",
            "+5 dB",
            os.path.join(OUT_DIR, "results", "cwru", "noise_white_5dB", "calibration_outputs.npz"),
        ),
        (
            "CWRU",
            "+10 dB",
            os.path.join(OUT_DIR, "results", "cwru", "noise_white_10dB", "calibration_outputs.npz"),
        ),
    ]
    output = {}
    for dataset, condition, path in sources:
        if not os.path.exists(path):
            continue
        data = np.load(path, allow_pickle=True)
        logits, labels = (data["logits"], data["labels"])
        temperature = float(data["temperature"][0])
        output.setdefault(dataset, {})[condition] = {
            "temperature": temperature,
            "uncalibrated": calibration_metrics(logits, labels, 1.0),
            "calibrated": calibration_metrics(logits, labels, temperature),
            "n": int(len(labels)),
        }
    return output


def analyse_selective_prediction_classifier():
    sources = [
        ("TEP", "Clean", os.path.join(OUT_DIR, "calibration_outputs.npz")),
        ("TEP", "0 dB", os.path.join(OUT_DIR, "final_runs/tep_snr/calib_snr0.npz")),
        ("TEP", "+5 dB", os.path.join(OUT_DIR, "final_runs/tep_snr/calib_snr5.npz")),
        ("TEP", "+10 dB", os.path.join(OUT_DIR, "final_runs/tep_snr/calib_snr10.npz")),
        ("CWRU", "Clean", os.path.join(OUT_DIR, "results", "cwru", "calibration_outputs.npz")),
        (
            "CWRU",
            "0 dB",
            os.path.join(OUT_DIR, "results", "cwru", "noise_white_0dB", "calibration_outputs.npz"),
        ),
        (
            "CWRU",
            "+5 dB",
            os.path.join(OUT_DIR, "results", "cwru", "noise_white_5dB", "calibration_outputs.npz"),
        ),
        (
            "CWRU",
            "+10 dB",
            os.path.join(OUT_DIR, "results", "cwru", "noise_white_10dB", "calibration_outputs.npz"),
        ),
    ]
    output = {}
    for dataset, condition, path in sources:
        if not os.path.exists(path):
            continue
        data = np.load(path, allow_pickle=True)
        logits, labels = (data["logits"], data["labels"])
        temperature = float(data["temperature"][0])

        def softmax_max(temp):
            scaled = logits / temp
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            probs = np.exp(scaled)
            probs /= probs.sum(axis=1, keepdims=True)
            return (probs.max(axis=1), probs.argmax(axis=1))

        confidence, predictions = softmax_max(temperature)
        correct = (predictions == labels).astype(float)
        coverage, risk, aurc = risk_coverage_curve(confidence, correct)
        entry = {
            "n": int(len(labels)),
            "full_risk": float(1.0 - correct.mean()),
            "aurc": aurc,
            "risk_at_50": selective_risk_at_coverage(confidence, correct, 0.5),
            "risk_at_70": selective_risk_at_coverage(confidence, correct, 0.7),
            "risk_at_90": selective_risk_at_coverage(confidence, correct, 0.9),
            "coverage": coverage.tolist(),
            "risk": risk.tolist(),
        }
        output.setdefault(dataset, {})[condition] = entry
    return output


def analyse_selective_prediction():
    blocks = {
        "TEP": [
            ("final_runs/tep_clean_rep*/llm_experiment_results.csv", "cases/tep_xai_reliability.json"),
            (
                "final_runs/tep_noise_0.05/llm_experiment_results.csv",
                "cases/tep_noise_0.05_xai_reliability.json",
            ),
            (
                "final_runs/tep_noise_0.10/llm_experiment_results.csv",
                "cases/tep_noise_0.10_xai_reliability.json",
            ),
            (
                "final_runs/tep_noise_0.20/llm_experiment_results.csv",
                "cases/tep_noise_0.20_xai_reliability.json",
            ),
        ],
        "CWRU": [
            ("final_runs/cwru_clean_rep*/llm_experiment_results.csv", "cases/cwru_xai_reliability.json"),
            ("final_runs/cwru_snr0/llm_experiment_results.csv", "cases/cwru_snr0_xai_reliability.json"),
            ("final_runs/cwru_snr5/llm_experiment_results.csv", "cases/cwru_snr5_xai_reliability.json"),
            ("final_runs/cwru_snr10/llm_experiment_results.csv", "cases/cwru_snr10_xai_reliability.json"),
        ],
    }
    output = {}
    for dataset, sources in blocks.items():
        rows = []
        for pattern, reliability_path in sources:
            reliability = load_json(reliability_path)
            frame = load_result_files(pattern)
            for _, record in frame.iterrows():
                entry = reliability.get(str(record["case_id"]), {})
                payload = DiagnosisPayload(
                    case_id="",
                    scenario="",
                    true_fault=0,
                    predicted_fault=0,
                    calibrated_confidence=float(record["calibrated_confidence"]),
                    raw_confidence=0.0,
                    temperature=1.0,
                    xai_stability=entry.get("xai_stability"),
                    xai_agreement=entry.get("xai_agreement"),
                    data_quality=float(record.get("data_quality", 1.0)),
                )
                rows.append(
                    {
                        "case_id": record["case_id"],
                        "confidence": float(record["calibrated_confidence"]),
                        "ers": compute_ers(payload)[0],
                        "correct": float(record["prediction_correct"]),
                    }
                )
        if not rows:
            continue
        frame = pd.DataFrame(rows)
        unique = frame.groupby("case_id").first().reset_index()
        entry = {"n_cases": len(unique), "full_risk": float(1 - unique.correct.mean())}
        for name, column in (("confidence", "confidence"), ("ers", "ers")):
            coverage, risk, aurc = risk_coverage_curve(unique[column].to_numpy(), unique.correct.to_numpy())
            entry[name] = {
                "aurc": aurc,
                "risk_at_50": selective_risk_at_coverage(
                    unique[column].to_numpy(), unique.correct.to_numpy(), 0.5
                ),
                "risk_at_80": selective_risk_at_coverage(
                    unique[column].to_numpy(), unique.correct.to_numpy(), 0.8
                ),
                "coverage": coverage.tolist(),
                "risk": risk.tolist(),
            }
        output[dataset] = entry
    return output


def analyse_ablation():
    sources = {
        "TEP": os.path.join(OUT_DIR, "ablation_corrected_results.csv"),
        "CWRU": os.path.join(OUT_DIR, "results", "cwru", "cwru_ablation_results.csv"),
    }
    output = {}
    for dataset, path in sources.items():
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        models = {}
        for model, subset in frame.groupby("Model"):
            values = subset.Val_F1.to_numpy()
            rng = np.random.default_rng(42)
            means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(2000)]
            models[model] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "ci_low": float(np.percentile(means, 2.5)),
                "ci_high": float(np.percentile(means, 97.5)),
                "n": int(len(values)),
            }
        ranked = sorted(models.items(), key=lambda kv: kv[1]["mean"], reverse=True)
        if len(ranked) >= 2:
            best_name, second_name = (ranked[0][0], ranked[1][0])
            best = frame[frame.Model == best_name].Val_F1.to_numpy()
            second = frame[frame.Model == second_name].Val_F1.to_numpy()
            paired = frame[frame.Model.isin([best_name, second_name])].copy()
            paired["pair_id"] = paired.Seed.astype(str) + "_" + paired.Fold.astype(str)
            _, p_value = paired_permutation_test(
                paired, "Val_F1", "Model", best_name, second_name, cluster_column="pair_id"
            )
            output_entry = {
                "best": best_name,
                "second": second_name,
                "difference": float(best.mean() - second.mean()),
                "p_value": p_value,
                "cohens_d": cohens_d(best, second),
            }
        else:
            output_entry = {}
        pair_tests = {}
        for left, right in (("1D-CNN", "KAN (Kolmogorov-Arnold)"),):
            if left not in models or right not in models:
                continue
            a = frame[frame.Model == left].sort_values(["Seed", "Fold"]).Val_F1.to_numpy()
            b = frame[frame.Model == right].sort_values(["Seed", "Fold"]).Val_F1.to_numpy()
            if len(a) != len(b) or len(a) < 2:
                continue
            rng = np.random.default_rng(42)
            diff = a - b
            observed = float(diff.mean())
            n_perm = 10000
            count = 0
            for _ in range(n_perm):
                signs = rng.choice([-1.0, 1.0], size=len(diff))
                if abs(float((diff * signs).mean())) >= abs(observed):
                    count += 1
            pair_tests[f"{left} vs {right}"] = {
                "mean_difference": observed,
                "p_value": (count + 1) / (n_perm + 1),
                "cohens_d": cohens_d(a.tolist(), b.tolist()),
                "n_folds": int(len(a)),
            }
        output[dataset] = {"models": models, "best_vs_second": output_entry, "pair_tests": pair_tests}
    return output


def stats_command(args):
    print("Running statistical analysis (bootstrap 10,000 x, permutation 10,000 x)...\n")
    results = {}
    print("Ablation - confidence intervals and best-vs-second comparison")
    results["ablation"] = analyse_ablation()
    for dataset, entry in results["ablation"].items():
        comparison = entry.get("best_vs_second", {})
        if comparison:
            print(
                f"  {dataset}: {comparison['best']} vs {comparison['second']} Δ={comparison['difference']:+.4f} p={format_p(comparison['p_value'])} d={comparison['cohens_d']:+.3f}"
            )
    print("\nCalibration - ECE, MCE, NLL, Brier")
    results["calibration"] = analyse_calibration()
    for dataset, conditions in results["calibration"].items():
        for condition, entry in conditions.items():
            u, c = (entry["uncalibrated"], entry["calibrated"])
            print(
                f"  {dataset} {condition:<7} ECE {u['ece']:.4f}->{c['ece']:.4f}  MCE {u['mce']:.4f}->{c['mce']:.4f}  NLL {u['nll']:.3f}->{c['nll']:.3f}  Brier {u['brier']:.4f}->{c['brier']:.4f}"
            )
    print("\nGrounding effect - CI, p, effect size")
    results["grounding"] = {}
    for dataset, pattern in (
        ("TEP", "final_runs/tep_clean_rep*/llm_experiment_results.csv"),
        ("CWRU", "final_runs/cwru_clean_rep*/llm_experiment_results.csv"),
    ):
        frame = load_result_files(pattern)
        if frame.empty:
            continue
        analysis = analyse_grounding(dataset, frame)
        results["grounding"][dataset] = analysis
        comparison = analysis["comparisons"].get("plain_vs_grounded", {})
        if comparison:
            print(
                f"  {dataset}: {comparison['plain']:.4f} -> {comparison['grounded']:.4f}  Δ={format_ci(comparison['delta'], comparison['delta_ci_low'], comparison['delta_ci_high'])}  p={format_p(comparison['p_value'])}  h={comparison['cohens_h']:.3f}  OR={comparison['odds_ratio']:.1f}"
            )
    print("\nSelective prediction - risk-coverage and AURC")
    results["selective_classifier"] = analyse_selective_prediction_classifier()
    for dataset, conditions in results["selective_classifier"].items():
        for condition, entry in conditions.items():
            print(
                f"  {dataset} {condition:<7} n={entry['n']:>5} full risk {entry['full_risk']:.4f} -> risk@70% {entry['risk_at_70']:.4f} | AURC {entry['aurc']:.4f}"
            )
    results["selective"] = analyse_selective_prediction()
    for dataset, entry in results["selective"].items():
        print(
            f"  {dataset}: full risk {entry['full_risk']:.4f} | AURC confidence {entry['confidence']['aurc']:.4f} / ERS {entry['ers']['aurc']:.4f} | risk@50% confidence {entry['confidence']['risk_at_50']:.4f} / ERS {entry['ers']['risk_at_50']:.4f}"
        )
    with open(OUTPUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWritten: {OUTPUT}")


TEP_CSV = os.path.join(OUT_DIR, "ablation_corrected_results.csv")
CWRU_CSV = os.path.join(OUT_DIR, "results", "cwru", "cwru_ablation_results.csv")
OUT_CSV = os.path.join(OUT_DIR, "table2_ablation.csv")
OUT_MD = os.path.join(OUT_DIR, "table2_ablation.md")
MODEL_ORDER = ["Logistic Regression", "XGBoost", "LSTM", "KAN (Kolmogorov-Arnold)", "1D-CNN"]
N_BOOT = 10000
SEED = 42


def bootstrap_ci(values, n_boot=N_BOOT, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(SEED)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = draws.mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))


def summarise_ablation(csv_path, label):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found - run the {label} ablation first.")
    frame = pd.read_csv(csv_path)
    missing = [c for c in ("Val_Accuracy", "Val_ECE") if c not in frame.columns]
    if missing:
        raise KeyError(
            f"{label}: {csv_path} lacks {missing}. This CSV predates the accuracy/ECE columns; re-run that ablation."
        )
    rows = {}
    for model, group in frame.groupby("Model"):
        entry = {"n_folds": len(group)}
        for metric, column in (("Accuracy", "Val_Accuracy"), ("F1", "Val_F1"), ("ECE", "Val_ECE")):
            values = group[column].to_numpy(dtype=float)
            low, high = bootstrap_ci(values)
            entry[f"{label}_{metric}_mean"] = float(np.nanmean(values))
            entry[f"{label}_{metric}_sd"] = float(np.nanstd(values, ddof=1))
            entry[f"{label}_{metric}_lo"] = low
            entry[f"{label}_{metric}_hi"] = high
        rows[model] = entry
    return pd.DataFrame.from_dict(rows, orient="index")


def format_cell(row, label, metric):
    mean = row.get(f"{label}_{metric}_mean")
    sd = row.get(f"{label}_{metric}_sd")
    if mean is None or np.isnan(mean):
        return "—"
    return f"{mean:.4f} ± {sd:.4f}"


def table2_command(args):
    tep = summarise_ablation(TEP_CSV, "TEP")
    cwru = summarise_ablation(CWRU_CSV, "CWRU")
    merged = tep.join(cwru, how="outer", lsuffix="_tep", rsuffix="_cwru")
    order = [m for m in MODEL_ORDER if m in merged.index]
    order += [m for m in merged.index if m not in order]
    merged = merged.loc[order]
    merged.to_csv(OUT_CSV, index_label="Model")
    header = "| Model | TEP Accuracy ↑ | TEP Macro-F1 ↑ | TEP ECE ↓ | CWRU Accuracy ↑ | CWRU Macro-F1 ↑ | CWRU ECE ↓ |"
    lines = [header, "|" + "---|" * 7]
    for model, row in merged.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(model),
                    format_cell(row, "TEP", "Accuracy"),
                    format_cell(row, "TEP", "F1"),
                    format_cell(row, "TEP", "ECE"),
                    format_cell(row, "CWRU", "Accuracy"),
                    format_cell(row, "CWRU", "F1"),
                    format_cell(row, "CWRU", "ECE"),
                ]
            )
            + " |"
        )
    table = "\n".join(lines)
    with open(OUT_MD, "w", encoding="utf-8") as handle:
        handle.write(table + "\n")
    print(table)
    print(f"\n{OUT_CSV}\n{OUT_MD}")


def main():
    parser = argparse.ArgumentParser(
        description="Analysis of the LLM explanation experiments and classifier ablations."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    tables_parser = sub.add_parser("tables")
    add_tables_arguments(tables_parser)
    tables_parser.set_defaults(func=tables_command)
    stats_parser = sub.add_parser("stats")
    stats_parser.set_defaults(func=stats_command)
    table2_parser = sub.add_parser("table2")
    table2_parser.set_defaults(func=table2_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
