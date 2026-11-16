from __future__ import annotations
import argparse
import glob
import json
import os
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import torch
from llm import DiagnosisPayload, compute_ers, kendall_tau_b, pearson_correlation
from paths import OUT_DIR, TEP_DIR, out
from stat_utils import cluster_bootstrap_ci

warnings.filterwarnings("ignore", category=FutureWarning)
N_PERTURBATIONS = 5
PERTURBATION_SCALE = 0.05
SNR_LEVELS = (None, 10.0, 5.0, 0.0)
TOP_K = (1, 3)
FRACTIONS = (0.05, 0.1, 0.2)
STRATEGIES = ("top", "random", "bottom")
RANDOM_REPEATS = 20
ABSTAIN_THRESHOLD = 0.85
MASK_FRACTIONS = (0.02, 0.05, 0.1)
AGAR_RANDOM_REPEATS = 10
SCALES = (0.05, 0.25, 0.5, 1.0, 1.5, 2.0)
BREAKDOWN_THRESHOLD = 0.5
N_BOOT = 10000
ERS_THREE = {"calibrated_confidence": 0.4, "xai_stability": 0.25, "xai_agreement": 0.2, "data_quality": 0.0}
COMPONENTS = {
    "calibrated_confidence": "− Confidence",
    "xai_stability": "− XAI Stability",
    "xai_agreement": "− XAI Agreement",
}
ABLATION_BLOCKS = (
    [
        ("TEP", "final_runs/tep_clean_rep*/llm_experiment_results.csv", "cases/tep_xai_reliability.json"),
        ("CWRU", "final_runs/cwru_clean_rep*/llm_experiment_results.csv", "cases/cwru_xai_reliability.json"),
    ]
    + [
        (
            "TEP",
            f"final_runs/tep_noise_{level}/llm_experiment_results.csv",
            f"cases/tep_noise_{level}_xai_reliability.json",
        )
        for level in ("0.05", "0.10", "0.20")
    ]
    + [
        (
            "CWRU",
            f"final_runs/cwru_snr{snr}/llm_experiment_results.csv",
            f"cases/cwru_snr{snr}_xai_reliability.json",
        )
        for snr in (0, 5, 10)
    ]
)


def ranking_correlation(a, b):
    tau = kendall_tau_b(list(a.astype(float)), list(b.astype(float)))
    return (tau + 1.0) / 2.0


def attribution_importance(attributions):
    return np.sum(np.abs(attributions), axis=0)


def gradient_times_input(model, x, target):
    leaf = torch.tensor(x.detach().cpu().numpy(), dtype=torch.float32, device=x.device).requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(leaf)
    logits[0, target].backward()
    if leaf.grad is None:
        raise RuntimeError("Gradient x Input received no gradient; the input was not a leaf.")
    return (leaf.grad * leaf).detach().cpu().numpy()[0]


def analyse_window(model, explainer, x, target, sensor_names, n_perturbations, scale, rng):
    _, base_attr = explainer.explain_prediction(x, target, sensor_names)
    base_importance = attribution_importance(base_attr)
    base_np = x.detach().cpu().numpy()
    channel_std = base_np.std(axis=(0, 1), keepdims=True)
    channel_std[channel_std < 1e-08] = 1.0
    correlations = []
    for _ in range(n_perturbations):
        noise = rng.standard_normal(base_np.shape) * channel_std * scale
        perturbed = torch.tensor(base_np + noise, dtype=torch.float32).to(x.device)
        _, attr = explainer.explain_prediction(perturbed, target, sensor_names)
        correlations.append(ranking_correlation(base_importance, attribution_importance(attr)))
    stability = float(np.mean(correlations)) if correlations else 0.0
    gxi = attribution_importance(gradient_times_input(model, x, target))
    agreement = ranking_correlation(base_importance, gxi)
    return (stability, agreement)


def build_tep(args):
    import joblib
    from models import Sensor1DCNN, XAIExplainer
    from tep_data import FastTEPLoader, SensorDegradator

    checkpoint_path = args.checkpoint or str(OUT_DIR / "best_1dcnn_fixed.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Sensor1DCNN(checkpoint["num_sensors"], checkpoint["num_classes"], checkpoint["window_size"]).to(
        device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    cases_path = args.cases or str(OUT_DIR / "cases" / "tep_clean.json")
    with open(cases_path, "r", encoding="utf-8") as fh:
        cases = json.load(fh)
    loader = FastTEPLoader(base_dir=str(TEP_DIR))
    test_df = loader.load_rdata_file("TEP_Faulty_Testing.RData")
    sensor_cols = [c for c in test_df.columns if c.startswith("xmeas") or c.startswith("xmv")]
    encoder = joblib.load(OUT_DIR / "fault_label_encoder.pkl")
    windows = []
    for case in cases:
        block = test_df[test_df["faultNumber"] == case["true_fault"]]
        start = case["window_start"]
        rows = block.iloc[start : start + checkpoint["window_size"]]
        if len(rows) < checkpoint["window_size"]:
            continue
        values = rows[sensor_cols].values.astype(np.float32)
        if case.get("degradation") == "noise" and case.get("degradation_level", 0) > 0:
            frame = rows[sensor_cols].copy()
            degraded = SensorDegradator(random_seed=42).apply_gaussian_noise(
                frame, sensor_cols, noise_level=float(case["degradation_level"])
            )
            values = degraded.values.astype(np.float32)
        if "scaler_mean" in checkpoint:
            values = (values - checkpoint["scaler_mean"]) / checkpoint["scaler_std"]
        x = torch.tensor(values, dtype=torch.float32).unsqueeze(0).to(device)
        target = int(encoder.transform([case["predicted_fault"]])[0])
        windows.append((case["case_id"], x, target))
    return (model, XAIExplainer(model, device), sensor_cols, windows, device)


def build_cwru(args):
    from models import Sensor1DCNN, XAIExplainer

    ckpt_dir = OUT_DIR / "results" / "cwru"
    if args.snr_db is not None:
        ckpt_dir = ckpt_dir / f"noise_white_{args.snr_db:g}dB"
    checkpoint_path = args.checkpoint or str(ckpt_dir / "cwru_1dcnn_calibrated.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    num_sensors = checkpoint.get("num_sensors", 3)
    num_classes = checkpoint.get("num_classes", 10)
    window_size = checkpoint.get("window_size", 1024)
    model = Sensor1DCNN(num_sensors, num_classes, window_size).to(device)
    model.load_state_dict(state)
    model.eval()
    prepared_dir = OUT_DIR / "results" / "cwru"
    if args.snr_db is not None:
        prepared_dir = prepared_dir / f"noise_white_{args.snr_db:g}dB"
        print(f"Using degraded windows from {prepared_dir}")
    prepared = np.load(prepared_dir / "prepared_windows.npz", allow_pickle=True)
    x_test, y_test = (prepared["x_test"], prepared["y_test"])
    sensor_cols = [str(c) for c in prepared["sensor_cols"]]
    cases_path = args.cases or str(OUT_DIR / "cases" / "cwru_clean.json")
    with open(cases_path, "r", encoding="utf-8") as fh:
        cases = json.load(fh)
    windows = []
    for case in cases:
        try:
            idx = int(str(case["case_id"]).rsplit("win", 1)[1])
        except (IndexError, ValueError):
            continue
        if idx >= len(x_test):
            continue
        x = torch.tensor(x_test[idx : idx + 1], dtype=torch.float32).to(device)
        with torch.no_grad():
            target = int(model(x).argmax(1).item())
        windows.append((case["case_id"], x, target))
    return (model, XAIExplainer(model, device), sensor_cols, windows, device)


def cmd_reliability(args):
    rng = np.random.default_rng(args.seed)
    builder = build_tep if args.dataset == "tep" else build_cwru
    model, explainer, sensor_cols, windows, device = builder(args)
    print(f"{args.dataset.upper()}: {len(windows)} windows, device {device}")
    print(f"{args.perturbations} perturbations at {args.scale:g} x channel sigma")
    results = {}
    for case_id, x, target in windows:
        stability, agreement = analyse_window(
            model, explainer, x, target, sensor_cols, args.perturbations, args.scale, rng
        )
        results[case_id] = {"xai_stability": stability, "xai_agreement": agreement}
        print(f"  {case_id:<28} stability {stability:.3f}  agreement {agreement:.3f}")
    output = args.output or str(out("cases", f"{args.dataset}_xai_reliability.json"))
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    stabilities = [v["xai_stability"] for v in results.values()]
    agreements = [v["xai_agreement"] for v in results.values()]
    print(
        f"\nMean stability {np.mean(stabilities):.3f} +/- {np.std(stabilities):.3f} | mean agreement {np.mean(agreements):.3f} +/- {np.std(agreements):.3f}"
    )
    print(f"Wrote {output}")


def load_tep(n_windows, device):
    from models import Sensor1DCNN

    checkpoint = torch.load(OUT_DIR / "best_1dcnn_fixed.pth", map_location=device, weights_only=False)
    model = Sensor1DCNN(checkpoint["num_sensors"], checkpoint["num_classes"], checkpoint["window_size"]).to(
        device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    cache = np.load(OUT_DIR / "tep_windows_cache.npz")
    x_raw, y = (cache["x_test"], cache["y_test"])
    x = (x_raw - checkpoint["scaler_mean"]) / checkpoint["scaler_std"]
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x))[:n_windows]
    return (model, x[idx].astype(np.float32), y[idx].astype(np.int64))


def load_cwru(n_windows, device):
    from models import Sensor1DCNN

    results_dir = OUT_DIR / "results" / "cwru"
    checkpoint = torch.load(
        results_dir / "cwru_1dcnn_calibrated.pth", map_location=device, weights_only=False
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    num_sensors = int(checkpoint.get("num_sensors", 3))
    num_classes = int(checkpoint.get("num_classes", 10))
    window_size = int(checkpoint.get("window_size", 1024))
    model = Sensor1DCNN(num_sensors, num_classes, window_size).to(device)
    model.load_state_dict(state)
    model.eval()
    prepared = np.load(results_dir / "prepared_windows.npz")
    x, y = (prepared["x_test"], prepared["y_test"])
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x))[:n_windows]
    return (model, x[idx].astype(np.float32), y[idx].astype(np.int64))


def load_dataset(dataset, n_windows, device):
    return load_tep(n_windows, device) if dataset == "TEP" else load_cwru(n_windows, device)


def integrated_gradients(model, x, target, steps=32):
    baseline = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for step in range(steps):
        alpha = (step + 0.5) / steps
        point = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        logits = model(point)
        selected = logits.gather(1, target.unsqueeze(1)).sum()
        grad = torch.autograd.grad(selected, point)[0]
        total = total + grad
    return ((x - baseline) * total / steps).abs()


def masked_accuracy(model, x, y, attributions, fraction, strategy, rng):
    batch, n_time, n_channel = x.shape
    n_inputs = n_time * n_channel
    k = max(1, int(round(fraction * n_inputs)))
    flat_attr = attributions.reshape(batch, n_inputs)
    if strategy == "top":
        order = torch.argsort(flat_attr, dim=1, descending=True)[:, :k]
    elif strategy == "bottom":
        order = torch.argsort(flat_attr, dim=1, descending=False)[:, :k]
    elif strategy == "random":
        picks = np.stack([rng.permutation(n_inputs)[:k] for _ in range(batch)])
        order = torch.tensor(picks, device=x.device, dtype=torch.long)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    masked = x.reshape(batch, n_inputs).clone()
    masked.scatter_(1, order, 0.0)
    masked = masked.reshape(batch, n_time, n_channel)
    with torch.no_grad():
        predictions = model(masked).argmax(dim=1)
    return float((predictions == y).float().mean().item())


def deletion_run(dataset, n_windows, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, x_all, y_all = load_dataset(dataset, n_windows, device)
    print(f"[{dataset}] {len(x_all)} windows | device: {device}", flush=True)
    rng = np.random.default_rng(42)
    totals = {(s, f): 0.0 for s in STRATEGIES for f in FRACTIONS}
    spread = {(s, f): [] for s in STRATEGIES for f in FRACTIONS}
    baseline_hits = 0.0
    seen = 0
    for start in range(0, len(x_all), batch_size):
        x = torch.tensor(x_all[start : start + batch_size], device=device)
        y = torch.tensor(y_all[start : start + batch_size], device=device)
        with torch.no_grad():
            predicted = model(x).argmax(dim=1)
        baseline_hits += float((predicted == y).float().sum().item())
        attributions = integrated_gradients(model, x, predicted)
        for strategy in STRATEGIES:
            for fraction in FRACTIONS:
                if strategy == "random":
                    draws = [
                        masked_accuracy(model, x, y, attributions, fraction, strategy, rng)
                        for _ in range(RANDOM_REPEATS)
                    ]
                    accuracy = float(np.mean(draws))
                    spread[strategy, fraction].append(float(np.std(draws, ddof=1)))
                else:
                    accuracy = masked_accuracy(model, x, y, attributions, fraction, strategy, rng)
                totals[strategy, fraction] += accuracy * len(x)
        seen += len(x)
        print(f"  {seen}/{len(x_all)}", flush=True)
    baseline = baseline_hits / seen
    result = {"baseline_accuracy": baseline, "n_windows": seen, "fractions": {}}
    for fraction in FRACTIONS:
        entry = {}
        for strategy in STRATEGIES:
            accuracy = totals[strategy, fraction] / seen
            entry[strategy] = {"accuracy": accuracy, "drop": baseline - accuracy}
            if spread[strategy, fraction]:
                entry[strategy]["sd"] = float(np.mean(spread[strategy, fraction]))
                entry[strategy]["repeats"] = RANDOM_REPEATS
        result["fractions"][f"{fraction:.2f}"] = entry
    return result


def cmd_deletion(args):
    output = out("xai_deletion_results.json")
    datasets = ["TEP", "CWRU"] if args.dataset == "both" else [args.dataset]
    existing = {}
    if output.exists():
        with open(output, encoding="utf-8") as fh:
            existing = json.load(fh)
    for dataset in datasets:
        existing[dataset] = deletion_run(dataset, args.n_windows, args.batch_size)
        print(f"\n[{dataset}] baseline accuracy {existing[dataset]['baseline_accuracy']:.4f}")
        for fraction, entry in existing[dataset]["fractions"].items():
            parts = " | ".join(
                (f"{s} {entry[s]['accuracy']:.4f} (-{entry[s]['drop']:.4f})" for s in STRATEGIES)
            )
            print(f"  {float(fraction) * 100:.0f}% masked: {parts}", flush=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)
    print(f"\nWrote {output}")


def spearman(a, b):
    if len(a) < 2:
        return float("nan")
    rank_a = np.argsort(np.argsort(-a)).astype(float)
    rank_b = np.argsort(np.argsort(-b)).astype(float)
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denominator = np.sqrt((rank_a**2).sum() * (rank_b**2).sum())
    if denominator == 0:
        return float("nan")
    return float((rank_a * rank_b).sum() / denominator)


def jaccard_top_k(a, b, k):
    k = min(k, len(a))
    set_a = set(np.argsort(-a)[:k].tolist())
    set_b = set(np.argsort(-b)[:k].tolist())
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else float("nan")


def add_noise(x, snr_db, rng):
    signal_power = float(np.mean(x**2))
    noise_power = signal_power / 10.0 ** (snr_db / 10.0)
    noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape)
    return (x + noise).astype(np.float32)


def sensor_importance(attributions):
    return attributions.sum(dim=1).detach().cpu().numpy()


def robustness_run(dataset, n_windows, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, x_all, y_all = load_dataset(dataset, n_windows, device)
    print(f"[{dataset}] {len(x_all)} windows | device: {device}", flush=True)
    rng = np.random.default_rng(42)
    result = {"n_windows": int(len(x_all)), "levels": {}}
    clean_importance = []
    for start in range(0, len(x_all), batch_size):
        x = torch.tensor(x_all[start : start + batch_size], device=device)
        with torch.no_grad():
            predicted = model(x).argmax(dim=1)
        clean_importance.append(sensor_importance(integrated_gradients(model, x, predicted)))
    clean = np.concatenate(clean_importance)
    for snr in SNR_LEVELS:
        if snr is None:
            label = "Clean"
            x_level = x_all
        else:
            label = f"{snr:.0f} dB"
            x_level = add_noise(x_all, snr, rng)
        importances = []
        for start in range(0, len(x_level), batch_size):
            x = torch.tensor(x_level[start : start + batch_size], device=device)
            with torch.no_grad():
                predicted = model(x).argmax(dim=1)
            importances.append(sensor_importance(integrated_gradients(model, x, predicted)))
        degraded = np.concatenate(importances)
        rhos = [spearman(clean[i], degraded[i]) for i in range(len(clean))]
        entry = {"spearman_rho": float(np.nanmean(rhos)), "spearman_sd": float(np.nanstd(rhos, ddof=1))}
        for k in TOP_K:
            values = [jaccard_top_k(clean[i], degraded[i], k) for i in range(len(clean))]
            entry[f"jaccard_top{k}"] = float(np.nanmean(values))
            entry[f"jaccard_top{k}_sd"] = float(np.nanstd(values, ddof=1))
        result["levels"][label] = entry
        parts = " | ".join((f"J@{k} {entry[f'jaccard_top{k}']:.4f}" for k in TOP_K))
        print(
            f"  {label:<7} rho {entry['spearman_rho']:.4f} +/- {entry['spearman_sd']:.4f} | {parts}",
            flush=True,
        )
    return result


def cmd_robustness(args):
    output = out("xai_robustness_results.json")
    datasets = ["TEP", "CWRU"] if args.dataset == "both" else [args.dataset]
    results = {}
    if output.exists():
        with open(output, encoding="utf-8") as fh:
            results = json.load(fh)
    for dataset in datasets:
        results[dataset] = robustness_run(dataset, args.n_windows, args.batch_size)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {output}")


def softmax_confidence(logits):
    probs = torch.softmax(logits, dim=1)
    confidence, prediction = probs.max(dim=1)
    return (prediction, confidence)


def repair(model, x, attributions, fraction, strategy, rng):
    batch, n_time, n_channel = x.shape
    n_inputs = n_time * n_channel
    k = max(1, int(round(fraction * n_inputs)))
    if strategy == "attribution":
        order = torch.argsort(attributions.reshape(batch, n_inputs), dim=1, descending=True)[:, :k]
    else:
        picks = np.stack([rng.permutation(n_inputs)[:k] for _ in range(batch)])
        order = torch.tensor(picks, device=x.device, dtype=torch.long)
    masked = x.reshape(batch, n_inputs).clone()
    masked.scatter_(1, order, 0.0)
    masked = masked.reshape(batch, n_time, n_channel)
    with torch.no_grad():
        return softmax_confidence(model(masked))


def agar_run(dataset, n_windows, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, x_all, y_all = load_dataset(dataset, n_windows, device)
    rng = np.random.default_rng(42)
    stats = {
        f"{f:.2f}": {
            "attribution": {"repaired": 0, "correct": 0},
            "random": {"repaired": 0.0, "correct": 0.0},
        }
        for f in MASK_FRACTIONS
    }
    n_abstained = 0
    n_abstained_correct = 0
    for start in range(0, len(x_all), batch_size):
        x = torch.tensor(x_all[start : start + batch_size], device=device)
        y = torch.tensor(y_all[start : start + batch_size], device=device)
        with torch.no_grad():
            base_pred, base_conf = softmax_confidence(model(x))
        mask = base_conf < ABSTAIN_THRESHOLD
        if not bool(mask.any()):
            continue
        x_low, y_low = (x[mask], y[mask])
        n_abstained += int(mask.sum().item())
        n_abstained_correct += int((base_pred[mask] == y_low).sum().item())
        with torch.no_grad():
            low_pred = model(x_low).argmax(dim=1)
        attributions = integrated_gradients(model, x_low, low_pred)
        for fraction in MASK_FRACTIONS:
            key = f"{fraction:.2f}"
            pred, conf = repair(model, x_low, attributions, fraction, "attribution", rng)
            recovered = conf >= ABSTAIN_THRESHOLD
            stats[key]["attribution"]["repaired"] += int(recovered.sum().item())
            stats[key]["attribution"]["correct"] += int(((pred == y_low) & recovered).sum().item())
            rep, cor = (0.0, 0.0)
            for _ in range(AGAR_RANDOM_REPEATS):
                pred_r, conf_r = repair(model, x_low, attributions, fraction, "random", rng)
                rec_r = conf_r >= ABSTAIN_THRESHOLD
                rep += float(rec_r.sum().item())
                cor += float(((pred_r == y_low) & rec_r).sum().item())
            stats[key]["random"]["repaired"] += rep / AGAR_RANDOM_REPEATS
            stats[key]["random"]["correct"] += cor / AGAR_RANDOM_REPEATS
    result = {
        "n_windows": int(len(x_all)),
        "n_abstained": n_abstained,
        "abstained_baseline_accuracy": n_abstained_correct / n_abstained if n_abstained else float("nan"),
        "fractions": {},
    }
    for key, entry in stats.items():
        row = {}
        for strategy in ("attribution", "random"):
            repaired = entry[strategy]["repaired"]
            correct = entry[strategy]["correct"]
            row[strategy] = {
                "repair_rate": repaired / n_abstained if n_abstained else float("nan"),
                "precision": correct / repaired if repaired else float("nan"),
                "n_repaired": repaired,
            }
        result["fractions"][key] = row
    print(
        f"[{dataset}] {len(x_all)} windows | abstained {n_abstained} (baseline accuracy {result['abstained_baseline_accuracy']:.4f})",
        flush=True,
    )
    for key, row in result["fractions"].items():
        a, r = (row["attribution"], row["random"])
        print(
            f"  {float(key) * 100:>4.0f}% masked | attribution: repair {a['repair_rate']:.3f} precision {a['precision']:.3f} | random: repair {r['repair_rate']:.3f} precision {r['precision']:.3f}",
            flush=True,
        )
    return result


def cmd_agar(args):
    output = out("agar_results.json")
    datasets = ["TEP", "CWRU"] if args.dataset == "both" else [args.dataset]
    results = {}
    for dataset in datasets:
        results[dataset] = agar_run(dataset, args.n_windows, args.batch_size)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {output}")


def profile_batch(model, x, rng):
    with torch.no_grad():
        predicted = model(x).argmax(dim=1)
    clean = sensor_importance(integrated_gradients(model, x, predicted))
    channel_std = x.std(dim=(0, 1), keepdim=True)
    profile = np.zeros((len(x), len(SCALES)), dtype=float)
    for column, scale in enumerate(SCALES):
        correlations = np.zeros((len(x), N_PERTURBATIONS), dtype=float)
        for repeat in range(N_PERTURBATIONS):
            noise = torch.tensor(rng.standard_normal(tuple(x.shape)), dtype=x.dtype, device=x.device)
            perturbed = x + noise * channel_std * scale
            attributions = sensor_importance(integrated_gradients(model, perturbed, predicted))
            for i in range(len(x)):
                correlations[i, repeat] = spearman(clean[i], attributions[i])
        profile[:, column] = np.nanmean(correlations, axis=1)
    breakdown = np.full(len(x), float(SCALES[-1]), dtype=float)
    for i in range(len(x)):
        below = np.where(profile[i] < BREAKDOWN_THRESHOLD)[0]
        if len(below):
            breakdown[i] = float(SCALES[below[0]])
    scales = np.asarray(SCALES, dtype=float)
    slope = np.array([float(np.polyfit(scales, profile[i], 1)[0]) for i in range(len(x))])
    return (profile, breakdown, slope)


def stability_run(dataset, n_windows, batch_size, input_snr=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, x_all, y_all = load_dataset(dataset, n_windows, device)
    if input_snr is not None:
        noise_rng = np.random.default_rng(7)
        power = float(np.mean(x_all**2))
        sigma = float(np.sqrt(power / 10.0 ** (input_snr / 10.0)))
        x_all = (x_all + noise_rng.normal(0.0, sigma, x_all.shape)).astype(np.float32)
    label = dataset if input_snr is None else f"{dataset} @{input_snr:.0f}dB"
    print(f"[{label}] {len(x_all)} windows | device: {device}", flush=True)
    rng = np.random.default_rng(42)
    profiles = []
    breakdowns = []
    slopes = []
    correct = []
    for start in range(0, len(x_all), batch_size):
        x = torch.tensor(x_all[start : start + batch_size], device=device)
        y = torch.tensor(y_all[start : start + batch_size], device=device)
        with torch.no_grad():
            predicted = model(x).argmax(dim=1)
        correct.append((predicted == y).cpu().numpy().astype(float))
        profile, breakdown, slope = profile_batch(model, x, rng)
        profiles.append(profile)
        breakdowns.append(breakdown)
        slopes.append(slope)
        print(f"  {min(start + batch_size, len(x_all))}/{len(x_all)}", flush=True)
    profile = np.concatenate(profiles)
    breakdown = np.concatenate(breakdowns)
    slope = np.concatenate(slopes)
    is_correct = np.concatenate(correct)
    result = {
        "n_windows": int(len(profile)),
        "scales": list(SCALES),
        "profile_mean": [float(v) for v in profile.mean(axis=0)],
        "profile_sd": [float(v) for v in profile.std(axis=0, ddof=1)],
        "breakdown_mean": float(breakdown.mean()),
        "breakdown_sd": float(breakdown.std(ddof=1)),
        "slope_mean": float(slope.mean()),
        "slope_sd": float(slope.std(ddof=1)),
        "accuracy": float(is_correct.mean()),
    }
    for name, values in (("breakdown", breakdown), ("slope", slope)):
        if len(set(values.tolist())) < 2 or len(set(is_correct.tolist())) < 2:
            result[f"r_{name}_correct"] = float("nan")
            continue
        result[f"r_{name}_correct"] = pearson_correlation(values.tolist(), is_correct.tolist())
    print(
        "  profile: " + " ".join((f"σ{s:.2f}={m:.3f}" for s, m in zip(SCALES, result["profile_mean"]))),
        flush=True,
    )
    print(
        f"  breakdown {result['breakdown_mean']:.3f} +/- {result['breakdown_sd']:.3f} | slope {result['slope_mean']:+.3f} +/- {result['slope_sd']:.3f}",
        flush=True,
    )
    print(
        f"  correlation with accuracy: breakdown r={result['r_breakdown_correct']:+.4f} | slope r={result['r_slope_correct']:+.4f}",
        flush=True,
    )
    return result


def cmd_stability(args):
    output = out("stability_profile_results.json")
    datasets = ["TEP", "CWRU"] if args.dataset == "both" else [args.dataset]
    results = {}
    if output.exists():
        with open(output, encoding="utf-8") as fh:
            results = json.load(fh)
    for dataset in datasets:
        key = dataset if args.input_snr is None else f"{dataset} @{args.input_snr:.0f}dB"
        results[key] = stability_run(dataset, args.n_windows, args.batch_size, args.input_snr)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {output}")


def bootstrap_ci(x, y, seed=42):
    point = pearson_correlation(x.tolist(), y.tolist())
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(x), len(x))
        xs, ys = (x[idx], y[idx])
        if len(set(ys.tolist())) < 2 or len(set(xs.tolist())) < 2:
            continue
        estimates.append(pearson_correlation(xs.tolist(), ys.tolist()))
    if not estimates:
        return (point, float("nan"), float("nan"))
    return (point, float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5)))


def measure(dataset, n_windows, batch_size=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, x_all, y_all = load_dataset(dataset, n_windows, device)
    rng = np.random.default_rng(42)
    single, breakdown, slope, correct = ([], [], [], [])
    for start in range(0, len(x_all), batch_size):
        x = torch.tensor(x_all[start : start + batch_size], device=device)
        y = torch.tensor(y_all[start : start + batch_size], device=device)
        with torch.no_grad():
            predicted = model(x).argmax(dim=1)
        correct.append((predicted == y).cpu().numpy().astype(float))
        clean = sensor_importance(integrated_gradients(model, x, predicted))
        channel_std = x.std(dim=(0, 1), keepdim=True)
        profile = np.zeros((len(x), len(SCALES)))
        for column, scale in enumerate(SCALES):
            correlations = np.zeros((len(x), N_PERTURBATIONS))
            for repeat in range(N_PERTURBATIONS):
                noise = torch.tensor(rng.standard_normal(tuple(x.shape)), dtype=x.dtype, device=x.device)
                attributions = sensor_importance(
                    integrated_gradients(model, x + noise * channel_std * scale, predicted)
                )
                for i in range(len(x)):
                    correlations[i, repeat] = spearman(clean[i], attributions[i])
            profile[:, column] = np.nanmean(correlations, axis=1)
        single.append(profile[:, 0])
        bd = np.full(len(x), float(SCALES[-1]))
        for i in range(len(x)):
            below = np.where(profile[i] < BREAKDOWN_THRESHOLD)[0]
            if len(below):
                bd[i] = float(SCALES[below[0]])
        breakdown.append(bd)
        scales = np.asarray(SCALES, dtype=float)
        slope.append(np.array([float(np.polyfit(scales, profile[i], 1)[0]) for i in range(len(x))]))
        print(f"  {min(start + batch_size, len(x_all))}/{len(x_all)}", flush=True)
    is_correct = np.concatenate(correct)
    result = {"n": int(len(is_correct)), "accuracy": float(is_correct.mean())}
    for name, values in (
        ("stability@0.05", np.concatenate(single)),
        ("breakdown", np.concatenate(breakdown)),
        ("slope", np.concatenate(slope)),
    ):
        result[name] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}
        if len(set(is_correct.tolist())) < 2 or len(set(values.tolist())) < 2:
            result[name].update({"r": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")})
            continue
        r, low, high = bootstrap_ci(values, is_correct)
        result[name].update({"r": r, "ci_low": low, "ci_high": high})
        print(f"  {name:<16} sd={result[name]['sd']:.4f}  r={r:+.4f} [{low:+.4f}, {high:+.4f}]", flush=True)
    return result


def cmd_spcompare(args):
    output = out("sp_compare_results.json")
    results = {}
    for dataset in ("TEP", "CWRU"):
        print(f"[{dataset}]", flush=True)
        results[dataset] = measure(dataset, 600)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {output}")


def load_reliability(path):
    full = OUT_DIR / path
    if not full.exists():
        print(f"{full} missing - stability/agreement unavailable.")
        return {}
    with open(full, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_results(patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    files = [f for p in patterns for f in glob.glob(str(OUT_DIR / p)) if "SYNTHETIC" not in f]
    if not files:
        return pd.DataFrame()
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return frame[frame["error"].fillna("").astype(str).str.strip().str.len() == 0]


def recompute_ers(row, reliability):
    entry = reliability.get(str(row["case_id"]), {})
    payload = DiagnosisPayload(
        case_id=str(row["case_id"]),
        scenario="",
        true_fault=int(row["true_fault"]),
        predicted_fault=int(row["predicted_fault"]),
        calibrated_confidence=float(row["calibrated_confidence"]),
        raw_confidence=float(row.get("raw_confidence", 0.0)),
        temperature=1.0,
        xai_stability=entry.get("xai_stability"),
        xai_agreement=entry.get("xai_agreement"),
        data_quality=float(row.get("data_quality", 1.0)),
    )
    return compute_ers(payload)[0]


def ers_report(name, results, reliability):
    if results.empty:
        print(f"\n{name}: no results found.")
        return
    answered = results[results["abstained"] == 0].copy()
    answered["ers_full"] = answered.apply(lambda r: recompute_ers(r, reliability), axis=1)
    covered = answered["case_id"].astype(str).isin(reliability).mean()
    print(f"\n{name}  (n={len(answered)}, cases with stability/agreement: {covered:.0%})")
    print(
        f"  ERS (2 signals, as previously reported): {answered['ers'].mean():.4f} +/- {answered['ers'].std():.4f}"
    )
    print(
        f"  ERS (4 signals, fully defined)        : {answered['ers_full'].mean():.4f} +/- {answered['ers_full'].std():.4f}"
    )
    print("\n  Correlation with measured faithfulness:")
    for label, column in (
        ("top-1 faithfulness", "top1_faithfulness"),
        ("recall faithfulness", "faithfulness"),
    ):
        if column not in answered.columns:
            continue
        old = pearson_correlation(answered["ers"].tolist(), answered[column].tolist())
        new = pearson_correlation(answered["ers_full"].tolist(), answered[column].tolist())
        print(f"    {label:<22} 2-signal r = {old:+.4f}    4-signal r = {new:+.4f}")
    spread = answered["ers_full"].std()
    if spread < 0.01:
        print(
            f"\n  ERS varies by only {spread:.4f} across cases. With no spread\n         the correlation is undefined in practice - a near-zero r here\n         means the score is constant, not that it fails to predict."
        )


def cmd_ers(args):
    datasets = [
        ("TEP", ["final_runs/tep_clean_rep*/llm_experiment_results.csv"], "cases/tep_xai_reliability.json"),
        (
            "CWRU",
            ["final_runs/cwru_clean_rep*/llm_experiment_results.csv"],
            "cases/cwru_xai_reliability.json",
        ),
    ]
    for name, patterns, reliability_path in datasets:
        ers_report(name, load_results(patterns), load_reliability(reliability_path))


def build_rows(dataset):
    rows = []
    for block_dataset, pattern, reliability_path in ABLATION_BLOCKS:
        if block_dataset != dataset:
            continue
        reliability = load_reliability(reliability_path)
        frame = load_results(pattern)
        for _, record in frame.iterrows():
            entry = reliability.get(str(record["case_id"]), {})
            rows.append(
                {
                    "case_id": str(record["case_id"]),
                    "calibrated_confidence": float(record["calibrated_confidence"]),
                    "xai_stability": entry.get("xai_stability"),
                    "xai_agreement": entry.get("xai_agreement"),
                    "data_quality": float(record.get("data_quality", 1.0)),
                    "top1": float(record["top1_faithfulness"]),
                    "acc": int(record["prediction_correct"]),
                    "abstain": int(record["abstained"]),
                }
            )
    return rows


def score_frame(rows, dropped):
    weights = dict(ERS_THREE)
    if dropped is not None:
        weights[dropped] = 0.0
    scored = []
    for row in rows:
        payload = DiagnosisPayload(
            case_id="",
            scenario="",
            true_fault=0,
            predicted_fault=0,
            calibrated_confidence=row["calibrated_confidence"],
            raw_confidence=0.0,
            temperature=1.0,
            xai_stability=row["xai_stability"],
            xai_agreement=row["xai_agreement"],
            data_quality=row["data_quality"],
        )
        value, _ = compute_ers(payload, weights=weights)
        scored.append({**row, "ers": value})
    return pd.DataFrame(scored)


def correlate(frame, column):

    def statistic(d, c=column):
        values = d[c].tolist()
        if len(set(values)) < 2 or len(set(d.ers.tolist())) < 2:
            return float("nan")
        return pearson_correlation(d.ers.tolist(), values)

    point, low, high = cluster_bootstrap_ci(frame, statistic, cluster_column="case_id")
    return {"r": point, "ci_low": low, "ci_high": high}


def cmd_ersablation(args):
    output = out("ers_ablation_results.json")
    results = {}
    for dataset in ("TEP", "CWRU"):
        rows = build_rows(dataset)
        if not rows:
            print(f"{dataset}: no results found, skipping.")
            continue
        entry = {}
        for label, dropped in [("Full ERS", None)] + [(name, key) for key, name in COMPONENTS.items()]:
            frame = score_frame(rows, dropped)
            answered = frame[frame.abstain == 0]
            entry[label] = {
                "prediction_correctness": correlate(frame, "acc"),
                "evidence_adherence": correlate(answered, "top1"),
                "n": int(len(frame)),
                "n_cases": int(frame.case_id.nunique()),
            }
            pc = entry[label]["prediction_correctness"]
            ea = entry[label]["evidence_adherence"]
            print(
                f"  {dataset} {label:<18} correctness r={pc['r']:+.4f} [{pc['ci_low']:+.4f}, {pc['ci_high']:+.4f}] | adherence r={ea['r']:+.4f} [{ea['ci_low']:+.4f}, {ea['ci_high']:+.4f}]",
                flush=True,
            )
        results[dataset] = entry
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {output}")


def add_window_args(p, n_windows, batch_size):
    p.add_argument("--dataset", choices=["TEP", "CWRU", "both"], default="both")
    p.add_argument("--n-windows", type=int, default=n_windows)
    p.add_argument("--batch-size", type=int, default=batch_size)


def main():
    parser = argparse.ArgumentParser(description="XAI reliability and evaluation experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("reliability")
    p.add_argument("--dataset", type=str, required=True, choices=["tep", "cwru"])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--cases", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--snr-db", type=float, default=None)
    p.add_argument("--perturbations", type=int, default=N_PERTURBATIONS)
    p.add_argument("--scale", type=float, default=PERTURBATION_SCALE)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_reliability)
    p = sub.add_parser("deletion")
    add_window_args(p, 200, 32)
    p.set_defaults(func=cmd_deletion)
    p = sub.add_parser("robustness")
    add_window_args(p, 150, 32)
    p.set_defaults(func=cmd_robustness)
    p = sub.add_parser("agar")
    add_window_args(p, 1500, 128)
    p.set_defaults(func=cmd_agar)
    p = sub.add_parser("stability")
    add_window_args(p, 600, 64)
    p.add_argument("--input-snr", type=float, default=None)
    p.set_defaults(func=cmd_stability)
    p = sub.add_parser("spcompare")
    p.set_defaults(func=cmd_spcompare)
    p = sub.add_parser("ers")
    p.set_defaults(func=cmd_ers)
    p = sub.add_parser("ersablation")
    p.set_defaults(func=cmd_ersablation)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
