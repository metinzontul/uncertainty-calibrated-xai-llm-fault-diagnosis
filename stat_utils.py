from __future__ import annotations
import math
import numpy as np
import pandas as pd

N_BOOTSTRAP = 10000
N_PERMUTATIONS = 10000
RANDOM_SEED = 42


def cohens_h(p1, p2):
    p1 = min(max(float(p1), 0.0), 1.0)
    p2 = min(max(float(p2), 0.0), 1.0)
    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))


def odds_ratio(p1, p2, correction=0.5, n=1):
    a = p1 * n + correction
    b = (1.0 - p1) * n + correction
    c = p2 * n + correction
    d = (1.0 - p2) * n + correction
    if b <= 0 or c <= 0 or d <= 0:
        return float("inf")
    return a / b / (c / d)


def cohens_d(a, b):
    a_arr, b_arr = (np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    if len(a_arr) < 2 or len(b_arr) < 2:
        return 0.0
    pooled_var = ((len(a_arr) - 1) * a_arr.var(ddof=1) + (len(b_arr) - 1) * b_arr.var(ddof=1)) / (
        len(a_arr) + len(b_arr) - 2
    )
    if pooled_var <= 0:
        return 0.0
    return float((a_arr.mean() - b_arr.mean()) / math.sqrt(pooled_var))


def cluster_bootstrap_ci(
    frame, statistic, cluster_column="case_id", n_boot=N_BOOTSTRAP, alpha=0.05, seed=RANDOM_SEED
):
    point = statistic(frame)
    clusters = frame[cluster_column].unique()
    if len(clusters) < 2:
        return (point, float("nan"), float("nan"))
    grouped = {c: sub for c, sub in frame.groupby(cluster_column)}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_boot):
        drawn = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([grouped[c] for c in drawn], ignore_index=True)
        value = statistic(sample)
        if not math.isnan(value):
            estimates.append(value)
    if not estimates:
        return (point, float("nan"), float("nan"))
    lower = float(np.percentile(estimates, 100 * alpha / 2))
    upper = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    return (point, lower, upper)


def paired_permutation_test(
    frame,
    value_column,
    condition_column,
    condition_a,
    condition_b,
    cluster_column="case_id",
    n_perm=N_PERMUTATIONS,
    seed=RANDOM_SEED,
):
    subset = frame[frame[condition_column].isin([condition_a, condition_b])].copy()
    if subset.empty:
        return (float("nan"), float("nan"))

    def difference(data):
        a = data[data[condition_column] == condition_a][value_column].mean()
        b = data[data[condition_column] == condition_b][value_column].mean()
        return float(a - b)

    observed = difference(subset)
    rng = np.random.default_rng(seed)
    labels = subset[condition_column].to_numpy(copy=True)
    clusters = subset[cluster_column].to_numpy()
    cluster_positions = [np.where(clusters == c)[0] for c in np.unique(clusters)]
    extreme = 0
    shuffled = subset.copy()
    for _ in range(n_perm):
        permuted = labels.copy()
        for positions in cluster_positions:
            permuted[positions] = rng.permutation(permuted[positions])
        shuffled[condition_column] = permuted
        if abs(difference(shuffled)) >= abs(observed) - 1e-12:
            extreme += 1
    p_value = (extreme + 1) / (n_perm + 1)
    return (observed, float(p_value))


def calibration_metrics(logits, labels, temperature=1.0, n_bins=10):
    scaled = np.asarray(logits, dtype=float) / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = np.asarray(labels).astype(int)
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        mask = (
            (confidence > edges[i]) & (confidence <= edges[i + 1])
            if i > 0
            else (confidence >= edges[i]) & (confidence <= edges[i + 1])
        )
        if not mask.sum():
            continue
        gap = abs(correct[mask].mean() - confidence[mask].mean())
        ece += mask.mean() * gap
        mce = max(mce, gap)
    true_probs = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    nll = float(-np.mean(np.log(true_probs)))
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return {
        "accuracy": float(correct.mean()),
        "confidence": float(confidence.mean()),
        "ece": float(ece),
        "mce": float(mce),
        "nll": nll,
        "brier": brier,
    }


def risk_coverage_curve(scores, correct, n_points=100):
    scores_arr = np.asarray(scores, dtype=float)
    correct_arr = np.asarray(correct, dtype=float)
    if len(scores_arr) == 0:
        return (np.array([]), np.array([]), float("nan"))
    order = np.argsort(-scores_arr)
    errors = 1.0 - correct_arr[order]
    cumulative = np.cumsum(errors)
    n = len(scores_arr)
    indices = np.unique(np.linspace(1, n, min(n_points, n)).astype(int))
    coverage = indices / n
    risk = cumulative[indices - 1] / indices
    aurc = float(np.trapezoid(risk, coverage)) if len(coverage) > 1 else float("nan")
    return (coverage, risk, aurc)


def selective_risk_at_coverage(scores, correct, target_coverage):
    scores_arr = np.asarray(scores, dtype=float)
    correct_arr = np.asarray(correct, dtype=float)
    keep = max(1, int(round(target_coverage * len(scores_arr))))
    if keep == 0 or len(scores_arr) == 0:
        return float("nan")
    order = np.argsort(-scores_arr)[:keep]
    return float(1.0 - correct_arr[order].mean())


def format_ci(point, lower, upper, decimals=4):
    if math.isnan(lower) or math.isnan(upper):
        return f"{point:.{decimals}f}"
    return f"{point:.{decimals}f} [{lower:.{decimals}f}, {upper:.{decimals}f}]"


def format_p(p_value):
    if math.isnan(p_value):
        return "—"
    if p_value < 0.001:
        return "< 0.001"
    return f"{p_value:.4f}"
