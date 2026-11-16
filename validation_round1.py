import argparse
import csv
import glob
import itertools
import json
import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
import cwru_data
import xai_evaluation as xe
from llm import ERS_WEIGHTS, DiagnosisPayload, compute_ers, pearson_correlation
from models import ModelTrainer, Sensor1DCNN, TemperatureScaler
from paths import DATA_DIR, OUT_DIR, TEP_DIR, out
from stat_utils import (
    calibration_metrics,
    cluster_bootstrap_ci,
    risk_coverage_curve,
    selective_risk_at_coverage,
)
from train_classifiers import make_run_folds, macro_f1, standardise

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
W, S, SPC, ONSET = (60, 10, 4000, 160)
ERS_THREE = dict(ERS_WEIGHTS, data_quality=0.0)
BLOCKS = [("clean", "tep_clean_rep*", "cases/tep_xai_reliability.json")] + [
    (f"noise_{l}", f"tep_noise_{l}", f"cases/tep_noise_{l}_xai_reliability.json")
    for l in ("0.05", "0.10", "0.20")
]
EXCLUDE = "fault2_at100"
BA = 2
SRC = {
    "TEP": OUT_DIR / "ablation_corrected_results.csv",
    "CWRU": OUT_DIR / "results" / "cwru" / "cwru_ablation_results.csv",
}
RATIO = {"TEP": 7200 / 28800, "CWRU": float(np.mean([0.2 / (0.4 + 0.1 * k) for k in range(5)]))}
PAIRS = {
    "TEP": [("XGBoost", "1D-CNN"), ("1D-CNN", "KAN (Kolmogorov-Arnold)")],
    "CWRU": [("1D-CNN", "KAN (Kolmogorov-Arnold)"), ("KAN (Kolmogorov-Arnold)", "XGBoost")],
}
WINDOW, STRIDE, TRAIN_RATIO, EPOCHS, BATCH, LR = (1024, 128, 0.7, 10, 64, 0.001)
SEEDS_SUPP = (43, 44, 45, 46)
N_BOOT, BLOCK = (10000, 8)
XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
)
ONSET_LAST_PREFAULT = 20


def r1(*parts):
    return out("validation_round1", *parts)


def xdir(*parts):
    return r1("xgboost_statistics_final", *parts)


def window_meta(block):
    runs, samp = (block.simulationRun.to_numpy(), block["sample"].to_numpy())
    rows = []
    for s in range(0, len(block) - W + 1, S):
        rr, ss = (runs[s : s + W], samp[s : s + W])
        pre = int((ss <= ONSET).sum())
        rows.append(
            {
                "start": s,
                "fully_prefault": pre == W,
                "crosses_onset": 0 < pre < W,
                "crosses_run": len(np.unique(rr)) > 1,
            }
        )
    return rows


def softmax_conf(logits, T):
    z = logits / T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return (p.max(1), p.argmax(1))


def panel(logits, labels, T):
    unc = calibration_metrics(logits, labels, 1.0)
    cal = calibration_metrics(logits, labels, T)
    conf, pred = softmax_conf(logits, T)
    correct = (pred == labels).astype(float)
    _, _, aurc = risk_coverage_curve(conf, correct)
    return {
        "n": int(len(labels)),
        "T": float(T),
        "accuracy": cal["accuracy"],
        "confidence_uncal": unc["confidence"],
        "confidence_cal": cal["confidence"],
        "ece_uncal": unc["ece"],
        "ece_cal": cal["ece"],
        "nll_uncal": unc["nll"],
        "nll_cal": cal["nll"],
        "brier_uncal": unc["brier"],
        "brier_cal": cal["brier"],
        "full_risk": float(1 - correct.mean()),
        "risk_at_50": selective_risk_at_coverage(conf, correct, 0.5),
        "risk_at_70": selective_risk_at_coverage(conf, correct, 0.7),
        "risk_at_90": selective_risk_at_coverage(conf, correct, 0.9),
        "aurc": aurc,
    }


def onset_main():
    import joblib

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT_DIR / "best_1dcnn_fixed.pth", map_location=device, weights_only=False)
    model = Sensor1DCNN(ck["num_sensors"], ck["num_classes"], ck["window_size"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    df = pd.read_parquet(TEP_DIR / "parquet_cache" / "faulty_testing.parquet")
    sensors = [c for c in df.columns if c.startswith(("xmeas", "xmv"))]
    df = df.groupby("faultNumber", group_keys=False).head(SPC).reset_index(drop=True)
    le = joblib.load(OUT_DIR / "fault_label_encoder.pkl")
    df["label"] = le.transform(df["faultNumber"])
    halves = {"calibration": [], "evaluation": []}
    meta = {"calibration": [], "evaluation": []}
    labels = {"calibration": [], "evaluation": []}
    for lab, block in df.groupby("label", sort=True):
        cut = int(len(block) * 0.5)
        for name, part in (("calibration", block.iloc[:cut]), ("evaluation", block.iloc[cut:])):
            vals = part[sensors].values.astype(np.float32)
            for m in window_meta(part):
                halves[name].append(vals[m["start"] : m["start"] + W])
                meta[name].append(m)
                labels[name].append(int(lab))
    mean = np.asarray(ck["scaler_mean"], dtype=np.float32)
    std = np.asarray(ck["scaler_std"], dtype=np.float32)
    logits = {}
    for name in halves:
        x = (np.stack(halves[name]) - mean) / std
        with torch.no_grad():
            outs = [
                model(torch.tensor(x[i : i + 512], dtype=torch.float32, device=device)).cpu().numpy()
                for i in range(0, len(x), 512)
            ]
        logits[name] = np.concatenate(outs).astype(np.float64)
    saved = np.load(OUT_DIR / "calibration_outputs.npz")
    y_eval = np.array(labels["evaluation"])
    recon = {
        "labels_identical": bool(np.array_equal(saved["labels"], y_eval)),
        "max_abs_logit_diff": float(np.abs(saved["logits"] - logits["evaluation"]).max()),
    }

    def masks(name):
        m = meta[name]
        pre = np.array([x["fully_prefault"] for x in m])
        onset = np.array([x["crosses_onset"] for x in m])
        run = np.array([x["crosses_run"] for x in m])
        valid = ~pre & ~onset & ~run
        return (pre, onset, run, valid)

    counts = {}
    for name in halves:
        pre, onset, run, valid = masks(name)
        counts[name] = {
            "total": int(len(valid)),
            "fully_prefault": int(pre.sum()),
            "crosses_onset": int(onset.sum()),
            "crosses_run": int(run.sum()),
            "crosses_run_and_prefault_or_onset": int((run & (pre | onset)).sum()),
            "excluded_any": int((~valid).sum()),
            "retained_valid": int(valid.sum()),
            "retained_per_class": int(valid.sum() // 20),
        }
    _, _, _, vcal = masks("calibration")
    y_cal = np.array(labels["calibration"])
    scaler = TemperatureScaler().to(device)
    lt = torch.tensor(logits["calibration"][vcal], dtype=torch.float32, device=device)
    ly = torch.tensor(y_cal[vcal], device=device)
    scaler.fit(torch.nn.Identity(), [(lt, ly)], device)
    T_valid = float(scaler.temperature.item())
    scaler_all = TemperatureScaler().to(device)
    scaler_all.fit(
        torch.nn.Identity(),
        [
            (
                torch.tensor(logits["calibration"], dtype=torch.float32, device=device),
                torch.tensor(y_cal, device=device),
            )
        ],
        device,
    )
    T_all_refit = float(scaler_all.temperature.item())
    T_orig = float(saved["temperature"][0])
    _, _, _, veval = masks("evaluation")
    results = {
        "reconstruction_check": recon,
        "window_counts": counts,
        "temperatures": {
            "original_published": T_orig,
            "refit_all_calibration_windows": T_all_refit,
            "refit_valid_calibration_windows": T_valid,
        },
        "clean": {},
        "snr": {},
    }
    results["clean"]["original_all_windows_T_orig"] = panel(saved["logits"], saved["labels"], T_orig)
    results["clean"]["valid_subset_T_orig"] = panel(saved["logits"][veval], saved["labels"][veval], T_orig)
    results["clean"]["valid_subset_T_valid"] = panel(saved["logits"][veval], saved["labels"][veval], T_valid)
    excl = ~veval
    results["clean"]["excluded_windows_only_T_orig"] = panel(
        saved["logits"][excl], saved["labels"][excl], T_orig
    )
    for snr in ("0", "5", "10"):
        d = np.load(OUT_DIR / "final_runs" / "tep_snr" / f"calib_snr{snr}.npz")
        assert np.array_equal(d["labels"], saved["labels"])
        results["snr"][f"{snr}dB"] = {
            "original_all_windows_T_orig": panel(d["logits"], d["labels"], T_orig),
            "valid_subset_T_orig": panel(d["logits"][veval], d["labels"][veval], T_orig),
            "valid_subset_T_valid": panel(d["logits"][veval], d["labels"][veval], T_valid),
        }
    pre, onset, run, valid = masks("evaluation")
    conf, pred = softmax_conf(saved["logits"], T_orig)
    err = pred != saved["labels"]
    results["clean_error_by_category"] = {
        k: {
            "n": int(m.sum()),
            "error_rate": float(err[m].mean()) if m.sum() else None,
            "errors": int(err[m].sum()),
        }
        for k, m in (
            ("fully_prefault", pre),
            ("crosses_onset", onset),
            ("crosses_run", run),
            ("valid", valid),
        )
    }
    results["clean_error_by_category"]["share_of_all_errors_in_excluded"] = float(
        err[~valid].sum() / err.sum()
    )
    with open(r1("task1_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(
        json.dumps(
            {
                k: results[k]
                for k in ("reconstruction_check", "window_counts", "temperatures", "clean_error_by_category")
            },
            indent=1,
        )
    )
    for k, v in results["clean"].items():
        print(
            f"{k:<34} n={v['n']} acc={v['accuracy']:.4f} ECE {v['ece_uncal']:.4f}->{v['ece_cal']:.4f} NLL {v['nll_cal']:.3f} Brier {v['brier_cal']:.4f} R50 {v['risk_at_50']:.4f} R70 {v['risk_at_70']:.4f} R90 {v['risk_at_90']:.4f} AURC {v['aurc']:.4f} T={v['T']:.4f}"
        )
    for s, e in results["snr"].items():
        for k, v in e.items():
            print(
                f"{s:<5}{k:<30} acc={v['accuracy']:.4f} ECE {v['ece_uncal']:.4f}->{v['ece_cal']:.4f} R70 {v['risk_at_70']:.4f} AURC {v['aurc']:.4f}"
            )


def ers_load():
    rows = []
    for block, pat, rel in BLOCKS:
        reliability = json.load(open(OUT_DIR / rel, encoding="utf-8"))
        for f in glob.glob(str(OUT_DIR / "final_runs" / pat / "llm_experiment_results.csv")):
            d = pd.read_csv(f)
            d = d[d["error"].fillna("").astype(str).str.strip().str.len() == 0].copy()
            d["block"] = block
            d["rep"] = os.path.basename(os.path.dirname(f))
            for i, r in d.iterrows():
                e = reliability.get(str(r.case_id), {})
                p = DiagnosisPayload(
                    case_id="",
                    scenario="",
                    true_fault=0,
                    predicted_fault=0,
                    calibrated_confidence=float(r.calibrated_confidence),
                    raw_confidence=0.0,
                    temperature=1.0,
                    xai_stability=e.get("xai_stability"),
                    xai_agreement=e.get("xai_agreement"),
                    data_quality=float(r.data_quality),
                )
                d.loc[i, "ers"] = compute_ers(p, weights=ERS_THREE)[0]
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def ers_corr(frame, col):
    vals, ers = (frame[col].astype(float), frame["ers"].astype(float))
    if frame.empty:
        return {"r": None, "status": "undefined: no observations"}
    if vals.nunique() < 2:
        return {
            "r": None,
            "status": f"undefined: target has zero variance (all = {vals.iloc[0]:g})",
            "n": len(frame),
        }
    if ers.nunique() < 2:
        return {"r": None, "status": "undefined: ERS has zero variance", "n": len(frame)}

    def stat(d, c=col):
        if d[c].nunique() < 2 or d.ers.nunique() < 2:
            return float("nan")
        return pearson_correlation(d.ers.tolist(), d[c].astype(float).tolist())

    point, lo, hi = cluster_bootstrap_ci(frame, stat, cluster_column="case_id")
    return {
        "r": point,
        "ci_low": None if np.isnan(lo) else lo,
        "ci_high": None if np.isnan(hi) else hi,
        "n": int(len(frame)),
        "n_clusters": int(frame.case_id.nunique()),
        "n_distinct_case_block_units": int(frame.groupby(["case_id", "block"]).ngroups),
        "status": "defined",
    }


def ers_metrics(frame):
    ans = frame[frame.abstained == 0]
    return {
        "n_rows": int(len(frame)),
        "n_answered": int(len(ans)),
        "abstentions": int(frame.abstained.sum()),
        "evidence_adherence_at1": float(ans.top1_faithfulness.mean()) if len(ans) else None,
        "recall_faithfulness": float(ans.faithfulness.mean()) if len(ans) else None,
        "rank_correlation": float(ans.rank_correlation.mean()) if len(ans) else None,
        "hallucination_rate": float(ans.hallucination_rate.mean()) if len(ans) else None,
    }


def ers_main():
    A = ers_load()
    sets = {"original": A, "excluding_fault2_at100": A[~A.case_id.str.startswith(EXCLUDE)]}
    res = {
        "rows": {k: int(len(v)) for k, v in sets.items()},
        "excluded_rows": int(A.case_id.str.startswith(EXCLUDE).sum()),
    }
    for name, F in sets.items():
        o = {}
        o["by_block_condition"] = {
            f"{b}|{c}": ers_metrics(g) for (b, c), g in F.groupby(["block", "condition"])
        }
        o["clean_by_model_condition"] = {
            f"{m}|{c}": ers_metrics(g)
            for (m, c), g in F[F.block == "clean"].groupby(["model_name", "condition"])
        }

        def delta(G):
            a = G[(G.abstained == 0) & (G.condition == "plain")].top1_faithfulness.mean()
            b = G[(G.abstained == 0) & (G.condition == "xai_grounded")].top1_faithfulness.mean()
            return {"plain": a, "xai_grounded": b, "delta": b - a}

        o["delta_by_model_clean"] = {m: delta(g) for m, g in F[F.block == "clean"].groupby("model_name")}
        o["delta_by_block"] = {b: delta(g) for b, g in F.groupby("block")}
        cases = (
            F.groupby(["block", "case_id"])
            .agg(
                correct=("prediction_correct", "first"),
                conf=("calibrated_confidence", "first"),
                ers=("ers", "first"),
                pred=("predicted_fault", "first"),
                true=("true_fault", "first"),
            )
            .reset_index()
        )
        o["incorrect_case_units"] = cases[cases.correct == 0].to_dict("records")
        o["n_incorrect_distinct_windows"] = int(
            cases[cases.correct == 0].case_id.str.replace("_noise", "").str.replace("_none", "").nunique()
        )
        ans = F[F.abstained == 0]
        o["ers_associations_pooled"] = {
            "prediction_correctness": ers_corr(F, "prediction_correct"),
            "abstention": ers_corr(F, "abstained"),
            "evidence_adherence_at1": ers_corr(ans, "top1_faithfulness"),
            "recall_faithfulness": ers_corr(ans, "faithfulness"),
        }
        o["ers_associations_within_block"] = {
            b: {
                "prediction_correctness": ers_corr(g, "prediction_correct"),
                "evidence_adherence_at1": ers_corr(g[g.abstained == 0], "top1_faithfulness"),
            }
            for b, g in F.groupby("block")
        }
        o["ers_summary"] = {"mean": float(F.ers.mean()), "sd": float(F.ers.std())}
        res[name] = o
    with open(r1("task2_results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for name in sets:
        r = res[name]
        print(f"\n{name} (rows {res['rows'][name]})")
        for k, v in r["ers_associations_pooled"].items():
            print(f"  ERS vs {k:<24}", {x: round(y, 4) if isinstance(y, float) else y for x, y in v.items()})
        print(
            "  incorrect windows:",
            r["n_incorrect_distinct_windows"],
            [(u["block"], u["case_id"], u["pred"]) for u in r["incorrect_case_units"]],
        )
        for k, v in r["by_block_condition"].items():
            print(
                f"  {k:<36} n={v['n_rows']:>3} abst={v['abstentions']} EA@1={v['evidence_adherence_at1']:.4f} rec={v['recall_faithfulness']:.4f} rank={v['rank_correlation']:.4f} hall={v['hallucination_rate']:.4f}"
            )
        print("  delta by block:", {b: round(d["delta"], 4) for b, d in r["delta_by_block"].items()})
        print("  delta by model:", {b: round(d["delta"], 4) for b, d in r["delta_by_model_clean"].items()})
        print("  ERS", r["ers_summary"])


def ba_predict(model, x, T, device):
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            outs.append(model(torch.tensor(x[i : i + 256], device=device)).cpu().numpy())
    z = np.concatenate(outs).astype(np.float64) / T
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    return (p.argmax(1), p.max(1), p[:, 0])


def ba_summary(y, pred, conf, p_normal, base_pred, base_conf, subset):
    tp = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 0) & (y != 0)).sum())
    fn = int(((pred != 0) & (y == 0)).sum())
    s = subset
    return {
        "n_modified": int(s.sum()),
        "accuracy_all": float((pred == y).mean()),
        "accuracy_modified": float((pred[s] == y[s]).mean()),
        "changed_prediction_modified": int((pred[s] != base_pred[s]).sum()),
        "changed_to_normal_modified": int(((pred[s] == 0) & (base_pred[s] != 0)).sum()),
        "fraction_modified_predicted_normal": float((pred[s] == 0).mean()),
        "normal_precision": tp / (tp + fp) if tp + fp else None,
        "normal_recall": tp / (tp + fn) if tp + fn else None,
        "mean_conf_modified_before": float(base_conf[s].mean()),
        "mean_conf_modified_after": float(conf[s].mean()),
        "mean_p_normal_modified_after": float(p_normal[s].mean()),
        "predicted_class_counts_modified": {
            int(k): int(v) for k, v in zip(*np.unique(pred[s], return_counts=True))
        },
    }


def ba_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rd = OUT_DIR / "results" / "cwru"
    ck = torch.load(rd / "cwru_1dcnn_calibrated.pth", map_location=device, weights_only=False)
    model = Sensor1DCNN(ck["num_sensors"], ck["num_classes"], ck["window_size"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    T = float(ck["temperature"])
    d = np.load(rd / "prepared_windows.npz")
    x, y = (d["x_test"].astype(np.float32), d["y_test"])
    normal, faulty = (y == 0, y != 0)
    const = float(np.unique(x[normal][:, :, BA].round(6))[0])
    eval_mask = np.zeros(len(y), bool)
    for c in np.unique(y):
        pos = np.where(y == c)[0]
        eval_mask[pos[len(pos) // 2 :]] = True
    bp, bc, bn = ba_predict(model, x, T, device)
    np.savez(r1("task3_baseline_predictions.npz"), y=y, pred=bp, conf=bc, p_normal=bn, eval_mask=eval_mask)
    res = {
        "temperature": T,
        "normal_BA_constant": const,
        "n_test": int(len(y)),
        "n_normal": int(normal.sum()),
        "n_faulty": int(faulty.sum()),
        "baseline": ba_summary(y, bp, bc, bn, bp, bc, np.ones(len(y), bool)),
        "baseline_eval_half_accuracy": float((bp[eval_mask] == y[eval_mask]).mean()),
    }
    res["baseline"]["per_class_recall"] = {int(c): float((bp[y == c] == c).mean()) for c in np.unique(y)}
    x1 = x.copy()
    x1[faulty, :, BA] = const
    p, c, pn = ba_predict(model, x1, T, device)
    res["I1_faulty_BA_set_to_normal_constant"] = ba_summary(y, p, c, pn, bp, bc, faulty)
    res["I1_faulty_BA_set_to_normal_constant"]["per_fault_class_fraction_predicted_normal"] = {
        int(k): float((p[y == k] == 0).mean()) for k in range(1, 10)
    }
    rng = np.random.default_rng(43)
    fidx = np.where(faulty)[0]
    xc = x.copy()
    xc[fidx, :, BA] = x[rng.choice(fidx, size=len(fidx), replace=True), :, BA]
    p, c, pn = ba_predict(model, xc, T, device)
    res["C1_faulty_BA_from_random_faulty_window"] = ba_summary(y, p, c, pn, bp, bc, faulty)
    rng = np.random.default_rng(42)
    nidx = np.where(normal)[0]
    donors = rng.choice(fidx, size=len(nidx), replace=True)
    x2 = x.copy()
    x2[nidx, :, BA] = x[donors, :, BA]
    p, c, pn = ba_predict(model, x2, T, device)
    res["I2_normal_BA_from_random_faulty_window"] = ba_summary(y, p, c, pn, bp, bc, normal)
    res["I2_normal_BA_from_random_faulty_window"]["donor_class_counts"] = {
        int(k): int(v) for k, v in zip(*np.unique(y[donors], return_counts=True))
    }
    res["I2s_by_donor_class"] = {}
    rng = np.random.default_rng(42)
    for k in range(1, 10):
        pool = np.where(y == k)[0]
        dn = rng.choice(pool, size=len(nidx), replace=True)
        xs = x.copy()
        xs[nidx, :, BA] = x[dn, :, BA]
        p, c, pn = ba_predict(model, xs, T, device)
        s = ba_summary(y, p, c, pn, bp, bc, normal)
        res["I2s_by_donor_class"][k] = {
            kk: s[kk]
            for kk in (
                "fraction_modified_predicted_normal",
                "normal_recall",
                "mean_conf_modified_after",
                "predicted_class_counts_modified",
            )
        }
    with open(r1("task3_results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=1))


def signflip(d):
    obs = abs(d.mean())
    vals = [abs((d * np.array(s)).mean()) for s in itertools.product([-1, 1], repeat=len(d))]
    return sum((v >= obs - 1e-15 for v in vals)) / len(vals)


def fold_main():
    orig = json.load(open(OUT_DIR / "statistical_results.json", encoding="utf-8"))["ablation"]
    res = {}
    for ds, path in SRC.items():
        f = pd.read_csv(path)
        fold = f.groupby(["Model", "Fold"]).Val_F1.mean().unstack("Fold")
        models = {}
        for m, row in fold.iterrows():
            v = row.to_numpy()
            half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            models[m] = {
                "fold_means": v.round(6).tolist(),
                "mean": float(v.mean()),
                "sd_between_folds": float(v.std(ddof=1)),
                "t95_ci": [float(v.mean() - half), float(v.mean() + half)],
                "sd_within_fold_across_seeds_mean": float(
                    f[f.Model == m].groupby("Fold").Val_F1.std(ddof=1).mean()
                ),
                "original_ci_25_iid": [orig[ds]["models"][m]["ci_low"], orig[ds]["models"][m]["ci_high"]],
            }
        comps = {}
        for a, b in PAIRS[ds]:
            d = (fold.loc[a] - fold.loc[b]).to_numpy()
            half = stats.t.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5)
            fa = f[f.Model == a].set_index(["Seed", "Fold"]).Val_F1
            fb = f[f.Model == b].set_index(["Seed", "Fold"]).Val_F1
            dd = (fa - fb).dropna().to_numpy()
            ratio = float(np.mean(RATIO[ds]))
            var = dd.var(ddof=1)
            t_corr = dd.mean() / np.sqrt((1 / len(dd) + ratio) * var) if var > 0 else float("nan")
            p_corr = 2 * stats.t.sf(abs(t_corr), len(dd) - 1)
            comps[f"{a} vs {b}"] = {
                "fold_differences": d.round(6).tolist(),
                "mean_difference": float(d.mean()),
                "t95_ci_df4": [float(d.mean() - half), float(d.mean() + half)],
                "sign_flip_exact_p_two_sided": signflip(d),
                "min_attainable_p": 2 / 32,
                "folds_favouring_first": int((d > 0).sum()),
                "d_z_paired": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else None,
                "corrected_repeated_cv": {
                    "t": float(t_corr),
                    "df": len(dd) - 1,
                    "p_two_sided": float(p_corr),
                    "n_val_over_n_train_used": ratio,
                },
            }
        ob = orig[ds]["best_vs_second"]
        res[ds] = {
            "models": models,
            "comparisons": comps,
            "original_best_vs_second": {
                "best": ob["best"],
                "second": ob["second"],
                "difference": ob["difference"],
                "p_value": ob["p_value"],
                "cohens_d_unpaired": ob["cohens_d"],
            },
            "original_pair_tests": orig[ds].get("pair_tests", {}),
        }
    with open(r1("task4_results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    for ds, r in res.items():
        print(f"\n{ds}")
        for m, v in sorted(r["models"].items(), key=lambda kv: -kv[1]["mean"]):
            print(
                f"  {m:<25} mean {v['mean']:.4f} foldSD {v['sd_between_folds']:.4f} t95 [{v['t95_ci'][0]:.4f}, {v['t95_ci'][1]:.4f}]  orig iid CI {np.round(v['original_ci_25_iid'], 4).tolist()}"
            )
        for k, v in r["comparisons"].items():
            print(
                f"  {k}: diffs {v['fold_differences']} mean {v['mean_difference']:+.4f} CI [{v['t95_ci_df4'][0]:+.4f}, {v['t95_ci_df4'][1]:+.4f}] signflip p={v['sign_flip_exact_p_two_sided']:.4f} dz={v['d_z_paired']} corrCV p={v['corrected_repeated_cv']['p_two_sided']:.4g}"
            )
        print("  original:", r["original_best_vs_second"], r["original_pair_tests"])


def defe_windows(df, cols):
    feats = df[cols].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy()
    rows = df["row_in_record"].to_numpy()
    xs, ys, spans = ([], [], [])
    for s in range(0, len(feats) - WINDOW + 1, STRIDE):
        b = labels[s : s + WINDOW]
        if b[0] != b[-1]:
            continue
        xs.append(feats[s : s + WINDOW])
        ys.append(int(b[-1]))
        spans.append((int(b[-1]), int(rows[s]), int(rows[s + WINDOW - 1])))
    return (np.asarray(xs, np.float32), np.asarray(ys, np.int64), np.asarray(spans, np.int64))


def defe_prepare(cols):
    loader = cwru_data.CWRUDataLoader(base_dir=str(DATA_DIR / "CWRU"))
    cwru_data.SENSOR_COLS = ["DE", "FE", "BA"]
    df = loader.load_all()
    df["row_in_record"] = df.groupby("label").cumcount()
    tr, te = loader.train_test_split(df, train_ratio=TRAIN_RATIO)
    xtr, ytr, str_ = defe_windows(tr, cols)
    xte, yte, ste = defe_windows(te, cols)
    mean = xtr.mean(axis=(0, 1), dtype=np.float64)
    std = xtr.std(axis=(0, 1), dtype=np.float64)
    std = np.where(std < 1e-12, 1.0, std)
    return (
        ((xtr - mean) / std).astype(np.float32),
        ytr,
        str_,
        ((xte - mean) / std).astype(np.float32),
        yte,
        ste,
        mean,
        std,
    )


def defe_train(x, y, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=BATCH,
        shuffle=True,
        drop_last=True,
    )
    model = Sensor1DCNN(x.shape[2], 10, WINDOW).to(device)
    tr = ModelTrainer(model, device, learning_rate=LR)
    losses = [tr.train_epoch(loader) for _ in range(EPOCHS)]
    return (model, losses)


def defe_predict(model, x, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            outs.append(model(torch.from_numpy(x[i : i + 256]).to(device)).cpu().numpy())
    return np.concatenate(outs)


def defe_report(y, p):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=range(10), zero_division=0)
    return {
        "accuracy": float((y == p).mean()),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "per_class": {
            int(c): {
                "precision": float(pr[c]),
                "recall": float(rc[c]),
                "f1": float(f1[c]),
                "support": int(sup[c]),
            }
            for c in range(10)
        },
        "confusion_matrix": confusion_matrix(y, p, labels=range(10)).tolist(),
    }


def block_bootstrap(y, pa, pb, seed=42):
    rng = np.random.default_rng(seed)
    blocks = []
    for c in range(10):
        pos = np.where(y == c)[0]
        blocks.append([pos[i : i + BLOCK] for i in range(0, len(pos), BLOCK)])
    acc_a, acc_b, f_a, f_b = ([], [], [], [])
    for _ in range(N_BOOT):
        idx = np.concatenate(
            [np.concatenate([bl[j] for j in rng.integers(0, len(bl), len(bl))]) for bl in blocks]
        )
        yy = y[idx]
        acc_a.append((pa[idx] == yy).mean())
        acc_b.append((pb[idx] == yy).mean())
        f_a.append(f1_score(yy, pa[idx], average="macro", zero_division=0))
        f_b.append(f1_score(yy, pb[idx], average="macro", zero_division=0))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    acc_a, acc_b, f_a, f_b = map(np.asarray, (acc_a, acc_b, f_a, f_b))
    return {
        "accuracy_A_ci": q(acc_a),
        "accuracy_B_ci": q(acc_b),
        "accuracy_diff_A_minus_B_ci": q(acc_a - acc_b),
        "macro_f1_A_ci": q(f_a),
        "macro_f1_B_ci": q(f_b),
        "macro_f1_diff_A_minus_B_ci": q(f_a - f_b),
        "n_boot": N_BOOT,
        "block_windows": BLOCK,
    }


def defe_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    res = {
        "protocol": {
            "window": WINDOW,
            "stride": STRIDE,
            "train_ratio": TRAIN_RATIO,
            "epochs": EPOCHS,
            "batch": BATCH,
            "lr": LR,
            "primary_seed": 42,
            "supplementary_seeds": list(SEEDS_SUPP),
            "device": str(device),
            "note": "cudnn nondeterminism not disabled, as in the original pipeline",
        }
    }
    data = {}
    for tag, cols in (("DEFE", ["DE", "FE"]), ("3ch", ["DE", "FE", "BA"])):
        xtr, ytr, str_, xte, yte, ste, mean, std = defe_prepare(cols)
        data[tag] = (xtr, ytr, xte, yte)
        leak = {}
        for c in range(10):
            trmax = int(str_[str_[:, 0] == c][:, 2].max())
            temin = int(ste[ste[:, 0] == c][:, 1].min())
            leak[c] = {
                "last_train_sample": trmax,
                "first_test_sample": temin,
                "gap_samples": temin - trmax - 1,
                "overlap": temin <= trmax,
            }
        hashes_tr = {hash(w.tobytes()) for w in xtr}
        dup = sum((hash(w.tobytes()) in hashes_tr for w in xte))
        res[f"data_{tag}"] = {
            "channels": cols,
            "train_windows": int(len(ytr)),
            "test_windows": int(len(yte)),
            "train_per_class": np.bincount(ytr).tolist(),
            "test_per_class": np.bincount(yte).tolist(),
            "channel_mean": mean.tolist(),
            "channel_std": std.tolist(),
            "raw_sample_overlap_any_class": any((v["overlap"] for v in leak.values())),
            "per_class_split": leak,
            "identical_train_test_windows": int(dup),
        }
        if tag == "DEFE":
            np.savez_compressed(
                r1("cwru_DEFE", "prepared_windows_DEFE.npz"),
                x_train=xtr,
                y_train=ytr,
                x_test=xte,
                y_test=yte,
                channel_mean=mean,
                channel_std=std,
                sensor_cols=np.array(cols),
                train_spans=str_,
                test_spans=ste,
            )
            ref = np.load(OUT_DIR / "results" / "cwru" / "prepared_windows.npz")
            res["data_DEFE"]["same_labels_and_windows_as_original"] = bool(
                np.array_equal(ref["y_test"], yte) and np.array_equal(ref["y_train"], ytr)
            )
    yte = data["DEFE"][3]
    eval_mask = np.zeros(len(yte), bool)
    for c in range(10):
        pos = np.where(yte == c)[0]
        eval_mask[pos[len(pos) // 2 :]] = True
    preds = {}
    ck = torch.load(
        OUT_DIR / "results" / "cwru" / "cwru_1dcnn_calibrated.pth", map_location=device, weights_only=False
    )
    m = Sensor1DCNN(3, 10, WINDOW).to(device)
    m.load_state_dict(ck["model_state_dict"])
    preds["3ch_original_checkpoint"] = defe_predict(m, data["3ch"][2], device).argmax(1)
    runs = [("DEFE", 42)] + [("3ch", 42)] + [(t, s) for s in SEEDS_SUPP for t in ("DEFE", "3ch")]
    res["training_losses"] = {}
    for tag, seed in runs:
        xtr, ytr, xte, _ = data[tag]
        model, losses = defe_train(xtr, ytr, seed, device)
        key = f"{tag}_seed{seed}"
        preds[key] = defe_predict(model, xte, device).argmax(1)
        res["training_losses"][key] = losses
        if key == "DEFE_seed42":
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_sensors": 2,
                    "num_classes": 10,
                    "window_size": WINDOW,
                    "sensor_cols": ["DE", "FE"],
                    "scaler_mean": res["data_DEFE"]["channel_mean"],
                    "scaler_std": res["data_DEFE"]["channel_std"],
                    "seed": 42,
                    "epochs": EPOCHS,
                },
                r1("cwru_DEFE", "cwru_DEFE_1dcnn_seed42.pth"),
            )
        print(
            f"[{key}] final loss {losses[-1]:.4f} | eval-half acc {(preds[key][eval_mask] == yte[eval_mask]).mean():.4f} | full-test acc {(preds[key] == yte).mean():.4f}",
            flush=True,
        )
    np.savez_compressed(r1("cwru_DEFE", "test_predictions.npz"), y_test=yte, eval_mask=eval_mask, **preds)
    res["results"] = {
        k: {"eval_half": defe_report(yte[eval_mask], p[eval_mask]), "full_test": defe_report(yte, p)}
        for k, p in preds.items()
    }
    res["paired_block_bootstrap"] = {
        "eval_half": block_bootstrap(
            yte[eval_mask], preds["3ch_original_checkpoint"][eval_mask], preds["DEFE_seed42"][eval_mask]
        ),
        "full_test": block_bootstrap(yte, preds["3ch_original_checkpoint"], preds["DEFE_seed42"]),
        "A": "3ch_original_checkpoint",
        "B": "DEFE_seed42",
    }
    seeds = [42, *SEEDS_SUPP]
    sd = {}
    for split, msk in (("eval_half", eval_mask), ("full_test", np.ones(len(yte), bool))):
        a = np.array([f1_score(yte[msk], preds[f"3ch_seed{s}"][msk], average="macro") for s in seeds])
        b = np.array([f1_score(yte[msk], preds[f"DEFE_seed{s}"][msk], average="macro") for s in seeds])
        na = np.array([res["results"][f"3ch_seed{s}"][split]["per_class"][0]["f1"] for s in seeds])
        nb = np.array([res["results"][f"DEFE_seed{s}"][split]["per_class"][0]["f1"] for s in seeds])
        d = a - b
        sd[split] = {
            "seeds": seeds,
            "macro_f1_3ch": a.tolist(),
            "macro_f1_DEFE": b.tolist(),
            "diff_3ch_minus_DEFE": d.tolist(),
            "mean_diff": float(d.mean()),
            "min_diff": float(d.min()),
            "max_diff": float(d.max()),
            "normal_f1_3ch": na.tolist(),
            "normal_f1_DEFE": nb.tolist(),
        }
    res["seed_level_comparison"] = sd
    with open(r1("cwru_DEFE", "defe_results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    for k, v in res["results"].items():
        e = v["eval_half"]
        f = v["full_test"]
        print(
            f"{k:<28} eval acc {e['accuracy']:.4f} F1 {e['macro_f1']:.4f} NormalF1 {e['per_class'][0]['f1']:.4f} | full acc {f['accuracy']:.4f} F1 {f['macro_f1']:.4f} NormalP/R {f['per_class'][0]['precision']:.4f}/{f['per_class'][0]['recall']:.4f}"
        )
    print(json.dumps({k: res[k] for k in ("paired_block_bootstrap", "seed_level_comparison")}, indent=1))
    print(
        "leakage DEFE:",
        res["data_DEFE"]["raw_sample_overlap_any_class"],
        res["data_DEFE"]["identical_train_test_windows"],
        "same windows as original:",
        res["data_DEFE"]["same_labels_and_windows_as_original"],
    )


def tables_main():
    import joblib

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT_DIR / "best_1dcnn_fixed.pth", map_location=device, weights_only=False)
    model = Sensor1DCNN(ck["num_sensors"], ck["num_classes"], ck["window_size"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    df = pd.read_parquet(TEP_DIR / "parquet_cache" / "faulty_testing.parquet")
    sensors = [c for c in df.columns if c.startswith(("xmeas", "xmv"))]
    df = df.groupby("faultNumber", group_keys=False).head(SPC).reset_index(drop=True)
    df["label"] = joblib.load(OUT_DIR / "fault_label_encoder.pkl").transform(df["faultNumber"])
    win, meta, lab = ({"cal": [], "eval": []}, {"cal": [], "eval": []}, {"cal": [], "eval": []})
    for c, block in df.groupby("label", sort=True):
        cut = int(len(block) * 0.5)
        for name, part in (("cal", block.iloc[:cut]), ("eval", block.iloc[cut:])):
            v = part[sensors].values.astype(np.float32)
            for m in window_meta(part):
                win[name].append(v[m["start"] : m["start"] + W])
                meta[name].append(m)
                lab[name].append(int(c))
    valid = {
        k: np.array([not (m["fully_prefault"] or m["crosses_onset"] or m["crosses_run"]) for m in meta[k]])
        for k in meta
    }
    mean = np.asarray(ck["scaler_mean"], np.float32)
    std = np.asarray(ck["scaler_std"], np.float32)
    x = (np.stack(win["cal"]) - mean) / std
    with torch.no_grad():
        lc = np.concatenate(
            [model(torch.tensor(x[i : i + 512], device=device)).cpu().numpy() for i in range(0, len(x), 512)]
        ).astype(np.float64)
    yc = np.array(lab["cal"])
    sc = TemperatureScaler().to(device)
    sc.fit(
        torch.nn.Identity(),
        [
            (
                torch.tensor(lc[valid["cal"]], dtype=torch.float32, device=device),
                torch.tensor(yc[valid["cal"]], device=device),
            )
        ],
        device,
    )
    T = float(sc.temperature.item())
    saved = np.load(OUT_DIR / "calibration_outputs.npz")
    assert np.array_equal(saved["labels"], np.array(lab["eval"]))
    ve = valid["eval"]
    sources = [("Clean", OUT_DIR / "calibration_outputs.npz")] + [
        (f"{('+' if s != '0' else '')}{s} dB", OUT_DIR / "final_runs" / "tep_snr" / f"calib_snr{s}.npz")
        for s in ("0", "5", "10")
    ]
    t4, t5, panels = ([], [], {})
    for cond, path in sources:
        d = np.load(path)
        assert np.array_equal(d["labels"], saved["labels"])
        p = panel(d["logits"][ve], d["labels"][ve], T)
        unc = calibration_metrics(d["logits"][ve], d["labels"][ve], 1.0)
        assert abs(p["full_risk"] - (1 - p["accuracy"])) < 1e-12
        panels[cond] = {
            **p,
            "mce_uncal": unc["mce"],
            "mce_cal": calibration_metrics(d["logits"][ve], d["labels"][ve], T)["mce"],
        }
        t4.append(
            [
                "TEP",
                cond,
                f"{p['accuracy']:.4f}",
                f"{p['confidence_uncal']:.4f}",
                f"{T:.4f}",
                f"{p['ece_uncal']:.4f}",
                f"{p['ece_cal']:.4f}",
            ]
        )
        t5.append(
            [
                "TEP",
                cond,
                f"{p['n']:,}",
                f"{p['full_risk']:.4f}",
                f"{p['risk_at_50']:.4f}",
                f"{p['risk_at_70']:.4f}",
                f"{p['risk_at_90']:.4f}",
                f"{p['aurc']:.4f}",
            ]
        )
    sr = json.load(open(OUT_DIR / "statistical_results.json", encoding="utf-8"))["selective_classifier"][
        "CWRU"
    ]
    cw_files = {
        "Clean": "calibration_outputs.npz",
        "0 dB": "noise_white_0dB/calibration_outputs.npz",
        "+5 dB": "noise_white_5dB/calibration_outputs.npz",
        "+10 dB": "noise_white_10dB/calibration_outputs.npz",
    }
    for cond, f in cw_files.items():
        d = np.load(OUT_DIR / "results" / "cwru" / f)
        u, c = (
            calibration_metrics(d["logits"], d["labels"], 1.0),
            calibration_metrics(d["logits"], d["labels"], 0.442),
        )
        t4.append(
            [
                "CWRU",
                cond,
                f"{u['accuracy']:.4f}",
                f"{u['confidence']:.4f}",
                "0.442",
                f"{u['ece']:.4f}",
                f"{c['ece']:.4f}",
            ]
        )
        e = sr[cond]
        t5.append(
            [
                "CWRU",
                cond,
                f"{e['n']:,}",
                f"{e['full_risk']:.4f}",
                f"{e['risk_at_50']:.4f}",
                f"{e['risk_at_70']:.4f}",
                f"{e['risk_at_90']:.4f}",
                f"{e['aurc']:.4f}",
            ]
        )
    with open(r1("tep_correction", "table4_calibration.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "Dataset",
                "SNR",
                "Accuracy",
                "Confidence (uncalibrated mean max softmax)",
                "T",
                "ECE (uncal.)",
                "ECE (cal.)",
            ]
        )
        w.writerows(t4)
    with open(r1("tep_correction", "table5_selective.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Dataset", "Condition", "n", "Full risk", "Risk@50%", "Risk@70%", "Risk@90%", "AURC"])
        w.writerows(t5)
    np.savez(r1("tep_correction", "valid_masks.npz"), calibration=valid["cal"], evaluation=ve)
    expected = {
        "T": 1.6497,
        "accuracy": 0.971,
        "ece_uncal": 0.0181,
        "ece_cal": 0.0133,
        "full_risk": 0.029,
        "risk_at_70": 0.0148,
        "aurc": 0.0118,
    }
    got = {"T": T, **{k: panels["Clean"][k] for k in expected if k != "T"}}
    t1json = json.load(open(r1("task1_results.json"), encoding="utf-8"))
    ver = {
        "n_calibration_total": int(len(valid["cal"])),
        "n_calibration_valid": int(valid["cal"].sum()),
        "n_evaluation_total": int(len(ve)),
        "n_evaluation_valid": int(ve.sum()),
        "recomputed": got,
        "expected_from_brief": expected,
        "agreement_4dp": {k: round(got[k], 4) == v for k, v in expected.items()},
        "agreement_with_task1_json_T": abs(T - t1json["temperatures"]["refit_valid_calibration_windows"]),
        "panels": panels,
    }
    json.dump(ver, open(r1("tep_correction", "step1_verification.json"), "w", encoding="utf-8"), indent=2)
    print(json.dumps({k: ver[k] for k in ver if k != "panels"}, indent=1))
    for r in t4:
        print("T4", r)
    for r in t5:
        print("T5", r)


def step6_load():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT_DIR / "best_1dcnn_fixed.pth", map_location=device, weights_only=False)
    model = Sensor1DCNN(ck["num_sensors"], ck["num_classes"], ck["window_size"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    c = np.load(OUT_DIR / "tep_windows_cache.npz")
    x = ((c["x_test"] - ck["scaler_mean"]) / ck["scaler_std"]).astype(np.float32)
    y, runs = (c["y_test"].astype(np.int64), c["runs_test"])
    start = np.zeros(len(y), int)
    for cl, ru in set(zip(y.tolist(), runs.tolist())):
        idx = np.where((y == cl) & (runs == ru))[0]
        start[idx] = np.arange(len(idx)) * S
    cat = np.where(start + W <= ONSET, "fully_prefault", np.where(start < ONSET, "crosses_onset", "valid"))
    return (model, x, y, cat, device)


def sens_deletion(model, x_all, y_all, device, batch=32):
    rng = np.random.default_rng(42)
    tot = {(s, f): 0.0 for s in xe.STRATEGIES for f in xe.FRACTIONS}
    spread = {(s, f): [] for s in xe.STRATEGIES for f in xe.FRACTIONS}
    hits = seen = 0
    for st in range(0, len(x_all), batch):
        x = torch.tensor(x_all[st : st + batch], device=device)
        y = torch.tensor(y_all[st : st + batch], device=device)
        with torch.no_grad():
            pred = model(x).argmax(1)
        hits += float((pred == y).float().sum().item())
        att = xe.integrated_gradients(model, x, pred)
        for s in xe.STRATEGIES:
            for f in xe.FRACTIONS:
                if s == "random":
                    draws = [
                        xe.masked_accuracy(model, x, y, att, f, s, rng) for _ in range(xe.RANDOM_REPEATS)
                    ]
                    acc = float(np.mean(draws))
                    spread[s, f].append(float(np.std(draws, ddof=1)))
                else:
                    acc = xe.masked_accuracy(model, x, y, att, f, s, rng)
                tot[s, f] += acc * len(x)
        seen += len(x)
    base = hits / seen
    o = {"baseline_accuracy": base, "n_windows": seen, "fractions": {}}
    for f in xe.FRACTIONS:
        e = {}
        for s in xe.STRATEGIES:
            a = tot[s, f] / seen
            e[s] = {"accuracy": a, "drop": base - a}
            if spread[s, f]:
                e[s]["sd"] = float(np.mean(spread[s, f]))
        o["fractions"][f"{f:.2f}"] = e
    return o


def sens_robustness(model, x_all, device, batch=32):
    rng = np.random.default_rng(42)

    def imp(xx):
        res = []
        for st in range(0, len(xx), batch):
            x = torch.tensor(xx[st : st + batch], device=device)
            with torch.no_grad():
                p = model(x).argmax(1)
            res.append(xe.sensor_importance(xe.integrated_gradients(model, x, p)))
        return np.concatenate(res)

    clean = imp(x_all)
    o = {"n_windows": int(len(x_all)), "levels": {}}
    for snr in xe.SNR_LEVELS:
        lab = "Clean" if snr is None else f"{snr:.0f} dB"
        deg = imp(x_all if snr is None else xe.add_noise(x_all, snr, rng))
        rhos = [xe.spearman(clean[i], deg[i]) for i in range(len(clean))]
        e = {"spearman_rho": float(np.nanmean(rhos)), "spearman_sd": float(np.nanstd(rhos, ddof=1))}
        for k in xe.TOP_K:
            v = [xe.jaccard_top_k(clean[i], deg[i], k) for i in range(len(clean))]
            e[f"jaccard_top{k}"] = float(np.nanmean(v))
            e[f"jaccard_top{k}_sd"] = float(np.nanstd(v, ddof=1))
        o["levels"][lab] = e
    return o


def step6_main():
    model, x, y, cat, device = step6_load()
    perm = np.random.default_rng(42).permutation(len(x))
    res = {
        "test_cache_categories": {
            k: int((cat == k).sum()) for k in ("valid", "fully_prefault", "crosses_onset")
        },
        "run_crossing_windows_in_cache": 0,
    }
    saved_del = json.load(open(OUT_DIR / "xai_deletion_results.json", encoding="utf-8"))["TEP"]
    saved_rob = json.load(open(OUT_DIR / "xai_robustness_results.json", encoding="utf-8"))["TEP"]
    for name, n, fn, saved in (
        ("deletion", 300, sens_deletion, saved_del),
        ("robustness", 150, sens_robustness, saved_rob),
    ):
        orig = perm[:n]
        valid_first = perm[cat[perm] == "valid"][:n]
        orig_valid = orig[cat[orig] == "valid"]
        audit = {k: int((cat[orig] == k).sum()) for k in ("valid", "fully_prefault", "crosses_onset")}
        args = (model, x[orig], y[orig], device) if name == "deletion" else (model, x[orig], device)
        r0 = fn(*args)
        r1_ = fn(
            *(
                (model, x[valid_first], y[valid_first], device)
                if name == "deletion"
                else (model, x[valid_first], device)
            )
        )
        r2 = fn(
            *(
                (model, x[orig_valid], y[orig_valid], device)
                if name == "deletion"
                else (model, x[orig_valid], device)
            )
        )
        res[name] = {
            "original_sample_audit": audit,
            "saved_original": saved,
            "R0_reproduction": r0,
            "R1_first_valid_in_original_permutation": r1_,
            "R2_original_sample_valid_only": r2,
        }
        if name == "deletion":
            diffs = [abs(r0["baseline_accuracy"] - saved["baseline_accuracy"])] + [
                abs(r0["fractions"][f][s]["accuracy"] - saved["fractions"][f][s]["accuracy"])
                for f in r0["fractions"]
                for s in xe.STRATEGIES
            ]
        else:
            diffs = [
                abs(r0["levels"][l][m] - saved["levels"][l][m])
                for l in r0["levels"]
                for m in ("spearman_rho", "jaccard_top1", "jaccard_top3")
            ]
        res[name]["reproduction_max_abs_diff"] = float(max(diffs))
        res[name]["reproduced_within_0.005"] = bool(max(diffs) <= 0.005)
        print(name, "audit", audit, "| reproduction max diff", round(max(diffs), 5), flush=True)
    json.dump(res, open(r1("tep_correction", "step6_xai_sensitivity.json"), "w", encoding="utf-8"), indent=2)
    d = res["deletion"]
    for k in (
        "saved_original",
        "R0_reproduction",
        "R1_first_valid_in_original_permutation",
        "R2_original_sample_valid_only",
    ):
        v = d[k]
        print(
            f"DEL {k:<40} n={v['n_windows']} base={v['baseline_accuracy']:.4f} "
            + " ".join(
                (
                    f"{f}: top {v['fractions'][f]['top']['accuracy']:.4f} rnd {v['fractions'][f]['random']['accuracy']:.4f}+/-{v['fractions'][f]['random'].get('sd', 0):.4f} bot {v['fractions'][f]['bottom']['accuracy']:.4f}"
                    for f in v["fractions"]
                )
            )
        )
    r = res["robustness"]
    for k in (
        "saved_original",
        "R0_reproduction",
        "R1_first_valid_in_original_permutation",
        "R2_original_sample_valid_only",
    ):
        v = r[k]
        print(
            f"ROB {k:<40} n={v['n_windows']} "
            + " | ".join(
                (
                    f"{l}: rho {e['spearman_rho']:.4f}+/-{e['spearman_sd']:.4f} J1 {e['jaccard_top1']:.4f} J3 {e['jaccard_top3']:.4f}"
                    for l, e in v["levels"].items()
                )
            )
        )


def ece(p, y, bins=10):
    conf, pred = (p.max(1), p.argmax(1))
    corr = (pred == y).astype(float)
    e = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for i in range(bins):
        m = (conf > e[i]) & (conf <= e[i + 1]) if i else (conf >= e[i]) & (conf <= e[i + 1])
        if m.any():
            tot += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return tot


def f1(y, p):
    return float(f1_score(y, p, average="macro", zero_division=0))


def compare(name, ds, a_lab, b_lab, fa, fb, sa, sb):
    d = np.asarray(fa) - np.asarray(fb)
    n = len(d)
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    dd = (sa - sb).dropna().to_numpy()
    var = dd.var(ddof=1)
    t = dd.mean() / np.sqrt((1 / len(dd) + RATIO[ds]) * var) if var > 0 else np.nan
    return {
        "comparison": name,
        "dataset": ds,
        "model_A": a_lab,
        "model_B": b_lab,
        "fold_diff_1": d[0],
        "fold_diff_2": d[1],
        "fold_diff_3": d[2],
        "fold_diff_4": d[3],
        "fold_diff_5": d[4],
        "mean_diff_A_minus_B": d.mean(),
        "folds_favouring_A": int((d > 0).sum()),
        "signflip_exact_p_two_sided": signflip(d),
        "signflip_resolution_min_p": 2 / 32,
        "n_sign_patterns": 32,
        "descriptive_t95_low_df4": d.mean() - half,
        "descriptive_t95_high_df4": d.mean() + half,
        "paired_dz_folds": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan,
        "corrected_cv_t": t,
        "corrected_cv_df": len(dd) - 1,
        "corrected_cv_p_two_sided": 2 * stats.t.sf(abs(t), len(dd) - 1) if np.isfinite(t) else np.nan,
        "corrected_cv_ratio_nval_ntrain": RATIO[ds],
    }


def xgb_analyse_main():
    rows = []
    orig = pd.read_csv(SRC["TEP"])
    ox = orig[orig.Model == "XGBoost"].set_index(["Seed", "Fold"]).Val_F1
    for f in sorted(glob.glob(str(xdir("runs") / "seed*_fold*.npz"))):
        b = os.path.basename(f)[:-4]
        seed, fold = (int(b[4:6]), int(b.split("fold")[1]))
        z = np.load(f)
        y, po = (z["y_val"], z["post_onset"])
        pr, pl, pi, pn = (z[k].argmax(1) for k in ("p_raw", "p_leaky", "p_inner_raw", "p_nested"))
        rows.append(
            {
                "Seed": seed,
                "Fold": fold,
                "original_reported_Val_F1": ox.loc[seed, fold],
                "reproduced_leaky_isotonic_F1": f1(y, pl),
                "raw_F1_full_training": f1(y, pr),
                "inner_raw_F1_24runs": f1(y, pi),
                "nested_isotonic_F1": f1(y, pn),
                "labels_changed_by_leaky_isotonic": int((pr != pl).sum()),
                "labels_changed_by_nested_isotonic_vs_inner_raw": int((pi != pn).sum()),
                "n_val": int(len(y)),
                "raw_accuracy": float((pr == y).mean()),
                "leaky_accuracy": float((pl == y).mean()),
                "raw_F1_post_onset_only": f1(y[po], pr[po]),
                "leaky_F1_post_onset_only": f1(y[po], pl[po]),
                "n_val_post_onset": int(po.sum()),
                "ECE_raw": ece(z["p_raw"], y),
                "ECE_leaky_isotonic": ece(z["p_leaky"], y),
                "ECE_inner_raw": ece(z["p_inner_raw"], y),
                "ECE_nested_isotonic": ece(z["p_nested"], y),
            }
        )
    X = pd.DataFrame(rows)
    if not X.empty:
        X = X.sort_values(["Seed", "Fold"])
        X.to_csv(xdir("XGBoost_Corrected_Results.csv"), index=False, float_format="%.6f")
    fold_rows, comps, summary = ([], [], {})
    seedfold = {}
    for ds, path in SRC.items():
        f = pd.read_csv(path)
        for m, g in f.groupby("Model"):
            fm = g.groupby("Fold").Val_F1.mean()
            seedfold[ds, m] = g.set_index(["Seed", "Fold"]).Val_F1
            fold_rows.append(
                {
                    "dataset": ds,
                    "model": m,
                    "variant": "original CSV",
                    **{f"fold{k}": fm.loc[k] for k in range(1, 6)},
                    "mean_25": g.Val_F1.mean(),
                    "sd_25_descriptive": g.Val_F1.std(ddof=1),
                    "sd_between_fold_means": fm.std(ddof=1),
                    "n_runs": len(g),
                }
            )
    if len(X) == 25:
        for col, lab in (
            ("raw_F1_full_training", "XGBoost raw (no calibration)"),
            ("nested_isotonic_F1", "XGBoost nested isotonic"),
            ("reproduced_leaky_isotonic_F1", "XGBoost leaky isotonic (reproduction)"),
            ("raw_F1_post_onset_only", "XGBoost raw, post-onset windows only"),
        ):
            fm = X.groupby("Fold")[col].mean()
            seedfold["TEP", lab] = X.set_index(["Seed", "Fold"])[col]
            fold_rows.append(
                {
                    "dataset": "TEP",
                    "model": "XGBoost",
                    "variant": lab,
                    **{f"fold{k}": fm.loc[k] for k in range(1, 6)},
                    "mean_25": X[col].mean(),
                    "sd_25_descriptive": X[col].std(ddof=1),
                    "sd_between_fold_means": fm.std(ddof=1),
                    "n_runs": len(X),
                }
            )
    F = pd.DataFrame(fold_rows)
    F.to_csv(xdir("Fold_Level_Results.csv"), index=False, float_format="%.6f")

    def fm(ds, m):
        r = F[
            (F.dataset == ds)
            & (F.variant == m if m.startswith("XGBoost ") else (F.model == m) & (F.variant == "original CSV"))
        ].iloc[0]
        return [r[f"fold{k}"] for k in range(1, 6)]

    pairs = [
        ("TEP: XGBoost (original, leaky) vs 1D-CNN", "TEP", "XGBoost", "1D-CNN"),
        ("TEP: 1D-CNN vs KAN", "TEP", "1D-CNN", "KAN (Kolmogorov-Arnold)"),
        ("CWRU: 1D-CNN vs KAN", "CWRU", "1D-CNN", "KAN (Kolmogorov-Arnold)"),
    ]
    if len(X) == 25:
        pairs.insert(
            1, ("TEP: XGBoost raw (corrected) vs 1D-CNN", "TEP", "XGBoost raw (no calibration)", "1D-CNN")
        )
        pairs.insert(
            2, ("TEP: XGBoost nested isotonic vs 1D-CNN", "TEP", "XGBoost nested isotonic", "1D-CNN")
        )
    for name, ds, a, b in pairs:
        comps.append(compare(name, ds, a, b, fm(ds, a), fm(ds, b), seedfold[ds, a], seedfold[ds, b]))
    C = pd.DataFrame(comps)
    C.to_csv(xdir("Corrected_Model_Comparisons.csv"), index=False, float_format="%.6f")
    summary["xgboost"] = (
        {}
        if X.empty
        else {
            "n_runs": len(X),
            "mean_original_reported": X.original_reported_Val_F1.mean(),
            "mean_reproduced_leaky": X.reproduced_leaky_isotonic_F1.mean(),
            "max_abs_diff_reproduced_vs_original": float(
                (X.reproduced_leaky_isotonic_F1 - X.original_reported_Val_F1).abs().max()
            ),
            "mean_raw": X.raw_F1_full_training.mean(),
            "sd_raw": X.raw_F1_full_training.std(ddof=1),
            "mean_inner_raw": X.inner_raw_F1_24runs.mean(),
            "mean_nested": X.nested_isotonic_F1.mean(),
            "sd_nested": X.nested_isotonic_F1.std(ddof=1),
            "runs_with_label_changes_leaky": int((X.labels_changed_by_leaky_isotonic > 0).sum()),
            "mean_labels_changed_leaky": X.labels_changed_by_leaky_isotonic.mean(),
            "mean_raw_post_onset": X.raw_F1_post_onset_only.mean(),
            "mean_ECE_raw": X.ECE_raw.mean(),
            "mean_ECE_leaky": X.ECE_leaky_isotonic.mean(),
            "mean_ECE_inner_raw": X.ECE_inner_raw.mean(),
            "mean_ECE_nested": X.ECE_nested_isotonic.mean(),
        }
    )
    json.dump(summary, open(xdir("summary.json"), "w"), indent=2, default=float)
    pd.set_option("display.width", 250)
    print(F.round(4).to_string())
    print(C.round(4).to_string())
    print(json.dumps(summary, indent=1, default=float))


def compare_raw5(name, ds, a, b, fa, fb, sa, sb):
    d = np.asarray(fa) - np.asarray(fb)
    half = stats.t.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5)
    dd = (sa - sb).dropna().to_numpy()
    var = dd.var(ddof=1)
    t = dd.mean() / np.sqrt((1 / len(dd) + RATIO[ds]) * var) if var > 0 else np.nan
    return {
        "comparison": name,
        "dataset": ds,
        "model_A": a,
        "model_B": b,
        **{f"fold_diff_{k + 1}": d[k] for k in range(5)},
        "mean_diff_A_minus_B": d.mean(),
        "folds_favouring_A": int((d > 0).sum()),
        "signflip_exact_p_two_sided": signflip(d),
        "signflip_min_attainable_p": 2 / 32,
        "descriptive_t95_low_df4": d.mean() - half,
        "descriptive_t95_high_df4": d.mean() + half,
        "paired_dz_folds_descriptive": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan,
        "corrected_cv_t_sensitivity": t,
        "corrected_cv_df": len(dd) - 1,
        "corrected_cv_p_two_sided_sensitivity": (
            2 * stats.t.sf(abs(t), len(dd) - 1) if np.isfinite(t) else np.nan
        ),
        "corrected_cv_nval_over_ntrain": RATIO[ds],
    }


def xgb_analyse_raw5_main():
    orig = pd.read_csv(SRC["TEP"])
    ox = orig[orig.Model == "XGBoost"].pivot(index="Fold", columns="Seed", values="Val_F1")
    assert (
        ox.max(axis=1) - ox.min(axis=1) < 1e-09
    ).all(), "XGBoost is not seed-invariant in the original CSV"
    rows = []
    for k in range(1, 6):
        z = np.load(xdir("runs_raw5", f"fold{k}.npz"))
        y, po = (z["y_val"], z["post_onset"])
        pr, pl = (z["p_raw"].argmax(1), z["p_leaky"].argmax(1))
        rows.append(
            {
                "Fold": k,
                "original_reported_Val_F1_isotonic": ox.loc[k].iloc[0],
                "reproduced_leaky_isotonic_F1": f1(y, pl),
                "reproduction_abs_diff": abs(f1(y, pl) - ox.loc[k].iloc[0]),
                "raw_uncalibrated_F1": f1(y, pr),
                "raw_minus_original": f1(y, pr) - ox.loc[k].iloc[0],
                "raw_accuracy": float((pr == y).mean()),
                "isotonic_accuracy": float((pl == y).mean()),
                "labels_changed_by_isotonic": int((pr != pl).sum()),
                "n_val": int(len(y)),
                "n_val_post_onset": int(po.sum()),
                "n_val_onset_crossing": int((~po).sum()),
                "raw_F1_post_onset_only": f1(y[po], pr[po]),
                "isotonic_F1_post_onset_only": f1(y[po], pl[po]),
                "ECE_raw": ece(z["p_raw"], y),
                "ECE_isotonic_leaky": ece(z["p_leaky"], y),
                "fit_seconds": float(z["fit_seconds"]),
                "device": str(z["device"]),
            }
        )
    X = pd.DataFrame(rows)
    X.to_csv(xdir("XGBoost_Corrected_Results.csv"), index=False, float_format="%.6f")
    F, seedfold = ([], {})
    for ds, path in SRC.items():
        f = pd.read_csv(path)
        for m, g in f.groupby("Model"):
            fm = g.groupby("Fold").Val_F1.mean()
            seedfold[ds, m] = g.set_index(["Seed", "Fold"]).Val_F1
            F.append(
                {
                    "dataset": ds,
                    "model": m,
                    "variant": "original CSV"
                    + (
                        " (isotonic fitted+evaluated on validation fold)"
                        if (ds, m) == ("TEP", "XGBoost")
                        else ""
                    ),
                    **{f"fold{k}": fm.loc[k] for k in range(1, 6)},
                    "mean_25_descriptive": g.Val_F1.mean(),
                    "sd_25_descriptive": g.Val_F1.std(ddof=1),
                    "mean_of_fold_means": fm.mean(),
                    "sd_between_fold_means": fm.std(ddof=1),
                    "within_fold_seed_sd_mean": g.groupby("Fold").Val_F1.std(ddof=1).mean(),
                    "n_runs": len(g),
                }
            )
    for col, lab in (
        ("raw_uncalibrated_F1", "corrected: raw uncalibrated (one fit per fold; seed-invariant)"),
        ("raw_F1_post_onset_only", "sensitivity: raw uncalibrated, post-onset validation windows only"),
    ):
        fm = X.set_index("Fold")[col]
        seedfold["TEP", lab] = pd.Series(
            {(s, k): fm.loc[k] for s in (42, 43, 44, 45, 46) for k in range(1, 6)}
        )
        rep = np.repeat(fm.to_numpy(), 5)
        F.append(
            {
                "dataset": "TEP",
                "model": "XGBoost",
                "variant": lab,
                **{f"fold{k}": fm.loc[k] for k in range(1, 6)},
                "mean_25_descriptive": rep.mean(),
                "sd_25_descriptive": rep.std(ddof=1),
                "mean_of_fold_means": fm.mean(),
                "sd_between_fold_means": fm.std(ddof=1),
                "within_fold_seed_sd_mean": 0.0,
                "n_runs": 5,
            }
        )
    F = pd.DataFrame(F)
    F.to_csv(xdir("Fold_Level_Results.csv"), index=False, float_format="%.6f")

    def fold(ds, model, variant_prefix):
        r = F[(F.dataset == ds) & (F.model == model) & F.variant.str.startswith(variant_prefix)].iloc[0]
        return (
            [r[f"fold{k}"] for k in range(1, 6)],
            seedfold[ds, r.variant if not r.variant.startswith("original") else model],
        )

    C = []
    for name, ds, (ma, va), (mb, vb) in (
        ("TEP: XGBoost corrected (raw) vs 1D-CNN", "TEP", ("XGBoost", "corrected"), ("1D-CNN", "original")),
        (
            "TEP: XGBoost original (leaky isotonic) vs 1D-CNN [superseded]",
            "TEP",
            ("XGBoost", "original"),
            ("1D-CNN", "original"),
        ),
        ("TEP: 1D-CNN vs KAN", "TEP", ("1D-CNN", "original"), ("KAN (Kolmogorov-Arnold)", "original")),
        ("CWRU: 1D-CNN vs KAN", "CWRU", ("1D-CNN", "original"), ("KAN (Kolmogorov-Arnold)", "original")),
    ):
        fa, sa = fold(ds, ma, va)
        fb, sb = fold(ds, mb, vb)
        C.append(compare_raw5(name, ds, ma if va == "original" else f"{ma} ({va})", mb, fa, fb, sa, sb))
    C = pd.DataFrame(C)
    C.to_csv(xdir("Corrected_Model_Comparisons.csv"), index=False, float_format="%.6f")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    print(X.round(4).to_string())
    print(F.round(4).to_string())
    print(C.round(4).T.to_string())
    tep = F[(F.dataset == "TEP") & ~F.variant.str.contains("isotonic|sensitivity")].sort_values(
        "mean_of_fold_means", ascending=False
    )
    print(
        "TEP ranking (corrected XGBoost):",
        tep[["model", "variant", "mean_of_fold_means"]].round(4).values.tolist(),
    )


def fit_xgb(x, y, seed, device):
    import xgboost as xgb

    m = xgb.XGBClassifier(device=device, random_state=seed, **XGB_PARAMS)
    try:
        m.fit(x, y)
    except Exception:
        m = xgb.XGBClassifier(n_jobs=4, random_state=seed, **XGB_PARAMS)
        m.fit(x, y)
    return m


def isotonic(base, x, y):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    cal = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    cal.fit(x, y)
    return cal


def post_onset_flags(y_all, runs_all):
    start = np.zeros(len(y_all), int)
    for cl, ru in set(zip(y_all.tolist(), runs_all.tolist())):
        idx = np.where((y_all == cl) & (runs_all == ru))[0]
        start[idx] = np.arange(len(idx)) * 10
    return start >= ONSET_LAST_PREFAULT


def xgb_rerun_main(a):
    warnings.filterwarnings("ignore")
    run_dir = xdir("runs")
    os.makedirs(run_dir, exist_ok=True)
    c = np.load(OUT_DIR / "tep_windows_cache.npz")
    x_all, y_all, runs_all = (c["x_train_full"], c["y_train_full"], c["runs_train"])
    post_onset = post_onset_flags(y_all, runs_all)
    masks = make_run_folds(y_all, runs_all, 5)
    done = 0
    for seed in (42, 43, 44, 45, 46):
        for fold in range(5):
            path = run_dir / f"seed{seed}_fold{fold + 1}.npz"
            if path.exists():
                continue
            if done >= a.max_runs:
                return
            t0 = time.time()
            np.random.seed(seed)
            vm = masks[fold]
            xtr, ytr, xva, yva = (x_all[~vm], y_all[~vm], x_all[vm], y_all[vm])
            xtr, xva = standardise(xtr, xva)
            ftr, fva = (xtr.reshape(len(xtr), -1), xva.reshape(len(xva), -1))
            rtr = runs_all[~vm]
            inner_cal = np.zeros(len(ytr), bool)
            for cl in np.unique(ytr):
                cr = np.unique(rtr[ytr == cl])
                inner_cal |= (ytr == cl) & np.isin(rtr, np.array_split(cr, 4)[-1])
            full = fit_xgb(ftr, ytr, seed, a.device)
            p_raw = full.predict_proba(fva).astype(np.float32)
            leaky = isotonic(full, fva, yva)
            p_leaky = leaky.predict_proba(fva).astype(np.float32)
            inner = fit_xgb(ftr[~inner_cal], ytr[~inner_cal], seed, a.device)
            p_inner_raw = inner.predict_proba(fva).astype(np.float32)
            nested = isotonic(inner, ftr[inner_cal], ytr[inner_cal])
            p_nested = nested.predict_proba(fva).astype(np.float32)
            np.savez_compressed(
                path,
                y_val=yva,
                post_onset=post_onset[vm],
                p_raw=p_raw,
                p_leaky=p_leaky,
                p_inner_raw=p_inner_raw,
                p_nested=p_nested,
                n_train=len(ytr),
                n_inner_train=int((~inner_cal).sum()),
                n_inner_cal=int(inner_cal.sum()),
                held_runs=np.unique(runs_all[vm]),
                inner_cal_runs=np.unique(rtr[inner_cal]),
            )
            done += 1
            print(
                f"seed {seed} fold {fold + 1}: raw F1 {macro_f1(yva, p_raw.argmax(1)):.4f} | leaky-iso F1 {macro_f1(yva, p_leaky.argmax(1)):.4f} | inner raw {macro_f1(yva, p_inner_raw.argmax(1)):.4f} | nested-iso {macro_f1(yva, p_nested.argmax(1)):.4f} | {time.time() - t0:.0f}s",
                flush=True,
            )


def xgb_raw5_main():
    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    warnings.filterwarnings("ignore")
    params = dict(XGB_PARAMS, random_state=42)
    run_dir = xdir("runs_raw5")
    os.makedirs(run_dir, exist_ok=True)
    c = np.load(OUT_DIR / "tep_windows_cache.npz")
    x_all, y_all, runs_all = (c["x_train_full"], c["y_train_full"], c["runs_train"])
    post_onset = post_onset_flags(y_all, runs_all)
    masks = make_run_folds(y_all, runs_all, 5)
    for k in range(5):
        path = run_dir / f"fold{k + 1}.npz"
        if path.exists():
            continue
        t0 = time.time()
        np.random.seed(42)
        vm = masks[k]
        xtr, xva = standardise(x_all[~vm], x_all[vm])
        ytr, yva = (y_all[~vm], y_all[vm])
        ftr, fva = (xtr.reshape(len(xtr), -1), xva.reshape(len(xva), -1))
        device = "cuda"
        try:
            m = xgb.XGBClassifier(device=device, **params)
            m.fit(ftr, ytr)
        except Exception as exc:
            print(f"CUDA failed ({exc}); CPU fallback", flush=True)
            device = "cpu"
            m = xgb.XGBClassifier(n_jobs=4, **params)
            m.fit(ftr, ytr)
        p_raw = m.predict_proba(fva).astype(np.float32)
        t_fit = time.time() - t0
        leaky = CalibratedClassifierCV(FrozenEstimator(m), method="isotonic").fit(fva, yva)
        p_leaky = leaky.predict_proba(fva).astype(np.float32)
        np.savez_compressed(
            path,
            y_val=yva,
            post_onset=post_onset[vm],
            p_raw=p_raw,
            p_leaky=p_leaky,
            device=device,
            held_runs=np.unique(runs_all[vm]),
            fit_seconds=t_fit,
        )
        print(
            f"fold {k + 1}: raw F1 {macro_f1(yva, p_raw.argmax(1)):.6f} | leaky-isotonic F1 {macro_f1(yva, p_leaky.argmax(1)):.6f} | labels changed {(p_raw.argmax(1) != p_leaky.argmax(1)).sum()} | fit {t_fit:.0f}s total {time.time() - t0:.0f}s | {device}",
            flush=True,
        )


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("onset", onset_main),
        ("ers", ers_main),
        ("ba-shortcut", ba_main),
        ("fold-stats", fold_main),
        ("defe", defe_main),
        ("tables", tables_main),
        ("xai-sens", step6_main),
        ("xgb-analyse", xgb_analyse_main),
        ("xgb-analyse-raw5", xgb_analyse_raw5_main),
        ("xgb-raw5", xgb_raw5_main),
    ):
        sub.add_parser(name).set_defaults(fn=lambda a, fn=fn: fn())
    p = sub.add_parser("xgb-rerun")
    p.add_argument("--max-runs", type=int, default=25)
    p.add_argument("--device", default="cuda")
    p.set_defaults(fn=xgb_rerun_main)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.fn(args)
