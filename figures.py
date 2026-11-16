import argparse
import json
import os
import sys
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.lines import Line2D
from sklearn.metrics import f1_score
from paths import OUT_DIR, TEP_DIR, out

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
STATS_DIR = OUT_DIR / "validation_round1" / "xgboost_statistics_final"
CORRECTION_DIR = OUT_DIR / "validation_round1" / "tep_correction"
FIG1_OUT = str(out("figures", "Figure1_TEP_Publication_Final"))
FIG2_OUT = str(out("figures", "Figure2_TEP_Calibration_Final"))
ATTRIBUTION_CACHE = OUT_DIR / "xai_attributions.npz"
W, S, SPC, ONSET, NB = (60, 10, 4000, 160, 10)
INK, MUTED, GRID = ("#1a1a1a", "#595959", "#e6e6e6")


def save_all(fig, base):
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".svg")
    fig.savefig(base + ".png", dpi=600)
    fig.savefig(base + ".tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    for ext, kw in ((".png", {}), (".tiff", {"compression": "tiff_lzw"})):
        im = Image.open(base + ext)
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[3])
        bg.save(base + ext, dpi=(600, 600), **kw)


def fig1(args):
    orig = pd.read_csv(OUT_DIR / "ablation_corrected_results.csv")
    names = {
        "1D-CNN": "1D-CNN",
        "KAN (Kolmogorov-Arnold)": "KAN",
        "LSTM": "LSTM",
        "Logistic Regression": "Logistic Regression",
    }
    fold_means = {}
    for csv_name, label in names.items():
        g = orig[orig.Model == csv_name]
        assert len(g) == 25 and g.Val_F1.notna().all()
        fold_means[label] = g.groupby("Fold").Val_F1.mean().sort_index().to_numpy(dtype=float)
    xgb = []
    for k in range(1, 6):
        z = np.load(STATS_DIR / "runs_raw5" / f"fold{k}.npz")
        xgb.append(f1_score(z["y_val"], z["p_raw"].argmax(1), average="macro", zero_division=0))
    fold_means["XGBoost"] = np.asarray(xgb, dtype=float)
    ref = pd.read_csv(STATS_DIR / "Fold_Level_Results.csv")
    ref = ref[ref.dataset == "TEP"]
    refrows = {
        "XGBoost": ref[(ref.model == "XGBoost") & ref.variant.str.startswith("corrected")],
        **{
            lab: ref[(ref.model == csv) & ref.variant.str.startswith("original")]
            for csv, lab in names.items()
        },
    }
    for lab, r in refrows.items():
        assert len(r) == 1
        diff = np.abs(r[[f"fold{k}" for k in range(1, 6)]].to_numpy(dtype=float)[0] - fold_means[lab]).max()
        assert diff < 5e-07, (lab, diff)
    assert not np.any(
        np.isclose(fold_means["XGBoost"].mean(), 0.9212, atol=5e-05)
    ), "leaky XGBoost value detected"
    rows = sorted(((lab, v, v.mean(), v.std(ddof=1)) for lab, v in fold_means.items()), key=lambda r: -r[2])
    pd.DataFrame(
        [
            {
                "model": lab,
                **{f"fold{k + 1}": v[k] for k in range(5)},
                "mean_of_fold_means": m,
                "sd_fold_means_ddof1": s,
                "label": f"{m:.4f} ± {s:.4f}",
            }
            for lab, v, m, s in rows
        ]
    ).to_csv(FIG1_OUT.replace("_Final", "_values") + ".csv", index=False, float_format="%.10f")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "xtick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    accent = "#1f4e79"
    fold = "#8c8c8c"
    fig = plt.figure(figsize=(160 / 25.4, 78 / 25.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[6.0, 0.95], wspace=0.0)
    ax, axt = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]))
    n = len(rows)
    ys = np.arange(n)[::-1]
    for y, (lab, v, m, s) in zip(ys, rows):
        ax.scatter(
            v, np.full(5, y), s=22, facecolors="none", edgecolors=fold, linewidths=0.8, alpha=0.9, zorder=2
        )
        ax.errorbar(
            m, y, xerr=s, fmt="none", ecolor=accent, elinewidth=1.2, capsize=3, capthick=1.2, zorder=3
        )
        ax.scatter([m], [y], s=30, color=accent, edgecolors="white", linewidths=0.6, zorder=4)
        axt.text(
            0.0, y, f"{m:.4f} ± {s:.4f}", va="center", ha="left", color=INK, fontsize=8.5, family="sans-serif"
        )
    ax.set_xlim(0.55, 1.0)
    ax.set_xticks(np.arange(0.55, 1.0001, 0.05))
    ax.set_xticklabels([f"{t:.2f}" for t in np.arange(0.55, 1.0001, 0.05)])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], color=INK)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", colors=MUTED, labelcolor=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.set_xlabel("Validation macro-F1", color=INK)
    axt.set_ylim(ax.get_ylim())
    axt.set_xlim(0, 1)
    axt.axis("off")
    axt.text(0.0, n - 0.45, "Mean ± SD", va="bottom", ha="left", color=MUTED, fontsize=8)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=fold,
            markersize=4.5,
            label="Fold-level mean (5 seeds averaged)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=accent,
            markersize=5,
            linewidth=1.2,
            label="Mean of 5 folds ± 1 SD (descriptive)",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=7.5,
        handletextpad=0.4,
        columnspacing=1.6,
        borderaxespad=0.1,
    )
    save_all(fig, FIG1_OUT)
    for lab, v, m, s in rows:
        print(
            f"{lab:<20} folds {np.array2string(v, precision=8, separator=', ')}  mean {m:.10f}  sd {s:.10f}  label {m:.4f} +/- {s:.4f}"
        )


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def bins(logits, y, T):
    p = softmax(logits / T)
    conf, correct = (p.max(1), (p.argmax(1) == y).astype(float))
    edges = np.linspace(0, 1, NB + 1)
    rows = []
    for i in range(NB):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= edges[i]) & (conf <= edges[i + 1])
        n = int(m.sum())
        mc = float(conf[m].mean()) if n else np.nan
        acc = float(correct[m].mean()) if n else np.nan
        gap = abs(acc - mc) if n else np.nan
        rows.append(
            {
                "bin": i + 1,
                "lower": edges[i],
                "upper": edges[i + 1],
                "n": n,
                "mean_confidence": mc,
                "accuracy": acc,
                "abs_gap": gap,
                "ece_contribution": n / len(y) * gap if n else 0.0,
            }
        )
    return (pd.DataFrame(rows), conf, correct)


def verify_and_compute():
    import joblib
    import torch
    from models import Sensor1DCNN, TemperatureScaler

    df = pd.read_parquet(TEP_DIR / "parquet_cache" / "faulty_testing.parquet")
    sensors = [c for c in df.columns if c.startswith(("xmeas", "xmv"))]
    df = df.groupby("faultNumber", group_keys=False).head(SPC).reset_index(drop=True)
    df["label"] = joblib.load(OUT_DIR / "fault_label_encoder.pkl").transform(df["faultNumber"])
    win = {"cal": [], "eval": []}
    valid = {"cal": [], "eval": []}
    lab = {"cal": [], "eval": []}
    rowsets = {"cal": set(), "eval": set()}
    for c, block in df.groupby("label", sort=True):
        cut = int(len(block) * 0.5)
        for name, part in (("cal", block.iloc[:cut]), ("eval", block.iloc[cut:])):
            v = part[sensors].values.astype(np.float32)
            runs = part.simulationRun.to_numpy()
            smp = part["sample"].to_numpy()
            for s in range(0, len(part) - W + 1, S):
                ok = bool((smp[s : s + W] > ONSET).all() and len(np.unique(runs[s : s + W])) == 1)
                win[name].append(v[s : s + W])
                valid[name].append(ok)
                lab[name].append(int(c))
                if ok:
                    rowsets[name].update(
                        (
                            (int(part.faultNumber.iloc[0]), int(r), int(q))
                            for r, q in zip(runs[s : s + W], smp[s : s + W])
                        )
                    )
    valid = {k: np.array(v) for k, v in valid.items()}
    lab = {k: np.array(v) for k, v in lab.items()}
    saved_masks = np.load(CORRECTION_DIR / "valid_masks.npz")
    chk = {
        "masks_equal_saved": bool(
            np.array_equal(valid["cal"], saved_masks["calibration"])
            and np.array_equal(valid["eval"], saved_masks["evaluation"])
        ),
        "n_calibration_valid": int(valid["cal"].sum()),
        "n_evaluation_valid": int(valid["eval"].sum()),
        "calibration_evaluation_rows_disjoint": len(rowsets["cal"] & rowsets["eval"]) == 0,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(OUT_DIR / "best_1dcnn_fixed.pth", map_location=device, weights_only=False)
    model = Sensor1DCNN(ck["num_sensors"], ck["num_classes"], ck["window_size"]).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    x = (np.stack(win["cal"]) - np.asarray(ck["scaler_mean"], np.float32)) / np.asarray(
        ck["scaler_std"], np.float32
    )
    with torch.no_grad():
        lc = np.concatenate(
            [model(torch.tensor(x[i : i + 512], device=device)).cpu().numpy() for i in range(0, len(x), 512)]
        ).astype(np.float64)
    yc = lab["cal"]
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
    from scipy.optimize import minimize_scalar

    zc, ycv = (lc[valid["cal"]], yc[valid["cal"]])

    def nll(t):
        z = zc / t
        z = z - z.max(1, keepdims=True)
        return float(-(z[np.arange(len(ycv)), ycv] - np.log(np.exp(z).sum(1))).mean())

    opt = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded", options={"xatol": 1e-08})
    chk["T_exact_float64_minimiser"] = float(opt.x)
    chk["nll_at_T_lbfgs"] = nll(T)
    chk["nll_at_exact_T"] = float(opt.fun)
    chk["nll_at_1.6494"] = nll(1.6494)
    chk["nll_at_1.6497"] = nll(1.6497)
    ev = np.load(OUT_DIR / "calibration_outputs.npz")
    chk["evaluation_labels_match_rebuilt_windows"] = bool(np.array_equal(ev["labels"], lab["eval"]))
    logits, y = (ev["logits"][valid["eval"]].astype(np.float64), ev["labels"][valid["eval"]])
    return (logits, y, T, chk)


def plot_calibration(b_unc, b_cal, T, ece_u, ece_c):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    c_u, c_c = ("#D55E00", "#0072B2")
    min_n = 10
    fig = plt.figure(figsize=(120 / 25.4, 182 / 25.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.75, 1.0])
    ax, axh = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]))
    ax.plot([0, 1], [0, 1], linestyle=(0, (4, 3)), color=MUTED, linewidth=0.8, zorder=1)
    specs = (
        (b_unc, c_u, "o", (0, (5, 2)), f"Uncalibrated (ECE = {ece_u:.4f})", (4, -12)),
        (b_cal, c_c, "s", "-", f"Temperature-scaled, T = {T:.4f} (ECE = {ece_c:.4f})", (-8, 5)),
    )
    for b, col, mk, ls, lab, off in specs:
        d = b[b.n > 0]
        for (_, r1), (_, r2) in zip(d.iloc[:-1].iterrows(), d.iloc[1:].iterrows()):
            if r2.bin - r1.bin == 1 and r1.n >= min_n and (r2.n >= min_n):
                ax.plot(
                    [r1.mean_confidence, r2.mean_confidence],
                    [r1.accuracy, r2.accuracy],
                    color=col,
                    linestyle=ls,
                    linewidth=1.3,
                    zorder=2,
                    solid_capstyle="round",
                    dash_capstyle="round",
                )
        big, small = (d[d.n >= min_n], d[d.n < min_n])
        ax.scatter(
            big.mean_confidence,
            big.accuracy,
            marker=mk,
            s=26,
            color=col,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        ax.scatter(
            small.mean_confidence,
            small.accuracy,
            marker=mk,
            s=26,
            facecolors="white",
            edgecolors=col,
            linewidths=1.1,
            zorder=3,
        )
        for _, r in small.iterrows():
            ax.annotate(
                f"n = {int(r.n)}",
                (r.mean_confidence, r.accuracy),
                textcoords="offset points",
                xytext=off,
                ha="right" if off[0] < 0 else "left",
                fontsize=7,
                color=col,
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Mean predicted confidence", color=INK)
    ax.set_ylabel("Empirical accuracy", color=INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [
        Line2D([0], [0], color=MUTED, linestyle=(0, (4, 3)), linewidth=0.8, label="Perfect calibration"),
        Line2D(
            [0],
            [0],
            color=c_u,
            linestyle=(0, (5, 2)),
            marker="o",
            markersize=4.5,
            markeredgecolor="white",
            label=specs[0][4],
        ),
        Line2D(
            [0],
            [0],
            color=c_c,
            linestyle="-",
            marker="s",
            markersize=4.5,
            markeredgecolor="white",
            label=specs[1][4],
        ),
        Line2D(
            [0],
            [0],
            color=MUTED,
            linestyle="none",
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            label=f"Bin with n < {min_n} windows (not connected)",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.115),
        ncol=2,
        frameon=False,
        fontsize=7.2,
        handlelength=2.4,
        columnspacing=1.2,
        handletextpad=0.5,
        borderaxespad=0.0,
    )
    ax.text(
        -0.13, 1.02, "(a)", transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", color=INK
    )
    edges = np.linspace(0, 1, NB + 1)
    w = 0.1 * 0.38
    for j, (b, col, hatch, lab) in enumerate(
        ((b_unc, c_u, "////", "Uncalibrated"), (b_cal, c_c, None, "Temperature-scaled"))
    ):
        nz = b[b.n > 0]
        x = nz.lower + 0.05 + (j - 0.5) * w
        axh.bar(
            x,
            nz.n,
            width=w,
            color=col,
            alpha=0.35 if hatch else 0.55,
            edgecolor=col,
            linewidth=0.7,
            hatch=hatch,
            label=lab,
        )
    axh.set_yscale("log")
    axh.set_ylim(0.7, 6000)
    axh.set_xlim(0, 1)
    axh.set_xticks(edges)
    axh.set_xticklabels([f"{e:.1f}" for e in edges])
    axh.set_xlabel("Maximum softmax confidence (10 equal-width bins)", color=INK)
    axh.set_ylabel("Number of windows\n(log scale)", color=INK)
    axh.grid(axis="y", color=GRID, linewidth=0.6)
    axh.set_axisbelow(True)
    for sp in ("top", "right"):
        axh.spines[sp].set_visible(False)
    axh.legend(loc="upper left", frameon=False, fontsize=7.2)
    axh.text(
        -0.13, 1.04, "(b)", transform=axh.transAxes, fontsize=9.5, fontweight="bold", va="bottom", color=INK
    )
    save_all(fig, FIG2_OUT)


def fig2(args):
    from stat_utils import calibration_metrics

    logits, y, T, chk = verify_and_compute()
    b_unc, conf_u, corr_u = bins(logits, y, 1.0)
    b_cal, conf_c, corr_c = bins(logits, y, T)
    ref_u, ref_c = (calibration_metrics(logits, y, 1.0), calibration_metrics(logits, y, T))
    chk.update(
        {
            "T": T,
            "accuracy": float(corr_u.mean()),
            "accuracy_calibrated": float(corr_c.mean()),
            "ece_uncalibrated": ref_u["ece"],
            "ece_calibrated": ref_c["ece"],
            "bin_sum_reproduces_ece_uncal": abs(b_unc.ece_contribution.sum() - ref_u["ece"]) < 1e-12,
            "bin_sum_reproduces_ece_cal": abs(b_cal.ece_contribution.sum() - ref_c["ece"]) < 1e-12,
            "bin_counts_sum_uncal": int(b_unc.n.sum()),
            "bin_counts_sum_cal": int(b_cal.n.sum()),
            "targets_4dp": {
                "T": round(T, 4) == 1.6497,
                "accuracy": round(float(corr_u.mean()), 4) == 0.971,
                "ece_uncal": round(ref_u["ece"], 4) == 0.0181,
                "ece_cal": round(ref_c["ece"], 4) == 0.0133,
            },
        }
    )
    must = [
        chk["masks_equal_saved"],
        chk["n_evaluation_valid"] == 3000,
        chk["n_calibration_valid"] == 3000,
        chk["calibration_evaluation_rows_disjoint"],
        chk["evaluation_labels_match_rebuilt_windows"],
        chk["bin_sum_reproduces_ece_uncal"],
        chk["bin_sum_reproduces_ece_cal"],
        chk["bin_counts_sum_uncal"] == 3000,
        chk["bin_counts_sum_cal"] == 3000,
        *chk["targets_4dp"].values(),
    ]
    with open(out("figures", "Figure2_TEP_verification.json"), "w") as fh:
        json.dump(chk, fh, indent=2, default=float)
    if not all(must):
        print(json.dumps(chk, indent=1, default=float))
        sys.exit("Verification failed, no figure produced")
    tab = pd.concat(
        [
            b_unc.assign(condition="uncalibrated (T = 1)"),
            b_cal.assign(condition=f"temperature-scaled (T = {T:.4f})"),
        ]
    )
    tab[
        [
            "condition",
            "bin",
            "lower",
            "upper",
            "n",
            "mean_confidence",
            "accuracy",
            "abs_gap",
            "ece_contribution",
        ]
    ].to_csv(out("figures", "Figure2_TEP_Calibration_Bins.csv"), index=False, float_format="%.10f")
    plot_calibration(b_unc, b_cal, T, ref_u["ece"], ref_c["ece"])
    print(json.dumps(chk, indent=1, default=float))


def compute_real_ig_attributions(args):
    import torch
    import torch.nn.functional as F
    import joblib
    from tep_data import FastTEPLoader as TEPDataLoader
    from models import Sensor1DCNN, XAIExplainer

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}. Train the final TEP model first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing real Integrated Gradients attributions on: {device}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_sensors = checkpoint["num_sensors"]
    num_classes = checkpoint["num_classes"]
    window_size = checkpoint["window_size"]
    temperature = float(checkpoint["temperature"])
    model = Sensor1DCNN(num_sensors, num_classes, window_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader_obj = TEPDataLoader(base_dir=args.data_dir)
    datasets = loader_obj.load_all_datasets()
    test_df = datasets["faulty_testing"]
    sensor_cols = [c for c in test_df.columns if c.startswith("xmeas") or c.startswith("xmv")]
    label_col = "faultNumber" if "faultNumber" in test_df.columns else test_df.columns[0]
    block = test_df[test_df[label_col] == args.fault]
    window = block.iloc[args.start : args.start + window_size]
    if len(window) < window_size:
        raise ValueError(
            f"Fault {args.fault} has only {len(block)} rows; window starting at {args.start} with size {window_size} is out of range."
        )
    features = window[sensor_cols].values
    if "scaler_mean" in checkpoint and "scaler_std" in checkpoint:
        features = (features - checkpoint["scaler_mean"]) / checkpoint["scaler_std"]
        print("Applied the checkpoint's per-sensor standardisation.")
    else:
        raise SystemExit(
            f"{args.checkpoint} carries no scaler statistics. An attribution map computed on unstandardised input is invalid. Use best_1dcnn_fixed.pth."
        )
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    le = joblib.load(OUT_DIR / "fault_label_encoder.pkl")
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits / temperature, dim=1)
        pred_class_idx = int(torch.argmax(probs).item())
        confidence = float(torch.max(probs).item())
    pred_fault = int(le.inverse_transform([pred_class_idx])[0])
    print(
        f"Window: fault {args.fault} @ row {args.start} | predicted fault {pred_fault} | calibrated confidence {confidence:.4f}"
    )
    explainer = XAIExplainer(model=model, device=device)
    _, attr_matrix = explainer.explain_prediction(
        input_tensor=x, target_class=pred_class_idx, sensor_names=sensor_cols
    )
    result = {
        "attr_matrix": np.asarray(attr_matrix, dtype=np.float64),
        "sensor_names": np.array(sensor_cols),
        "true_fault": np.array([args.fault]),
        "pred_fault": np.array([pred_fault]),
        "confidence": np.array([confidence]),
        "temperature": np.array([temperature]),
    }
    np.savez(args.cache, **result)
    print(f"Cached attributions to '{args.cache}'")
    return result


def load_cached_attributions(cache_path):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No cached attributions at '{cache_path}'. Run this command without --from-cache (requires the TEP dataset and the trained checkpoint). The figure must show real attributions, never simulated ones."
        )
    data = np.load(cache_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def select_top_sensors(attr_matrix, sensor_names, top_n):
    importance = np.sum(np.abs(attr_matrix), axis=0)
    order = np.argsort(importance)[::-1][:top_n]
    sub = attr_matrix[:, order].T
    labels = [str(sensor_names[i]) for i in order]
    return (sub, labels)


def plot_xai_heatmap(data, output_path, top_n=15):
    import seaborn as sns

    attr_matrix = data["attr_matrix"]
    sensor_names = data["sensor_names"]
    window_size = attr_matrix.shape[0]
    matrix, labels = select_top_sensors(attr_matrix, sensor_names, top_n)
    magnitude = np.abs(matrix)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        magnitude,
        cmap="Reds",
        cbar_kws={"label": "Integrated Gradients Attribution (|attribution|)"},
        xticklabels=5,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel(f"Time Step (Window Size = {window_size})", fontweight="bold", labelpad=10)
    ax.set_ylabel("Process Sensors (ranked by total attribution)", fontweight="bold", labelpad=10)
    true_fault = int(np.asarray(data["true_fault"]).ravel()[0])
    pred_fault = int(np.asarray(data["pred_fault"]).ravel()[0])
    confidence = float(np.asarray(data["confidence"]).ravel()[0])
    ax.set_title(
        f"Spatio-Temporal Attribution Analysis via Integrated Gradients\n(Ground truth: Fault {true_fault} | Prediction: Fault {pred_fault} | Calibrated confidence: {confidence:.2%})",
        fontweight="bold",
        pad=20,
    )
    plt.yticks(rotation=0, fontweight="bold")
    plt.xticks(fontweight="bold")
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Figure saved at 600 DPI: {output_path}")
    print("\n[Figure source values - top sensors by total |attribution|]")
    importance = np.sum(np.abs(attr_matrix), axis=0)
    total = importance.sum()
    order = np.argsort(importance)[::-1][:10]
    for rank, i in enumerate(order, 1):
        share = importance[i] / total if total > 0 else 0.0
        print(f"  {rank:>2}. {str(sensor_names[i]):<12} {share * 100:6.2f}% contribution")


def fig3(args):
    import seaborn as sns

    plt.style.use("default")
    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "serif",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "figure.autolayout": True,
        }
    )
    if args.from_cache:
        data = load_cached_attributions(args.cache)
    else:
        data = compute_real_ig_attributions(args)
    target_path = str(out("figures", "Figure3_XAI_Heatmap.png"))
    plot_xai_heatmap(data, target_path, top_n=args.top_n)
    print(f"Figure written to {target_path}")


def main():
    parser = argparse.ArgumentParser(description="Manuscript figures")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fig1")
    p.set_defaults(func=fig1)
    p = sub.add_parser("fig2")
    p.set_defaults(func=fig2)
    p = sub.add_parser("fig3")
    p.add_argument("--checkpoint", type=str, default=str(OUT_DIR / "best_1dcnn_fixed.pth"))
    p.add_argument("--data-dir", type=str, default=str(TEP_DIR))
    p.add_argument("--fault", type=int, default=4)
    p.add_argument("--start", type=int, default=500)
    p.add_argument("--top-n", type=int, default=15)
    p.add_argument("--from-cache", action="store_true")
    p.add_argument("--cache", type=str, default=str(ATTRIBUTION_CACHE))
    p.set_defaults(func=fig3)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
