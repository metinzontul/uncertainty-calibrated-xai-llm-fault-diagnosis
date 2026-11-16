import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
import cwru_data
import cwru_experiments as ra
import train_classifiers as ac
import xai_evaluation as xe
from cwru_data import add_noise, measured_snr_db
from llm import TEP_FAULT_DESCRIPTIONS, DiagnosisPayload, summarise_window
from llm_experiment import dump_cases, load_cases
from models import ModelTrainer, Sensor1DCNN, XAIExplainer
from paths import CWRU_DIR, CWRU_LOADS_DIR, OUT_DIR, TEP_DIR, out
from stat_utils import calibration_metrics, risk_coverage_curve, selective_risk_at_coverage
from validation_round1 import step6_load

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CWRU_RES = OUT_DIR / "results" / "cwru"
WIN, STR, SEEDS = (1024, 128, (42, 43, 44, 45, 46))
LOADS = {
    1: [98, 106, 170, 210, 119, 186, 223, 131, 198, 235],
    2: [99, 107, 171, 211, 120, 187, 224, 132, 199, 236],
    3: [100, 108, 172, 212, 121, 188, 225, 133, 200, 237],
}
CHSETS = {"DE+FE+BA": ["DE", "FE", "BA"], "DE+FE": ["DE", "FE"], "DE": ["DE"]}
W, S, ONSET_TR, ONSET_TE = (60, 10, 20, 160)
MODELS = {
    "gpt5_mini": "GPT-5 Mini",
    "qwen35_9b": "Qwen3.5-9B",
    "deepseek_v41_flash": "DeepSeek-V4.1-Flash-1017",
    "glm53_flash": "GLM-5.3-Flash",
}
SHUF = ("rank_only_shuffled", "percentages_only_shuffled")
HEAD = {
    "rank_only_shuffled": "EXPLAINABILITY EVIDENCE (Integrated Gradients attribution; the number before each sensor is its rank by contribution to this prediction; sensors are listed in arbitrary order):",
    "percentages_only_shuffled": "EXPLAINABILITY EVIDENCE (Integrated Gradients attribution share of each sensor; sensors are listed in arbitrary order):",
}
INSTR = "Explain to the operator why this fault was reported. Ground your root-cause statement in the attribution evidence above and cite the specific sensors it identifies."
FINAL = OUT_DIR / "llm_revision" / "llm_final_consolidated" / "final_results.csv"
ERS_W = {"c": 0.4, "s": 0.25, "a": 0.2}


def r2(*parts):
    return out("validation_round2", *parts)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(logits, y, T):
    p = softmax(logits.astype(np.float64) / T)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))


def fit_T(logits, y):
    return float(minimize_scalar(lambda t: nll(logits, y, t), bounds=(0.02, 1000), method="bounded").x)


def evaluate(logits, y, T):
    p = softmax(logits.astype(np.float64) / T)
    pred = p.argmax(1)
    correct = (pred == y).astype(float)
    cm = calibration_metrics(logits, y, temperature=T)
    _, _, aurc = risk_coverage_curve(p.max(1), correct)
    res = {
        "n": int(len(y)),
        "T": float(T),
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "mean_conf": float(p.max(1).mean()),
        "full_risk": float(1 - correct.mean()),
        "risk_at_70": float(selective_risk_at_coverage(p.max(1), correct, 0.7)),
        "aurc": float(aurc),
    }
    res.update({k: float(v) for k, v in cm.items() if isinstance(v, (int, float, np.floating))})
    return res


def save(name, obj):
    path = r2("results", name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"saved {path}", flush=True)


def purged_halves(y, gap):
    a, b = ([], [])
    for c in np.unique(y):
        pos = np.where(y == c)[0]
        cut = len(pos) // 2
        a.append(pos[: max(cut - gap, 0)])
        b.append(pos[cut:])
    return (np.concatenate(a), np.concatenate(b))


def windows(block, cols, stride=STR):
    v = block[cols].to_numpy(np.float32)
    return np.stack([v[s : s + WIN] for s in range(0, len(v) - WIN + 1, stride)])


def fold_data(train_df, fold, cols):
    xa, ya, xb, yb = ([], [], [], [])
    for lab, blk in train_df.groupby("label", sort=True):
        n = len(blk)
        a, b = (int(n * (0.4 + 0.1 * fold)), int(n * (0.6 + 0.1 * fold)))
        wa, wb = (windows(blk.iloc[:a], cols), windows(blk.iloc[a:b], cols))
        xa.append(wa)
        ya.append(np.full(len(wa), lab))
        xb.append(wb)
        yb.append(np.full(len(wb), lab))
    xa, xb = (np.concatenate(xa), np.concatenate(xb))
    mu, sd = (xa.mean((0, 1)), xa.std((0, 1)))
    sd[sd < 1e-08] = 1.0
    return (
        np.ascontiguousarray((xa - mu) / sd, dtype=np.float32),
        np.concatenate(ya).astype(np.int64),
        np.ascontiguousarray((xb - mu) / sd, dtype=np.float32),
        np.concatenate(yb).astype(np.int64),
    )


def e1_csv():
    return r2("results", os.environ.get("E1_OUT", "e1_cwru_cv_rows.csv"))


def append(row):
    path = e1_csv()
    new = not os.path.exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=new, index=False)


def done_keys():
    path = e1_csv()
    if not os.path.exists(path):
        return set()
    d = pd.read_csv(path)
    return set(zip(d.part, d.channels, d.Model, d.Seed, d.Fold))


def part_ab(train_df):
    torch_models, _ = ra.build_torch_models()
    done = done_keys()
    for part, chsets in [
        p for p in (("A", ["DE+FE+BA"]), ("B", ["DE+FE", "DE"])) if p[0] in os.environ.get("E1_PARTS", "AB")
    ]:
        for chn in chsets:
            cols = CHSETS[chn]
            for seed in SEEDS:
                for fold in range(5):
                    names = (
                        ["Logistic Regression", "XGBoost"] + list(torch_models) if part == "A" else ["1D-CNN"]
                    )
                    if all(((part, chn, n, seed, fold + 1) in done for n in names)):
                        continue
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    xtr, ytr, xva, yva = fold_data(train_df, fold, cols)
                    ra.SENSOR_COLS = list(cols)
                    recs = ra.train_sklearn_models(xtr, ytr, xva, yva, 10, seed) if part == "A" else []
                    for name, cls in torch_models.items():
                        if part == "A" or name == "1D-CNN":
                            recs.append(
                                ra.train_torch_model(name, cls, xtr, ytr, xva, yva, 10, dev, 10, 64, WIN)
                            )
                    for r in recs:
                        append(
                            {
                                "part": part,
                                "channels": chn,
                                "Model": r["Model"],
                                "Seed": seed,
                                "Fold": fold + 1,
                                "Val_F1": r["Val_F1"],
                                "Val_Accuracy": r["Val_Accuracy"],
                                "Val_ECE": r["Val_ECE"],
                                "n_train": len(ytr),
                                "n_val": len(yva),
                            }
                        )
                        print(part, chn, seed, fold + 1, r["Model"], round(r["Val_F1"], 4), flush=True)
                    torch.cuda.empty_cache()


def original_overlap(train_df):
    res = {}
    for fold in range(5):
        shared = total = 0
        for _, blk in train_df.groupby("label", sort=True):
            n_w = (len(blk) - WIN) // STR + 1
            a, b = (int(n_w * (0.4 + 0.1 * fold)), int(n_w * (0.6 + 0.1 * fold)))
            last_train_end = (a - 1) * STR + WIN
            shared += sum((1 for i in range(a, b) if i * STR < last_train_end))
            total += b - a
        res[f"fold{fold + 1}"] = {
            "val_windows": total,
            "sharing_samples_with_train": shared,
            "fraction": shared / total,
        }
    return res


def load_file(path, label, cols):
    m = sio.loadmat(path)
    ch = {}
    for c in ("DE", "FE", "BA"):
        k = next((k for k in m if k.endswith(f"_{c}_time")), None)
        if k is not None:
            ch[c] = m[k].ravel()
    n = min((len(v) for v in ch.values()))
    df = pd.DataFrame({c: ch[c][:n] if c in ch else np.zeros(n) for c in ("DE", "FE", "BA")})
    return (
        windows(df, cols, stride=WIN),
        np.full((n - WIN) // WIN + 1, label),
        [c for c in ("DE", "FE", "BA") if c not in ch],
    )


def logits_of(m, x):
    m.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            outs.append(m(torch.tensor(x[i : i + 256], device=dev)).cpu().numpy())
    return np.concatenate(outs).astype(np.float64)


def ig_shares(m, x, cols, n=200):
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(m)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x), min(n, len(x)), replace=False)
    sh = []
    for i in range(0, len(idx), 20):
        xb = torch.tensor(x[idx[i : i + 20]], device=dev)
        with torch.no_grad():
            t = m(xb).argmax(1)
        a = (
            ig.attribute(xb, baselines=torch.zeros_like(xb), target=t, n_steps=50, method="gausslegendre")
            .abs()
            .sum(1)
            .cpu()
            .numpy()
        )
        sh.append(a / a.sum(1, keepdims=True))
    sh = np.concatenate(sh)
    return {
        "mean_share": dict(zip(cols, sh.mean(0).round(4).tolist())),
        "top1_freq": dict(zip(cols, np.bincount(sh.argmax(1), minlength=len(cols)) / len(sh))),
    }


def part_c(train_df, test_df):
    res = {}
    for chn, cols in CHSETS.items():
        xtr, ytr = ([], [])
        for lab, blk in train_df.groupby("label", sort=True):
            w = windows(blk, cols)
            xtr.append(w)
            ytr.append(np.full(len(w), lab))
        xtr = np.concatenate(xtr)
        ytr = np.concatenate(ytr).astype(np.int64)
        mu, sd = (xtr.mean((0, 1)), xtr.std((0, 1)))
        sd[sd < 1e-08] = 1.0
        xtr = ((xtr - mu) / sd).astype(np.float32)
        xte, yte = ([], [])
        for lab, blk in test_df.groupby("label", sort=True):
            w = windows(blk, cols)
            xte.append(w)
            yte.append(np.full(len(w), lab))
        xte = ((np.concatenate(xte) - mu) / sd).astype(np.float32)
        yte = np.concatenate(yte).astype(np.int64)
        cal, ev = purged_halves(yte, WIN // STR - 1)
        other = {}
        for load, files in LOADS.items():
            xs, ys, missing = ([], [], {})
            for lab, fn in enumerate(files):
                p = CWRU_LOADS_DIR / f"{fn}.mat"
                if not os.path.exists(p):
                    continue
                w, yy, miss = load_file(p, lab, cols)
                xs.append(w)
                ys.append(yy)
                if miss:
                    missing[fn] = miss
            if xs:
                other[load] = (
                    ((np.concatenate(xs) - mu) / sd).astype(np.float32),
                    np.concatenate(ys).astype(np.int64),
                    missing,
                )
        res[chn] = {}
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            net = Sensor1DCNN(len(cols), 10, WIN).to(dev)
            tr = ModelTrainer(net, dev, learning_rate=0.001)
            ld = DataLoader(
                TensorDataset(torch.tensor(xtr), torch.tensor(ytr)),
                batch_size=64,
                shuffle=True,
                drop_last=True,
            )
            for _ in range(10):
                tr.train_epoch(ld)
            L = logits_of(net, xte)
            T = fit_T(L[cal], yte[cal])
            r = {
                "T": T,
                "same_recording_0hp": evaluate(L[ev], yte[ev], T),
                "ig_0hp": ig_shares(net, xte[ev], cols),
            }
            for load, (xo, yo, miss) in other.items():
                Lo = logits_of(net, xo)
                r[f"load_{load}hp"] = evaluate(Lo, yo, T)
                r[f"ig_{load}hp"] = ig_shares(net, xo, cols)
                r[f"missing_channels_{load}hp"] = miss
            res[chn][f"seed{seed}"] = r
            print(
                chn,
                seed,
                round(r["same_recording_0hp"]["macro_f1"], 4),
                {k: round(v["macro_f1"], 4) for k, v in r.items() if k.startswith("load_")},
                flush=True,
            )
            del net
            torch.cuda.empty_cache()
        save("e1c_cwru_cross_load.json", res)
    return res


def e1_main():
    loader = cwru_data.CWRUDataLoader(base_dir=str(CWRU_DIR))
    df = loader.load_all()
    train_df, test_df = loader.train_test_split(df, train_ratio=0.7)
    save("e1_original_overlap.json", original_overlap(train_df))
    if not os.path.exists(r2("results", "e1c_cwru_cross_load.json")):
        part_c(train_df, test_df)
    part_ab(train_df)
    print("done", flush=True)


def starts(y, runs):
    st = np.zeros(len(y), int)
    for c, r in set(zip(y.tolist(), runs.tolist())):
        idx = np.where((y == c) & (runs == r))[0]
        assert np.all(np.diff(idx) == 1), "cache not contiguous per class/run"
        st[idx] = np.arange(len(idx)) * S
    return st


def tensor_loader(x, y, shuffle, bs):
    return DataLoader(
        TensorDataset(torch.tensor(x), torch.tensor(y)), batch_size=bs, shuffle=shuffle, drop_last=shuffle
    )


def torch_logits(m, x):
    m.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            outs.append(m(torch.tensor(x[i : i + 512], device=dev)).cpu().numpy())
    return np.concatenate(outs).astype(np.float64)


def train_torch(cls, x, y, n_cls, epochs, bs, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = cls(x.shape[2], n_cls, W).to(dev)
    tr = ModelTrainer(m, dev, learning_rate=0.001)
    ld = tensor_loader(x, y, True, bs)
    for _ in range(epochs):
        tr.train_epoch(ld)
    return m


def e2_part_a(c, rows_path):
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression

    x_all, y_all, runs = (c["x_train_full"], c["y_train_full"].astype(np.int64), c["runs_train"])
    post = starts(y_all, runs) >= ONSET_TR
    n_cls = int(len(np.unique(y_all)))
    folds = ac.make_run_folds(y_all, runs, 5)
    done = set()
    if os.path.exists(rows_path):
        done = {tuple(l.split(",")[:4]) for l in open(rows_path, encoding="utf-8").read().splitlines()[1:]}
    else:
        open(rows_path, "w", encoding="utf-8").write(
            "regime,model,seed,fold,f1_val_all,f1_val_post,acc_val_all,acc_val_post,n_train,n_val_all,n_val_post\n"
        )
    torch_models = {"1D-CNN": Sensor1DCNN, "LSTM": ac.SensorLSTM, "KAN": ac.SensorKAN}
    for regime in ("ORIG", "CONS"):
        for f, vm in enumerate(folds, 1):
            tm = ~vm if regime == "ORIG" else ~vm & post
            xtr, ytr = (x_all[tm], y_all[tm])
            xva, yva, pv = (x_all[vm], y_all[vm], post[vm])
            xtr, xva = ac.standardise(xtr, xva)

            def log(model, seed, pred):
                line = [
                    regime,
                    model,
                    str(seed),
                    str(f),
                    f"{ac.macro_f1(yva, pred):.6f}",
                    f"{ac.macro_f1(yva[pv], pred[pv]):.6f}",
                    f"{(pred == yva).mean():.6f}",
                    f"{(pred[pv] == yva[pv]).mean():.6f}",
                    str(len(ytr)),
                    str(len(yva)),
                    str(int(pv.sum())),
                ]
                open(rows_path, "a", encoding="utf-8").write(",".join(line) + "\n")
                print(time.strftime("%H:%M:%S"), line[:6], flush=True)

            for name, cls in torch_models.items():
                for seed in (42, 43, 44, 45, 46):
                    if (regime, name, str(seed), str(f)) in done:
                        continue
                    m = train_torch(cls, xtr, ytr, n_cls, 10, 64, seed)
                    log(name, seed, torch_logits(m, xva).argmax(1))
                    del m
                    torch.cuda.empty_cache()
            ftr, fva = (xtr.reshape(len(xtr), -1), xva.reshape(len(xva), -1))
            if (regime, "XGBoost", "42", str(f)) not in done:
                xm = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    objective="multi:softprob",
                    random_state=42,
                    eval_metric="mlogloss",
                    tree_method="hist",
                    device="cuda" if dev.type == "cuda" else "cpu",
                )
                xm.fit(ftr, ytr)
                log("XGBoost", 42, xm.predict(fva))
                del xm
            if (regime, "Logistic Regression", "42", str(f)) not in done:
                lr = LogisticRegression(max_iter=500, n_jobs=-1, solver="lbfgs", random_state=42).fit(
                    ftr, ytr
                )
                log("Logistic Regression", 42, lr.predict(fva))
                del lr


def e2_part_b(c, epochs=15, suffix=""):
    x_tr, y_tr, r_tr = (c["x_train_full"], c["y_train_full"].astype(np.int64), c["runs_train"])
    sel = r_tr <= 10
    post_tr = starts(y_tr, r_tr) >= ONSET_TR
    x_te, y_te, r_te = (c["x_test"], c["y_test"].astype(np.int64), c["runs_test"])
    st_te = starts(y_te, r_te)
    cal, ev = purged_halves(y_te, W // S - 1)
    cal, ev = (cal[st_te[cal] >= ONSET_TE], ev[st_te[ev] >= ONSET_TE])
    n_cls = int(len(np.unique(y_tr)))
    res = {"n_cal_post_onset": int(len(cal)), "n_eval_post_onset": int(len(ev)), "runs": {}}
    ck = torch.load(OUT_DIR / "best_1dcnn_fixed.pth", map_location=dev, weights_only=False)
    m0 = Sensor1DCNN(ck["num_sensors"], ck["num_classes"], ck["window_size"]).to(dev)
    m0.load_state_dict(ck["model_state_dict"])
    L = torch_logits(m0, ((x_te - ck["scaler_mean"]) / ck["scaler_std"]).astype(np.float32))
    T = fit_T(L[cal], y_te[cal])
    res["published_checkpoint"] = {
        "refit_T": evaluate(L[ev], y_te[ev], T),
        "T_1.6497": evaluate(L[ev], y_te[ev], 1.649679183959961),
    }
    for regime in ("ORIG", "CONS"):
        m = sel if regime == "ORIG" else sel & post_tr
        xr = x_tr[m]
        mu = xr.mean((0, 1))
        sd = xr.std((0, 1))
        sd[sd < 1e-08] = 1.0
        xs, xt = (((xr - mu) / sd).astype(np.float32), ((x_te - mu) / sd).astype(np.float32))
        for seed in (42, 43, 44, 45, 46):
            net = train_torch(Sensor1DCNN, xs, y_tr[m], n_cls, epochs, 128, seed)
            L = torch_logits(net, xt)
            T = fit_T(L[cal], y_te[cal])
            res["runs"][f"{regime}_seed{seed}"] = dict(evaluate(L[ev], y_te[ev], T), n_train=int(m.sum()))
            print(
                regime,
                seed,
                round(res["runs"][f"{regime}_seed{seed}"]["accuracy"], 4),
                round(T, 3),
                flush=True,
            )
    for regime in ("ORIG", "CONS"):
        rs = [v for k, v in res["runs"].items() if k.startswith(regime)]
        res[f"summary_{regime}"] = {
            k: {"mean": float(np.mean([r[k] for r in rs])), "sd": float(np.std([r[k] for r in rs], ddof=1))}
            for k in rs[0]
            if isinstance(rs[0][k], float)
        }
    save(f"e2b_tep_calibration_model{suffix}.json", res)


def e2_main():
    c = np.load(OUT_DIR / "tep_windows_cache.npz")
    c = {k: c[k] for k in c.files}
    e2_part_b(c)
    e2_part_a(c, r2("results", "e2a_tep_table3_rows.csv"))
    print("done", flush=True)


def e2b_main():
    c = np.load(OUT_DIR / "tep_windows_cache.npz")
    e2_part_b({k: c[k] for k in c.files}, epochs=20, suffix="_epochs20")


def load_model(path):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = Sensor1DCNN(
        int(ck.get("num_sensors", 3)), int(ck.get("num_classes", 10)), int(ck.get("window_size", 1024))
    ).to(dev)
    m.load_state_dict(ck.get("model_state_dict", ck))
    m.eval()
    return (m, float(ck["temperature"]))


def e3_main():
    P = np.load(CWRU_RES / "prepared_windows.npz")
    x, y, mu, sd = (P["x_test"], P["y_test"], P["channel_mean"], P["channel_std"])
    gap = int(np.ceil(int(P["window_size"][0]) / int(P["stride"][0]))) - 1
    cal, ev = purged_halves(y, gap)
    model, T_pub = load_model(CWRU_RES / "cwru_1dcnn_calibrated.pth")
    raw = x * sd + mu
    L0 = logits_of(model, x)
    T_clean = fit_T(L0[cal], y[cal])
    res = {
        "purge_gap_windows": gap,
        "n_calibration": int(len(cal)),
        "n_evaluation": int(len(ev)),
        "T_published": T_pub,
        "T_clean_refit_purged": T_clean,
        "clean": {"published_T": evaluate(L0[ev], y[ev], T_pub), "refit_T": evaluate(L0[ev], y[ev], T_clean)},
        "snr": {},
    }
    for snr in (10, 5, 0):
        noisy = add_noise(raw, snr, "white", seed=43)
        xs = ((noisy - mu) / sd).astype(np.float32)
        L = logits_of(model, xs)
        T_noisy = fit_T(L[cal], y[cal])
        r = {
            "measured_snr_db": float(measured_snr_db(raw, noisy)),
            "F_cleanT": evaluate(L[ev], y[ev], T_clean),
            "F_publishedT": evaluate(L[ev], y[ev], T_pub),
            "F_noisyT": evaluate(L[ev], y[ev], T_noisy),
        }
        d = CWRU_RES / f"noise_white_{snr}dB"
        if os.path.exists(d / "cwru_1dcnn_calibrated.pth"):
            rm, T_own_stored = load_model(d / "cwru_1dcnn_calibrated.pth")
            Q = np.load(d / "prepared_windows.npz")
            yq = Q["y_test"]
            cq, eq = purged_halves(yq, gap)
            Lq = logits_of(rm, Q["x_test"])
            T_own = fit_T(Lq[cq], yq[cq])
            r.update(
                {
                    "R_cleanT_0.4420": evaluate(Lq[eq], yq[eq], T_pub),
                    "R_ownT": evaluate(Lq[eq], yq[eq], T_own),
                    "R_ownT_stored_checkpoint": T_own_stored,
                }
            )
        res["snr"][f"{snr}dB"] = r
        print(
            snr,
            {
                k: (round(v["accuracy"], 4), round(v.get("ece", np.nan), 4), round(v["T"], 3))
                for k, v in r.items()
                if isinstance(v, dict)
            },
            flush=True,
        )
    save("e3_cwru_fixed_noise.json", res)


def riemann_signed(model, x, t, steps=32):
    tot = torch.zeros_like(x)
    for s in range(steps):
        pt = ((s + 0.5) / steps * x).detach().requires_grad_(True)
        g = torch.autograd.grad(model(pt).gather(1, t.unsqueeze(1)).sum(), pt)[0]
        tot = tot + g
    return x * tot / steps


def attributions(model, x_all, batch=16):
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(model)
    atts = {"R32": [], "GL50": [], "GL300": []}
    gaps = {k: [] for k in atts}
    preds = []
    for st in range(0, len(x_all), batch):
        x = torch.tensor(x_all[st : st + batch], device=dev)
        with torch.no_grad():
            t = model(x).argmax(1)
            fx = model(x).gather(1, t.unsqueeze(1)).squeeze(1)
            f0 = model(torch.zeros_like(x)).gather(1, t.unsqueeze(1)).squeeze(1)
        preds.append(t.cpu().numpy())
        a = {
            "R32": riemann_signed(model, x, t, 32),
            "GL50": ig.attribute(
                x,
                baselines=torch.zeros_like(x),
                target=t,
                n_steps=50,
                method="gausslegendre",
                internal_batch_size=batch * 50,
            ),
            "GL300": ig.attribute(
                x,
                baselines=torch.zeros_like(x),
                target=t,
                n_steps=300,
                method="gausslegendre",
                internal_batch_size=batch * 50,
            ),
        }
        for k, v in a.items():
            v = v.detach()
            gaps[k].append(
                (v.sum((1, 2)) - (fx - f0)).abs().div((fx - f0).abs().clamp_min(1e-08)).cpu().numpy()
            )
            atts[k].append(v.cpu().numpy())
    return (
        {k: np.concatenate(v) for k, v in atts.items()},
        {k: np.concatenate(v) for k, v in gaps.items()},
        np.concatenate(preds),
    )


def agreement(A, B, k):
    sa, sb = (np.abs(A).sum(1), np.abs(B).sum(1))
    sa, sb = (sa / sa.sum(1, keepdims=True), sb / sb.sum(1, keepdims=True))
    r = {
        "kendall_tau_b": [],
        "spearman": [],
        "top1": [],
        f"top{k}_set_jaccard": [],
        f"top{k}_same_order": [],
        "entry_top10pct_overlap": [],
        "max_abs_share_diff": [],
    }
    for i in range(len(sa)):
        r["kendall_tau_b"].append(kendalltau(sa[i], sb[i]).statistic)
        r["spearman"].append(spearmanr(sa[i], sb[i]).statistic)
        oa, ob = (np.argsort(-sa[i], kind="stable")[:k], np.argsort(-sb[i], kind="stable")[:k])
        r["top1"].append(float(oa[0] == ob[0]))
        r[f"top{k}_set_jaccard"].append(len(set(oa) & set(ob)) / len(set(oa) | set(ob)))
        r[f"top{k}_same_order"].append(float(np.array_equal(oa, ob)))
        fa, fb = (np.abs(A[i]).ravel(), np.abs(B[i]).ravel())
        n = int(round(0.1 * fa.size))
        r["entry_top10pct_overlap"].append(len(set(np.argsort(-fa)[:n]) & set(np.argsort(-fb)[:n])) / n)
        r["max_abs_share_diff"].append(float(np.abs(sa[i] - sb[i]).max()))
    return {m: {"mean": float(np.nanmean(v)), "min": float(np.nanmin(v))} for m, v in r.items()}


def ig_deletion(model, x_all, y_all, att_all, batch=32):
    rng = np.random.default_rng(42)
    tot = {(s, f): 0.0 for s in xe.STRATEGIES for f in xe.FRACTIONS}
    spread = {k: [] for k in tot}
    hits = 0
    for st in range(0, len(x_all), batch):
        x = torch.tensor(x_all[st : st + batch], device=dev)
        y = torch.tensor(y_all[st : st + batch], device=dev)
        att = torch.tensor(np.abs(att_all[st : st + batch]), device=dev)
        with torch.no_grad():
            hits += float((model(x).argmax(1) == y).float().sum())
        for s in xe.STRATEGIES:
            for f in xe.FRACTIONS:
                if s == "random":
                    d = [xe.masked_accuracy(model, x, y, att, f, s, rng) for _ in range(xe.RANDOM_REPEATS)]
                    a = float(np.mean(d))
                    spread[s, f].append(float(np.std(d, ddof=1)))
                else:
                    a = xe.masked_accuracy(model, x, y, att, f, s, rng)
                tot[s, f] += a * len(x)
    base = hits / len(x_all)
    res = {"baseline_accuracy": base, "n_windows": int(len(x_all)), "fractions": {}}
    for f in xe.FRACTIONS:
        res["fractions"][f"{f:.2f}"] = {
            s: dict(
                {"accuracy": tot[s, f] / len(x_all), "drop": base - tot[s, f] / len(x_all)},
                **{"sd": float(np.mean(spread[s, f]))} if spread[s, f] else {},
            )
            for s in xe.STRATEGIES
        }
    return res


def e4_main():
    res = {}
    model, x, y, cat, _ = step6_load()
    perm = np.random.default_rng(42).permutation(len(x))
    idx = perm[cat[perm] == "valid"][:300]
    data = {"TEP": (model, x[idx], y[idx], 5)}
    m2, x2, y2 = xe.load_cwru(300, dev)
    data["CWRU"] = (m2, x2, y2, 3)
    for ds, (m, xx, yy, k) in data.items():
        m.eval()
        A, gaps, pred = attributions(m, xx)
        r = {
            "completeness_rel_error": {
                kk: {"median": float(np.median(v)), "p95": float(np.percentile(v, 95))}
                for kk, v in gaps.items()
            },
            "agreement_R32_vs_GL50": agreement(A["R32"], A["GL50"], k),
            "agreement_GL50_vs_GL300": agreement(A["GL50"], A["GL300"], k),
            "agreement_R32_vs_GL300": agreement(A["R32"], A["GL300"], k),
            "deletion_GL50_supplied_to_LLMs": ig_deletion(m, xx, yy, A["GL50"]),
            "deletion_R32_reproduction": ig_deletion(m, xx, yy, A["R32"]),
        }
        res[ds] = r
        print(
            ds,
            r["agreement_R32_vs_GL50"]["top1"],
            r["deletion_GL50_supplied_to_LLMs"]["fractions"]["0.10"],
            flush=True,
        )
    save("e4_ig_convergence.json", res)


def predict_case(model, T, xs):
    x = torch.tensor(xs, dtype=torch.float32).unsqueeze(0).to(dev)
    with torch.no_grad():
        lg = model(x)
        raw = float(F.softmax(lg, 1).max())
        p = F.softmax(lg / T, 1)
    return (x, int(p.argmax()), float(p.max()), raw)


def case_reliability(model, expl, x, target, names):
    return xe.analyse_window(
        model, expl, x, target, names, xe.N_PERTURBATIONS, xe.PERTURBATION_SCALE, np.random.default_rng(42)
    )


def tep_cases():
    import joblib

    ck = torch.load(OUT_DIR / "best_1dcnn_fixed.pth", map_location=dev, weights_only=False)
    m = Sensor1DCNN(ck["num_sensors"], ck["num_classes"], ck["window_size"]).to(dev)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    T = float(ck["temperature"])
    le = joblib.load(OUT_DIR / "fault_label_encoder.pkl")
    ex = XAIExplainer(model=m, device=dev)
    df = pd.read_parquet(TEP_DIR / "parquet_cache" / "faulty_testing.parquet")
    cols = [c for c in df.columns if c.startswith("xmeas") or c.startswith("xmv")]
    rng = np.random.default_rng(2026)
    payloads = []
    for f in range(1, 21):
        run = 6 + (f - 1) % 10
        blk = df[(df.faultNumber == f) & (df.simulationRun == run)].sort_values("sample")
        start = int(rng.integers(160, len(blk) - 60 + 1))
        w = blk.iloc[start : start + 60]
        assert w["sample"].min() > 160 and len(w) == 60
        raw = w[cols].values.astype(float)
        xs = (raw - ck["scaler_mean"]) / ck["scaler_std"]
        x, pi, conf, rawc = predict_case(m, T, xs)
        imp, _ = ex.explain_prediction(input_tensor=x, target_class=pi, sensor_names=cols)
        stab, agr = case_reliability(m, ex, x, pi, cols)
        payloads.append(
            DiagnosisPayload(
                case_id=f"new_tep_fault{f}_run{run}_s{start}",
                scenario=TEP_FAULT_DESCRIPTIONS.get(f, f"Fault {f}"),
                true_fault=f,
                predicted_fault=int(le.inverse_transform([pi])[0]),
                calibrated_confidence=conf,
                raw_confidence=rawc,
                temperature=T,
                top_evidence=summarise_window(xs, cols, attributions=imp, top_k=5),
                degradation="none",
                degradation_level=0.0,
                xai_stability=stab,
                xai_agreement=agr,
                data_quality=1.0,
                window_start=start,
            )
        )
        print(
            payloads[-1].case_id,
            payloads[-1].predicted_fault,
            round(conf, 3),
            round(stab, 3),
            round(agr, 3),
            flush=True,
        )
    dump_cases(payloads, str(r2("cases", "tep_new.json")))


def cwru_cases():
    cwru_data.activate()
    ck = torch.load(CWRU_RES / "cwru_1dcnn_calibrated.pth", map_location=dev, weights_only=False)
    m = Sensor1DCNN(3, 10, 1024).to(dev)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    T = float(ck["temperature"])
    P = np.load(CWRU_RES / "prepared_windows.npz")
    mu, sd = (P["channel_mean"], P["channel_std"])
    ex = XAIExplainer(model=m, device=dev)
    cols = ["DE", "FE", "BA"]
    files = {1: LOADS[1], 2: LOADS[2]}
    rng = np.random.default_rng(2026)
    payloads = []
    for load, fl in files.items():
        for lab, fn in enumerate(fl):
            mat = sio.loadmat(CWRU_LOADS_DIR / f"{fn}.mat")
            ch = {}
            for c in cols:
                k = next((k for k in mat if k.endswith(f"_{c}_time")), None)
                if k is not None:
                    ch[c] = mat[k].ravel()
            n = min((len(v) for v in ch.values()))
            sig = np.stack([ch[c][:n] if c in ch else np.zeros(n) for c in cols], 1)
            start = int(rng.integers(0, n - 1024 + 1))
            xs = ((sig[start : start + 1024] - mu) / sd).astype(np.float32)
            x, pi, conf, rawc = predict_case(m, T, xs)
            imp, _ = ex.explain_prediction(input_tensor=x, target_class=pi, sensor_names=cols)
            stab, agr = case_reliability(m, ex, x, pi, cols)
            payloads.append(
                DiagnosisPayload(
                    case_id=f"new_cwru_class{lab}_{load}hp_file{fn}_s{start}",
                    scenario=cwru_data.FAULT_DESCRIPTIONS[lab],
                    true_fault=lab,
                    predicted_fault=pi,
                    calibrated_confidence=conf,
                    raw_confidence=rawc,
                    temperature=T,
                    top_evidence=summarise_window(xs, cols, attributions=imp, top_k=3),
                    degradation="none",
                    degradation_level=0.0,
                    xai_stability=stab,
                    xai_agreement=agr,
                    data_quality=1.0,
                    window_start=start,
                )
            )
            print(payloads[-1].case_id, pi, round(conf, 3), round(stab, 3), round(agr, 3), flush=True)
    dump_cases(payloads, str(r2("cases", "cwru_new.json")))


def cases_main():
    tep_cases()
    cwru_cases()


def perm(n, key):
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
    while True:
        p = rng.permutation(n)
        if p[0] != 0:
            return p


def shuffled(payload, cond, rep):
    ev = list(payload.top_evidence)
    p = perm(len(ev), f"{payload.case_id}|{rep}|{cond}")
    return (dataclasses.replace(payload, top_evidence=[ev[i] for i in p]), [int(i) + 1 for i in p])


def build_shuffled(payload, condition):
    import llm as lp

    cond, rep = condition.split("@")
    pf, ranks = shuffled(payload, cond, int(rep))
    if cond == "rank_only_shuffled":
        lines = [
            f"  {r}. {e.sensor} ({lp.describe_sensor(e.sensor)})" for r, e in zip(ranks, pf.top_evidence)
        ]
    else:
        lines = [
            f"  - {e.sensor} - {e.attribution * 100:.1f}% of the model's attribution ({lp.describe_sensor(e.sensor)})"
            for e in pf.top_evidence
        ]
    return (
        f"{lp._render_header(pf)}\n\nPROCESS VARIABLE SUMMARY (most-changed variables in the window):\n{lp._render_sensor_summary(pf)}\n\n{HEAD[cond]}\n"
        + "\n".join(lines)
        + f"\n\n{INSTR}\n\n{lp._RESPONSE_FORMAT}"
    )


def build_manifests():
    rows = []
    for ds in ("TEP", "CWRU"):
        f = OUT_DIR / "cases" / f"{ds.lower()}_clean.json"
        ids = [c["case_id"] for c in json.load(open(f, encoding="utf-8"))]
        for rep in (1, 2, 3):
            for cid in ids:
                for cond in SHUF:
                    for k, n in MODELS.items():
                        c = f"{cond}@{rep}"
                        rows.append(
                            dict(
                                retry_id=f"e5|{cid}|{k}|{c}",
                                run_block=f"{ds.lower()}_clean_rep{rep}",
                                case_block=f"{ds.lower()}_clean",
                                dataset=ds,
                                rep=rep,
                                case_id=cid,
                                model=k,
                                model_name=n,
                                condition=c,
                                regime="clean",
                                case_file=str(f),
                                start_max_tokens=8192,
                            )
                        )
    pd.DataFrame(rows).to_csv(r2("e5_manifest.csv"), index=False)
    rows = []
    for ds in ("TEP", "CWRU"):
        f = r2("cases", f"{ds.lower()}_new.json")
        for c in json.load(open(f, encoding="utf-8")):
            for cond in ("plain", "xai_grounded", "xai_uncertainty_grounded"):
                if cond == "xai_uncertainty_grounded" and c["calibrated_confidence"] < 0.85:
                    continue
                for k, n in MODELS.items():
                    rows.append(
                        dict(
                            retry_id=f"e6|{c['case_id']}|{k}|{cond}",
                            run_block=f"{ds.lower()}_new",
                            case_block=f"{ds.lower()}_new",
                            dataset=ds,
                            rep=1,
                            case_id=c["case_id"],
                            model=k,
                            model_name=n,
                            condition=cond,
                            regime="clean",
                            case_file=str(f),
                            start_max_tokens=8192,
                        )
                    )
    pd.DataFrame(rows).to_csv(r2("e6_manifest.csv"), index=False)
    print("manifests written")


def run_llm(exp, dataset, models, workers):
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    import llm_revision as rc

    orig_setup = rc.setup_domain
    orig_case_path = rc.case_path

    def setup(ds):
        orig_setup(ds)
        import llm as lp

        base = lp.build_prompt
        lp.build_prompt = lambda p, c, *a, **k: (
            build_shuffled(p, c) if c.split("@")[0] in SHUF else base(p, c, *a, **k)
        )

    rc.setup_domain = setup
    rc.case_path = lambda name: Path(name) if Path(name).is_absolute() else orig_case_path(name)
    rc.ThreadPoolExecutor = lambda max_workers=None, **kw: ThreadPoolExecutor(
        max_workers=min(max_workers or workers, workers), **kw
    )
    out_dir = r2("results", exp)
    os.makedirs(out_dir, exist_ok=True)
    rc.cmd_run(
        argparse.Namespace(
            dataset=dataset,
            models=models,
            limit=0,
            qwen_workers=workers,
            gpt_workers=workers,
            manifest=str(r2(f"{exp}_manifest.csv")),
            out_dir=str(out_dir),
            blocks=None,
        )
    )


def llm_run_main(a):
    run_llm(a.exp, a.dataset, list(MODELS), 2)


def llm_seq_main(a):
    run_llm(a.exp, a.dataset, [a.model], 1)


def boot(df, stat, cluster="window", B=10000, seed=42):
    ids = sorted(df[cluster].unique())
    by = {k: v for k, v in df.groupby(cluster)}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        vals.append(stat(pd.concat([by[k] for k in rng.choice(ids, len(ids))])))
    return (float(stat(df)), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))


def per_case():
    from scipy.stats import binomtest, wilcoxon
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = OUT_DIR / "figures"
    os.makedirs(fig_dir, exist_ok=True)
    R = pd.read_csv(FINAL)
    R = R[R.run_block.str.contains("clean") & (R.final_status == "response")].copy()
    R["window"] = R.case_id.str.replace("_none$", "", regex=True)
    T = R.pivot_table(
        index=["dataset", "window"], columns="condition", values="top1_faithfulness", aggfunc="mean"
    ).reset_index()
    N = R.groupby(["dataset", "window"]).size().rename("n_responses").reset_index()
    T = T.merge(N)
    T["delta_xai_minus_plain"] = T.xai_grounded - T.plain
    T.to_csv(r2("results", "r2_1_per_case_EA1.csv"), index=False)
    summ = {}
    for ds, sub in T.groupby("dataset"):
        d = sub.delta_xai_minus_plain
        summ[ds] = {
            "windows": int(len(sub)),
            "xai_gt_plain": int((d > 0).sum()),
            "equal": int((d == 0).sum()),
            "xai_lt_plain": int((d < 0).sum()),
            "sign_test_p": (
                float(binomtest(int((d > 0).sum()), int((d != 0).sum())).pvalue) if (d != 0).any() else None
            ),
            "wilcoxon_p": float(wilcoxon(d[d != 0]).pvalue) if (d != 0).sum() > 1 else None,
        }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [8, 10]})
    for ax, ds in zip(axes, ("TEP", "CWRU")):
        sub = T[T.dataset == ds].reset_index(drop=True)
        x = np.arange(len(sub))
        for cond, mk, col, lab in (
            ("plain", "o", "#8c8c8c", "Plain (IG-ordered summary)"),
            ("xai_grounded", "s", "#1f6fb2", "XAI-Grounded"),
            ("xai_uncertainty_grounded", "^", "#d1792b", "XAI + Uncertainty"),
        ):
            ax.scatter(
                x + {"plain": -0.18, "xai_grounded": 0, "xai_uncertainty_grounded": 0.18}[cond],
                sub[cond],
                marker=mk,
                color=col,
                label=lab,
                s=36,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                w.replace("fault", "F").replace("cwru_class", "C").split("_at")[0].split("_win")[0]
                for w in sub.window
            ],
            fontsize=8,
        )
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(ds)
        ax.set_ylabel("Evidence Adherence@1 (mean over models and repetitions)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "Figure_R2_1_per_case_EA1.png", dpi=600)
    plt.close(fig)
    save("r2_1_per_case_summary.json", summ)


def load_runs(exp):
    fr = []
    for d in ("tep", "cwru"):
        p = r2("results", exp, f"_partial_results_{d}.csv")
        if os.path.exists(p):
            fr.append(pd.read_csv(p))
    R = pd.concat(fr) if fr else pd.DataFrame()
    return R[R.status == "success"].copy() if len(R) else R


def shuffled_ablation():
    R = load_runs("e5")
    if R.empty:
        return
    R["cond"] = R.condition.str.split("@").str[0]
    R["window"] = R.case_id
    cases = {}
    for ds in ("TEP", "CWRU"):
        if ds == "CWRU":
            cwru_data.activate()
        cases.update({c.case_id: c for c in load_cases(str(OUT_DIR / "cases" / f"{ds.lower()}_clean.json"))})
    first_presented = []
    for r in R.itertuples():
        pf, _ = shuffled(cases[r.case_id], r.cond, int(r.rep))
        first_presented.append(pf.top_evidence[0].sensor)
    R["first_presented"] = first_presented
    R["cited1"] = R.cited_sensors.fillna("").str.split(";").str[0]
    R["follows_position"] = (R.cited1 == R.first_presented).astype(float)
    res = {}
    for (ds, c), sub in R.groupby(["dataset", "cond"]):
        res[f"{ds}|{c}"] = {
            "n": int(len(sub)),
            "models": sorted(sub.model.unique()),
            "EA1_true_IG": boot(sub, lambda d: d.top1_faithfulness.mean()),
            "position_following": boot(sub, lambda d: d.follows_position.mean()),
            "recall": boot(sub, lambda d: d.faithfulness.mean()),
            "rank_corr": boot(sub, lambda d: d.rank_correlation.mean()),
        }
        for m, sm in sub.groupby("model"):
            res[f"{ds}|{c}"][f"EA1_{m}"] = float(sm.top1_faithfulness.mean())
    A = pd.read_csv(OUT_DIR / "llm_revision" / "rank_pct_ablation_final" / "Ablation_Summary.csv")
    res["reference_unshuffled_summary"] = A[
        ["dataset", "condition", "n", "EA@1", "EA@1_ci_low", "EA@1_ci_high"]
    ].to_dict("records")
    save("e5_shuffled_ablation.json", res)


def heldout():
    from sklearn.metrics import roc_auc_score

    R = load_runs("e6")
    cases = {}
    for ds in ("tep", "cwru"):
        if ds == "cwru":
            cwru_data.activate()
        cases.update({c.case_id: c for c in load_cases(str(r2("cases", f"{ds}_new.json")))})
    C = pd.DataFrame(
        [
            dict(
                case_id=k,
                dataset="TEP" if k.startswith("new_tep") else "CWRU",
                correct=int(c.is_correct),
                conf=c.calibrated_confidence,
                stab=c.xai_stability,
                agr=c.xai_agreement,
            )
            for k, c in cases.items()
        ]
    )
    C["ERS"] = (ERS_W["c"] * C.conf + ERS_W["s"] * C.stab + ERS_W["a"] * C.agr) / sum(ERS_W.values())
    C["abstain_runtime"] = (C.conf < 0.85).astype(int)
    res = {"cases": C.round(4).to_dict("records")}
    for ds, sub in list(C.groupby("dataset")) + [("ALL", C)]:
        r = {
            "n": int(len(sub)),
            "errors": int((1 - sub.correct).sum()),
            "abstained_runtime": int(sub.abstain_runtime.sum()),
            "errors_not_abstained": int(((1 - sub.correct) * (1 - sub.abstain_runtime)).sum()),
        }
        if 0 < sub.correct.sum() < len(sub):
            r["AUROC_correct_conf"] = float(roc_auc_score(sub.correct, sub.conf))
            r["AUROC_correct_ERS"] = float(roc_auc_score(sub.correct, sub.ERS))
        res[f"case_level_{ds}"] = r
    if not R.empty:
        R["window"] = R.case_id
        R = R.merge(C[["case_id", "ERS", "conf"]], on="case_id")
        for (ds, c), sub in R.groupby(["dataset", "condition"]):
            res[f"{ds}|{c}"] = {
                "n": int(len(sub)),
                "windows": int(sub.window.nunique()),
                "models": sorted(sub.model.unique()),
                "EA1": boot(sub, lambda d: d.top1_faithfulness.mean()),
                "recall": boot(sub, lambda d: d.faithfulness.mean()),
                "halluc": float(sub.hallucination_rate.mean()),
            }
        for ds, sub in R.groupby("dataset"):
            p = (
                sub.pivot_table(index=["window", "model"], columns="condition", values="top1_faithfulness")
                .dropna(subset=["plain", "xai_grounded"])
                .reset_index()
            )
            res[f"{ds}|delta_xai_minus_plain"] = boot(p, lambda d: (d.xai_grounded - d.plain).mean())
            a = sub[sub.condition != "plain"]
            if a.top1_faithfulness.nunique() > 1:
                res[f"{ds}|r_EA1_conf"] = float(np.corrcoef(a.conf, a.top1_faithfulness)[0, 1])
                res[f"{ds}|r_EA1_ERS"] = float(np.corrcoef(a.ERS, a.top1_faithfulness)[0, 1])
    save("e6_heldout.json", res)


ANALYSES = {"per_case": per_case, "shuffled_ablation": shuffled_ablation, "heldout": heldout}


def analyse_main(a):
    for name in a.steps or list(ANALYSES):
        ANALYSES[name]()


def build_parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (
        ("e1", e1_main),
        ("e2", e2_main),
        ("e2b", e2b_main),
        ("e3", e3_main),
        ("e4", e4_main),
        ("cases", cases_main),
        ("llm-manifest", build_manifests),
    ):
        sub.add_parser(name).set_defaults(fn=lambda a, fn=fn: fn())
    p = sub.add_parser("llm-run")
    p.add_argument("--exp")
    p.add_argument("--dataset")
    p.set_defaults(fn=llm_run_main)
    p = sub.add_parser("llm-seq")
    p.add_argument("exp")
    p.add_argument("dataset")
    p.add_argument("model")
    p.set_defaults(fn=llm_seq_main)
    p = sub.add_parser("analyse")
    p.add_argument("steps", nargs="*", choices=list(ANALYSES))
    p.set_defaults(fn=analyse_main)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.fn(args)
