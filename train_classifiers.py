import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import csv
import gc
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from models import ModelTrainer, Sensor1DCNN, TemperatureScaler
from paths import OUT_DIR, TEP_DIR, out
from tep_data import FastTEPLoader as TEPDataLoader

try:
    from efficient_kan import KAN

    KAN_AVAILABLE = True
except ImportError:
    try:
        from kan import KAN

        KAN_AVAILABLE = True
    except ImportError:
        print("No KAN library found; the KAN arm will be skipped.")
        KAN_AVAILABLE = False
WINDOW_SIZE = 60
STRIDE = 10
CACHE_FILE = str(OUT_DIR / "tep_windows_cache.npz")
RESULTS_CSV = str(OUT_DIR / "ablation_corrected_results.csv")
CSV_HEADER = [
    "Seed",
    "Fold",
    "Model",
    "Train_Loss",
    "Val_Loss",
    "Train_F1",
    "Val_F1",
    "Val_Accuracy",
    "Val_ECE",
    "Confidence",
    "Distinct_Classes",
]


def build_windows(df, sensor_cols, label_col, window_size=WINDOW_SIZE, stride=STRIDE):
    has_runs = "simulationRun" in df.columns
    group_cols = [label_col, "simulationRun"] if has_runs else [label_col]
    features = []
    labels = []
    runs = []
    for keys, block in df.groupby(group_cols, sort=True):
        if isinstance(keys, tuple):
            label = int(keys[0])
            run = int(keys[1]) if has_runs else 0
        else:
            label, run = (int(keys), 0)
        values = block[sensor_cols].values.astype(np.float32)
        if len(values) < window_size:
            continue
        for start in range(0, len(values) - window_size + 1, stride):
            features.append(values[start : start + window_size])
            labels.append(label)
            runs.append(run)
    if not features:
        raise ValueError("No segment was long enough to form a window.")
    return (np.stack(features), np.asarray(labels, dtype=np.int64), np.asarray(runs, dtype=np.int64))


def fit_scaler(x):
    mean = x.mean(axis=(0, 1))
    std = x.std(axis=(0, 1))
    std[std < 1e-08] = 1.0
    return (mean, std)


def split_train_val_by_run(x, y, runs, val_runs):
    val_mask = np.zeros(len(y), dtype=bool)
    for label in np.unique(y):
        class_mask = y == label
        class_runs = np.unique(runs[class_mask])
        held_out = class_runs[-val_runs:] if val_runs < len(class_runs) else class_runs[-1:]
        val_mask |= class_mask & np.isin(runs, held_out)
    return (x[~val_mask], y[~val_mask], x[val_mask], y[val_mask])


def audit_no_leakage(x_train, x_val, x_test, y_train, y_val, y_test):
    print("\n[AUDIT] Checking for leakage between splits...")
    for name, array in (("train", x_train), ("val", x_val), ("test", x_test)):
        assert len(array) > 0, f"{name} split is empty"

    def hashes(array):
        return {hash(w.tobytes()) for w in array}

    h_train, h_val, h_test = (hashes(x_train), hashes(x_val), hashes(x_test))
    overlaps = {
        "train/val": len(h_train & h_val),
        "train/test": len(h_train & h_test),
        "val/test": len(h_val & h_test),
    }
    for pair, count in overlaps.items():
        assert count == 0, f"{count} identical windows shared between {pair}"
        print(f"  {pair:<12} no shared windows  OK")
    classes_train = set(np.unique(y_train).tolist())
    classes_val = set(np.unique(y_val).tolist())
    classes_test = set(np.unique(y_test).tolist())
    assert (
        classes_train == classes_val == classes_test
    ), f"Class vocabulary differs: train {sorted(classes_train)}, val {sorted(classes_val)}, test {sorted(classes_test)}"
    print(f"  class vocabulary identical across splits ({len(classes_train)}) OK")
    print(f"  sizes: train {len(x_train):,} | val {len(x_val):,} | test {len(x_test):,}")


def evaluate(model, loader, device):
    model.eval()
    preds, targets = ([], [])
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device, non_blocking=True))
            preds.append(logits.argmax(1).cpu())
            targets.append(batch_y)
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    _, _, f1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "f1_score": float(f1),
        "distinct_predictions": int(len(np.unique(preds))),
    }


def load_or_build_cache(args):
    signature = f"{args.samples_per_class}_{args.test_samples_per_class}_{WINDOW_SIZE}_{STRIDE}"
    if os.path.exists(args.cache) and (not args.rebuild_cache):
        cached = np.load(args.cache, allow_pickle=True)
        if str(cached.get("signature", "")) == signature:
            print(f"[CACHE] Reusing windows from {args.cache}")
            return {k: cached[k] for k in cached.files}
        print("[CACHE] Settings changed; rebuilding.")
    print("[CACHE] Building windows from the .RData files (slow, one time)...")
    loader = TEPDataLoader(base_dir=args.data_dir)
    train_df = loader.load_rdata_file("TEP_Faulty_Training.RData")
    test_df = loader.load_rdata_file("TEP_Faulty_Testing.RData")
    sensor_cols = [c for c in train_df.columns if c.startswith("xmeas") or c.startswith("xmv")]
    label_col = "faultNumber"
    train_df = (
        train_df.groupby(label_col, group_keys=False).head(args.samples_per_class).reset_index(drop=True)
    )
    test_df = (
        test_df.groupby(label_col, group_keys=False).head(args.test_samples_per_class).reset_index(drop=True)
    )
    encoder = LabelEncoder()
    train_df[label_col] = encoder.fit_transform(train_df[label_col])
    test_df[label_col] = encoder.transform(test_df[label_col])
    joblib.dump(encoder, out("fault_label_encoder.pkl"))
    x_tr, y_tr, runs_tr = build_windows(train_df, sensor_cols, label_col)
    x_te, y_te, runs_te = build_windows(test_df, sensor_cols, label_col)
    payload = {
        "x_train_full": x_tr,
        "y_train_full": y_tr,
        "runs_train": runs_tr,
        "x_test": x_te,
        "y_test": y_te,
        "runs_test": runs_te,
        "sensor_cols": np.array(sensor_cols),
        "signature": np.array(signature),
    }
    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    np.savez_compressed(args.cache, **payload)
    print(f"[CACHE] Wrote {args.cache}")
    return payload


def run_final(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    loader = TEPDataLoader(base_dir=args.data_dir)
    train_df = loader.load_rdata_file("TEP_Faulty_Training.RData")
    test_df = loader.load_rdata_file("TEP_Faulty_Testing.RData")
    sensor_cols = [c for c in train_df.columns if c.startswith("xmeas") or c.startswith("xmv")]
    label_col = "faultNumber"
    train_df = (
        train_df.groupby(label_col, group_keys=False).head(args.samples_per_class).reset_index(drop=True)
    )
    test_df = (
        test_df.groupby(label_col, group_keys=False).head(args.test_samples_per_class).reset_index(drop=True)
    )
    encoder = LabelEncoder()
    train_df[label_col] = encoder.fit_transform(train_df[label_col])
    test_df[label_col] = encoder.transform(test_df[label_col])
    joblib.dump(encoder, out("fault_label_encoder.pkl"))
    print("\nWindowing (per class and simulation run)...")
    x_train, y_train, _ = build_windows(train_df, sensor_cols, label_col)
    x_test, y_test, _ = build_windows(test_df, sensor_cols, label_col)
    print(f"  train: {x_train.shape}  classes {len(np.unique(y_train))}")
    print(f"  test : {x_test.shape}  classes {len(np.unique(y_test))}")
    mean, std = fit_scaler(x_train)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    print(f"  standardised: train mean {x_train.mean():.4f}, std {x_train.std():.4f}")
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=args.batch_size, shuffle=False
    )
    num_classes = len(encoder.classes_)
    model = Sensor1DCNN(len(sensor_cols), num_classes, WINDOW_SIZE).to(device)
    trainer = ModelTrainer(model, device, learning_rate=args.learning_rate)
    print(f"\nTraining {args.epochs} epochs...")
    for epoch in range(args.epochs):
        trainer.train_epoch(train_loader)
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            metrics = trainer.evaluate(test_loader)
            print(f"  epoch {epoch + 1:>2}: test acc {metrics['accuracy']:.4f} F1 {metrics['f1_score']:.4f}")
    train_metrics = trainer.evaluate(train_loader)
    test_metrics = trainer.evaluate(test_loader)
    print(
        f"\nFinal: train F1 {train_metrics['f1_score']:.4f} | test acc {test_metrics['accuracy']:.4f} F1 {test_metrics['f1_score']:.4f}"
    )
    with torch.no_grad():
        preds = []
        for batch_x, _ in test_loader:
            preds.append(model(batch_x.to(device)).argmax(1).cpu())
        preds = torch.cat(preds)
    distinct = len(torch.unique(preds))
    print(f"Distinct predicted classes: {distinct} / {num_classes}")
    if distinct < num_classes // 2:
        print("The model is collapsing onto a few classes.")
    scaler = TemperatureScaler().to(device)
    scaler.fit(model, test_loader, device)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "temperature": scaler.temperature.item(),
            "num_sensors": len(sensor_cols),
            "num_classes": num_classes,
            "window_size": WINDOW_SIZE,
            "scaler_mean": mean,
            "scaler_std": std,
        },
        args.output,
    )
    print(f"\nWrote {args.output} (T = {scaler.temperature.item():.4f})")


def run_tep(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device} | mixed precision: {use_amp}")
    cache = load_or_build_cache(args)
    sensor_cols = [str(c) for c in cache["sensor_cols"]]
    x_train, y_train, x_val, y_val = split_train_val_by_run(
        cache["x_train_full"], cache["y_train_full"], cache["runs_train"], args.val_runs
    )
    x_test, y_test = (cache["x_test"], cache["y_test"])
    audit_no_leakage(x_train, x_val, x_test, y_train, y_val, y_test)
    mean, std = fit_scaler(x_train)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std
    print(f"[SCALE] Fitted on train only: mean {x_train.mean():.4f}, std {x_train.std():.4f}")

    def make_loader(x, y, shuffle):
        return DataLoader(
            TensorDataset(torch.tensor(x), torch.tensor(y)),
            batch_size=args.batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )

    train_loader = make_loader(x_train, y_train, True)
    val_loader = make_loader(x_val, y_val, False)
    test_loader = make_loader(x_test, y_test, False)
    num_classes = len(np.unique(y_train))
    model = Sensor1DCNN(len(sensor_cols), num_classes, WINDOW_SIZE).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.weight_decay > 0:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1 = -1.0
    best_state = None
    epochs_without_gain = 0
    started = time.time()
    print(f"\nTraining up to {args.epochs} epochs (early stopping on validation F1)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(batch_x), batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["f1_score"])
        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_gain = 0
            marker = " *"
        else:
            epochs_without_gain += 1
            marker = ""
        if epoch % 2 == 0 or marker:
            print(
                f"  epoch {epoch:>3}: val acc {val_metrics['accuracy']:.4f} F1 {val_metrics['f1_score']:.4f}{marker}"
            )
        if epochs_without_gain >= args.patience:
            print(f"  early stop at epoch {epoch} (no gain for {args.patience})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Training took {time.time() - started:.0f}s; best validation F1 {best_f1:.4f}")
    train_metrics = evaluate(model, make_loader(x_train, y_train, False), device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    print("\nFinal results")
    for name, m in (("train", train_metrics), ("val", val_metrics), ("test", test_metrics)):
        print(
            f"  {name:<6} acc {m['accuracy']:.4f} | F1 {m['f1_score']:.4f} | distinct predictions {m['distinct_predictions']}/{num_classes}"
        )
    if test_metrics["distinct_predictions"] < num_classes // 2:
        print("The model is collapsing onto a few classes.")
    print("\nFitting temperature on the validation split (test untouched)...")
    temp_scaler = TemperatureScaler().to(device)
    temp_scaler.fit(model, val_loader, device)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "temperature": temp_scaler.temperature.item(),
            "num_sensors": len(sensor_cols),
            "num_classes": num_classes,
            "window_size": WINDOW_SIZE,
            "scaler_mean": mean,
            "scaler_std": std,
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1_score"],
        },
        args.output,
    )
    print(f"\nWrote {args.output} (T = {temp_scaler.temperature.item():.4f})")


def expected_calibration_error(probabilities, labels, n_bins=10):
    if len(labels) == 0:
        return float("nan")
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        mask = (
            (confidence > edges[i]) & (confidence <= edges[i + 1])
            if i > 0
            else (confidence >= edges[i]) & (confidence <= edges[i + 1])
        )
        if mask.sum():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def collect_probabilities_torch(model, loader, device, temperature=1.0):
    model.eval()
    probs, targets = ([], [])
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device)) / temperature
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
            targets.append(batch_y.numpy())
    return (np.concatenate(probs), np.concatenate(targets))


class SensorLSTM(nn.Module):

    def __init__(self, num_sensors, num_classes, window_size=60):
        super(SensorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=num_sensors, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class SensorKAN(nn.Module):

    def __init__(self, num_sensors, num_classes, window_size=60):
        super(SensorKAN, self).__init__()
        input_dim = num_sensors * window_size
        if KAN_AVAILABLE:
            self.kan = KAN([input_dim, 64, num_classes])
        else:
            raise RuntimeError("KAN library is not installed.")

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.kan(x)


def make_run_folds(y, runs, n_splits):
    masks = [np.zeros(len(y), dtype=bool) for _ in range(n_splits)]
    for label in np.unique(y):
        class_mask = y == label
        class_runs = np.unique(runs[class_mask])
        if len(class_runs) < n_splits:
            raise ValueError(
                f"Class {label} has only {len(class_runs)} runs; cannot build {n_splits} disjoint run folds."
            )
        for fold, held_out in enumerate(np.array_split(class_runs, n_splits)):
            masks[fold] |= class_mask & np.isin(runs, held_out)
    return masks


def standardise(x_train, x_val):
    mean = x_train.mean(axis=(0, 1))
    std = x_train.std(axis=(0, 1))
    std[std < 1e-08] = 1.0
    return (((x_train - mean) / std).astype(np.float32), ((x_val - mean) / std).astype(np.float32))


def macro_f1(y_true, y_pred):
    return float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2])


def predict_torch(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch_x, _ in loader:
            preds.append(model(batch_x.to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy()


def append_row(row):
    with open(RESULTS_CSV, mode="a", newline="") as handle:
        csv.writer(handle).writerow(row)


def run_ablation(args):
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running CORRECTED ablation (5 models) on: {device}")
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, mode="w", newline="") as handle:
        csv.writer(handle).writerow(CSV_HEADER)
    cache = load_or_build_cache(args)
    x_all = cache["x_train_full"]
    y_all = cache["y_train_full"]
    runs_all = cache["runs_train"]
    sensor_cols = [str(c) for c in cache["sensor_cols"]]
    x_test_audit = cache["x_test"]
    y_test_audit = cache["y_test"]
    num_sensors = len(sensor_cols)
    num_classes = int(len(np.unique(y_all)))
    print(
        f"Windows: {x_all.shape} | sensors {num_sensors} | classes {num_classes} | window {WINDOW_SIZE} stride {STRIDE}"
    )
    fold_masks = make_run_folds(y_all, runs_all, args.n_splits)
    xgb_device = "cuda" if device.type == "cuda" and (not args.xgb_cpu) else "cpu"
    print(f"XGBoost fitting on: {xgb_device}")
    print(f"\nStarting corrected ablation: {len(args.seeds)}-seed x {args.n_splits}-fold run-grouped CV")
    started = time.time()
    for seed in args.seeds:
        print(f"\nSeed {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        for fold in range(args.n_splits):
            print(f"\n--- Seed {seed} | FOLD {fold + 1}/{args.n_splits} ---")
            fold_started = time.time()
            val_mask = fold_masks[fold]
            x_train, y_train = (x_all[~val_mask], y_all[~val_mask])
            x_val, y_val = (x_all[val_mask], y_all[val_mask])
            held_runs = sorted(np.unique(runs_all[val_mask]).tolist())
            print(f"  held-out runs per class: {held_runs}")
            audit_no_leakage(x_train, x_val, x_test_audit, y_train, y_val, y_test_audit)
            x_train, x_val = standardise(x_train, x_val)
            flat_train = x_train.reshape(len(x_train), -1)
            flat_val = x_val.reshape(len(x_val), -1)
            fault_indices = np.where(y_val > 0)[0]
            probe = int(fault_indices[0]) if len(fault_indices) else 0
            probe_flat = flat_val[probe].reshape(1, -1)
            probe_tensor = torch.tensor(
                x_val[probe].reshape(1, WINDOW_SIZE, num_sensors), dtype=torch.float32
            ).to(device)

            def make_loader(x, y, shuffle):
                return DataLoader(
                    TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y)),
                    batch_size=args.batch_size,
                    shuffle=shuffle,
                    drop_last=shuffle,
                )

            train_loader = make_loader(x_train, y_train, True)
            train_eval_loader = make_loader(x_train, y_train, False)
            val_loader = make_loader(x_val, y_val, False)

            def train_and_log_pytorch(model_name, model_class):
                try:
                    model = model_class(num_sensors, num_classes, WINDOW_SIZE).to(device)
                    trainer = ModelTrainer(model, device, learning_rate=args.learning_rate)
                    for _ in range(args.epochs):
                        trainer.train_epoch(train_loader)
                    t_metrics = trainer.evaluate(train_eval_loader)
                    v_metrics = trainer.evaluate(val_loader)
                    temp_scaler = TemperatureScaler().to(device)
                    temp_scaler.fit(model, val_loader, device)
                    with torch.no_grad():
                        logits = temp_scaler(model(probe_tensor))
                        conf = float(torch.max(F.softmax(logits, dim=1)).item())
                    fold_probs, fold_labels = collect_probabilities_torch(
                        model, val_loader, device, temperature=float(temp_scaler.temperature.item())
                    )
                    v_ece = expected_calibration_error(fold_probs, fold_labels)
                    distinct = int(len(np.unique(predict_torch(model, val_loader, device))))
                    t_loss = t_metrics.get("loss", float("nan"))
                    v_loss = v_metrics.get("loss", float("nan"))
                    print(
                        f"> {model_name:<23} | Train Loss: {t_loss:.4f} - Val Loss: {v_loss:.4f} | Train F1: {t_metrics['f1_score']:.4f} - Val F1: {v_metrics['f1_score']:.4f} | Conf: {conf:.4f} | Classes: {distinct}/{num_classes}"
                    )
                    append_row(
                        [
                            seed,
                            fold + 1,
                            model_name,
                            t_loss,
                            v_loss,
                            t_metrics["f1_score"],
                            v_metrics["f1_score"],
                            v_metrics.get("accuracy", float("nan")),
                            v_ece,
                            conf,
                            distinct,
                        ]
                    )
                    del model, trainer, temp_scaler
                except Exception as exc:
                    print(f"> {model_name:<23} | FAILED: {exc}")
                    append_row(
                        [seed, fold + 1, model_name, "nan", "nan", "nan", "nan", "nan", "nan", "nan", 0]
                    )
                finally:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            try:
                lr_model = LogisticRegression(
                    max_iter=500, n_jobs=args.n_jobs, solver="lbfgs", random_state=seed
                )
                lr_model.fit(flat_train, y_train)
                lr_train_pred = lr_model.predict(flat_train)
                lr_val_pred = lr_model.predict(flat_val)
                lr_t_f1 = macro_f1(y_train, lr_train_pred)
                lr_v_f1 = macro_f1(y_val, lr_val_pred)
                lr_t_loss = log_loss(
                    y_train, lr_model.predict_proba(flat_train), labels=np.arange(num_classes)
                )
                lr_val_proba = lr_model.predict_proba(flat_val)
                lr_v_loss = log_loss(y_val, lr_val_proba, labels=np.arange(num_classes))
                lr_conf = float(np.max(lr_model.predict_proba(probe_flat)))
                lr_distinct = int(len(np.unique(lr_val_pred)))
                lr_v_acc = float((lr_val_pred == y_val).mean())
                lr_v_ece = expected_calibration_error(lr_val_proba, y_val)
                print(
                    f"> {'Logistic Reg (LR)':<23} | Train Loss: {lr_t_loss:.4f} - Val Loss: {lr_v_loss:.4f} | Train F1: {lr_t_f1:.4f} - Val F1: {lr_v_f1:.4f} | Conf: {lr_conf:.4f} | Classes: {lr_distinct}/{num_classes}"
                )
                append_row(
                    [
                        seed,
                        fold + 1,
                        "Logistic Regression",
                        lr_t_loss,
                        lr_v_loss,
                        lr_t_f1,
                        lr_v_f1,
                        lr_v_acc,
                        lr_v_ece,
                        lr_conf,
                        lr_distinct,
                    ]
                )
            except Exception as exc:
                print(f"> {'Logistic Reg (LR)':<23} | FAILED: {exc}")
                append_row(
                    [
                        seed,
                        fold + 1,
                        "Logistic Regression",
                        "nan",
                        "nan",
                        "nan",
                        "nan",
                        "nan",
                        "nan",
                        "nan",
                        0,
                    ]
                )
            try:
                xgb_params = dict(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    objective="multi:softprob",
                    random_state=seed,
                    eval_metric="mlogloss",
                    tree_method="hist",
                )
                try:
                    xgb_base = xgb.XGBClassifier(device=xgb_device, **xgb_params)
                    xgb_base.fit(flat_train, y_train)
                except Exception as gpu_exc:
                    if xgb_device == "cpu":
                        raise
                    print(f"  [XGB] CUDA fit failed ({gpu_exc}); falling back to CPU.")
                    xgb_device = "cpu"
                    xgb_base = xgb.XGBClassifier(n_jobs=args.n_jobs, **xgb_params)
                    xgb_base.fit(flat_train, y_train)
                xgb_t_f1 = macro_f1(y_train, xgb_base.predict(flat_train))
                xgb_t_loss = log_loss(
                    y_train, xgb_base.predict_proba(flat_train), labels=np.arange(num_classes)
                )
                xgb_calibrated = CalibratedClassifierCV(xgb_base, method="isotonic", cv="prefit")
                xgb_calibrated.fit(flat_val, y_val)
                xgb_val_proba = xgb_calibrated.predict_proba(flat_val)
                xgb_val_pred = xgb_calibrated.predict(flat_val)
                xgb_v_f1 = macro_f1(y_val, xgb_val_pred)
                xgb_v_acc = float((xgb_val_pred == y_val).mean())
                xgb_v_ece = expected_calibration_error(xgb_val_proba, y_val)
                xgb_v_loss = log_loss(y_val, xgb_val_proba, labels=np.arange(num_classes))
                xgb_conf = float(np.max(xgb_calibrated.predict_proba(probe_flat)))
                xgb_distinct = int(len(np.unique(xgb_val_pred)))
                print(
                    f"> {'XGBoost':<23} | Train Loss: {xgb_t_loss:.4f} - Val Loss: {xgb_v_loss:.4f} | Train F1: {xgb_t_f1:.4f} - Val F1: {xgb_v_f1:.4f} | Conf: {xgb_conf:.4f} | Classes: {xgb_distinct}/{num_classes}"
                )
                append_row(
                    [
                        seed,
                        fold + 1,
                        "XGBoost",
                        xgb_t_loss,
                        xgb_v_loss,
                        xgb_t_f1,
                        xgb_v_f1,
                        xgb_v_acc,
                        xgb_v_ece,
                        xgb_conf,
                        xgb_distinct,
                    ]
                )
            except Exception as exc:
                print(f"> {'XGBoost':<23} | FAILED: {exc}")
                append_row([seed, fold + 1, "XGBoost", "nan", "nan", "nan", "nan", "nan", "nan", "nan", 0])
            train_and_log_pytorch("1D-CNN", Sensor1DCNN)
            train_and_log_pytorch("LSTM", SensorLSTM)
            if KAN_AVAILABLE:
                train_and_log_pytorch("KAN (Kolmogorov-Arnold)", SensorKAN)
            print(f"  [TIME] fold wall clock: {time.time() - fold_started:.0f}s")
            gc.collect()
    print(f"\nCorrected ablation results (elapsed {time.time() - started:.0f}s)")
    results = pd.read_csv(RESULTS_CSV)
    summary = results.groupby("Model").agg(
        {
            "Train_F1": ["mean", "std"],
            "Val_F1": ["mean", "std"],
            "Confidence": ["mean", "std"],
            "Distinct_Classes": ["mean", "min"],
        }
    )
    for model_name in summary.index:
        row = summary.loc[model_name]
        print(f"[{model_name}]")
        print(
            f"  Train F1: {row['Train_F1', 'mean']:.4f} +/- {row['Train_F1', 'std']:.4f} | Val F1: {row['Val_F1', 'mean']:.4f} +/- {row['Val_F1', 'std']:.4f} | Confidence: {row['Confidence', 'mean']:.4f} +/- {row['Confidence', 'std']:.4f}"
        )
        print(
            f"  Distinct classes predicted: mean {row['Distinct_Classes', 'mean']:.1f} | worst {int(row['Distinct_Classes', 'min'])} of {num_classes}"
        )
        if row["Distinct_Classes", "min"] < num_classes / 2:
            print("  This model collapsed on at least one fold; its F1 is not meaningful.")
        print()
    failures = int(results["Val_F1"].isna().sum())
    if failures:
        print(f"{failures} run(s) failed and were recorded as NaN, not fabricated.")
    print(f"Wrote {RESULTS_CSV}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and compare the TEP fault classifiers.")
    sub = parser.add_subparsers(dest="command", required=True)
    final = sub.add_parser("final")
    final.add_argument("--data-dir", type=str, default=str(TEP_DIR))
    final.add_argument("--output", type=str, default=str(OUT_DIR / "best_1dcnn_fixed.pth"))
    final.add_argument("--epochs", type=int, default=15)
    final.add_argument("--batch-size", type=int, default=128)
    final.add_argument("--learning-rate", type=float, default=0.001)
    final.add_argument("--samples-per-class", type=int, default=5000)
    final.add_argument("--test-samples-per-class", type=int, default=1000)
    final.add_argument("--seed", type=int, default=42)
    final.set_defaults(func=run_final)
    tep = sub.add_parser("tep")
    tep.add_argument("--data-dir", type=str, default=str(TEP_DIR))
    tep.add_argument("--output", type=str, default=str(OUT_DIR / "best_1dcnn_fixed.pth"))
    tep.add_argument("--cache", type=str, default=CACHE_FILE)
    tep.add_argument("--rebuild-cache", action="store_true")
    tep.add_argument("--epochs", type=int, default=40)
    tep.add_argument("--batch-size", type=int, default=256)
    tep.add_argument("--learning-rate", type=float, default=0.001)
    tep.add_argument("--samples-per-class", type=int, default=20000)
    tep.add_argument("--test-samples-per-class", type=int, default=4000)
    tep.add_argument("--val-runs", type=int, default=2)
    tep.add_argument("--patience", type=int, default=8)
    tep.add_argument("--weight-decay", type=float, default=0.0001)
    tep.add_argument("--label-smoothing", type=float, default=0.05)
    tep.add_argument("--workers", type=int, default=4)
    tep.add_argument("--seed", type=int, default=42)
    tep.add_argument("--no-amp", action="store_true")
    tep.set_defaults(func=run_tep)
    ablation = sub.add_parser("ablation")
    ablation.add_argument("--data-dir", type=str, default=str(TEP_DIR))
    ablation.add_argument("--cache", type=str, default=CACHE_FILE)
    ablation.add_argument("--rebuild-cache", action="store_true")
    ablation.add_argument("--samples-per-class", type=int, default=20000)
    ablation.add_argument("--test-samples-per-class", type=int, default=4000)
    ablation.add_argument("--batch-size", type=int, default=64)
    ablation.add_argument("--epochs", type=int, default=10)
    ablation.add_argument("--learning-rate", type=float, default=0.001)
    ablation.add_argument("--n-splits", type=int, default=5)
    ablation.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ablation.add_argument("--n-jobs", type=int, default=4)
    ablation.add_argument("--xgb-cpu", action="store_true")
    ablation.set_defaults(func=run_ablation)
    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    import torch.multiprocessing

    torch.multiprocessing.freeze_support()
    main()
