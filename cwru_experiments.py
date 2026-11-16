from __future__ import annotations
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import csv
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import cwru_data
from cwru_data import (
    CWRUDataLoader,
    FAULT_DESCRIPTIONS,
    LABEL_COL,
    SAMPLING_RATE_HZ,
    SENSOR_COLS,
    add_noise,
    measured_snr_db,
)
from llm import DiagnosisPayload, active_domain_name, summarise_window
from models import ModelTrainer, Sensor1DCNN, TemperatureScaler, XAIExplainer
from paths import CWRU_DIR, OUT_DIR
from train_classifiers import KAN_AVAILABLE, SensorKAN, SensorLSTM, expected_calibration_error

warnings.filterwarnings("ignore", category=FutureWarning)
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")
RESULTS_DIR = str(OUT_DIR / "results" / "cwru")
WINDOW_SIZE = 1024
STRIDE = 128
SAMPLES_PER_CLASS = 0
TRAIN_RATIO = 0.7
SEEDS = (42, 43, 44, 45, 46)
N_FOLDS = 5
METRICS_COLUMNS = [
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
]
PREPARED_NPZ = "prepared_windows.npz"
METRICS_CSV = "cwru_ablation_results.csv"
CALIBRATION_NPZ = "calibration_outputs.npz"
XAI_NPZ = "xai_attributions.npz"
CHECKPOINT_PTH = "cwru_1dcnn_calibrated.pth"
LLM_RESULTS_DIR = "llm_results"
_NOISE_TAG = ""


def set_noise_tag(args):
    global _NOISE_TAG
    if args.snr_db is None:
        _NOISE_TAG = ""
    else:
        _NOISE_TAG = f"noise_{args.noise_type}_{args.snr_db:g}dB"
    return _NOISE_TAG


def stage_path(filename):
    if _NOISE_TAG:
        return os.path.join(RESULTS_DIR, _NOISE_TAG, filename)
    return os.path.join(RESULTS_DIR, filename)


def should_skip(path, force, stage_name):
    if force:
        return False
    if os.path.exists(path):
        print(f"[SKIP] {stage_name}: {path} already exists.")
        print("       Use --force to recompute.")
        return True
    return False


def make_windows(df, sensor_cols, window_size, stride, label_col):
    features = df[sensor_cols].to_numpy(dtype=np.float32)
    labels = df[label_col].to_numpy()
    if len(features) < window_size:
        raise ValueError(f"Need at least {window_size} rows to make one window, got {len(features)}.")
    windows = []
    window_labels = []
    for start in range(0, len(features) - window_size + 1, stride):
        end = start + window_size
        block = labels[start:end]
        if block[0] != block[-1]:
            continue
        windows.append(features[start:end])
        window_labels.append(int(block[-1]))
    return (np.asarray(windows, dtype=np.float32), np.asarray(window_labels, dtype=np.int64))


def normalise(train, test):
    mean = train.mean(axis=(0, 1), dtype=np.float64)
    std = train.std(axis=(0, 1), dtype=np.float64)
    safe_std = np.where(std < 1e-12, 1.0, std)
    scaled_train = ((train - mean) / safe_std).astype(np.float32)
    scaled_test = ((test - mean) / safe_std).astype(np.float32)
    return (scaled_train, scaled_test, mean.astype(np.float64), safe_std.astype(np.float64))


def stage_prepare(args):
    output = stage_path(PREPARED_NPZ)
    if should_skip(output, args.force, "Stage 1 (prepare)"):
        return
    print("\nStage 1 - prepare")
    loader = CWRUDataLoader(base_dir=args.data_dir)
    df = loader.load_all()
    if args.samples_per_class > 0:
        df = df.groupby(LABEL_COL, group_keys=False).head(args.samples_per_class)
        df = df.reset_index(drop=True)
        print(f"Capped to {args.samples_per_class:,} rows per class: {len(df):,} rows")
    train_df, test_df = loader.train_test_split(df, train_ratio=TRAIN_RATIO)
    x_train, y_train = make_windows(train_df, SENSOR_COLS, args.window_size, args.stride, LABEL_COL)
    x_test, y_test = make_windows(test_df, SENSOR_COLS, args.window_size, args.stride, LABEL_COL)
    if args.snr_db is not None:
        clean_test = x_test.copy()
        x_train = add_noise(x_train, args.snr_db, args.noise_type, seed=args.noise_seed)
        x_test = add_noise(x_test, args.snr_db, args.noise_type, seed=args.noise_seed + 1)
        print(
            f"Injected {args.noise_type} noise at {args.snr_db:+g} dB (measured on test: {measured_snr_db(clean_test, x_test):+.3f} dB)"
        )
    x_train, x_test, mean, std = normalise(x_train, x_test)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez_compressed(
        output,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        channel_mean=mean,
        channel_std=std,
        sensor_cols=np.array(SENSOR_COLS),
        window_size=np.array([args.window_size], dtype=np.int64),
        stride=np.array([args.stride], dtype=np.int64),
        sampling_rate_hz=np.array([SAMPLING_RATE_HZ], dtype=np.int64),
        snr_db=np.array([np.nan if args.snr_db is None else args.snr_db], dtype=np.float64),
        noise_type=np.array([args.noise_type if args.snr_db is not None else "none"]),
    )
    print(f"\n  train windows : {x_train.shape}  labels {y_train.shape}")
    print(f"  test  windows : {x_test.shape}  labels {y_test.shape}")
    print(f"  classes       : {sorted(np.unique(y_train).tolist())}")
    print(f"Wrote {output}")


def load_prepared():
    path = stage_path(PREPARED_NPZ)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run stage 1 first:  py cwru_experiments.py --stage 1")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def build_torch_models():
    registry = {"1D-CNN": Sensor1DCNN, "LSTM": SensorLSTM}
    if KAN_AVAILABLE:
        registry["KAN (Kolmogorov-Arnold)"] = SensorKAN
    return (registry, KAN_AVAILABLE)


def fold_split(x, y, fold):
    train_end_pct = 0.4 + fold * 0.1
    val_end_pct = 0.6 + fold * 0.1
    train_idx = []
    val_idx = []
    for label in np.unique(y):
        positions = np.where(y == label)[0]
        size = len(positions)
        train_idx.append(positions[: int(size * train_end_pct)])
        val_idx.append(positions[int(size * train_end_pct) : int(size * val_end_pct)])
    train_positions = np.concatenate(train_idx)
    val_positions = np.concatenate(val_idx)
    return (x[train_positions], y[train_positions], x[val_positions], y[val_positions])


def train_sklearn_models(x_train, y_train, x_val, y_val, num_classes, seed):
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, precision_recall_fscore_support

    flat_train = x_train.reshape(len(x_train), -1)
    flat_val = x_val.reshape(len(x_val), -1)
    labels = np.arange(num_classes)
    sample = flat_val[:1]
    records = []

    def macro_f1(y_true, y_pred):
        return float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2])

    lr = LogisticRegression(max_iter=500, n_jobs=-1, solver="lbfgs", random_state=seed)
    lr.fit(flat_train, y_train)
    lr_val_proba = lr.predict_proba(flat_val)
    lr_val_pred = lr_val_proba.argmax(axis=1)
    records.append(
        {
            "Model": "Logistic Regression",
            "Train_Loss": float(log_loss(y_train, lr.predict_proba(flat_train), labels=labels)),
            "Val_Loss": float(log_loss(y_val, lr_val_proba, labels=labels)),
            "Train_F1": macro_f1(y_train, lr.predict(flat_train)),
            "Val_F1": macro_f1(y_val, lr_val_pred),
            "Val_Accuracy": float(np.mean(lr_val_pred == y_val)),
            "Val_ECE": expected_calibration_error(lr_val_proba, y_val),
            "Confidence": float(np.max(lr_val_proba[:1])),
        }
    )
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
        booster = xgb.XGBClassifier(device="cuda", **xgb_params)
        booster.fit(flat_train, y_train)
    except Exception as gpu_exc:
        print(f"  [XGB] CUDA fit failed ({gpu_exc}); falling back to CPU.")
        booster = xgb.XGBClassifier(n_jobs=-1, **xgb_params)
        booster.fit(flat_train, y_train)
    xgb_val_proba = booster.predict_proba(flat_val)
    xgb_val_pred = xgb_val_proba.argmax(axis=1)
    records.append(
        {
            "Model": "XGBoost",
            "Train_Loss": float(log_loss(y_train, booster.predict_proba(flat_train), labels=labels)),
            "Val_Loss": float(log_loss(y_val, xgb_val_proba, labels=labels)),
            "Train_F1": macro_f1(y_train, booster.predict(flat_train)),
            "Val_F1": macro_f1(y_val, xgb_val_pred),
            "Val_Accuracy": float(np.mean(xgb_val_pred == y_val)),
            "Val_ECE": expected_calibration_error(xgb_val_proba, y_val),
            "Confidence": float(np.max(xgb_val_proba[:1])),
        }
    )
    return records


def train_torch_model(
    name, model_class, x_train, y_train, x_val, y_val, num_classes, device, epochs, batch_size, window_size
):
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    model = model_class(len(SENSOR_COLS), num_classes, window_size).to(device)
    trainer = ModelTrainer(model, device, learning_rate=0.001)
    for _ in range(epochs):
        trainer.train_epoch(train_loader)
    train_metrics = trainer.evaluate(train_loader)
    val_metrics = trainer.evaluate(val_loader)
    scaler = TemperatureScaler().to(device)
    scaler.fit(model, val_loader, device)
    sample = torch.from_numpy(x_val[:1]).to(device)
    with torch.no_grad():
        confidence = float(torch.max(F.softmax(scaler(model(sample)), dim=1)).item())
    model.eval()
    val_probs, val_targets = ([], [])
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            logits = scaler(model(batch_x.to(device)))
            val_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            val_targets.append(batch_y.numpy())
    val_probs = np.concatenate(val_probs)
    val_targets = np.concatenate(val_targets)
    record = {
        "Model": name,
        "Train_Loss": float(train_metrics["loss"]),
        "Val_Loss": float(val_metrics["loss"]),
        "Train_F1": float(train_metrics["f1_score"]),
        "Val_F1": float(val_metrics["f1_score"]),
        "Val_Accuracy": float(val_metrics["accuracy"]),
        "Val_ECE": expected_calibration_error(val_probs, val_targets),
        "Confidence": confidence,
    }
    del model, trainer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def stage_train(args):
    output = stage_path(METRICS_CSV)
    if should_skip(output, args.force, "Stage 2 (train)"):
        return
    print("\nStage 2 - train")
    import gc

    data = load_prepared()
    x_all, y_all = (data["x_train"], data["y_train"])
    window_size = int(data["window_size"][0])
    num_classes = int(np.max(y_all)) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch_models, kan_available = build_torch_models()
    if not kan_available:
        print("KAN unavailable (install 'efficient-kan' or 'pykan').")
    seeds = SEEDS[: args.seeds]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(METRICS_COLUMNS)
    for seed in seeds:
        print(f"\n{'*' * 25} SEED {seed} {'*' * 25}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        for fold in range(args.folds):
            print(f"\n--- Seed {seed} | Fold {fold + 1}/{args.folds} ---")
            x_tr, y_tr, x_va, y_va = fold_split(x_all, y_all, fold)
            print(f"  train {x_tr.shape[0]:,} windows | val {x_va.shape[0]:,} windows")
            records = train_sklearn_models(x_tr, y_tr, x_va, y_va, num_classes, seed)
            for name, model_class in torch_models.items():
                records.append(
                    train_torch_model(
                        name,
                        model_class,
                        x_tr,
                        y_tr,
                        x_va,
                        y_va,
                        num_classes,
                        device,
                        args.epochs,
                        args.batch_size,
                        window_size,
                    )
                )
            with open(output, "a", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                for record in records:
                    print(
                        f"> {record['Model']:<23} | Train F1: {record['Train_F1']:.4f} - Val F1: {record['Val_F1']:.4f} | Conf: {record['Confidence']:.4f}"
                    )
                    writer.writerow(
                        [
                            seed,
                            fold + 1,
                            record["Model"],
                            record["Train_Loss"],
                            record["Val_Loss"],
                            record["Train_F1"],
                            record["Val_F1"],
                            record["Val_Accuracy"],
                            record["Val_ECE"],
                            record["Confidence"],
                        ]
                    )
            gc.collect()
    print(f"\nWrote {output}")
    summary = (
        pd.read_csv(output)
        .groupby("Model")[["Val_F1", "Val_Accuracy", "Val_ECE", "Confidence"]]
        .agg(["mean", "std"])
    )
    print("\nSummary:\n", summary)


def stage_calibrate(args):
    output = stage_path(CALIBRATION_NPZ)
    if should_skip(output, args.force, "Stage 3 (calibrate)"):
        return
    print("\nStage 3 - calibrate")
    data = load_prepared()
    x_train, y_train = (data["x_train"], data["y_train"])
    x_test, y_test = (data["x_test"], data["y_test"])
    window_size = int(data["window_size"][0])
    num_classes = int(np.max(y_train)) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(SEEDS[0])
    np.random.seed(SEEDS[0])
    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = Sensor1DCNN(len(SENSOR_COLS), num_classes, window_size).to(device)
    trainer = ModelTrainer(model, device, learning_rate=0.001)
    print(f"Training the final 1D-CNN for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(train_loader)
        print(f"  epoch {epoch + 1}/{args.epochs}  train loss {loss:.4f}")
    calib_idx, eval_idx = ([], [])
    for label in np.unique(y_test):
        positions = np.where(y_test == label)[0]
        cut = len(positions) // 2
        calib_idx.append(positions[:cut])
        eval_idx.append(positions[cut:])
    calib_positions = np.concatenate(calib_idx)
    eval_positions = np.concatenate(eval_idx)
    calib_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test[calib_positions]), torch.from_numpy(y_test[calib_positions])),
        batch_size=args.batch_size,
        shuffle=False,
    )
    eval_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test[eval_positions]), torch.from_numpy(y_test[eval_positions])),
        batch_size=args.batch_size,
        shuffle=False,
    )
    print(
        f"Disjoint split -> calibration {len(calib_positions):,} windows, evaluation {len(eval_positions):,} windows"
    )
    scaler = TemperatureScaler().to(device)
    scaler.fit(model, calib_loader, device)
    temperature = float(scaler.temperature.item())
    model.eval()
    logits_list, labels_list = ([], [])
    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            logits_list.append(model(batch_x.to(device)).cpu())
            labels_list.append(batch_y)
    logits = torch.cat(logits_list).numpy().astype(np.float64)
    labels = torch.cat(labels_list).numpy().astype(np.int64)
    np.savez(
        output,
        logits=logits,
        labels=labels,
        temperature=np.array([temperature], dtype=np.float64),
        temperature_source=np.array(["refit"]),
        stored_temperature=np.array([temperature], dtype=np.float64),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_sensors": len(SENSOR_COLS),
            "num_classes": num_classes,
            "window_size": window_size,
            "temperature": temperature,
        },
        stage_path(CHECKPOINT_PTH),
    )
    accuracy = float((logits.argmax(axis=1) == labels).mean())
    print(f"\n  logits      : {logits.shape}")
    print(f"  accuracy    : {accuracy:.4f}")
    print(f"  temperature : {temperature:.4f}")
    print(f"Wrote {output}")
    print(f"Wrote {stage_path(CHECKPOINT_PTH)}")


def stage_xai(args):
    output = stage_path(XAI_NPZ)
    if should_skip(output, args.force, "Stage 4 (xai)"):
        return
    print("\nStage 4 - integrated gradients")
    checkpoint_path = stage_path(CHECKPOINT_PTH)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"{checkpoint_path} not found. Run stage 3 first:  py cwru_experiments.py --stage 3"
        )
    data = load_prepared()
    x_test, y_test = (data["x_test"], data["y_test"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Sensor1DCNN(checkpoint["num_sensors"], checkpoint["num_classes"], checkpoint["window_size"]).to(
        device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    explainer = XAIExplainer(model=model, device=device)
    rng = np.random.default_rng(SEEDS[0])
    selected = []
    per_class = max(1, args.xai_samples // len(np.unique(y_test)))
    for label in np.unique(y_test):
        positions = np.where(y_test == label)[0]
        take = min(per_class, len(positions))
        selected.extend(rng.choice(positions, size=take, replace=False).tolist())
    selected_idx = np.array(sorted(selected))
    print(f"Explaining {len(selected_idx)} windows ({per_class} per class)...")
    importances = []
    attribution_maps = []
    predictions = []
    for count, index in enumerate(selected_idx, 1):
        window = torch.from_numpy(x_test[index : index + 1]).to(device)
        with torch.no_grad():
            predicted = int(torch.argmax(model(window), dim=1).item())
        importance, attr_matrix = explainer.explain_prediction(
            input_tensor=window, target_class=predicted, sensor_names=list(SENSOR_COLS)
        )
        importances.append(np.array([importance[name] for name in SENSOR_COLS], dtype=np.float64))
        attribution_maps.append(attr_matrix.astype(np.float32))
        predictions.append(predicted)
        if count % 20 == 0 or count == len(selected_idx):
            print(f"  {count}/{len(selected_idx)}")
    np.savez_compressed(
        output,
        window_index=selected_idx.astype(np.int64),
        true_label=y_test[selected_idx].astype(np.int64),
        predicted_label=np.array(predictions, dtype=np.int64),
        sensor_importance=np.stack(importances),
        attribution_maps=np.stack(attribution_maps),
        sensor_cols=np.array(SENSOR_COLS),
    )
    mean_importance = np.stack(importances).mean(axis=0)
    print("\nMean attribution share per channel:")
    for name, value in zip(SENSOR_COLS, mean_importance):
        print(f"  {name}: {value * 100:.1f}%")
    print(f"Wrote {output}")


def build_cwru_cases(args, top_k):
    checkpoint_path = stage_path(CHECKPOINT_PTH)
    xai_path = stage_path(XAI_NPZ)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} not found. Run stage 3 first.")
    if not os.path.exists(xai_path):
        raise FileNotFoundError(f"{xai_path} not found. Run stage 4 first.")
    data = load_prepared()
    x_test, y_test = (data["x_test"], data["y_test"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    temperature = float(checkpoint["temperature"])
    model = Sensor1DCNN(checkpoint["num_sensors"], checkpoint["num_classes"], checkpoint["window_size"]).to(
        device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with np.load(xai_path, allow_pickle=False) as xai:
        xai_index = xai["window_index"]
        xai_importance = xai["sensor_importance"]
    payloads = []
    for label in np.unique(y_test):
        positions = np.where(y_test == label)[0]
        if len(positions) == 0:
            continue
        explained = [p for p in positions if p in set(xai_index.tolist())]
        chosen = explained[len(explained) // 2] if explained else int(positions[len(positions) // 2])
        window = torch.from_numpy(x_test[chosen : chosen + 1]).to(device)
        with torch.no_grad():
            logits = model(window)
            raw_confidence = float(torch.max(F.softmax(logits, dim=1)).item())
            probabilities = F.softmax(logits / temperature, dim=1)
            predicted = int(torch.argmax(probabilities).item())
            calibrated_confidence = float(torch.max(probabilities).item())
        class_rows = np.array([xai_importance[i] for i, idx in enumerate(xai_index) if y_test[idx] == label])
        if len(class_rows) > 0:
            attributions = {name: float(value) for name, value in zip(SENSOR_COLS, class_rows.mean(axis=0))}
            spread = class_rows.std(axis=0).mean()
            stability = float(max(0.0, 1.0 - spread * len(SENSOR_COLS)))
        else:
            attributions = None
            stability = None
        evidence = summarise_window(x_test[chosen], list(SENSOR_COLS), attributions=attributions, top_k=top_k)
        payloads.append(
            DiagnosisPayload(
                case_id=f"cwru_class{int(label)}_win{int(chosen)}",
                scenario=FAULT_DESCRIPTIONS[int(label)],
                true_fault=int(label),
                predicted_fault=predicted,
                calibrated_confidence=calibrated_confidence,
                raw_confidence=raw_confidence,
                temperature=temperature,
                top_evidence=evidence,
                xai_stability=stability,
                data_quality=1.0,
                window_start=int(chosen),
            )
        )
        print(f"class {int(label)} -> predicted {predicted} (conf {calibrated_confidence:.4f})")
    if not payloads:
        raise RuntimeError("No usable CWRU cases could be built.")
    return payloads


def stage_llm(args):
    output_dir = stage_path(LLM_RESULTS_DIR)
    marker = os.path.join(output_dir, "llm_experiment_results.csv")
    mock_marker = os.path.join(output_dir, "SYNTHETIC_llm_experiment_results.csv")
    existing = mock_marker if args.mock_llm else marker
    if not args.dump_cases and should_skip(existing, args.force, "Stage 5 (llm)"):
        return
    print("\nStage 5 - LLM explanation experiment")
    cwru_data.activate()
    import llm_experiment as runner

    print(f"Active domain: {active_domain_name()}")
    runner.build_real_cases = lambda parsed: build_cwru_cases(parsed, parsed.top_k)
    llm_args = argparse.Namespace(
        models=args.models,
        conditions=list(runner.CONDITIONS),
        checkpoint=stage_path(CHECKPOINT_PTH),
        data_dir=args.data_dir,
        output_dir=output_dir,
        degradation="none",
        level=0.0,
        top_k=len(SENSOR_COLS),
        abstain_threshold=args.abstain_threshold,
        ers_threshold=args.ers_threshold,
        enable_abstain=args.enable_abstain,
        mock_cases=args.mock_llm,
        dump_cases=args.dump_cases,
        load_cases=args.load_cases,
    )
    if args.dump_cases:
        cases = build_cwru_cases(llm_args, llm_args.top_k)
        runner.dump_cases(cases, args.dump_cases)
        return
    records = runner.run_experiment(llm_args)
    synthetic = args.mock_llm or "mock" in args.models
    runner.write_results(records, output_dir, synthetic)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the checkpointed CWRU experiment pipeline.")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5], default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", type=str, default=str(CWRU_DIR))
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_CLASS)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument(
        "--noise-type", type=str, default="white", choices=["white", "pink", "brown", "impulsive"]
    )
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--dump-cases", type=str, default=None)
    parser.add_argument("--load-cases", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--xai-samples", type=int, default=200)
    parser.add_argument("--models", nargs="+", default=["mock"])
    parser.add_argument("--abstain-threshold", type=float, default=0.85)
    parser.add_argument("--ers-threshold", type=float, default=0.6)
    parser.add_argument("--enable-abstain", action="store_true")
    parser.add_argument("--mock-llm", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_noise_tag(args)
    os.makedirs(stage_path(""), exist_ok=True)
    stages = {
        1: ("prepare", stage_prepare),
        2: ("train", stage_train),
        3: ("calibrate", stage_calibrate),
        4: ("xai", stage_xai),
        5: ("llm", stage_llm),
    }
    selected = [args.stage] if args.stage else sorted(stages)
    print("CWRU experiment pipeline")
    print(f"Results directory: {os.path.abspath(stage_path(''))}")
    print(f"Stages to run    : {selected}")
    if args.snr_db is None:
        print("Noise condition  : clean")
    else:
        print(f"Noise condition  : {args.noise_type} @ {args.snr_db:+g} dB SNR")
    for number in selected:
        name, function = stages[number]
        try:
            function(args)
        except (FileNotFoundError, RuntimeError, ImportError) as exc:
            print(f"\nStage {number} ({name}) failed: {exc}")
            return 1
    print("\nPipeline complete")
    return 0


if __name__ == "__main__":
    import torch.multiprocessing

    torch.multiprocessing.freeze_support()
    sys.exit(main())
