import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from cwru_data import add_noise, measured_snr_db
from models import Sensor1DCNN, TemperatureScaler
from paths import OUT_DIR, TEP_DIR
from tep_data import FastTEPLoader as TEPDataLoader

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export real logits/labels/temperature for the reliability diagram."
    )
    parser.add_argument("--checkpoint", type=str, default=str(OUT_DIR / "best_1dcnn_fixed.pth"))
    parser.add_argument("--data-dir", type=str, default=str(TEP_DIR))
    parser.add_argument("--output", type=str, default=str(OUT_DIR / "calibration_outputs.npz"))
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--temperature-source", type=str, choices=["refit", "checkpoint"], default="refit")
    return parser.parse_args()


def get_stratified_data(full_df, label_col, samples_per_class):
    stratified = full_df.groupby(label_col, group_keys=False).head(samples_per_class)
    return stratified.reset_index(drop=True)


def split_by_class(df, label_col, frac=0.5):
    first, second = ([], [])
    for _, block in df.groupby(label_col, group_keys=False):
        cut = int(len(block) * frac)
        first.append(block.iloc[:cut])
        second.append(block.iloc[cut:])
    return (pd.concat(first).reset_index(drop=True), pd.concat(second).reset_index(drop=True))


def build_windows_per_class(df, sensor_cols, label_col, window_size, stride=10):
    features = []
    labels = []
    for label, block in df.groupby(label_col, sort=True):
        values = block[sensor_cols].values.astype(np.float32)
        if len(values) < window_size:
            print(f"  Class {label}: only {len(values)} rows, skipped.")
            continue
        for start in range(0, len(values) - window_size + 1, stride):
            features.append(values[start : start + window_size])
            labels.append(int(label))
    if not features:
        raise ValueError(
            "No class block was long enough to form a window. Increase --samples-per-class or reduce the window size."
        )
    x = torch.tensor(np.stack(features), dtype=torch.float32)
    y = torch.tensor(np.asarray(labels), dtype=torch.long)
    counts = {int(c): int((y == c).sum()) for c in torch.unique(y)}
    print(f"  Windows per class: {counts}")
    return (x, y)


def collect_logits(model, loader, device):
    model.eval()
    logits_list, labels_list = ([], [])
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits_list.append(model(batch_x.to(device)).cpu())
            labels_list.append(batch_y.cpu())
    return (torch.cat(logits_list), torch.cat(labels_list))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exporting calibration outputs on: {device}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Run train_classifiers.py final or tep first."
        )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_sensors = checkpoint["num_sensors"]
    num_classes = checkpoint["num_classes"]
    window_size = checkpoint["window_size"]
    stored_temperature = float(checkpoint["temperature"])
    model = Sensor1DCNN(num_sensors, num_classes, window_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(
        f"Loaded model: {num_sensors} sensors, {num_classes} classes, window {window_size}, stored T = {stored_temperature:.4f}"
    )
    loader_obj = TEPDataLoader(base_dir=args.data_dir)
    datasets = loader_obj.load_all_datasets()
    test_df = datasets["faulty_testing"]
    sensor_cols = [c for c in test_df.columns if c.startswith("xmeas") or c.startswith("xmv")]
    label_col = "faultNumber" if "faultNumber" in test_df.columns else test_df.columns[0]
    stratified = get_stratified_data(test_df, label_col, args.samples_per_class)
    le = joblib.load(OUT_DIR / "fault_label_encoder.pkl")
    stratified[label_col] = le.transform(stratified[label_col])
    calib_df, eval_df = split_by_class(stratified, label_col, frac=0.5)
    print(f"Disjoint split -> calibration rows: {len(calib_df)}, evaluation rows: {len(eval_df)}")
    print("Calibration half:")
    calib_x, calib_y = build_windows_per_class(calib_df, sensor_cols, label_col, window_size, stride=10)
    print("Evaluation half:")
    eval_x, eval_y = build_windows_per_class(eval_df, sensor_cols, label_col, window_size, stride=10)
    if "scaler_mean" in checkpoint and "scaler_std" in checkpoint:
        mean = torch.tensor(checkpoint["scaler_mean"], dtype=torch.float32)
        std = torch.tensor(checkpoint["scaler_std"], dtype=torch.float32)
        calib_x = (calib_x - mean) / std
        eval_x = (eval_x - mean) / std
        print("Applied the checkpoint's per-sensor standardisation.")
    else:
        print("Checkpoint carries no scaler; feeding raw sensor values.")
    if args.snr_db is not None:
        clean = eval_x.numpy().copy()
        noisy = add_noise(clean.transpose(0, 2, 1), args.snr_db, "white", seed=42)
        eval_x = torch.tensor(noisy.transpose(0, 2, 1), dtype=torch.float32)
        print(
            f"Injected white noise at {args.snr_db:+g} dB on the standardised signal (measured {measured_snr_db(clean.transpose(0, 2, 1), noisy):+.2f} dB)"
        )
    calib_loader = DataLoader(TensorDataset(calib_x, calib_y), batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(TensorDataset(eval_x, eval_y), batch_size=args.batch_size, shuffle=False)
    if args.temperature_source == "refit":
        print("Refitting temperature on the held-out calibration half...")
        scaler = TemperatureScaler().to(device)
        scaler.fit(model, calib_loader, device)
        temperature = float(scaler.temperature.item())
        print(f"Leak-free temperature: T = {temperature:.4f}")
    else:
        temperature = stored_temperature
        print(
            f"Reusing checkpoint temperature T = {temperature:.4f}. This T was fitted on the evaluation data (calibration leak)."
        )
    logits, labels = collect_logits(model, eval_loader, device)
    logits_np = logits.numpy().astype(np.float64)
    labels_np = labels.numpy().astype(np.int64)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        logits=logits_np,
        labels=labels_np,
        temperature=np.array([temperature], dtype=np.float64),
        temperature_source=np.array([args.temperature_source]),
        stored_temperature=np.array([stored_temperature], dtype=np.float64),
    )
    accuracy = float((logits_np.argmax(axis=1) == labels_np).mean())
    print(f"\nWrote {args.output}")
    print(f"  logits shape : {logits_np.shape}")
    print(f"  labels shape : {labels_np.shape}")
    print(f"  accuracy     : {accuracy:.4f}")
    print(f"  temperature  : {temperature:.4f} ({args.temperature_source})")


if __name__ == "__main__":
    import torch.multiprocessing

    torch.multiprocessing.freeze_support()
    main()
