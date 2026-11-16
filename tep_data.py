import argparse
import os
import time
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from paths import TEP_DIR


class TEPDataLoader:

    def __init__(self, base_dir=str(TEP_DIR)):
        self.base_dir = Path(base_dir)

    def load_rdata_file(self, filename):
        import pyreadr

        file_path = self.base_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")
        try:
            r_env: Dict[str, pd.DataFrame] = pyreadr.read_r(str(file_path))
            available_keys = list(r_env.keys())
            target_key = available_keys[0]
            df: pd.DataFrame = r_env[target_key]
            print(f"Loaded '{filename}' successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to read {filename}. Error: {str(e)}")

    def load_all_datasets(self):
        files_to_load = {
            "fault_free_training": "TEP_FaultFree_Training.RData",
            "fault_free_testing": "TEP_FaultFree_Testing.RData",
            "faulty_training": "TEP_Faulty_Training.RData",
            "faulty_testing": "TEP_Faulty_Testing.RData",
        }
        datasets: Dict[str, pd.DataFrame] = {}
        for key, filename in files_to_load.items():
            datasets[key] = self.load_rdata_file(filename)
        return datasets


class TEPTimeseriesDataset(Dataset):

    def __init__(self, df, sensor_cols, window_size=50, stride=10, label_col="faultNumber"):
        self.window_size = window_size
        self.stride = stride
        if "simulationRun" in df.columns:
            df = df.sort_values(by=["simulationRun", "sample"])
        self.features = df[sensor_cols].values
        self.labels = df[label_col].values
        self.valid_indices = self._calculate_indices()

    def _calculate_indices(self):
        num_samples = len(self.features)
        indices = []
        for i in range(0, num_samples - self.window_size + 1, self.stride):
            indices.append(i)
        return indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.window_size
        sequence = self.features[start_idx:end_idx, :]
        label = self.labels[end_idx - 1]
        x_tensor = torch.tensor(sequence, dtype=torch.float32)
        y_tensor = torch.tensor(label, dtype=torch.long)
        return (x_tensor, y_tensor)


TEP_FILES = {
    "TEP_FaultFree_Training.RData": "fault_free_training.parquet",
    "TEP_FaultFree_Testing.RData": "fault_free_testing.parquet",
    "TEP_Faulty_Training.RData": "faulty_training.parquet",
    "TEP_Faulty_Testing.RData": "faulty_testing.parquet",
}
CACHE_DIRNAME = "parquet_cache"


class FastTEPLoader:

    def __init__(self, base_dir=str(TEP_DIR)):
        self.base_dir = base_dir
        self.cache_dir = os.path.join(base_dir, CACHE_DIRNAME)

    def _cache_path(self, filename):
        if filename not in TEP_FILES:
            raise KeyError(f"Unknown TEP file '{filename}'. Expected one of {list(TEP_FILES)}.")
        return os.path.join(self.cache_dir, TEP_FILES[filename])

    def load_rdata_file(self, filename):
        cache_path = self._cache_path(filename)
        if os.path.exists(cache_path):
            started = time.time()
            df = pd.read_parquet(cache_path)
            print(
                f"Loaded '{filename}' from parquet cache in {time.time() - started:.1f}s. Shape: {df.shape}"
            )
            return df
        return self._convert_one(filename)

    def _convert_one(self, filename):
        import pyreadr

        source = os.path.join(self.base_dir, filename)
        if not os.path.exists(source):
            raise FileNotFoundError(f"TEP file not found: {source}")
        print(f"Converting '{filename}' (slow, one time)...")
        started = time.time()
        result = pyreadr.read_r(source)
        df = result[list(result.keys())[0]]
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_path = self._cache_path(filename)
        df.to_parquet(cache_path, index=False)
        print(
            f"  {df.shape[0]:,} rows in {time.time() - started:.1f}s -> {cache_path} ({os.path.getsize(cache_path) / 1000000.0:.0f} MB)"
        )
        return df

    def load_all_datasets(self):
        keys = {
            "TEP_FaultFree_Training.RData": "fault_free_training",
            "TEP_FaultFree_Testing.RData": "fault_free_testing",
            "TEP_Faulty_Training.RData": "faulty_training",
            "TEP_Faulty_Testing.RData": "faulty_testing",
        }
        return {key: self.load_rdata_file(name) for name, key in keys.items()}


def main():
    parser = argparse.ArgumentParser(description="Build the TEP parquet cache.")
    parser.add_argument("--base-dir", type=str, default=str(TEP_DIR))
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    loader = FastTEPLoader(base_dir=args.base_dir)
    if args.convert:
        started = time.time()
        for filename in TEP_FILES:
            if os.path.exists(loader._cache_path(filename)):
                print(f"Skipping '{filename}' - already cached.")
                continue
            loader._convert_one(filename)
        print(f"\nConversion finished in {time.time() - started:.0f}s.")
    if args.verify or not args.convert:
        print("\nCache status:")
        for filename in TEP_FILES:
            path = loader._cache_path(filename)
            if os.path.exists(path):
                started = time.time()
                rows = len(pd.read_parquet(path, columns=["faultNumber"]))
                print(
                    f"  {filename:<32} cached  {rows:>10,} rows  ({time.time() - started:.1f}s to read one column)"
                )
            else:
                print(f"  {filename:<32} NOT CACHED - run with --convert")


class SensorDegradator:

    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def apply_gaussian_noise(self, df, sensor_cols, noise_level=0.05):
        degraded_df = df.copy()
        for col in sensor_cols:
            col_std = degraded_df[col].std()
            noise = np.random.normal(0, col_std * noise_level, size=len(degraded_df))
            degraded_df[col] = degraded_df[col] + noise
        return degraded_df

    def apply_sensor_drift(self, df, sensor_cols, drift_rate=0.001):
        degraded_df = df.copy()
        time_steps = np.arange(len(degraded_df))
        for col in sensor_cols:
            drift = time_steps * drift_rate
            degraded_df[col] = degraded_df[col] + drift
        return degraded_df

    def apply_missing_data(self, df, sensor_cols, missing_ratio=0.1, fill_strategy=None):
        degraded_df = df.copy()
        for col in sensor_cols:
            mask = np.random.rand(len(degraded_df)) < missing_ratio
            degraded_df.loc[mask, col] = np.nan
            if fill_strategy == "zero":
                degraded_df[col] = degraded_df[col].fillna(0.0)
            elif fill_strategy == "ffill":
                degraded_df[col] = degraded_df[col].ffill()
        return degraded_df


if __name__ == "__main__":
    main()
