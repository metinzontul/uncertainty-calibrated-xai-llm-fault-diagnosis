import os
import re
from typing import Dict, List
import numpy as np
import pandas as pd
import scipy.io as sio
import llm
from paths import CWRU_DIR

SENSOR_COLS = ["DE", "FE", "BA"]
SENSOR_DESCRIPTIONS = {
    "DE": "Drive-end accelerometer (closest to the tested bearing)",
    "FE": "Fan-end accelerometer",
    "BA": "Base-plate accelerometer (motor housing)",
}
SAMPLING_RATE_HZ = 12000
LABEL_COL = "label"
FILE_MAP = {
    97: (0, "Normal baseline"),
    105: (1, "Inner race fault, 0.007 in"),
    169: (2, "Inner race fault, 0.014 in"),
    209: (3, "Inner race fault, 0.021 in"),
    118: (4, "Ball fault, 0.007 in"),
    185: (5, "Ball fault, 0.014 in"),
    222: (6, "Ball fault, 0.021 in"),
    130: (7, "Outer race fault, 0.007 in"),
    197: (8, "Outer race fault, 0.014 in"),
    234: (9, "Outer race fault, 0.021 in"),
}
FAULT_DESCRIPTIONS = {idx: desc for idx, desc in FILE_MAP.values()}


class CWRUDataLoader:

    def __init__(self, base_dir=str(CWRU_DIR)):
        self.base_dir = base_dir

    def load_mat_file(self, file_number):
        path = os.path.join(self.base_dir, f"{file_number}.mat")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"CWRU file not found: {path}\nDownload it from https://engineering.case.edu/bearingdatacenter"
            )
        mat = sio.loadmat(path)
        channels: Dict[str, np.ndarray] = {}
        for channel in SENSOR_COLS:
            key = next((k for k in mat if k.endswith(f"_{channel}_time")), None)
            if key is not None:
                channels[channel] = mat[key].ravel()
        if "DE" not in channels:
            raise KeyError(f"No drive-end channel found in {path}")
        length = min((len(v) for v in channels.values()))
        data = {}
        for channel in SENSOR_COLS:
            if channel in channels:
                data[channel] = channels[channel][:length]
            else:
                data[channel] = np.zeros(length, dtype=float)
        df = pd.DataFrame(data)
        df[LABEL_COL] = FILE_MAP[file_number][0]
        return df

    def load_all(self):
        frames: List[pd.DataFrame] = []
        for file_number in sorted(FILE_MAP, key=lambda n: FILE_MAP[n][0]):
            try:
                frame = self.load_mat_file(file_number)
                frames.append(frame)
                label, desc = FILE_MAP[file_number]
                print(f"  [{label}] {file_number}.mat  {len(frame):>7,} samples  {desc}")
            except (FileNotFoundError, KeyError) as exc:
                print(f"  [!] Skipping {file_number}.mat: {exc}")
        if not frames:
            raise RuntimeError("No CWRU files could be loaded.")
        df = pd.concat(frames, ignore_index=True)
        print(
            f"Loaded CWRU: {len(df):,} samples, {df[LABEL_COL].nunique()} classes, {len(SENSOR_COLS)} channels @ {SAMPLING_RATE_HZ} Hz"
        )
        return df

    def train_test_split(self, df, train_ratio=0.7):
        train_parts, test_parts = ([], [])
        for label, block in df.groupby(LABEL_COL, sort=True):
            cut = int(len(block) * train_ratio)
            train_parts.append(block.iloc[:cut])
            test_parts.append(block.iloc[cut:])
        train = pd.concat(train_parts, ignore_index=True)
        test = pd.concat(test_parts, ignore_index=True)
        print(
            f"Chronological within-class split at {train_ratio:.0%}: train {len(train):,} samples, test {len(test):,} samples"
        )
        return (train, test)


DEFAULT_SNR_LEVELS_DB = (-10.0, -5.0, 0.0, 5.0, 10.0)
NOISE_TYPES = ("white", "pink", "brown", "impulsive")


def signal_power(signal):
    return float(np.mean(np.square(np.asarray(signal, dtype=float))))


def _coloured_noise(shape, noise_type, rng):
    if noise_type == "white":
        return rng.standard_normal(shape)
    if noise_type == "impulsive":
        noise = rng.standard_normal(shape) * 0.1
        mask = rng.random(shape) < 0.01
        noise[mask] = rng.standard_normal(int(mask.sum())) * 5.0
        return noise / (np.std(noise) + 1e-12)
    if noise_type not in ("pink", "brown"):
        raise ValueError(f"Unknown noise type '{noise_type}'. Expected one of {NOISE_TYPES}.")
    exponent = 1.0 if noise_type == "pink" else 2.0
    n_time = shape[-1]
    white = rng.standard_normal(shape)
    spectrum = np.fft.rfft(white, axis=-1)
    freqs = np.fft.rfftfreq(n_time)
    scaling = np.ones_like(freqs)
    scaling[1:] = 1.0 / np.power(freqs[1:], exponent / 2.0)
    shaped = np.fft.irfft(spectrum * scaling, n=n_time, axis=-1)
    return shaped / (np.std(shaped) + 1e-12)


def add_noise(windows, snr_db, noise_type="white", seed=42):
    signal = np.asarray(windows, dtype=float)
    p_signal = signal_power(signal)
    if p_signal <= 0.0:
        raise ValueError("Signal has zero power; cannot set an SNR against it.")
    rng = np.random.default_rng(seed)
    noise = _coloured_noise(signal.shape, noise_type, rng)
    p_noise_target = p_signal / 10.0 ** (snr_db / 10.0)
    noise = noise * np.sqrt(p_noise_target / (signal_power(noise) + 1e-12))
    return signal + noise


def measured_snr_db(clean, noisy):
    clean = np.asarray(clean, dtype=float)
    noise = np.asarray(noisy, dtype=float) - clean
    p_noise = signal_power(noise)
    if p_noise <= 0.0:
        return float("inf")
    return 10.0 * np.log10(signal_power(clean) / p_noise)


CWRU_SENSOR_NAMES = tuple(SENSOR_COLS)
CWRU_SENSOR_DESCRIPTIONS = dict(SENSOR_DESCRIPTIONS)
CWRU_FAULT_DESCRIPTIONS = dict(FAULT_DESCRIPTIONS)
CWRU_SYSTEM_PROMPT = "You are a rotating-machinery condition-monitoring assistant supporting maintenance engineers of a bearing test rig.\n\nRules you must follow without exception:\n1. Base every statement only on the evidence given in the prompt. Never invent sensor readings, fault codes, or machine events.\n2. Refer to sensors by their exact identifiers (DE, FE or BA). Do not cite a sensor that does not appear in the evidence.\n3. Be concise and operational. The engineer is reading you under time pressure.\n4. If the evidence is insufficient to justify a root cause, say so explicitly instead of guessing.\n"
_SENSOR_MENTION_RE = re.compile("\\b(" + "|".join(CWRU_SENSOR_NAMES) + ")\\b")
_FAULT_CLAIM_RE = re.compile("\\b(?:faults?|class(?:es)?)[\\s_\\-#]*(\\d{1,2})\\b", re.IGNORECASE)


def describe_sensor(name):
    return CWRU_SENSOR_DESCRIPTIONS.get(name, name)


def describe_fault(fault_number):
    return CWRU_FAULT_DESCRIPTIONS.get(int(fault_number), "Unknown fault")


def extract_sensor_mentions(text):
    seen = []
    for name in _SENSOR_MENTION_RE.findall(text or ""):
        if name in CWRU_SENSOR_NAMES and name not in seen:
            seen.append(name)
    return seen


def extract_fault_claims(text):
    claims = []
    for match in _FAULT_CLAIM_RE.findall(text or ""):
        value = int(match)
        if value not in claims:
            claims.append(value)
    return claims


def activate():
    llm.register_domain(
        name="CWRU",
        sensor_names=CWRU_SENSOR_NAMES,
        sensor_descriptions=CWRU_SENSOR_DESCRIPTIONS,
        fault_descriptions=CWRU_FAULT_DESCRIPTIONS,
        sensor_mention_extractor=extract_sensor_mentions,
        fault_claim_extractor=extract_fault_claims,
    )
    llm.SYSTEM_PROMPT = CWRU_SYSTEM_PROMPT
