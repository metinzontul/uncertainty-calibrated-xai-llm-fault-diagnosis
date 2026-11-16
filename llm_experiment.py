from __future__ import annotations
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import csv
import json
import sys
import pandas as pd
import llm
from llm import (
    MODEL_REGISTRY,
    ModelSpec,
    build_backend,
    generate_response,
    compute_ers,
    score_response,
    should_abstain,
    DEFAULT_ABSTAIN_THRESHOLD,
    assert_condition_isolation,
    build_prompt,
    CONDITIONS,
    DiagnosisPayload,
    LLMResponse,
    SensorEvidence,
    summarise_window,
    active_domain_name,
)
from paths import OUT_DIR, TEP_DIR

DEFAULT_OUTPUT_DIR = str(OUT_DIR / "llm_results")
CHECKPOINT_EVERY = 10


def synthetic_run(args):
    return bool(args.mock_cases) or "mock" in args.models


def add_run_arguments(parser):
    parser.add_argument("--models", nargs="+", default=["mock"], choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--checkpoint", type=str, default=str(OUT_DIR / "best_1dcnn_fixed.pth"))
    parser.add_argument("--data-dir", type=str, default=str(TEP_DIR))
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--degradation", type=str, default="none", choices=["none", "noise", "drift", "missing"]
    )
    parser.add_argument("--level", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--abstain-threshold", type=float, default=DEFAULT_ABSTAIN_THRESHOLD)
    parser.add_argument("--ers-threshold", type=float, default=0.6)
    parser.add_argument("--enable-abstain", action="store_true")
    parser.add_argument("--mock-cases", action="store_true")
    parser.add_argument("--dump-cases", type=str, default=None)
    parser.add_argument("--load-cases", type=str, default=None)
    parser.add_argument("--domain", type=str, default=None, choices=["tep", "cwru"])


def dump_cases(cases, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([c.to_dict() for c in cases], fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(cases)} cases to {path}")


def load_cases(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Case file not found: {path}. Build it first with --dump-cases.")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    cases = []
    for item in raw:
        item = dict(item)
        item.pop("is_correct", None)
        item["top_evidence"] = [SensorEvidence(**e) for e in item.get("top_evidence", [])]
        cases.append(DiagnosisPayload(**item))
    print(f"Loaded {len(cases)} pre-built cases from {path}")
    return cases


DEFAULT_SCENARIOS = [
    {"name": "Step change (reactor cooling)", "fault": 4, "start": 500},
    {"name": "Composition step", "fault": 2, "start": 100},
    {"name": "Valve sticking (complex)", "fault": 14, "start": 300},
    {"name": "Feed ratio step", "fault": 1, "start": 600},
    {"name": "Random variation", "fault": 10, "start": 800},
    {"name": "Feed loss", "fault": 6, "start": 400},
    {"name": "Slow kinetics drift", "fault": 13, "start": 200},
    {"name": "Condenser cooling variation", "fault": 12, "start": 700},
]


def build_real_cases(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import joblib
    from tep_data import FastTEPLoader as TEPDataLoader, SensorDegradator
    from models import Sensor1DCNN, XAIExplainer

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}. Run train_classifiers.py first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building real cases on: {device}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_sensors = checkpoint["num_sensors"]
    num_classes = checkpoint["num_classes"]
    window_size = checkpoint["window_size"]
    temperature = float(checkpoint["temperature"])
    model = Sensor1DCNN(num_sensors, num_classes, window_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = TEPDataLoader(base_dir=args.data_dir)
    test_df = loader.load_rdata_file("TEP_Faulty_Testing.RData")
    sensor_cols = [c for c in test_df.columns if c.startswith("xmeas") or c.startswith("xmv")]
    label_col = "faultNumber" if "faultNumber" in test_df.columns else test_df.columns[0]
    le = joblib.load(OUT_DIR / "fault_label_encoder.pkl")
    explainer = XAIExplainer(model=model, device=device)
    degradator = SensorDegradator(random_seed=42)
    payloads = []
    for scenario in DEFAULT_SCENARIOS:
        block = test_df[test_df[label_col] == scenario["fault"]]
        window_df = block.iloc[scenario["start"] : scenario["start"] + window_size]
        if len(window_df) < window_size:
            print(f"Skipping '{scenario['name']}': insufficient rows.")
            continue
        window_df = window_df.copy()
        if args.degradation == "noise":
            window_df = degradator.apply_gaussian_noise(window_df, sensor_cols, noise_level=args.level)
        elif args.degradation == "drift":
            window_df = degradator.apply_sensor_drift(window_df, sensor_cols, drift_rate=args.level)
        elif args.degradation == "missing":
            window_df = degradator.apply_missing_data(
                window_df, sensor_cols, missing_ratio=args.level, fill_strategy="ffill"
            )
        features = window_df[sensor_cols].values.astype(float)
        valid_fraction = float(np.isfinite(features).mean())
        features = np.nan_to_num(features, nan=0.0)
        raw_features = features.copy()
        if "scaler_mean" in checkpoint and "scaler_std" in checkpoint:
            features = (features - checkpoint["scaler_mean"]) / checkpoint["scaler_std"]
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            raw_conf = float(torch.max(F.softmax(logits, dim=1)).item())
            probs = F.softmax(logits / temperature, dim=1)
            pred_idx = int(torch.argmax(probs).item())
            cal_conf = float(torch.max(probs).item())
        pred_fault = int(le.inverse_transform([pred_idx])[0])
        importance, _ = explainer.explain_prediction(
            input_tensor=x, target_class=pred_idx, sensor_names=sensor_cols
        )
        evidence = summarise_window(features, sensor_cols, attributions=importance, top_k=args.top_k)
        case_id = f"fault{scenario['fault']}_at{scenario['start']}_{args.degradation}"
        payloads.append(
            DiagnosisPayload(
                case_id=case_id,
                scenario=scenario["name"],
                true_fault=int(scenario["fault"]),
                predicted_fault=pred_fault,
                calibrated_confidence=cal_conf,
                raw_confidence=raw_conf,
                temperature=temperature,
                top_evidence=evidence,
                degradation=args.degradation,
                degradation_level=args.level,
                data_quality=valid_fraction,
                window_start=int(scenario["start"]),
            )
        )
        print(
            f"{case_id}: true Fault {scenario['fault']} -> predicted Fault {pred_fault} (conf {cal_conf:.4f})"
        )
    if not payloads:
        raise RuntimeError("No usable cases were built from the TEP test data.")
    return payloads


def build_mock_cases(args):
    print("--mock-cases: building SYNTHETIC cases for pipeline testing only.")
    specs = [
        (4, 4, 0.94, 0.88, 0.82, 1.0),
        (2, 10, 0.99, 0.71, 0.64, 1.0),
        (14, 10, 0.64, 0.42, 0.38, 0.97),
        (1, 1, 0.91, 0.79, 0.75, 1.0),
        (10, 20, 0.72, 0.51, 0.47, 0.88),
    ]
    payloads = []
    for idx, (true_f, pred_f, conf, stab, agree, qual) in enumerate(specs, 1):
        ranked = [
            ("xmeas_7", 0.34),
            ("xmeas_9", 0.21),
            ("xmeas_21", 0.14),
            ("xmv_10", 0.09),
            ("xmeas_13", 0.06),
        ]
        evidence = [
            SensorEvidence(
                sensor=sensor,
                attribution=attribution,
                mean=10.0 + i,
                std=0.5 + 0.1 * i,
                delta=(-1) ** i * (0.4 + 0.2 * i),
                trend="rising" if i % 2 == 0 else "falling",
            )
            for i, (sensor, attribution) in enumerate(ranked)
        ]
        payloads.append(
            DiagnosisPayload(
                case_id=f"SYNTHETIC_case{idx}",
                scenario=f"SYNTHETIC scenario {idx}",
                true_fault=true_f,
                predicted_fault=pred_f,
                calibrated_confidence=conf,
                raw_confidence=min(conf + 0.05, 1.0),
                temperature=1.5,
                top_evidence=evidence,
                xai_stability=stab,
                xai_agreement=agree,
                data_quality=qual,
            )
        )
    return payloads


def run_experiment(args):
    if args.domain == "cwru":
        import cwru_data

        cwru_data.activate()
        print("Scoring vocabulary: CWRU")
    if args.load_cases:
        cases = load_cases(args.load_cases)
    elif args.mock_cases:
        cases = build_mock_cases(args)
    else:
        cases = build_real_cases(args)
    assert_condition_isolation(cases[0])
    print("Condition isolation verified: no cross-condition prompt leakage.")
    backends = {}
    for key in args.models:
        spec = MODEL_REGISTRY[key]
        try:
            backends[key] = (spec, build_backend(spec))
            print(f"Backend ready: {spec.display_name}")
        except (RuntimeError, ValueError) as exc:
            print(f"Skipping {spec.display_name}: {exc}")
    if not backends:
        raise RuntimeError("No usable backends. Check API credentials (LLM_API_KEY, OPEN_LLM_API_KEY).")
    records = []
    total = len(cases) * len(backends) * len(args.conditions)
    done = 0
    for payload in cases:
        ers, _ = compute_ers(payload)
        abstain_flag, abstain_reason = (
            should_abstain(payload, args.ers_threshold, args.abstain_threshold)
            if args.enable_abstain
            else (False, "")
        )
        for condition in args.conditions:
            prompt = build_prompt(payload, condition, args.abstain_threshold)
            case_abstains = abstain_flag and condition == "xai_uncertainty_grounded"
            for key, (spec, backend) in backends.items():
                done += 1
                print(
                    f"[{done}/{total}] {spec.display_name} | {condition} | {payload.case_id}"
                    + (" | ABSTAIN" if case_abstains else "")
                )
                response = generate_response(
                    backend=backend,
                    spec=spec,
                    system_prompt=llm.SYSTEM_PROMPT,
                    user_prompt=prompt,
                    condition=condition,
                    case_id=payload.case_id,
                    abstained=case_abstains,
                    abstain_reason=abstain_reason,
                )
                scores = score_response(response, payload, top_k=3)
                records.append(
                    {
                        "case_id": payload.case_id,
                        "scenario": payload.scenario,
                        "model": spec.key,
                        "model_name": spec.display_name,
                        "condition": condition,
                        "true_fault": payload.true_fault,
                        "predicted_fault": payload.predicted_fault,
                        "prediction_correct": int(payload.is_correct),
                        "calibrated_confidence": payload.calibrated_confidence,
                        "raw_confidence": payload.raw_confidence,
                        "degradation": payload.degradation,
                        "degradation_level": payload.degradation_level,
                        "data_quality": payload.data_quality,
                        "ers": scores.ers,
                        "faithfulness": scores.faithfulness,
                        "top1_faithfulness": scores.top1_faithfulness,
                        "weighted_faithfulness": scores.weighted_faithfulness,
                        "rank_correlation": scores.rank_correlation,
                        "hallucination_rate": scores.hallucination_rate,
                        "fault_consistency": scores.fault_consistency,
                        "abstained": int(response.abstained),
                        "abstain_reason": response.abstain_reason,
                        "cited_sensors": ";".join(scores.cited_sensors),
                        "unsupported_sensors": ";".join(scores.unsupported_sensors),
                        "evidence_sensors": ";".join(payload.evidence_sensors(3)),
                        "latency_s": response.latency_s,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "error": response.error,
                        "response_text": response.text,
                        "finish_reason": response.finish_reason,
                    }
                )
                if len(records) % CHECKPOINT_EVERY == 0:
                    write_results(records, args.output_dir, synthetic_run(args), partial=True)
    return records


def write_results(records, output_dir, synthetic, partial=False):
    os.makedirs(output_dir, exist_ok=True)
    prefix = "SYNTHETIC_" if synthetic else ""
    json_path = os.path.join(output_dir, f"{prefix}llm_experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    csv_path = os.path.join(output_dir, f"{prefix}llm_experiment_results.csv")
    if records:
        columns = [k for k in records[0].keys() if k != "response_text"]
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    print(f"\nWrote {len(records)} records:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    if synthetic:
        print("\nThese results are SYNTHETIC (mock backend and/or mock cases).")


def run_command(args):
    if args.dump_cases:
        try:
            cases = build_mock_cases(args) if args.mock_cases else build_real_cases(args)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"\n{exc}")
            sys.exit(1)
        dump_cases(cases, args.dump_cases)
        return
    try:
        records = run_experiment(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\n{exc}")
        print("\nTo test the pipeline without the dataset or API access, run:")
        sys.exit(1)
    synthetic = args.mock_cases or "mock" in args.models
    write_results(records, args.output_dir, synthetic)
    print("\nNext: python llm_analysis.py tables")


def load_full_evidence(case_file):
    with open(case_file, "r", encoding="utf-8") as fh:
        return {c["case_id"]: [e["sensor"] for e in c.get("top_evidence", [])] for c in json.load(fh)}


def rescore_directory(directory, full_evidence=None):
    json_path = os.path.join(directory, "llm_experiment_results.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No results JSON in {directory}")
    with open(json_path, "r", encoding="utf-8") as fh:
        records = json.load(fh)
    for rec in records:
        names = (full_evidence or {}).get(rec["case_id"])
        if not names:
            names = [s for s in str(rec.get("evidence_sensors", "")).split(";") if s]
        evidence = [
            SensorEvidence(sensor=s, attribution=0.0, mean=0.0, std=0.0, delta=0.0, trend="") for s in names
        ]
        payload = DiagnosisPayload(
            case_id=rec["case_id"],
            scenario=rec.get("scenario", ""),
            true_fault=int(rec["true_fault"]),
            predicted_fault=int(rec["predicted_fault"]),
            calibrated_confidence=float(rec["calibrated_confidence"]),
            raw_confidence=float(rec.get("raw_confidence", 0.0)),
            temperature=1.0,
            top_evidence=evidence,
            data_quality=float(rec.get("data_quality", 1.0)),
        )
        response = LLMResponse(
            model=rec["model"],
            condition=rec["condition"],
            case_id=rec["case_id"],
            text=rec.get("response_text", "") or "",
            abstained=bool(rec.get("abstained", 0)),
            abstain_reason=rec.get("abstain_reason", "") or "",
            error=rec.get("error", "") or "",
        )
        scores = score_response(response, payload, top_k=3)
        rec["faithfulness"] = scores.faithfulness
        rec["top1_faithfulness"] = scores.top1_faithfulness
        rec["weighted_faithfulness"] = scores.weighted_faithfulness
        rec["rank_correlation"] = scores.rank_correlation
        rec["hallucination_rate"] = scores.hallucination_rate
        rec["fault_consistency"] = scores.fault_consistency
        rec["cited_sensors"] = ";".join(scores.cited_sensors)
        rec["unsupported_sensors"] = ";".join(scores.unsupported_sensors)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    frame = pd.DataFrame(records)
    csv_columns = [c for c in frame.columns if c != "response_text"]
    frame[csv_columns].to_csv(os.path.join(directory, "llm_experiment_results.csv"), index=False)
    return len(records)


def rescore_command(args):
    if args.domain == "cwru":
        import cwru_data

        cwru_data.activate()
    print(f"Active vocabulary: {active_domain_name()}")
    full_evidence = load_full_evidence(args.cases) if args.cases else None
    if full_evidence:
        print(f"Full evidence lists loaded from {args.cases}")
    for directory in args.directories:
        try:
            count = rescore_directory(directory, full_evidence)
            print(f"  {directory}: re-scored {count} records")
        except FileNotFoundError as exc:
            print(f"  {directory}: SKIPPED - {exc}")


def main():
    parser = argparse.ArgumentParser(description="LLM explanation-trustworthiness experiment.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    add_run_arguments(run_parser)
    run_parser.set_defaults(func=run_command)
    rescore_parser = sub.add_parser("rescore")
    rescore_parser.add_argument("directories", nargs="+")
    rescore_parser.add_argument("--domain", type=str, default=None, choices=["tep", "cwru"])
    rescore_parser.add_argument("--cases", type=str, default=None)
    rescore_parser.set_defaults(func=rescore_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
