import argparse
import csv
import dataclasses
import glob
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, rankdata
from sklearn.metrics import roc_auc_score
import llm
from llm import (
    ERS_WEIGHTS,
    MODEL_REGISTRY,
    DiagnosisPayload,
    LLMResponse,
    OpenAICompatibleBackend,
    build_backend,
    compute_ers,
    generate_response,
    score_response,
    should_abstain,
)
from llm_experiment import load_cases
from paths import ROOT, OUT_DIR, out

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
RV = OUT_DIR / "llm_revision"
CASES_DIR = OUT_DIR / "cases"
RUNS_DIR = OUT_DIR / "final_runs"
RETRY_DIR = "llm_retry_completion"
MANIFEST = RV / RETRY_DIR / "retry_manifest.csv"
B, SEED = (10000, 42)
W = dict(ERS_WEIGHTS, data_quality=0.0)
MET = {
    "top1_faithfulness": "EA@1",
    "faithfulness": "Recall",
    "rank_correlation": "RankCorr",
    "hallucination_rate": "Halluc",
}
LEVEL = {
    "tep_clean": "Clean",
    "tep_noise_0.05": "σ = 0.05",
    "tep_noise_0.10": "σ = 0.10",
    "tep_noise_0.20": "σ = 0.20",
    "cwru_clean": "Clean",
    "cwru_snr10": "+10 dB",
    "cwru_snr5": "+5 dB",
    "cwru_snr0": "0 dB",
}
ABL_MET = {
    "top1_faithfulness": "EA@1",
    "faithfulness": "Recall",
    "rank_correlation": "RankCorr",
    "hallucination_rate": "Hallucination",
}
ABL_ORDER = ["plain", "rank_only", "percentages_only", "xai_grounded"]
CTL_ORDER = ["plain", "random_order_plain", "xai_grounded", "shuffled_attribution"]
PHYS_ORDER = [
    "plain",
    "random_order_plain",
    "rank_only",
    "percentages_only",
    "xai_grounded",
    "xai_uncertainty_grounded",
    "shuffled_attribution",
]
COND = {"plain": "Plain", "xai_grounded": "XAI-Grounded", "xai_uncertainty_grounded": "XAI + Uncertainty"}
CASEFILE = {
    "tep_clean": "tep_clean.json",
    "tep_noise_0.05": "tep_noise_0.05.json",
    "tep_noise_0.10": "tep_noise_0.10.json",
    "tep_noise_0.20": "tep_noise_0.20.json",
    "cwru_clean": "cwru_clean.json",
    "cwru_snr0": "cwru_snr0.json",
    "cwru_snr5": "cwru_snr5.json",
    "cwru_snr10": "cwru_snr10.json",
}
TRANSIENT = (429, 500, 502, 503, 504)
QUOTA_MARKERS = (
    "insufficient_quota",
    "no credits",
    "credit",
    "billing",
    "quota",
    "invalid_api_key",
    "expired",
)
LONG_MAX_TOKENS, LONG_TIMEOUT = (32768, 600)
ABL_CASES = {"tep": CASES_DIR / "tep_clean.json", "cwru": CASES_DIR / "cwru_clean.json"}
ABL_NEW = ("rank_only", "percentages_only")
ABL_ALL = ("plain", "rank_only", "percentages_only", "xai_grounded")
XAI_HEAD = (
    "EXPLAINABILITY EVIDENCE (Integrated Gradients attribution, ranked by contribution to this prediction):"
)
PCT_HEAD = "EXPLAINABILITY EVIDENCE (Integrated Gradients attribution share of each sensor):"
INSTR = "Explain to the operator why this fault was reported. Ground your root-cause statement in the attribution evidence above and cite the specific sensors it identifies."
CONTROL_CONDITIONS = {"random_order_plain": "plain", "shuffled_attribution": "xai_grounded"}
RUNS = [
    ("1. Retry of failed requests (TEP + CWRU SNR)", "llm_retry_completion", 138),
    ("2. CWRU clean rerun (corrected CWRU prompt)", "cwru_clean_rerun", 180),
    ("3. DeepSeek-V4.1-Flash-1017 (12 blocks)", "deepseek_v41_replacement", 308),
    ("4. GLM-5.3-Flash (12 blocks)", "glm53_replacement", 308),
    ("5. Rank Only / Percentages Only ablation", "rank_pct_ablation_final", 432),
]
KEEP = [
    "record_id",
    "dataset",
    "run_block",
    "rep",
    "case_id",
    "model",
    "model_name",
    "condition",
    "calibrated_confidence",
    "top1_faithfulness",
    "faithfulness",
    "rank_correlation",
    "hallucination_rate",
    "fault_consistency",
    "cited_sensors",
    "evidence_sensors",
    "final_status",
    "generation_batch",
    "served_model",
    "prompt_tokens",
    "completion_tokens",
    "finish_reason",
    "max_tokens_used",
    "response_text",
]
NEW_MODELS = {"deepseek_v41_flash": "DeepSeek-V4.1-Flash-1017", "glm53_flash": "GLM-5.3-Flash"}
ERS_BLOCKS = (
    [("TEP", "clean", "tep_clean_rep*", "tep_xai_reliability.json")]
    + [
        ("TEP", f"noise_{l}", f"tep_noise_{l}", f"tep_noise_{l}_xai_reliability.json")
        for l in ("0.05", "0.10", "0.20")
    ]
    + [("CWRU", "clean", "cwru_clean_rep*", "cwru_xai_reliability.json")]
    + [("CWRU", f"snr{s}", f"cwru_snr{s}", f"cwru_snr{s}_xai_reliability.json") for s in (0, 5, 10)]
)
SCORES = ("S_conf", "S_ERS", "S_XAI")
OUTCOMES = (
    ("prediction_correct", "Prediction correctness", "all"),
    ("top1_faithfulness", "Evidence Adherence@1", "answered"),
    ("faithfulness", "Recall Faithfulness", "answered"),
)


def rv(folder, name):
    return out("llm_revision", folder, name)


def rv_glob(pattern):
    return glob.glob(str(RV / pattern))


def block_of(path):
    return os.path.basename(os.path.dirname(path))


def block_key(run_block):
    return run_block.split("_rep")[0]


def run_files():
    return sorted(glob.glob(str(RUNS_DIR / "*" / "llm_experiment_results.csv")))


def case_path(name):
    return CASES_DIR / Path(name).name


def pct_ci(v):
    return np.percentile(v, [2.5, 97.5])


def boot_contrast(g, a, b, col):
    w = g.window.unique()
    rng = np.random.default_rng(SEED)
    res = []
    ga = {x: g[(g.window == x) & (g.condition == a)][col].to_numpy() for x in w}
    gb = {x: g[(g.window == x) & (g.condition == b)][col].to_numpy() for x in w}
    for _ in range(B):
        s = rng.choice(w, len(w))
        xa = np.concatenate([ga[x] for x in s])
        xb = np.concatenate([gb[x] for x in s])
        if len(xa) and len(xb):
            res.append(xa.mean() - xb.mean())
    return pct_ci(res)


def successes(pattern):
    r = pd.concat([pd.read_csv(p) for p in rv_glob(pattern)], ignore_index=True)
    return r[r.status == "success"].drop_duplicates("retry_id")


def get_spec(key):
    return MODEL_REGISTRY[key]


def cmd_registry(a):
    for k in ("gpt5_mini", "qwen35_9b", "deepseek_v41_flash", "glm53_flash"):
        s = get_spec(k)
        print(
            f"{k:20s} {s.backend:9s} {s.model_id:42s} temp={getattr(s, 'temperature', None)} max_tokens={getattr(s, 'max_tokens', None)}"
        )


def cmd_newmodel_manifest(a):
    key = a.models[0]
    out_dir = Path(a.out_dir)
    name = NEW_MODELS[key]
    rows = []
    for p in run_files():
        blk = block_of(p)
        d = pd.read_csv(p)
        d = d[d.model == "gpt5_mini"]
        for i, r in d.iterrows():
            rows.append(
                {
                    "retry_id": f"{blk}|{r.case_id}|{key}|{r.condition}",
                    "run_block": blk,
                    "case_block": block_key(blk),
                    "dataset": "TEP" if blk.startswith("tep") else "CWRU",
                    "rep": int(blk.split("rep")[1]) if "rep" in blk else 1,
                    "case_id": r.case_id,
                    "model": key,
                    "model_name": name,
                    "condition": r.condition,
                    "regime": "clean" if "clean" in blk else "degradation",
                    "original_row_index": i,
                    "original_status": "abstained" if r.abstained == 1 else "to_generate",
                    "abstain_reason": r.abstain_reason if r.abstained == 1 else "",
                    "case_file": "cases/" + CASEFILE[block_key(blk)],
                    "source_csv": os.path.relpath(p, OUT_DIR).replace("\\", "/"),
                }
            )
    m = pd.DataFrame(rows)
    assert len(m) == 324 and m.retry_id.is_unique
    os.makedirs(out_dir, exist_ok=True)
    m.to_csv(out_dir / "newmodel_manifest_all324.csv", index=False)
    m[m.original_status != "abstained"].to_csv(out_dir / "newmodel_manifest.csv", index=False)
    print(m.groupby(["dataset", "original_status"]).size())


def cmd_manifest(a):
    rows = []
    for p in run_files():
        blk = block_of(p)
        d = pd.read_csv(p)
        j = json.load(open(p.replace(".csv", ".json"), encoding="utf-8"))
        assert len(j) == len(d), p
        for i, r in d.iterrows():
            assert (
                j[i]["case_id"] == r.case_id
                and j[i]["model"] == r.model
                and (j[i]["condition"] == r.condition)
            )
            if isinstance(r.error, str) and r.error.strip():
                rows.append(
                    {
                        "retry_id": f"{blk}|{r.case_id}|{r.model}|{r.condition}",
                        "run_block": blk,
                        "case_block": block_key(blk),
                        "dataset": "TEP" if blk.startswith("tep") else "CWRU",
                        "rep": int(blk.split("rep")[1]) if "rep" in blk else 1,
                        "case_id": r.case_id,
                        "model": r.model,
                        "model_name": r.model_name,
                        "condition": r.condition,
                        "regime": "clean" if "clean" in blk else "degradation",
                        "original_row_index": i,
                        "original_error": str(r.error)[:300].replace("\n", " "),
                        "case_file": "cases/" + CASEFILE[block_key(blk)],
                        "source_csv": os.path.relpath(p, OUT_DIR).replace("\\", "/"),
                    }
                )
    m = pd.DataFrame(rows)
    assert len(m) == 211 and m.retry_id.is_unique
    m.to_csv(out("llm_revision", RETRY_DIR, "retry_manifest.csv"), index=False)
    print(m.groupby(["model_name", "dataset"]).size(), "\ntotal", len(m))


def cmd_rerun_manifest(a):
    out_dir = Path(a.out_dir)
    rows = []
    for p in sorted(glob.glob(str(RUNS_DIR / "cwru_clean_rep*" / "llm_experiment_results.csv"))):
        blk = block_of(p)
        d = pd.read_csv(p)
        for i, r in d.iterrows():
            status = (
                "abstained"
                if r.abstained == 1
                else "api_failure" if isinstance(r.error, str) and r.error.strip() else "response"
            )
            rows.append(
                {
                    "retry_id": f"{blk}|{r.case_id}|{r.model}|{r.condition}",
                    "run_block": blk,
                    "case_block": "cwru_clean",
                    "dataset": "CWRU",
                    "rep": int(blk.split("rep")[1]),
                    "case_id": r.case_id,
                    "model": r.model,
                    "model_name": r.model_name,
                    "condition": r.condition,
                    "regime": "clean",
                    "original_row_index": i,
                    "original_status": status,
                    "original_error": (
                        str(r.error)[:300].replace("\n", " ") if isinstance(r.error, str) else ""
                    ),
                    "case_file": "cases/cwru_clean.json",
                    "source_csv": os.path.relpath(p, OUT_DIR).replace("\\", "/"),
                }
            )
    m = pd.DataFrame(rows)
    assert len(m) == 360 and m.retry_id.is_unique
    os.makedirs(out_dir, exist_ok=True)
    m.to_csv(out_dir / "rerun_manifest_all360.csv", index=False)
    m[m.original_status != "abstained"].to_csv(out_dir / "rerun_manifest.csv", index=False)
    print(m.groupby(["model_name", "original_status"]).size().unstack(fill_value=0))


def setup_domain(dataset):
    if dataset == "CWRU":
        import cwru_data

        cwru_data.activate()


def send(backend, system_prompt, user_prompt, max_tokens=None, timeout=None, stream=False):
    payload = {
        "model": backend.model_id,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    }
    if backend.model_id.startswith("gpt-5") or backend.model_id.startswith("o1"):
        payload["max_completion_tokens"] = max_tokens or backend.max_tokens
    else:
        payload["max_tokens"] = max_tokens or backend.max_tokens
        payload["temperature"] = backend.temperature
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        f"{backend.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {backend.api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout or backend.timeout) as resp:
        if not stream:
            return (json.loads(resp.read().decode("utf-8")), dict(resp.headers))
        content, reasoning, usage, finish, model = ([], [], {}, "", "")
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:].strip())
            model = ev.get("model", model)
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                dlt = ch.get("delta") or {}
                content.append(dlt.get("content") or "")
                reasoning.append(dlt.get("reasoning") or dlt.get("reasoning_content") or "")
                finish = ch.get("finish_reason") or finish
        msg = {"role": "assistant", "content": "".join(content), "reasoning": "".join(reasoning)}
        return (
            {"model": model, "choices": [{"message": msg, "finish_reason": finish}], "usage": usage},
            dict(resp.headers),
        )


def ablation_builders():

    def body(payload, head, lines):
        return (
            f"{llm._render_header(payload)}\n\nPROCESS VARIABLE SUMMARY (most-changed variables in the window):\n{llm._render_sensor_summary(payload)}\n\n{head}\n"
            + "\n".join(lines)
            + f"\n\n{INSTR}\n\n{llm._RESPONSE_FORMAT}"
        )

    def rank_only(p):
        return body(
            p,
            XAI_HEAD,
            [f"  {i}. {e.sensor} ({llm.describe_sensor(e.sensor)})" for i, e in enumerate(p.top_evidence, 1)],
        )

    def percentages_only(p):
        return body(
            p,
            PCT_HEAD,
            [
                f"  - {e.sensor} - {e.attribution * 100:.1f}% of the model's attribution ({llm.describe_sensor(e.sensor)})"
                for e in p.top_evidence
            ],
        )

    return {
        "plain": llm.build_plain_prompt,
        "xai_grounded": llm.build_xai_prompt,
        "rank_only": rank_only,
        "percentages_only": percentages_only,
    }


def verify_prompts(cases, bld):
    for p in cases:
        x, r, q = (bld["xai_grounded"](p), bld["rank_only"](p), bld["percentages_only"](p))
        assert INSTR in x, "XAI instruction text changed in llm_prompts"
        x_no_pct = re.sub(" - \\d+\\.\\d% of the model's attribution", "", x)
        assert r == x_no_pct, f"rank_only differs from XAI beyond percentages ({p.case_id})"
        x_no_rank = re.sub("^  \\d+\\. ", "  - ", x.replace(XAI_HEAD, PCT_HEAD), flags=re.M)
        assert q == x_no_rank, f"percentages_only differs from XAI beyond ranks ({p.case_id})"
        assert "%" not in r.split(XAI_HEAD)[1].split(INSTR)[0] and (not re.search("^  \\d+\\. ", q, re.M))
        assert bld["plain"](p).split("PROCESS VARIABLE SUMMARY")[1].split("Explain")[0] in x
    return True


def control_variant(payload, condition, rep):
    ev = list(payload.top_evidence)
    key = f"{payload.case_id}|{rep}|{condition}"
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
    while True:
        p = rng.permutation(len(ev))
        if p[0] != 0:
            break
    if condition == "random_order_plain":
        new = [ev[i] for i in p]
    elif condition == "shuffled_attribution":
        new = [dataclasses.replace(ev[j], attribution=ev[i].attribution) for i, j in enumerate(p)]
    else:
        raise ValueError(condition)
    return dataclasses.replace(payload, top_evidence=new)


def cmd_ablation_run(a):
    if a.dataset == "cwru":
        import cwru_data

        cwru_data.activate()
    cases = load_cases(str(ABL_CASES[a.dataset]))
    bld = ablation_builders()
    verify_prompts(cases, bld)
    print(f"prompt isolation verified on {len(cases)} {a.dataset.upper()} cases", flush=True)
    if a.print_example:
        for c in ABL_ALL:
            print(f"\n{c}\nSYSTEM:\n{llm.SYSTEM_PROMPT}\nUSER:\n{bld[c](cases[0])}")
        return
    synthetic = "mock" in a.models
    out_dir = "results" if not synthetic else "results_SYNTHETIC"
    tag = "_".join(a.models) + "_rep" + "-".join(map(str, a.reps))
    path = rv("rank_pct_ablation", f"{out_dir}/{a.dataset}_{tag}.csv")
    cols = [
        "dataset",
        "rep",
        "case_id",
        "model",
        "model_name",
        "condition",
        "true_fault",
        "predicted_fault",
        "prediction_correct",
        "calibrated_confidence",
        "top1_faithfulness",
        "faithfulness",
        "rank_correlation",
        "hallucination_rate",
        "fault_consistency",
        "cited_sensors",
        "unsupported_sensors",
        "evidence_sensors",
        "latency_s",
        "prompt_tokens",
        "completion_tokens",
        "finish_reason",
        "error",
        "timestamp",
        "response_text",
    ]
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if not r["error"]:
                    done.add((r["case_id"], r["model"], r["condition"], int(r["rep"])))
    backends = {k: (MODEL_REGISTRY[k], build_backend(MODEL_REGISTRY[k])) for k in a.models}
    todo = [
        (rep, p, c, k)
        for rep in a.reps
        for p in cases
        for c in a.conditions
        for k in a.models
        if (p.case_id, k, c, rep) not in done
    ]
    print(f"{len(todo)} generations to run ({len(done)} already done) -> {path}", flush=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new_file:
            w.writeheader()
        for i, (rep, p, c, k) in enumerate(todo, 1):
            spec, be = backends[k]
            resp = generate_response(
                backend=be,
                spec=spec,
                system_prompt=llm.SYSTEM_PROMPT,
                user_prompt=bld[c](p),
                condition=c,
                case_id=p.case_id,
            )
            s = score_response(resp, p, top_k=3)
            w.writerow(
                {
                    "dataset": a.dataset.upper(),
                    "rep": rep,
                    "case_id": p.case_id,
                    "model": spec.key,
                    "model_name": spec.display_name,
                    "condition": c,
                    "true_fault": p.true_fault,
                    "predicted_fault": p.predicted_fault,
                    "prediction_correct": int(p.is_correct),
                    "calibrated_confidence": p.calibrated_confidence,
                    "top1_faithfulness": s.top1_faithfulness,
                    "faithfulness": s.faithfulness,
                    "rank_correlation": s.rank_correlation,
                    "hallucination_rate": s.hallucination_rate,
                    "fault_consistency": s.fault_consistency,
                    "cited_sensors": ";".join(s.cited_sensors),
                    "unsupported_sensors": ";".join(s.unsupported_sensors),
                    "evidence_sensors": ";".join(p.evidence_sensors(3)),
                    "latency_s": resp.latency_s,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "finish_reason": resp.finish_reason,
                    "error": resp.error,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "response_text": resp.text,
                }
            )
            fh.flush()
            print(
                f"[{i}/{len(todo)}] rep{rep} {spec.display_name} | {c} | {p.case_id} | "
                + (f"ERROR {resp.error[:60]}" if resp.error else f"EA@1={s.top1_faithfulness:.0f}"),
                flush=True,
            )


def cmd_run(a):
    dataset, models, limit = (a.dataset, a.models, a.limit)
    workers = {
        "qwen35_9b": a.qwen_workers,
        "gpt5_mini": a.gpt_workers,
        "deepseek_v41_flash": 3,
        "glm53_flash": 3,
    }
    manifest, out_dir, blocks = (a.manifest, Path(a.out_dir), a.blocks)
    os.makedirs(out_dir, exist_ok=True)
    setup_domain(dataset)
    m = pd.read_csv(manifest)
    m = m[(m.dataset == dataset) & m.model.isin(models)]
    abl = {}
    use_ctl = bool(m.condition.isin(list(CONTROL_CONDITIONS)).any())
    if m.condition.isin(list(ABL_NEW)).any():
        bld = ablation_builders()
        abl = {c: bld[c] for c in ABL_NEW}
    if blocks:
        m = m[m.case_block.isin(blocks)]
    outp = out_dir / f"_partial_results_{dataset.lower()}.csv"
    logp = out_dir / f"_partial_log_{dataset.lower()}.jsonl"
    done = set(pd.read_csv(outp).query("status == 'success'").retry_id) if outp.exists() else set()
    m = m[~m.retry_id.isin(done)]
    if limit:
        m = m.groupby("model").head(limit)
    cases = {
        cb: {c.case_id: c for c in load_cases(str(case_path(f)))}
        for cb, f in m.groupby("case_block").case_file.first().items()
    }
    backends = {k: build_backend(get_spec(k)) for k in m.model.unique()}
    orig = pd.concat([pd.read_csv(p).assign(run_block=block_of(p)) for p in run_files()])
    ok = orig[orig.error.isna()]
    ok = ok.assign(case_block=ok.run_block.map(block_key))
    ref_tokens = ok.groupby(["case_block", "model", "case_id", "condition"]).prompt_tokens.agg(
        lambda s: sorted(set((int(x) for x in s)))
    )
    lock, stop = (threading.Lock(), {})
    cols = [
        "retry_id",
        "status",
        "attempts",
        "dataset",
        "run_block",
        "rep",
        "case_id",
        "model",
        "model_name",
        "condition",
        "served_model",
        "system_fingerprint",
        "prompt_tokens",
        "reference_prompt_tokens_original",
        "prompt_tokens_match",
        "completion_tokens",
        "finish_reason",
        "latency_s",
        "top1_faithfulness",
        "faithfulness",
        "rank_correlation",
        "hallucination_rate",
        "fault_consistency",
        "cited_sensors",
        "unsupported_sensors",
        "evidence_sensors",
        "calibrated_confidence",
        "prediction_correct",
        "abstain_check",
        "error",
        "timestamp_utc",
        "response_text",
    ]
    escalate = "start_max_tokens" in m.columns
    if escalate:
        cols.insert(cols.index("response_text"), "max_tokens_used")
    if use_ctl:
        for c in ("supplied_order", "ea1_vs_supplied", "recall_vs_supplied", "rankcorr_vs_supplied"):
            cols.insert(cols.index("response_text"), c)
    print(f"{dataset}: {len(m)} pending retries for {sorted(m.model.unique())}", flush=True)

    def log(ev):
        with lock, open(logp, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def write(row):
        with lock:
            new = not outp.exists()
            with open(outp, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                if new:
                    w.writeheader()
                w.writerow(row)
                fh.flush()
                os.fsync(fh.fileno())

    def one(r):
        if stop.get(r.model):
            return
        p = cases[r.case_block][r.case_id]
        be = backends[r.model]
        pf = control_variant(p, r.condition, r.rep) if use_ctl and r.condition in CONTROL_CONDITIONS else None
        if pf is not None:
            up = llm.build_prompt(pf, CONTROL_CONDITIONS[r.condition], llm.DEFAULT_ABSTAIN_THRESHOLD)
        else:
            up = (
                abl[r.condition](p)
                if r.condition in abl
                else llm.build_prompt(p, r.condition, llm.DEFAULT_ABSTAIN_THRESHOLD)
            )
        ab, reason = should_abstain(p, 0.6, 0.85)
        abstain_check = (
            "would_abstain:" + reason if ab and r.condition == "xai_uncertainty_grounded" else "no_abstention"
        )
        attempt, err, t0 = (0, "", time.time())
        mt = int(r.start_max_tokens) if escalate else be.max_tokens
        while attempt < 6:
            attempt += 1
            t0 = time.time()
            try:
                body, headers = send(
                    be,
                    llm.SYSTEM_PROMPT,
                    up,
                    mt,
                    LONG_TIMEOUT if mt > be.max_tokens else None,
                    stream=mt > be.max_tokens,
                )
                choice = body["choices"][0]
                usage = body.get("usage", {}) or {}
                text = OpenAICompatibleBackend._extract_text(choice, choice.get("message") or {}, usage)
                resp = LLMResponse(
                    model=r.model,
                    condition=r.condition,
                    case_id=r.case_id,
                    text=text,
                    latency_s=time.time() - t0,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    finish_reason=str(choice.get("finish_reason", "")),
                )
                if escalate and resp.finish_reason == "length" and (mt < LONG_MAX_TOKENS):
                    log(
                        {
                            "retry_id": r.retry_id,
                            "attempt": attempt,
                            "status": "escalate_max_tokens",
                            "from": mt,
                            "to": LONG_MAX_TOKENS,
                        }
                    )
                    print(
                        f"[ESCALATE] {r.model_name} | {r.case_id} | {r.condition} | length at {mt} -> {LONG_MAX_TOKENS}",
                        flush=True,
                    )
                    mt = LONG_MAX_TOKENS
                    attempt -= 1
                    continue
                s = score_response(resp, p, top_k=3)
                ref = ref_tokens.get((r.case_block, r.model, r.case_id, r.condition), [])
                row = {
                    "retry_id": r.retry_id,
                    "status": "success",
                    "attempts": attempt,
                    "dataset": dataset,
                    "run_block": r.run_block,
                    "rep": r.rep,
                    "case_id": r.case_id,
                    "model": r.model,
                    "model_name": r.model_name,
                    "condition": r.condition,
                    "served_model": body.get("model", ""),
                    "system_fingerprint": body.get("system_fingerprint", ""),
                    "prompt_tokens": resp.prompt_tokens,
                    "reference_prompt_tokens_original": ";".join(map(str, ref)),
                    "prompt_tokens_match": resp.prompt_tokens in ref if ref else "no_reference",
                    "completion_tokens": resp.completion_tokens,
                    "finish_reason": resp.finish_reason,
                    "latency_s": round(resp.latency_s, 2),
                    "top1_faithfulness": s.top1_faithfulness,
                    "faithfulness": s.faithfulness,
                    "rank_correlation": s.rank_correlation,
                    "hallucination_rate": s.hallucination_rate,
                    "fault_consistency": s.fault_consistency,
                    "cited_sensors": ";".join(s.cited_sensors),
                    "unsupported_sensors": ";".join(s.unsupported_sensors),
                    "evidence_sensors": ";".join(p.evidence_sensors(3)),
                    "calibrated_confidence": p.calibrated_confidence,
                    "prediction_correct": int(p.is_correct),
                    "abstain_check": abstain_check,
                    "error": "",
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "response_text": text,
                }
                if escalate:
                    row["max_tokens_used"] = mt
                if pf is not None:
                    sf = score_response(resp, pf, top_k=3)
                    row.update(
                        {
                            "supplied_order": ";".join(pf.evidence_sensors()),
                            "ea1_vs_supplied": sf.top1_faithfulness,
                            "recall_vs_supplied": sf.faithfulness,
                            "rankcorr_vs_supplied": sf.rank_correlation,
                        }
                    )
                write(row)
                log(
                    {
                        "retry_id": r.retry_id,
                        "attempt": attempt,
                        "status": "success",
                        "served_model": row["served_model"],
                        "t": row["timestamp_utc"],
                    }
                )
                print(
                    f"{r.model_name} | {r.condition} | {r.run_block} | {r.case_id} | EA@1={s.top1_faithfulness:.0f} | tok={resp.prompt_tokens} ref={row['reference_prompt_tokens_original']} | {row['served_model']}",
                    flush=True,
                )
                return
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:400].replace("\n", " ")
                err = f"HTTP {e.code}: {detail}"
                if any((q in detail.lower() for q in QUOTA_MARKERS)) or e.code in (401, 403):
                    stop[r.model] = err
                    log(
                        {
                            "retry_id": r.retry_id,
                            "attempt": attempt,
                            "status": "stopped_quota_or_auth",
                            "error": err,
                        }
                    )
                    print(f"[STOP] {r.model_name}: {err[:160]}", flush=True)
                    break
                wait = None
                retry_after = e.headers.get("Retry-After") if e.headers else None
                if retry_after:
                    try:
                        wait = min(float(retry_after), 300.0)
                    except ValueError:
                        wait = None
                if e.code not in TRANSIENT:
                    log(
                        {
                            "retry_id": r.retry_id,
                            "attempt": attempt,
                            "status": "permanent_error",
                            "error": err,
                        }
                    )
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                err, wait = (f"{type(e).__name__}: {str(e)[:200]}", None)
            except RuntimeError as e:
                err, wait = (f"RuntimeError: {str(e)[:300]}", None)
            wait = wait if wait is not None else min(120.0, 5.0 * 2 ** (attempt - 1))
            log(
                {
                    "retry_id": r.retry_id,
                    "attempt": attempt,
                    "status": "transient_error",
                    "error": err[:300],
                    "wait_s": wait,
                }
            )
            print(
                f"[RETRY {attempt}] {r.model_name} | {r.case_id} | {err[:100]} | wait {wait:.0f}s", flush=True
            )
            if attempt < 6:
                time.sleep(wait)
        write(
            {
                "retry_id": r.retry_id,
                "status": "failed",
                "attempts": attempt,
                "dataset": dataset,
                "run_block": r.run_block,
                "rep": r.rep,
                "case_id": r.case_id,
                "model": r.model,
                "model_name": r.model_name,
                "condition": r.condition,
                "error": (stop.get(r.model) or err)[:400],
                "abstain_check": abstain_check,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )

    jobs = list(m.itertuples(index=False))
    by_model = {k: [j for j in jobs if j.model == k] for k in m.model.unique()}

    def run_model(k):
        with ThreadPoolExecutor(max_workers=workers.get(k, 1)) as ex:
            list(ex.map(one, by_model[k]))

    threads = [threading.Thread(target=run_model, args=(k,)) for k in by_model]
    [t.start() for t in threads]
    [t.join() for t in threads]
    print(f"{dataset}; stopped providers: {stop}", flush=True)


def boot(g, fn, n=B):
    w = g.window.unique()
    pos = {x: np.flatnonzero(g.window.to_numpy() == x) for x in w}
    rng = np.random.default_rng(SEED)
    res = []
    for _ in range(n):
        v = fn(g.iloc[np.concatenate([pos[x] for x in rng.choice(w, len(w))])])
        if np.isfinite(v):
            res.append(v)
    return (np.percentile(res, 2.5), np.percentile(res, 97.5)) if res else (np.nan, np.nan)


def table8(R, label):
    rows = []
    for (ds, c), g in R[R.level == "Clean"].groupby(["dataset", "condition"]):
        r = {"version": label, "dataset": ds, "condition": c, "n": len(g)}
        for k, v in MET.items():
            r[v] = g[k].mean()
        r["EA@1_ci_low"], r["EA@1_ci_high"] = boot(g, lambda s: s.top1_faithfulness.mean(), 2000)
        rows.append(r)
    return rows


def table9(A, R, label):
    rows = []
    for (ds, lv), g in A.groupby(["dataset", "level"]):
        r_ = R[(R.dataset == ds) & (R.level == lv)]
        p = r_[r_.condition == "plain"].top1_faithfulness.mean()
        x = r_[r_.condition == "xai_grounded"].top1_faithfulness.mean()
        rows.append(
            {
                "version": label,
                "dataset": ds,
                "level": lv,
                "plain_EA1": p,
                "xai_EA1": x,
                "delta": x - p,
                "abstentions": int((g.final_status == "abstained").sum()),
                "n_error_free_records": int(g.final_status.isin(["response", "abstained"]).sum()),
                "n_responses": len(r_),
                "models_with_responses": r_.model_name.nunique(),
            }
        )
    return rows


def tableS4(R, label):
    rows = []
    for (ds, m), g in R[R.level == "Clean"].groupby(["dataset", "model_name"]):
        p = g[g.condition == "plain"].top1_faithfulness.mean()
        x = g[g.condition == "xai_grounded"].top1_faithfulness.mean()
        rows.append(
            {
                "version": label,
                "dataset": ds,
                "model": m,
                "plain": p,
                "xai": x,
                "delta": x - p,
                "n_plain": int((g.condition == "plain").sum()),
                "n_xai": int((g.condition == "xai_grounded").sum()),
            }
        )
    return rows


def sensitivity(R, label):
    key = ["dataset", "level", "rep", "case_id", "window", "model_name"]
    P = R[R.condition == "plain"][key + ["top1_faithfulness"]].merge(
        R[R.condition == "xai_grounded"][key + ["top1_faithfulness"]], on=key, suffixes=("_p", "_x")
    )
    P["diff"] = P.top1_faithfulness_x - P.top1_faithfulness_p
    rows = []
    for (ds, lv), g in P.groupby(["dataset", "level"]):
        rr = R[(R.dataset == ds) & (R.level == lv)]
        mb = rr.groupby(["condition", "model_name"]).top1_faithfulness.mean().unstack(0)
        lo, hi = boot(g, lambda s: s["diff"].mean())
        rows.append(
            {
                "version": label,
                "dataset": ds,
                "level": lv,
                "matched_pairs": len(g),
                "delta_matched": g["diff"].mean(),
                "delta_ci_low": lo,
                "delta_ci_high": hi,
                "plain_model_balanced": mb["plain"].mean(),
                "xai_model_balanced": mb["xai_grounded"].mean(),
                "delta_model_balanced": (mb["xai_grounded"] - mb["plain"]).mean(),
            }
        )
    return rows


def ers_rows(R, label):
    rel = {}
    for cb in LEVEL:
        f = CASES_DIR / (cb + "_xai_reliability.json").replace("_clean_xai", "_xai")
        rel[cb] = json.load(open(f, encoding="utf-8"))
    R = R.copy()
    cb = R.run_block.str.split("_rep").str[0]
    R["s"] = [rel[b][c]["xai_stability"] for b, c in zip(cb, R.case_id)]
    R["a"] = [rel[b][c]["xai_agreement"] for b, c in zip(cb, R.case_id)]
    R["ERS"] = (
        W["calibrated_confidence"] * R.calibrated_confidence
        + W["xai_stability"] * R.s
        + W["xai_agreement"] * R.a
    ) / 0.85
    res = []
    for ds, g in R.groupby("dataset"):
        if ds == "TEP":
            g = g[g.window != "fault2_at100"]
        for k, v in (("top1_faithfulness", "EA@1"), ("faithfulness", "Recall")):
            rc, re_ = (np.corrcoef(g.calibrated_confidence, g[k])[0, 1], np.corrcoef(g.ERS, g[k])[0, 1])
            lo, hi = boot(
                g,
                lambda s, k=k: (
                    np.corrcoef(s.ERS, s[k])[0, 1] - np.corrcoef(s.calibrated_confidence, s[k])[0, 1]
                    if s[k].nunique() > 1
                    else np.nan
                ),
                2000,
            )
            res.append(
                {
                    "version": label,
                    "dataset": ds,
                    "outcome": v,
                    "n": len(g),
                    "windows": g.window.nunique(),
                    "r_conf": rc,
                    "r_ERS": re_,
                    "delta_r": re_ - rc,
                    "delta_ci_low": lo,
                    "delta_ci_high": hi,
                }
            )
    return res


def final_originals():
    parts = []
    for p in run_files():
        blk = block_of(p)
        d = pd.read_csv(p)
        d["response_text"] = [
            x.get("response_text", "") for x in json.load(open(p.replace(".csv", ".json"), encoding="utf-8"))
        ]
        d["run_block"] = blk
        d["dataset"] = "TEP" if blk.startswith("tep") else "CWRU"
        d["rep"] = int(blk.split("rep")[1]) if "rep" in blk else 1
        parts.append(d)
    O = pd.concat(parts, ignore_index=True)
    O["final_status"] = np.where(
        O.error.notna(), "api_failure", np.where(O.abstained == 1, "abstained", "response")
    )
    O["record_id"] = O.run_block + "|" + O.case_id + "|" + O.model + "|" + O.condition
    O["generation_batch"] = "original_2026-09"
    O["served_model"] = ""
    return O


def final_successes(pattern):
    s = successes(pattern).rename(columns={"retry_id": "record_id"})
    s["final_status"] = "response"
    return s


def final_build():
    O = final_originals()
    assert len(O) == 1296 and O.record_id.is_unique
    retry = final_successes("llm_retry_completion/_partial_results_*.csv").assign(
        generation_batch="retry_2026-10"
    )
    rerun = final_successes("cwru_clean_rerun/_partial_results_cwru.csv").assign(
        generation_batch="cwru_clean_rerun_2026-10"
    )
    rerun = rerun[rerun.model.isin(["gpt5_mini", "qwen35_9b"])]
    new = pd.concat(
        [
            final_successes(f"{d}/_partial_results_*.csv")
            for d in ("deepseek_v41_replacement", "glm53_replacement")
        ]
    ).assign(generation_batch="replacement_model_2026-10")
    base = O[O.model.isin(["gpt5_mini", "qwen35_9b"])].copy()
    keep_orig = base[~(base.run_block.str.startswith("cwru_clean") | (base.final_status == "api_failure"))]
    assert retry.record_id.isin(base[base.final_status == "api_failure"].record_id).all()
    parts = [keep_orig, retry, rerun]
    for key in NEW_MODELS:
        ab = O[(O.model == "gpt5_mini") & (O.final_status == "abstained")].copy()
        ab["model"] = key
        ab["record_id"] = ab.run_block + "|" + ab.case_id + "|" + key + "|" + ab.condition
        ab["generation_batch"] = "replacement_model_2026-10 (abstention, no call)"
        parts.append(ab)
    parts.append(new)
    names = {
        "gpt5_mini": "GPT-5 Mini",
        "qwen35_9b": "Qwen3.5-9B",
        "deepseek_v41_flash": "DeepSeek-V4.1-Flash-1017",
        "glm53_flash": "GLM-5.3-Flash",
    }
    F = pd.concat([p.reindex(columns=KEEP) for p in parts], ignore_index=True)
    if rv_glob("qwen_fix/_partial_results_*.csv"):
        q = final_successes("qwen_fix/_partial_results_*.csv")
        q = q[~q.condition.isin(["rank_only", "percentages_only"])].assign(
            generation_batch="qwen_fix_2026-10"
        )
        assert q.record_id.isin(F.record_id).all()
        F = pd.concat([F[~F.record_id.isin(q.record_id)], q.reindex(columns=KEEP)], ignore_index=True)
    F["model_name"] = F.model.map(names)
    conf = O.groupby(["run_block", "case_id"]).calibrated_confidence.first()
    F["calibrated_confidence"] = [conf[b, c] for b, c in zip(F.run_block, F.case_id)]
    return (O, F)


def final_audit(O, F):
    key = F.run_block + "|" + F.case_id + "|" + F.model + "|" + F.condition
    o_abs = O[O.final_status == "abstained"]
    f_abs = []
    for m in F.model.unique():
        s = F[(F.final_status == "abstained") & (F.model == m)]
        f_abs += (s.run_block + s.case_id + s.condition).tolist()
    a = {
        "records": len(F),
        "unique_records": int(key.nunique()),
        "final_status": F.final_status.value_counts().to_dict(),
        "per_model_status": F.groupby(["model_name", "final_status"])
        .size()
        .unstack(fill_value=0)
        .to_dict("index"),
        "per_batch": F.generation_batch.value_counts().to_dict(),
        "abstentions_equal_to_published": bool(
            sorted((o_abs.run_block + o_abs.case_id + o_abs.condition).tolist()) == sorted(f_abs)
        ),
        "retained_original_responses_unchanged": True,
        "truncated_finish_length": F[F.final_status == "response"]
        .groupby("model_name")
        .finish_reason.apply(lambda x: int((x == "length").sum()))
        .to_dict(),
        "served_models": F[F.served_model.fillna("") != ""]
        .groupby("model_name")
        .served_model.apply(lambda x: sorted(set(x)))
        .to_dict(),
    }
    o = O.set_index("record_id")
    f = F[F.generation_batch == "original_2026-09"].set_index("record_id")
    a["retained_original_responses_unchanged"] = bool(
        (o.loc[f.index, "top1_faithfulness"].values == f.top1_faithfulness.values).all()
    )
    a["retained_original_count"] = len(f)
    assert a["records"] == 1296 and a["unique_records"] == 1296 and ("api_failure" not in a["final_status"])
    return a


def final_tables(O, F):

    def prep(A):
        A = A.copy()
        A["window"] = np.where(
            A.dataset == "TEP", A.case_id.str.replace("_(none|noise)$", "", regex=True), A.case_id
        )
        A["level"] = A.run_block.str.split("_rep").str[0].map(LEVEL)
        return A

    versions = {"published (2026-09)": prep(O), "final (2026-10)": prep(F)}
    res = {k: [] for k in ("Table8", "Table9_S5", "TableS4", "Sensitivity", "ERS_explanation")}
    for lab, A in versions.items():
        R = A[A.final_status == "response"]
        res["Table8"] += table8(R, lab)
        res["Table9_S5"] += table9(A, R, lab)
        res["TableS4"] += tableS4(R, lab)
        res["Sensitivity"] += sensitivity(R, lab)
        res["ERS_explanation"] += ers_rows(R, lab)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for k, rows in res.items():
        df = pd.DataFrame(rows)
        df.to_csv(
            rv("llm_final_consolidated", f"{k}_published_vs_final.csv"), index=False, float_format="%.4f"
        )
        print(f"\n{k}\n" + df.round(3).to_string())
    R = versions["final (2026-10)"]
    R = R[R.final_status == "response"]
    pm = (
        R.groupby(["dataset", "level", "model_name", "condition"])
        .top1_faithfulness.mean()
        .unstack("condition")
    )
    pm.to_csv(rv("llm_final_consolidated", "EA1_by_model_level_final.csv"), float_format="%.4f")
    print("\nEA@1 by model x level (final)\n" + pm.round(3).to_string())


def cmd_final(a):
    O, F = final_build()
    F.to_csv(rv("llm_final_consolidated", "final_results.csv"), index=False)
    au = final_audit(O, F)
    json.dump(
        au,
        open(rv("llm_final_consolidated", "final_audit.json"), "w", encoding="utf-8"),
        indent=2,
        default=str,
    )
    print(json.dumps(au, indent=1, default=str))
    final_tables(O, F)


def cmd_run_details(a):
    rows, cons = ([], [])
    for name, folder, planned in RUNS:
        R = pd.concat(
            [pd.read_csv(p) for p in rv_glob(f"{folder}/_partial_results_*.csv")], ignore_index=True
        )
        ev = []
        for p in rv_glob(f"{folder}/_partial_log_*.jsonl"):
            ev += [json.loads(l) for l in open(p, encoding="utf-8")]
        E = pd.DataFrame(ev).reindex(columns=["retry_id", "status", "error", "attempt"])
        E["model"] = E.retry_id.str.split("|").str[2]
        S = R[R.status == "success"].drop_duplicates("retry_id")
        for m, g in S.groupby("model"):
            e = E[E.model == m]
            rows.append(
                {
                    "run": name,
                    "folder": folder,
                    "model": g.model_name.iloc[0],
                    "served_model": "; ".join(sorted(set(g.served_model.astype(str)))),
                    "successful": len(g),
                    "unresolved": int(
                        R[
                            (R.status == "failed") & (R.model == m) & ~R.retry_id.isin(S.retry_id)
                        ].retry_id.nunique()
                    ),
                    "calls_with_retry": int((g.attempts > 1).sum()),
                    "transient_events": int((e.status == "transient_error").sum()),
                    "http429": int(e.error.astype(str).str.contains("HTTP 429").sum()),
                    "http5xx": int(e.error.astype(str).str.contains("HTTP 5\\d\\d").sum()),
                    "first_utc": g.timestamp_utc.min(),
                    "last_utc": g.timestamp_utc.max(),
                    "mean_latency_s": g.latency_s.mean(),
                    "mean_prompt_tokens": g.prompt_tokens.mean(),
                    "mean_completion_tokens": g.completion_tokens.mean(),
                    "finish_length": int((g.finish_reason == "length").sum()),
                }
            )
        S = S.assign(cb=S.run_block.str.split("_rep").str[0])
        k = S.groupby(["model", "cb", "case_id", "condition"]).prompt_tokens.nunique()
        match = S.get("prompt_tokens_match", pd.Series(dtype=str)).astype(str)
        cons.append(
            {
                "run": name,
                "groups_with_repeats": int(
                    (S.groupby(["model", "cb", "case_id", "condition"]).size() > 1).sum()
                ),
                "groups_with_differing_prompt_tokens": int((k > 1).sum()),
                "match_original_true": int((match == "True").sum()),
                "match_original_false": int((match == "False").sum()),
            }
        )
    D = pd.DataFrame(rows)
    D.to_csv(rv("llm_final_consolidated", "Run_Details.csv"), index=False, float_format="%.4f")
    C = pd.DataFrame(cons)
    C.to_csv(rv("llm_final_consolidated", "Prompt_Consistency.csv"), index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    print(D.round(2).to_string())
    print(C.to_string())


def ablation_boot_mean(g, col, n=B, seed=SEED):
    cases = g.case_id.unique()
    idx = {c: g[col].to_numpy()[g.case_id.to_numpy() == c] for c in cases}
    rng = np.random.default_rng(seed)
    m = []
    for _ in range(n):
        m.append(np.concatenate([idx[c] for c in rng.choice(cases, len(cases))]).mean())
    return pct_ci(m)


def ablation_boot_diff(g, col, a, b, n=B, seed=SEED):
    cases = g.case_id.unique()
    rng = np.random.default_rng(seed)
    d = []
    ga = {c: g[(g.case_id == c) & (g.condition == a)][col].to_numpy() for c in cases}
    gb = {c: g[(g.case_id == c) & (g.condition == b)][col].to_numpy() for c in cases}
    for _ in range(n):
        s = rng.choice(cases, len(cases))
        xa = np.concatenate([ga[c] for c in s])
        xb = np.concatenate([gb[c] for c in s])
        d.append(xa.mean() - xb.mean() if len(xa) and len(xb) else np.nan)
    return np.nanpercentile(d, [2.5, 97.5])


def ablation_load():
    folder = "rank_pct_ablation_final"
    r = pd.concat([pd.read_csv(p) for p in rv_glob(f"{folder}/_partial_results_*.csv")], ignore_index=True)
    new = r[r.status == "success"].drop_duplicates("retry_id").assign(source="new ablation run (2026-10)")
    fx = rv_glob("qwen_fix/_partial_results_*.csv")
    if fx:
        q = pd.concat([pd.read_csv(p) for p in fx])
        q = q[(q.status == "success") & q.condition.isin(["rank_only", "percentages_only"])]
        q = q.drop_duplicates("retry_id").assign(source="new ablation run (2026-10)")
        new = pd.concat([new[~new.retry_id.isin(q.retry_id)], q], ignore_index=True)
    loop = new[new.finish_reason == "length"]
    print(f"excluded (no final answer within 32,768 tokens): {len(loop)} -> {sorted(loop.retry_id)}")
    pd.Series(sorted(loop.retry_id), name="retry_id").to_csv(
        rv(folder, "Ablation_Excluded_NoFinalAnswer.csv"), index=False
    )
    new = new[new.finish_reason != "length"]
    F = pd.read_csv(RV / "llm_final_consolidated" / "final_results.csv")
    old = F[
        F.run_block.str.contains("clean")
        & F.condition.isin(["plain", "xai_grounded"])
        & (F.final_status == "response")
    ]
    old = old.assign(source="final dataset (2026-10)")
    cols = [
        "dataset",
        "rep",
        "case_id",
        "model",
        "model_name",
        "condition",
        "top1_faithfulness",
        "faithfulness",
        "rank_correlation",
        "hallucination_rate",
        "finish_reason",
        "source",
    ]
    A = pd.concat([new[cols], old[cols]], ignore_index=True)
    A["error"] = ""
    n_unresolved = int(r[(r.status == "failed") & ~r.retry_id.isin(new.retry_id)].retry_id.nunique())
    print(f"new ablation responses {len(new)}/432, unresolved {n_unresolved}; Plain/XAI rows {len(old)}")
    return A


def cmd_ablation(a):
    folder = "rank_pct_ablation_final"
    A = ablation_load()
    ok = A[A.error.str.len() == 0]
    miss = (
        A.groupby(["dataset", "condition", "model_name", "source"])
        .agg(rows=("error", "size"), errors=("error", lambda s: int((s.str.len() > 0).sum())))
        .reset_index()
    )
    miss.to_csv(rv(folder, "Ablation_Missingness.csv"), index=False)
    S = []
    for (ds, c), g in ok.groupby(["dataset", "condition"]):
        r = {
            "dataset": ds,
            "condition": c,
            "n": len(g),
            "n_cases": g.case_id.nunique(),
            "source": g.source.iloc[0],
        }
        for col, lab in ABL_MET.items():
            lo, hi = ablation_boot_mean(g, col)
            r[lab] = g[col].mean()
            r[f"{lab}_ci_low"], r[f"{lab}_ci_high"] = (lo, hi)
        S.append(r)
    S = pd.DataFrame(S)
    S["condition"] = pd.Categorical(S.condition, ABL_ORDER)
    S = S.sort_values(["dataset", "condition"])
    S.to_csv(rv(folder, "Ablation_Summary.csv"), index=False, float_format="%.4f")
    M = (
        ok.groupby(["dataset", "model_name", "condition"])[list(ABL_MET)]
        .mean()
        .rename(columns=ABL_MET)
        .reset_index()
    )
    M["n"] = ok.groupby(["dataset", "model_name", "condition"]).size().values
    M.to_csv(rv(folder, "Ablation_By_Model.csv"), index=False, float_format="%.4f")
    D = []
    for ds, g in ok.groupby("dataset"):
        for x, y in (
            ("rank_only", "plain"),
            ("percentages_only", "plain"),
            ("xai_grounded", "plain"),
            ("xai_grounded", "rank_only"),
            ("xai_grounded", "percentages_only"),
            ("rank_only", "percentages_only"),
        ):
            for col, lab in ABL_MET.items():
                lo, hi = ablation_boot_diff(g, col, x, y)
                D.append(
                    {
                        "dataset": ds,
                        "contrast": f"{x} - {y}",
                        "metric": lab,
                        "diff": g[g.condition == x][col].mean() - g[g.condition == y][col].mean(),
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
    D = pd.DataFrame(D)
    D.to_csv(rv(folder, "Ablation_Differences.csv"), index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(miss.to_string())
    print(S.round(4).to_string())
    print(M.round(3).to_string())
    print(D[D.metric.isin(["EA@1", "RankCorr"])].round(4).to_string())


def controls_load():
    r = pd.concat([pd.read_csv(p) for p in rv_glob("xai_controls/_partial_results_*.csv")], ignore_index=True)
    c = r[r.status == "success"].drop_duplicates("retry_id")
    excl = c[c.finish_reason == "length"]
    c = c[c.finish_reason != "length"]
    unresolved = r[(r.status == "failed") & ~r.retry_id.isin(c.retry_id)].retry_id.nunique()
    print(f"control responses {len(c)}/432; excluded no-final-answer {len(excl)}; unresolved {unresolved}")
    F = pd.read_csv(RV / "llm_final_consolidated" / "final_results.csv")
    F = F[
        (F.final_status == "response")
        & F.run_block.str.contains("clean")
        & F.condition.isin(["plain", "xai_grounded"])
    ].copy()
    F["ea1_vs_supplied"] = F.top1_faithfulness
    F["rankcorr_vs_supplied"] = F.rank_correlation
    cols = [
        "dataset",
        "rep",
        "case_id",
        "model_name",
        "condition",
        "top1_faithfulness",
        "rank_correlation",
        "ea1_vs_supplied",
        "rankcorr_vs_supplied",
    ]
    A = pd.concat([c[cols], F[cols]], ignore_index=True)
    A["window"] = A.case_id.str.replace("_(none|noise)$", "", regex=True)
    return (A, len(excl))


def cmd_controls(a):
    folder = "xai_controls"
    A, n_excl = controls_load()
    S = (
        A.groupby(["dataset", "condition"])
        .agg(
            n=("top1_faithfulness", "size"),
            EA1_true_IG=("top1_faithfulness", "mean"),
            EA1_supplied=("ea1_vs_supplied", "mean"),
            RankCorr_true_IG=("rank_correlation", "mean"),
            RankCorr_supplied=("rankcorr_vs_supplied", "mean"),
        )
        .reset_index()
    )
    S["condition"] = pd.Categorical(S.condition, CTL_ORDER)
    S = S.sort_values(["dataset", "condition"])
    S.to_csv(rv(folder, "Controls_Summary.csv"), index=False, float_format="%.4f")
    D = []
    for ds, g in A.groupby("dataset"):
        for x, y, col in (
            ("random_order_plain", "plain", "top1_faithfulness"),
            ("shuffled_attribution", "xai_grounded", "top1_faithfulness"),
            ("shuffled_attribution", "plain", "top1_faithfulness"),
            ("shuffled_attribution", "xai_grounded", "ea1_vs_supplied"),
        ):
            lo, hi = boot_contrast(g, x, y, col)
            D.append(
                {
                    "dataset": ds,
                    "contrast": f"{x} - {y}",
                    "metric": "EA@1 vs supplied" if col == "ea1_vs_supplied" else "EA@1 vs true IG",
                    "diff": g[g.condition == x][col].mean() - g[g.condition == y][col].mean(),
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    D = pd.DataFrame(D)
    D.to_csv(rv(folder, "Controls_Differences.csv"), index=False, float_format="%.4f")
    M = (
        A.groupby(["dataset", "model_name", "condition"])
        .agg(EA1_true=("top1_faithfulness", "mean"), EA1_supplied=("ea1_vs_supplied", "mean"))
        .unstack("condition")
    )
    M.to_csv(rv(folder, "Controls_By_Model.csv"), float_format="%.4f")
    pd.set_option("display.width", 250)
    print(S.round(3).to_string())
    print(D.round(3).to_string())
    print(f"excluded {n_excl}")


def ers_load():
    rows = []
    for ds, blk, pat, rel in ERS_BLOCKS:
        r = json.load(open(CASES_DIR / rel, encoding="utf-8"))
        for f in glob.glob(str(RUNS_DIR / pat / "llm_experiment_results.csv")):
            d = pd.read_csv(f)
            d = d[d["error"].fillna("").astype(str).str.strip().str.len() == 0].copy()
            d["dataset"], d["block"], d["run_dir"] = (ds, blk, block_of(f))
            d["xai_stability"] = d.case_id.map(lambda c: r[c]["xai_stability"])
            d["xai_agreement"] = d.case_id.map(lambda c: r[c]["xai_agreement"])
            rows.append(d)
    A = pd.concat(rows, ignore_index=True)
    A["window"] = np.where(
        A.dataset == "TEP", A.case_id.str.replace("_(none|noise)$", "", regex=True), A.case_id
    )
    c, s, ag = (A.calibrated_confidence, A.xai_stability, A.xai_agreement)
    A["S_conf"] = c
    A["S_ERS"] = (W["calibrated_confidence"] * c + W["xai_stability"] * s + W["xai_agreement"] * ag) / (
        W["calibrated_confidence"] + W["xai_stability"] + W["xai_agreement"]
    )
    A["S_XAI"] = (W["xai_stability"] * s + W["xai_agreement"] * ag) / (
        W["xai_stability"] + W["xai_agreement"]
    )
    ref = [
        compute_ers(
            DiagnosisPayload(
                case_id="",
                scenario="",
                true_fault=0,
                predicted_fault=0,
                calibrated_confidence=float(x),
                raw_confidence=0.0,
                temperature=1.0,
                xai_stability=float(y),
                xai_agreement=float(z),
                data_quality=float(q),
            ),
            weights=W,
        )[0]
        for x, y, z, q in zip(c, s, ag, A.data_quality)
    ]
    A["ERS_reference_compute_ers"] = ref
    A["ERS_stored_runtime_csv"] = A.ers
    return A


def auc_cols(y, X):
    pos = y == 1
    n1 = int(pos.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.full(X.shape[1], np.nan)
    R = rankdata(X, axis=0)
    return (R[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def pearson_cols(y, X):
    yc = y - y.mean()
    vy = (yc**2).sum()
    Xc = X - X.mean(0)
    vx = (Xc**2).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Xc * yc[:, None]).sum(0) / np.sqrt(vx * vy)
    r[(vx <= 1e-15) | (vy <= 1e-15)] = np.nan
    return r


def ers_analyse(F, label):
    res, bt = ([], [])
    clusters = F.window.unique()
    rng = np.random.default_rng(SEED)
    draws = [rng.choice(clusters, len(clusters), replace=True) for _ in range(B)]
    for col, name, elig in OUTCOMES:
        G = F if elig == "all" else F[F.abstained == 0]
        y = G[col].to_numpy(float)
        X = G[list(SCORES)].to_numpy(float)
        gidx = {c: np.where(G.window.to_numpy() == c)[0] for c in G.window.unique()}
        point = pearson_cols(y, X)
        auc = {}
        binary = set(np.unique(y)) <= {0.0, 1.0}
        if binary and len(np.unique(y)) == 2:
            auc = {s: roc_auc_score(y, G[s]) for s in SCORES}
            assert np.allclose(auc_cols(y, X), [auc[s] for s in SCORES]), "fast AUC differs from sklearn"
        bs_r, bs_auc = ([], [])
        for dr in draws:
            ii = (
                np.concatenate([gidx[c] for c in dr if c in gidx])
                if any((c in gidx for c in dr))
                else np.array([], int)
            )
            if len(ii) < 3:
                bs_r.append([np.nan] * 3)
                bs_auc.append([np.nan] * 3)
                continue
            yy, XX = (y[ii], X[ii])
            bs_r.append(pearson_cols(yy, XX))
            bs_auc.append(auc_cols(yy, XX) if auc else [np.nan] * 3)
        bs_r, bs_auc = (np.array(bs_r), np.array(bs_auc))
        d_r = bs_r[:, 1] - bs_r[:, 0]
        q = lambda v: (
            (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
            if np.isfinite(v).any()
            else (None, None)
        )
        rec = {
            "analysis": label,
            "dataset": F.dataset.iloc[0],
            "outcome": name,
            "n_records": int(len(G)),
            "n_original_windows": int(G.window.nunique()),
            "n_window_block_units": int(G.groupby(["window", "block"]).ngroups),
            "outcome_mean_or_prevalence": float(y.mean()),
            "outcome_variance": float(y.var(ddof=1)) if len(y) > 1 else None,
            "n_incorrect_or_zero_units": (
                int((G.groupby(["window", "block"])[col].first() < 1).sum())
                if col == "prediction_correct"
                else None
            ),
        }
        for k, s in enumerate(SCORES):
            lo, hi = q(bs_r[:, k])
            rec[f"r_{s}"] = None if np.isnan(point[k]) else float(point[k])
            rec[f"r_{s}_ci_low"] = lo
            rec[f"r_{s}_ci_high"] = hi
            rec[f"score_sd_{s}"] = float(G[s].std(ddof=1))
        dl, dh = q(d_r)
        rec["delta_r_ERS_minus_conf"] = (
            None if np.isnan(point[1]) or np.isnan(point[0]) else float(point[1] - point[0])
        )
        rec["delta_r_ci_low"], rec["delta_r_ci_high"] = (dl, dh)
        rec["delta_r_boot_share_gt0"] = float(np.nanmean(d_r > 0)) if np.isfinite(d_r).any() else None
        rec["boot_invalid_share_r"] = float(np.isnan(bs_r[:, 0]).mean())
        rec["boot_invalid_share_r_ERS"] = float(np.isnan(bs_r[:, 1]).mean())
        rec["boot_invalid_share_r_XAI"] = float(np.isnan(bs_r[:, 2]).mean())
        if auc:
            for k, s in enumerate(SCORES):
                lo, hi = q(bs_auc[:, k])
                rec[f"auc_{s}"] = float(auc[s])
                rec[f"auc_{s}_ci_low"] = lo
                rec[f"auc_{s}_ci_high"] = hi
            da = bs_auc[:, 1] - bs_auc[:, 0]
            lo, hi = q(da)
            rec["delta_auc_ERS_minus_conf"] = float(auc["S_ERS"] - auc["S_conf"])
            rec["delta_auc_ci_low"], rec["delta_auc_ci_high"] = (lo, hi)
            rec["boot_invalid_share_auc"] = float(np.isnan(bs_auc[:, 0]).mean())
        res.append(rec)
        for s_i, s in enumerate(SCORES):
            v = bs_r[:, s_i]
            bt.append(
                {
                    "analysis": label,
                    "outcome": name,
                    "statistic": f"r({s})",
                    "point": rec[f"r_{s}"],
                    "ci_low": q(v)[0],
                    "ci_high": q(v)[1],
                    "n_valid": int(np.isfinite(v).sum()),
                    "n_invalid": int(np.isnan(v).sum()),
                    "invalid_share": float(np.isnan(v).mean()),
                }
            )
        bt.append(
            {
                "analysis": label,
                "outcome": name,
                "statistic": "delta_r(ERS - conf)",
                "point": rec["delta_r_ERS_minus_conf"],
                "ci_low": dl,
                "ci_high": dh,
                "n_valid": int(np.isfinite(d_r).sum()),
                "n_invalid": int(np.isnan(d_r).sum()),
                "invalid_share": float(np.isnan(d_r).mean()),
            }
        )
        if auc:
            for s_i, s in enumerate(SCORES):
                v = bs_auc[:, s_i]
                bt.append(
                    {
                        "analysis": label,
                        "outcome": name,
                        "statistic": f"AUC({s})",
                        "point": rec[f"auc_{s}"],
                        "ci_low": q(v)[0],
                        "ci_high": q(v)[1],
                        "n_valid": int(np.isfinite(v).sum()),
                        "n_invalid": int(np.isnan(v).sum()),
                        "invalid_share": float(np.isnan(v).mean()),
                    }
                )
            da = bs_auc[:, 1] - bs_auc[:, 0]
            bt.append(
                {
                    "analysis": label,
                    "outcome": name,
                    "statistic": "delta_AUC(ERS - conf)",
                    "point": rec["delta_auc_ERS_minus_conf"],
                    "ci_low": q(da)[0],
                    "ci_high": q(da)[1],
                    "n_valid": int(np.isfinite(da).sum()),
                    "n_invalid": int(np.isnan(da).sum()),
                    "invalid_share": float(np.isnan(da).mean()),
                }
            )
    return (res, bt)


def ers_report(loader, folder):
    A = loader()
    info = {
        "n_records": int(len(A)),
        "by_dataset": A.groupby("dataset").size().to_dict(),
        "windows": A.groupby("dataset").window.nunique().to_dict(),
        "max_abs_diff_S_ERS_vs_compute_ers": float((A.S_ERS - A.ERS_reference_compute_ers).abs().max()),
        "max_abs_diff_S_ERS_vs_stored_runtime_ers_column": float(
            (A.S_ERS - A.ERS_stored_runtime_csv).abs().max()
        ),
        "exact_weights": {
            "conf": W["calibrated_confidence"] / 0.85,
            "stab": W["xai_stability"] / 0.85,
            "agree": W["xai_agreement"] / 0.85,
            "xai_only_stab": W["xai_stability"] / 0.45,
            "xai_only_agree": W["xai_agreement"] / 0.45,
        },
        "answered_rows": A[A.abstained == 0].groupby("dataset").size().to_dict(),
        "abstained_rows": A[A.abstained == 1].groupby("dataset").size().to_dict(),
    }
    assert info["max_abs_diff_S_ERS_vs_compute_ers"] < 1e-12
    rows, boots = ([], [])
    for label, F in (
        ("CWRU (primary)", A[A.dataset == "CWRU"]),
        ("TEP excluding fault2_at100 (primary)", A[(A.dataset == "TEP") & (A.window != "fault2_at100")]),
        ("TEP unfiltered (sensitivity)", A[A.dataset == "TEP"]),
    ):
        r, b = ers_analyse(F, label)
        rows += r
        boots += b
    F = A[A.dataset == "CWRU"]
    U = F.groupby(["window", "block"]).first().reset_index()
    U["abstained"] = 0
    r, b = ers_analyse(U, "CWRU unit-level (sensitivity)")
    rows += [x for x in r if x["outcome"] == "Prediction correctness"]
    boots += [x for x in b if x["outcome"] == "Prediction correctness"]
    C = pd.DataFrame(rows)
    C.to_csv(rv(folder, "ERS_Confidence_Comparison.csv"), index=False, float_format="%.6f")
    pd.DataFrame(boots).to_csv(rv(folder, "ERS_Cluster_Bootstrap.csv"), index=False, float_format="%.6f")
    sd = A.groupby(["dataset"])[["S_conf", "S_ERS", "S_XAI", "xai_stability", "xai_agreement"]].agg(
        ["mean", "std", "min", "max"]
    )
    info["score_summary"] = {
        f"{d}|{x}|{y}": float(v) for d, row in sd.iterrows() for (x, y), v in row.items()
    }
    info["corr_conf_ERS"] = {d: float(np.corrcoef(g.S_conf, g.S_ERS)[0, 1]) for d, g in A.groupby("dataset")}
    json.dump(info, open(rv(folder, "results.json"), "w"), indent=2, default=float)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(json.dumps({k: v for k, v in info.items() if k != "score_summary"}, indent=1, default=float))
    print(sd.round(4).to_string())
    show = [
        "analysis",
        "outcome",
        "n_records",
        "n_original_windows",
        "n_window_block_units",
        "outcome_mean_or_prevalence",
        "n_incorrect_or_zero_units",
        "r_S_conf",
        "r_S_conf_ci_low",
        "r_S_conf_ci_high",
        "r_S_ERS",
        "r_S_ERS_ci_low",
        "r_S_ERS_ci_high",
        "r_S_XAI",
        "r_S_XAI_ci_low",
        "r_S_XAI_ci_high",
        "delta_r_ERS_minus_conf",
        "delta_r_ci_low",
        "delta_r_ci_high",
        "delta_r_boot_share_gt0",
        "boot_invalid_share_r",
        "boot_invalid_share_r_XAI",
        "auc_S_conf",
        "auc_S_ERS",
        "auc_S_XAI",
        "delta_auc_ERS_minus_conf",
        "delta_auc_ci_low",
        "delta_auc_ci_high",
        "boot_invalid_share_auc",
    ]
    print(C[[c for c in show if c in C]].round(4).T.to_string())


def cmd_ers_baseline(a):
    ers_report(ers_load, "ers_confidence_baseline")


def ers_final_load():
    O = ers_load()
    F = pd.read_csv(RV / "llm_final_consolidated" / "final_results.csv")
    case_cols = [
        "prediction_correct",
        "data_quality",
        "ers",
        "xai_stability",
        "xai_agreement",
        "block",
        "dataset",
    ]
    O["run_block"] = O.run_dir
    case = O.groupby(["run_block", "case_id"])[case_cols].first()
    assert case.groupby(level=[0, 1]).size().max() == 1
    F = F.join(case, on=["run_block", "case_id"], rsuffix="_o")
    F["dataset"] = F.dataset_o
    F = F.drop(columns="dataset_o")
    F["abstained"] = (F.final_status == "abstained").astype(int)
    F["error"] = ""
    F["run_dir"] = F.run_block
    F["window"] = np.where(
        F.dataset == "TEP", F.case_id.str.replace("_(none|noise)$", "", regex=True), F.case_id
    )
    c, s, ag = (F.calibrated_confidence, F.xai_stability, F.xai_agreement)
    F["S_conf"] = c
    F["S_ERS"] = (W["calibrated_confidence"] * c + W["xai_stability"] * s + W["xai_agreement"] * ag) / (
        W["calibrated_confidence"] + W["xai_stability"] + W["xai_agreement"]
    )
    F["S_XAI"] = (W["xai_stability"] * s + W["xai_agreement"] * ag) / (
        W["xai_stability"] + W["xai_agreement"]
    )
    F["ERS_reference_compute_ers"] = F.S_ERS
    F["ERS_stored_runtime_csv"] = F.ers
    assert len(F) == 1296 and F.prediction_correct.notna().all()
    return F


def ers_s6(A, label):
    comp = {
        "c": ("calibrated_confidence", W["calibrated_confidence"]),
        "s": ("xai_stability", W["xai_stability"]),
        "a": ("xai_agreement", W["xai_agreement"]),
    }
    variants = {"Full ERS": "csa", "− Confidence": "sa", "− XAI Stability": "ca", "− XAI Agreement": "cs"}
    rows = []
    for v, keys in variants.items():
        r = {"version": label, "variant": v}
        score = sum((comp[k][1] * A[comp[k][0]] for k in keys)) / sum((comp[k][1] for k in keys))
        for ds in ("TEP", "CWRU"):
            m = A.dataset == ds
            ans = m & (A.abstained == 0)
            r[f"{ds} correctness"] = np.corrcoef(score[m], A.prediction_correct[m])[0, 1]
            r[f"{ds} adherence"] = np.corrcoef(score[ans], A.top1_faithfulness[ans])[0, 1]
        rows.append(r)
    return rows


def cmd_ers_final(a):
    O = ers_load()
    S = ers_s6(O, "published (2026-09) — reproduced")
    print(pd.DataFrame(S).round(4).to_string())
    F = ers_final_load()
    S += ers_s6(F, "final (2026-10)")
    pd.DataFrame(S).to_csv(
        rv("ers_final", "TableS6_published_vs_final.csv"), index=False, float_format="%.4f"
    )
    print(pd.DataFrame(S).round(4).to_string())
    ers_report(ers_final_load, "ers_final")


def phys_set(ref, ds, fault):
    if ds == "TEP":
        return set(ref["TEP"].get(str(int(fault)), {}).get("sensors", []))
    return set(ref["CWRU"]["fault_classes"]["sensors"]) if int(fault) != 0 else set()


def physical_load(ref):
    cols = [
        "dataset",
        "rep",
        "case_id",
        "model_name",
        "condition",
        "cited_sensors",
        "evidence_sensors",
        "top1_faithfulness",
        "finish_reason",
    ]
    F = pd.read_csv(RV / "llm_final_consolidated" / "final_results.csv")
    F = F[(F.final_status == "response") & F.run_block.str.contains("clean")]
    parts = [F.reindex(columns=cols + ["supplied_order"])]
    ab = successes("rank_pct_ablation_final/_partial_results_*.csv")
    fx = successes("qwen_fix/_partial_results_*.csv")
    fx = fx[fx.condition.isin(["rank_only", "percentages_only"])]
    ab = pd.concat([ab[~ab.retry_id.isin(fx.retry_id)], fx])
    ab = ab[ab.finish_reason != "length"]
    parts.append(ab.reindex(columns=cols + ["supplied_order"]))
    if rv_glob("xai_controls/_partial_results_*.csv"):
        ct = successes("xai_controls/_partial_results_*.csv")
        ct = ct[ct.finish_reason != "length"]
        parts.append(ct.reindex(columns=cols + ["supplied_order"]))
    A = pd.concat(parts, ignore_index=True)
    truth = {}
    for f in ("tep_clean.json", "cwru_clean.json"):
        for c in load_cases(str(CASES_DIR / f)):
            truth[c.case_id] = (c.true_fault, c.predicted_fault, c.evidence_sensors())
    A["true_fault"] = A.case_id.map(lambda c: truth[c][0])
    A["pred_fault"] = A.case_id.map(lambda c: truth[c][1])
    A["ig"] = A.case_id.map(lambda c: truth[c][2])
    A["phys"] = [phys_set(ref, d, f) for d, f in zip(A.dataset, A.true_fault)]
    A = A[(A.true_fault == A.pred_fault) & (A.phys.map(len) > 0)].copy()
    cited = A.cited_sensors.fillna("").map(lambda s: [x for x in s.split(";") if x])
    A["Phys@1"] = [float(len(c) > 0 and c[0] in p) for c, p in zip(cited, A.phys)]
    A["PhysHit3"] = [float(any((x in p for x in c[:3]))) for c, p in zip(cited, A.phys)]
    A["PhysPrec"] = [np.mean([x in p for x in c]) if c else 0.0 for c, p in zip(cited, A.phys)]
    A["IG@1"] = [float(i[0] in p) for i, p in zip(A.ig, A.phys)]
    A["IGHit3"] = [float(any((x in p for x in i[:3]))) for i, p in zip(A.ig, A.phys)]
    A["window"] = A.case_id.str.replace("_(none|noise)$", "", regex=True)
    return A


def cmd_physical(a):
    folder = "physical_validity"
    ref = json.load(open(ROOT / "resources" / "physical_reference.json", encoding="utf-8"))
    A = physical_load(ref)
    M = ["Phys@1", "PhysHit3", "PhysPrec"]
    S = (
        A.groupby(["dataset", "condition"])
        .agg(
            n=("Phys@1", "size"),
            cases=("window", "nunique"),
            **{m: (m, "mean") for m in M},
            EA1_vs_IG=("top1_faithfulness", "mean"),
        )
        .reset_index()
    )
    S["condition"] = pd.Categorical(S.condition, PHYS_ORDER)
    S = S.sort_values(["dataset", "condition"])
    S.to_csv(rv(folder, "Physical_Summary.csv"), index=False, float_format="%.4f")
    IG = A.drop_duplicates("case_id").groupby("dataset")[["IG@1", "IGHit3"]].agg(["mean", "size"])
    IG.to_csv(rv(folder, "IG_Physical.csv"), float_format="%.4f")
    D = []
    for ds, g in A.groupby("dataset"):
        for c in [x for x in PHYS_ORDER if x != "plain" and x in set(g.condition)]:
            for m in ("Phys@1", "PhysHit3"):
                lo, hi = boot_contrast(g, c, "plain", m)
                D.append(
                    {
                        "dataset": ds,
                        "contrast": f"{c} - plain",
                        "metric": m,
                        "diff": g[g.condition == c][m].mean() - g[g.condition == "plain"][m].mean(),
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
    D = pd.DataFrame(D)
    D.to_csv(rv(folder, "Physical_Differences.csv"), index=False, float_format="%.4f")
    ST = (
        A.assign(IG_top1_physical=A["IG@1"].map({1.0: "IG top-1 physical", 0.0: "IG top-1 not physical"}))
        .groupby(["dataset", "IG_top1_physical", "condition"])
        .agg(n=("Phys@1", "size"), cases=("window", "nunique"), Phys1=("Phys@1", "mean"))
        .reset_index()
    )
    ST.to_csv(rv(folder, "Physical_By_IG_Stratum.csv"), index=False, float_format="%.4f")
    print(ST.round(3).to_string())
    PM = A.groupby(["dataset", "model_name", "condition"])["Phys@1"].mean().unstack("condition")
    PM.to_csv(rv(folder, "Physical_By_Model.csv"), float_format="%.4f")
    pd.set_option("display.width", 250)
    print(S.round(3).to_string())
    print(IG.round(3))
    print(D.round(3).to_string())
    print(A.drop_duplicates("case_id")[["dataset", "case_id", "ig", "phys"]].to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("registry").set_defaults(fn=cmd_registry)
    sub.add_parser("manifest").set_defaults(fn=cmd_manifest)
    p = sub.add_parser("rerun-manifest")
    p.add_argument("--out-dir", default=str(RV / RETRY_DIR))
    p.set_defaults(fn=cmd_rerun_manifest)
    p = sub.add_parser("newmodel-manifest")
    p.add_argument("--models", nargs="+", default=["gpt5_mini", "qwen35_9b"])
    p.add_argument("--out-dir", default=str(RV / RETRY_DIR))
    p.set_defaults(fn=cmd_newmodel_manifest)
    p = sub.add_parser("run")
    p.add_argument("--dataset", choices=["TEP", "CWRU"])
    p.add_argument("--models", nargs="+", default=["gpt5_mini", "qwen35_9b"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--qwen-workers", type=int, default=2)
    p.add_argument("--gpt-workers", type=int, default=1)
    p.add_argument("--manifest", default=str(MANIFEST))
    p.add_argument("--out-dir", default=str(RV / RETRY_DIR))
    p.add_argument("--blocks", nargs="+", default=None)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("ablation-run")
    p.add_argument("--dataset", required=True, choices=["tep", "cwru"])
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--conditions", nargs="+", default=list(ABL_NEW), choices=list(ABL_ALL))
    p.add_argument("--reps", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--print-example", action="store_true")
    p.set_defaults(fn=cmd_ablation_run)
    for name, fn in (
        ("final", cmd_final),
        ("run-details", cmd_run_details),
        ("ablation", cmd_ablation),
        ("controls", cmd_controls),
        ("ers-baseline", cmd_ers_baseline),
        ("ers-final", cmd_ers_final),
        ("physical", cmd_physical),
    ):
        sub.add_parser(name).set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
