from __future__ import annotations
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

CONDITION_PLAIN = "plain"
CONDITION_XAI = "xai_grounded"
CONDITION_XAI_UNC = "xai_uncertainty_grounded"
CONDITIONS = (CONDITION_PLAIN, CONDITION_XAI, CONDITION_XAI_UNC)
CONDITION_LABELS = {
    CONDITION_PLAIN: "Plain LLM (sensor summary + prediction only)",
    CONDITION_XAI: "XAI-Grounded LLM (+ attribution evidence)",
    CONDITION_XAI_UNC: "XAI + Uncertainty-Grounded LLM (+ calibrated confidence)",
}
TEP_SENSOR_NAMES = tuple([f"xmeas_{i}" for i in range(1, 42)] + [f"xmv_{i}" for i in range(1, 12)])
TEP_SENSOR_DESCRIPTIONS = {
    "xmeas_1": "A feed (stream 1) flow",
    "xmeas_2": "D feed (stream 2) flow",
    "xmeas_3": "E feed (stream 3) flow",
    "xmeas_4": "A and C feed (stream 4) flow",
    "xmeas_5": "Recycle flow to separator",
    "xmeas_6": "Reactor feed rate (stream 6)",
    "xmeas_7": "Reactor pressure",
    "xmeas_8": "Reactor level",
    "xmeas_9": "Reactor temperature",
    "xmeas_10": "Purge rate (stream 9)",
    "xmeas_11": "Product separator temperature",
    "xmeas_12": "Product separator level",
    "xmeas_13": "Product separator pressure",
    "xmeas_14": "Product separator underflow",
    "xmeas_15": "Stripper level",
    "xmeas_16": "Stripper pressure",
    "xmeas_17": "Stripper underflow",
    "xmeas_18": "Stripper temperature",
    "xmeas_19": "Stripper steam flow",
    "xmeas_20": "Compressor work",
    "xmeas_21": "Reactor cooling water outlet temperature",
    "xmeas_22": "Separator cooling water outlet temperature",
    "xmv_1": "D feed flow valve",
    "xmv_2": "E feed flow valve",
    "xmv_3": "A feed flow valve",
    "xmv_4": "A and C feed flow valve",
    "xmv_5": "Compressor recycle valve",
    "xmv_6": "Purge valve",
    "xmv_7": "Separator pot liquid flow valve",
    "xmv_8": "Stripper liquid product flow valve",
    "xmv_9": "Stripper steam valve",
    "xmv_10": "Reactor cooling water flow",
    "xmv_11": "Condenser cooling water flow",
}
TEP_FAULT_DESCRIPTIONS = {
    1: "A/C feed ratio, B composition constant (step)",
    2: "B composition, A/C ratio constant (step)",
    3: "D feed temperature (step)",
    4: "Reactor cooling water inlet temperature (step)",
    5: "Condenser cooling water inlet temperature (step)",
    6: "A feed loss (step)",
    7: "C header pressure loss - reduced availability (step)",
    8: "A, B, C feed composition (random variation)",
    9: "D feed temperature (random variation)",
    10: "C feed temperature (random variation)",
    11: "Reactor cooling water inlet temperature (random variation)",
    12: "Condenser cooling water inlet temperature (random variation)",
    13: "Reaction kinetics (slow drift)",
    14: "Reactor cooling water valve (sticking)",
    15: "Condenser cooling water valve (sticking)",
    16: "Unknown",
    17: "Unknown",
    18: "Unknown",
    19: "Unknown",
    20: "Unknown",
}
_SENSOR_MENTION_RE = re.compile("\\b(xmeas|xmv)[\\s_\\-]*(\\d{1,2})\\b", re.IGNORECASE)


def _tep_describe_sensor(name):
    return TEP_SENSOR_DESCRIPTIONS.get(name, name)


def _tep_describe_fault(fault_number):
    return TEP_FAULT_DESCRIPTIONS.get(int(fault_number), "Unknown fault")


def _tep_extract_sensor_mentions(text):
    seen = []
    for prefix, number in _SENSOR_MENTION_RE.findall(text or ""):
        canonical = f"{prefix.lower()}_{int(number)}"
        if canonical in TEP_SENSOR_NAMES and canonical not in seen:
            seen.append(canonical)
    return seen


def _tep_extract_fault_claims(text):
    claims = []
    for match in re.findall("\\bfaults?[\\s_\\-#]*(\\d{1,2})\\b", text or "", re.IGNORECASE):
        value = int(match)
        if value not in claims:
            claims.append(value)
    return claims


_ACTIVE_DOMAIN = {
    "name": "TEP",
    "sensor_names": TEP_SENSOR_NAMES,
    "sensor_descriptions": TEP_SENSOR_DESCRIPTIONS,
    "fault_descriptions": TEP_FAULT_DESCRIPTIONS,
    "sensor_mention_extractor": _tep_extract_sensor_mentions,
    "fault_claim_extractor": _tep_extract_fault_claims,
}


def register_domain(
    name,
    sensor_names,
    sensor_descriptions,
    fault_descriptions,
    sensor_mention_extractor=None,
    fault_claim_extractor=None,
):
    if not sensor_names:
        raise ValueError(
            f"Domain '{name}' has an empty sensor vocabulary; every sensor citation would then be scored as a hallucination."
        )
    _ACTIVE_DOMAIN.update(
        name=name,
        sensor_names=tuple(sensor_names),
        sensor_descriptions=dict(sensor_descriptions),
        fault_descriptions=dict(fault_descriptions),
        sensor_mention_extractor=sensor_mention_extractor or _tep_extract_sensor_mentions,
        fault_claim_extractor=fault_claim_extractor or _tep_extract_fault_claims,
    )


def active_domain_name():
    return str(_ACTIVE_DOMAIN["name"])


def active_sensor_names():
    return tuple(_ACTIVE_DOMAIN["sensor_names"])


def describe_sensor(name):
    descriptions = _ACTIVE_DOMAIN["sensor_descriptions"]
    return descriptions.get(name, name)


def describe_fault(fault_number):
    descriptions = _ACTIVE_DOMAIN["fault_descriptions"]
    return descriptions.get(int(fault_number), "Unknown fault")


def extract_sensor_mentions(text):
    return _ACTIVE_DOMAIN["sensor_mention_extractor"](text)


def extract_fault_claims(text):
    return _ACTIVE_DOMAIN["fault_claim_extractor"](text)


@dataclass
class SensorEvidence:
    sensor: str
    attribution: float
    mean: float
    std: float
    delta: float
    trend: str

    def to_dict(self):
        return asdict(self)


@dataclass
class DiagnosisPayload:
    case_id: str
    scenario: str
    true_fault: int
    predicted_fault: int
    calibrated_confidence: float
    raw_confidence: float
    temperature: float
    top_evidence: List[SensorEvidence] = field(default_factory=list)
    degradation: str = "none"
    degradation_level: float = 0.0
    xai_stability: Optional[float] = None
    xai_agreement: Optional[float] = None
    data_quality: float = 1.0
    window_start: int = 0

    @property
    def is_correct(self):
        return int(self.true_fault) == int(self.predicted_fault)

    def evidence_sensors(self, top_k=None):
        names = [e.sensor for e in self.top_evidence]
        return names[:top_k] if top_k else names

    def to_dict(self):
        payload = asdict(self)
        payload["is_correct"] = self.is_correct
        return payload


@dataclass
class LLMResponse:
    model: str
    condition: str
    case_id: str
    text: str
    abstained: bool = False
    abstain_reason: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""
    finish_reason: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class ExplanationScores:
    faithfulness: float = 0.0
    top1_faithfulness: float = 0.0
    weighted_faithfulness: float = 0.0
    rank_correlation: float = 0.0
    hallucination_rate: float = 0.0
    fault_consistency: float = 0.0
    unsupported_sensors: List[str] = field(default_factory=list)
    cited_sensors: List[str] = field(default_factory=list)
    ers: float = 0.0
    ers_components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def summarise_window(values, sensor_names, attributions=None, top_k=5):
    import numpy as np

    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D window, got shape {array.shape}")
    if array.shape[1] != len(sensor_names):
        raise ValueError(f"Window has {array.shape[1]} columns but {len(sensor_names)} sensor names")
    records = []
    for idx, name in enumerate(sensor_names):
        column = array[:, idx]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            mean = std = delta = 0.0
        else:
            mean = float(np.mean(finite))
            std = float(np.std(finite))
            delta = float(finite[-1] - finite[0])
        scale = std if std > 1e-09 else 1.0
        if delta > 0.5 * scale:
            trend = "rising"
        elif delta < -0.5 * scale:
            trend = "falling"
        else:
            trend = "stable"
        records.append(
            SensorEvidence(
                sensor=name,
                attribution=float(attributions.get(name, 0.0)) if attributions else 0.0,
                mean=mean,
                std=std,
                delta=delta,
                trend=trend,
            )
        )
    if attributions:
        records.sort(key=lambda r: r.attribution, reverse=True)
    else:
        records.sort(key=lambda r: abs(r.delta), reverse=True)
    return records[:top_k]


SYSTEM_PROMPT = "You are an industrial process-safety assistant supporting control-room operators of a Tennessee Eastman chemical plant.\n\nRules you must follow without exception:\n1. Base every statement only on the evidence given in the prompt. Never invent sensor readings, fault codes, or plant events.\n2. Refer to sensors by their exact identifiers (for example xmeas_7 or xmv_10). Do not cite a sensor that does not appear in the evidence.\n3. Be concise and operational. The operator is reading you under time pressure.\n4. If the evidence is insufficient to justify a root cause, say so explicitly instead of guessing.\n"
DEFAULT_ABSTAIN_THRESHOLD = 0.85
_RESPONSE_FORMAT = "Answer in exactly these four sections, each on its own line:\nSTATUS: <the fault you are explaining>\nROOT CAUSE: <which sensors drive this diagnosis and why>\nACTION: <what the operator should do next>\nCONFIDENCE NOTE: <how much the operator should trust this explanation>\n"


def _render_sensor_summary(payload):
    lines = []
    for ev in payload.top_evidence:
        lines.append(
            f"  - {ev.sensor} ({describe_sensor(ev.sensor)}): mean {ev.mean:.3f}, sd {ev.std:.3f}, change across window {ev.delta:+.3f} ({ev.trend})"
        )
    return "\n".join(lines) if lines else "  (no sensor summary available)"


def _render_attribution_block(payload):
    lines = []
    for rank, ev in enumerate(payload.top_evidence, 1):
        lines.append(
            f"  {rank}. {ev.sensor} - {ev.attribution * 100:.1f}% of the model's attribution ({describe_sensor(ev.sensor)})"
        )
    return "\n".join(lines) if lines else "  (no attribution evidence available)"


_DOMAIN_INPUT_DESCRIPTIONS = {
    "TEP": "a 60-step window of 52 process variables",
    "CWRU": "a 1024-sample vibration window from 3 accelerometer channels",
}


def _render_header(payload):
    domain = active_domain_name()
    input_description = _DOMAIN_INPUT_DESCRIPTIONS.get(
        domain, f"a sensor window of {len(active_sensor_names())} channels"
    )
    return f"CASE: {payload.case_id}\nA 1D-CNN fault classifier analysed {input_description}.\nCLASSIFIER PREDICTION: Fault {payload.predicted_fault} ({describe_fault(payload.predicted_fault)})"


def build_plain_prompt(payload):
    return f"{_render_header(payload)}\n\nPROCESS VARIABLE SUMMARY (most-changed variables in the window):\n{_render_sensor_summary(payload)}\n\nExplain to the operator why this fault was reported and which process variables are responsible.\n\n{_RESPONSE_FORMAT}"


def build_xai_prompt(payload):
    return f"{_render_header(payload)}\n\nPROCESS VARIABLE SUMMARY (most-changed variables in the window):\n{_render_sensor_summary(payload)}\n\nEXPLAINABILITY EVIDENCE (Integrated Gradients attribution, ranked by contribution to this prediction):\n{_render_attribution_block(payload)}\n\nExplain to the operator why this fault was reported. Ground your root-cause statement in the attribution evidence above and cite the specific sensors it identifies.\n\n{_RESPONSE_FORMAT}"


def build_xai_uncertainty_prompt(payload, abstain_threshold=DEFAULT_ABSTAIN_THRESHOLD):
    confidence = payload.calibrated_confidence
    band = "ABOVE" if confidence >= abstain_threshold else "BELOW"
    reliability_lines = [
        f"  - Calibrated confidence: {confidence * 100:.2f}% ({band} the {abstain_threshold * 100:.0f}% action threshold)",
        f"  - Calibration temperature applied: T = {payload.temperature:.4f}",
    ]
    if payload.xai_stability is not None:
        reliability_lines.append(
            f"  - Attribution stability under sensor perturbation: {payload.xai_stability:.3f} (1.0 = perfectly stable)"
        )
    if payload.xai_agreement is not None:
        reliability_lines.append(
            f"  - Agreement between explainability methods: {payload.xai_agreement:.3f} (1.0 = identical rankings)"
        )
    if payload.degradation != "none":
        reliability_lines.append(
            f"  - Sensor data degradation present: {payload.degradation} at level {payload.degradation_level:g}"
        )
    reliability_lines.append(f"  - Usable sensor data: {payload.data_quality * 100:.1f}% of readings valid")
    guidance = (
        "Because calibrated confidence is below the action threshold, you must tell the operator the diagnosis is uncertain, must not recommend an autonomous corrective action, and must request manual verification of the implicated sensors."
        if confidence < abstain_threshold
        else "Calibrated confidence is above the action threshold, so a standard diagnostic procedure may be recommended - but still tell the operator which sensor to verify first."
    )
    return (
        f"{_render_header(payload)}\n\nPROCESS VARIABLE SUMMARY (most-changed variables in the window):\n{_render_sensor_summary(payload)}\n\nEXPLAINABILITY EVIDENCE (Integrated Gradients attribution, ranked by contribution to this prediction):\n{_render_attribution_block(payload)}\n\nRELIABILITY EVIDENCE:\n"
        + "\n".join(reliability_lines)
        + f"\n\n{guidance}\n\nExplain to the operator why this fault was reported, grounding your root cause in the attribution evidence and your trust statement in the reliability evidence.\n\n{_RESPONSE_FORMAT}"
    )


def build_prompt(payload, condition, abstain_threshold=DEFAULT_ABSTAIN_THRESHOLD):
    if condition == CONDITION_PLAIN:
        return build_plain_prompt(payload)
    if condition == CONDITION_XAI:
        return build_xai_prompt(payload)
    if condition == CONDITION_XAI_UNC:
        return build_xai_uncertainty_prompt(payload, abstain_threshold)
    raise ValueError(f"Unknown condition '{condition}'. Expected one of {CONDITIONS}.")


def assert_condition_isolation(payload):
    plain = build_plain_prompt(payload)
    xai = build_xai_prompt(payload)
    unc = build_xai_uncertainty_prompt(payload)
    assert "attribution" not in plain.lower(), "Plain prompt leaks attribution evidence"
    assert "Integrated Gradients" not in plain, "Plain prompt leaks the XAI method"
    assert "confidence" not in plain.lower().replace(
        "confidence note", ""
    ), "Plain prompt leaks confidence information"
    assert "attribution" in xai.lower(), "XAI prompt is missing attribution evidence"
    assert (
        "Calibrated confidence" not in xai
    ), "XAI prompt leaks calibrated confidence into the XAI-only condition"
    assert "attribution" in unc.lower(), "Uncertainty prompt is missing attribution"
    assert "Calibrated confidence" in unc, "Uncertainty prompt is missing calibrated confidence"


ERS_WEIGHTS = {
    "calibrated_confidence": 0.4,
    "xai_stability": 0.25,
    "xai_agreement": 0.2,
    "data_quality": 0.15,
}


def kendall_tau_b(rank_a, rank_b):
    n = len(rank_a)
    if n < 2 or len(rank_b) != n:
        return 0.0
    concordant = discordant = ties_a = ties_b = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            da = rank_a[i] - rank_a[j]
            db = rank_b[i] - rank_b[j]
            product = da * db
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            else:
                if da == 0:
                    ties_a += 1
                if db == 0:
                    ties_b += 1
    denom_a = concordant + discordant + ties_a
    denom_b = concordant + discordant + ties_b
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return (concordant - discordant) / (denom_a * denom_b) ** 0.5


def compute_faithfulness(cited, evidence, top_k=3):
    target = [s for s in evidence[:top_k]]
    if not target:
        return 0.0
    cited_set = set(cited)
    hits = sum((1 for sensor in target if sensor in cited_set))
    return hits / len(target)


def compute_top1_faithfulness(cited, evidence):
    if not cited or not evidence:
        return 0.0
    return 1.0 if cited[0] == evidence[0] else 0.0


def compute_weighted_faithfulness(cited, evidence, top_k=3):
    target = list(evidence[:top_k])
    if not target:
        return 0.0
    cited_set = set(cited)
    weights = [1.0 / (i + 1) for i in range(len(target))]
    earned = sum((w for s, w in zip(target, weights) if s in cited_set))
    return earned / sum(weights)


def compute_rank_correlation(cited, evidence, top_k=5):
    evidence_rank = {s: i for i, s in enumerate(evidence[:top_k])}
    shared = [s for s in cited if s in evidence_rank]
    if len(shared) < 2:
        return 0.0
    llm_order = list(range(len(shared)))
    xai_order = [evidence_rank[s] for s in shared]
    return kendall_tau_b(llm_order, xai_order)


def compute_hallucination(cited, supported):
    if not cited:
        return (0.0, [])
    supported_set = set(supported)
    unsupported = [s for s in cited if s not in supported_set]
    return (len(unsupported) / len(cited), unsupported)


def compute_fault_consistency(text, predicted_fault):
    claims = extract_fault_claims(text)
    if not claims:
        return 0.0
    if claims == [int(predicted_fault)]:
        return 1.0
    if int(predicted_fault) in claims:
        return 0.5
    return 0.0


def compute_ers(payload, weights=None):
    weights = dict(weights or ERS_WEIGHTS)
    signals = {
        "calibrated_confidence": payload.calibrated_confidence,
        "xai_stability": payload.xai_stability,
        "xai_agreement": payload.xai_agreement,
        "data_quality": payload.data_quality,
    }
    available = {k: v for k, v in signals.items() if v is not None}
    if not available:
        return (0.0, {})
    total_weight = sum((weights.get(k, 0.0) for k in available))
    if total_weight <= 0:
        return (0.0, {})
    components = {}
    score = 0.0
    for key, value in available.items():
        normalised_weight = weights.get(key, 0.0) / total_weight
        clamped = min(max(float(value), 0.0), 1.0)
        contribution = normalised_weight * clamped
        components[key] = contribution
        score += contribution
    return (min(max(score, 0.0), 1.0), components)


def score_response(response, payload, top_k=3):
    ers, ers_components = compute_ers(payload)
    if response.abstained or response.error:
        return ExplanationScores(
            faithfulness=0.0,
            rank_correlation=0.0,
            hallucination_rate=0.0,
            fault_consistency=0.0,
            unsupported_sensors=[],
            cited_sensors=[],
            ers=ers,
            ers_components=ers_components,
        )
    cited = extract_sensor_mentions(response.text)
    evidence = payload.evidence_sensors()
    faithfulness = compute_faithfulness(cited, evidence, top_k=top_k)
    top1 = compute_top1_faithfulness(cited, evidence)
    weighted = compute_weighted_faithfulness(cited, evidence, top_k=top_k)
    rank_corr = compute_rank_correlation(cited, evidence)
    hallucination, unsupported = compute_hallucination(cited, evidence)
    consistency = compute_fault_consistency(response.text, payload.predicted_fault)
    return ExplanationScores(
        faithfulness=faithfulness,
        top1_faithfulness=top1,
        weighted_faithfulness=weighted,
        rank_correlation=rank_corr,
        hallucination_rate=hallucination,
        fault_consistency=consistency,
        unsupported_sensors=unsupported,
        cited_sensors=cited,
        ers=ers,
        ers_components=ers_components,
    )


def should_abstain(payload, ers_threshold=0.6, confidence_threshold=0.85):
    ers, _ = compute_ers(payload)
    reasons = []
    if payload.calibrated_confidence < confidence_threshold:
        reasons.append(
            f"calibrated confidence {payload.calibrated_confidence:.3f} < {confidence_threshold:.2f}"
        )
    if ers < ers_threshold:
        reasons.append(f"ERS {ers:.3f} < {ers_threshold:.2f}")
    return (bool(reasons), "; ".join(reasons))


def pearson_correlation(xs, ys):
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum(((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)))
    var_x = sum(((x - mean_x) ** 2 for x in xs))
    var_y = sum(((y - mean_y) ** 2 for y in ys))
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return cov / (var_x * var_y) ** 0.5


@dataclass
class ModelSpec:
    key: str
    display_name: str
    backend: str
    model_id: str
    notes: str = ""


MODEL_REGISTRY = {
    "gpt5_mini": ModelSpec(
        key="gpt5_mini", display_name="GPT-5 Mini", backend="openai_compatible", model_id="gpt-5-mini"
    ),
    "qwen35_9b": ModelSpec(
        key="qwen35_9b", display_name="Qwen3.5-9B", backend="open_weight", model_id="Qwen/Qwen3.5-9B"
    ),
    "deepseek_v41_flash": ModelSpec(
        key="deepseek_v41_flash",
        display_name="DeepSeek-V4.1-Flash-1017",
        backend="open_weight",
        model_id="deepseek-ai/DeepSeek-V4.1-Flash-1017",
    ),
    "glm53_flash": ModelSpec(
        key="glm53_flash",
        display_name="GLM-5.3-Flash",
        backend="open_weight",
        model_id="zai-org/GLM-5.3-Flash",
    ),
    "mock": ModelSpec(key="mock", display_name="Mock backend", backend="mock", model_id="mock"),
}


def load_api_key(env_var="LLM_API_KEY", provider="openai"):
    return os.environ.get(env_var, "").strip()


class OpenAICompatibleBackend:
    PROVIDER_BASES: Dict[str, str] = {
        "openai": "https://api.openai.com/v1",
        "open_weight": "https://api.together.xyz/v1",
    }
    PROVIDER_ENV: Dict[str, str] = {"openai": "LLM_API_KEY", "open_weight": "OPEN_LLM_API_KEY"}
    PROVIDER_BASE_ENV: Dict[str, str] = {"openai": "LLM_API_BASE", "open_weight": "OPEN_LLM_BASE"}

    def __init__(
        self,
        model_id,
        provider="openai",
        api_key_env=None,
        base_url_env="LLM_API_BASE",
        default_base=None,
        temperature=0.0,
        max_tokens=8192,
        timeout=300.0,
    ):
        self.name = model_id
        self.model_id = model_id
        self.provider = provider
        env_var = api_key_env or self.PROVIDER_ENV.get(provider, "LLM_API_KEY")
        self.api_key = load_api_key(env_var, provider=provider)
        if not self.api_key and provider == "open_weight":
            self.api_key = load_api_key("TOGETHER_API_KEY")
        base = default_base or self.PROVIDER_BASES.get(provider, "https://api.openai.com/v1")
        base = os.environ.get(self.PROVIDER_BASE_ENV.get(provider, base_url_env), base)
        self.base_url = base.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError(
                f"No credential found for model '{model_id}' at {self.base_url}. Set {env_var}."
            )

    def generate(self, system_prompt, user_prompt):
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.model_id.startswith("gpt-5") or self.model_id.startswith("o1"):
            payload["max_completion_tokens"] = self.max_tokens
        else:
            payload["max_tokens"] = self.max_tokens
            payload["temperature"] = self.temperature
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code == 429:
                raise RuntimeError(f"RATE_LIMIT {self.base_url}: {detail}") from exc
            raise RuntimeError(f"HTTP {exc.code} from {self.base_url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach {self.base_url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"Timed out after {self.timeout:.0f}s waiting for {self.model_id}") from exc
        try:
            choice = body["choices"][0]
            message = choice.get("message") or {}
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected response shape: {str(body)[:300]}") from exc
        usage = body.get("usage", {}) or {}
        text = self._extract_text(choice, message, usage)
        details = usage.get("completion_tokens_details") or {}
        return {
            "text": text,
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "reasoning_tokens": int(details.get("reasoning_tokens", 0)),
            "finish_reason": str(choice.get("finish_reason", "")),
        }

    @staticmethod
    def _extract_text(choice, message, usage):
        content = (message.get("content") or "").strip()
        if content:
            return content
        reasoning = (message.get("reasoning") or "").strip()
        if reasoning:
            return reasoning
        finish_reason = choice.get("finish_reason", "unknown")
        raise RuntimeError(
            f"Empty completion: the response carried no 'content' and no 'reasoning' text (finish_reason={finish_reason!r}, usage={usage})."
        )


class MockBackend:
    SYNTHETIC_MARKER = "[SYNTHETIC MOCK OUTPUT - NOT A REAL MODEL]"

    def __init__(self, seed=42, hallucination_probability=0.3):
        self.name = "mock"
        self.seed = seed
        self.hallucination_probability = hallucination_probability

    def generate(self, system_prompt, user_prompt):
        rng = random.Random(f"{self.seed}:{user_prompt}")
        cited = re.findall("\\b(?:xmeas|xmv)_\\d{1,2}\\b", user_prompt)
        ordered: List[str] = []
        for sensor in cited:
            if sensor not in ordered:
                ordered.append(sensor)
        has_attribution = "attribution" in user_prompt.lower()
        has_confidence = "Calibrated confidence" in user_prompt
        fault_match = re.search("CLASSIFIER PREDICTION: Fault (\\d+)", user_prompt)
        fault = fault_match.group(1) if fault_match else "0"
        top = ordered[:3] if has_attribution else ordered[:2]
        if not has_attribution and rng.random() < self.hallucination_probability:
            invented = f"xmeas_{rng.randint(23, 41)}"
            if invented not in top:
                top = top + [invented]
        sensor_text = ", ".join(top) if top else "no specific sensor"
        confidence_note = (
            "Calibrated confidence supports acting on this diagnosis."
            if has_confidence and "ABOVE" in user_prompt
            else (
                "Confidence is below the action threshold; manual verification is required before acting."
                if has_confidence
                else "No calibrated confidence was provided with this diagnosis."
            )
        )
        text = f"{self.SYNTHETIC_MARKER}\nSTATUS: Fault {fault} detected.\nROOT CAUSE: The diagnosis is driven by {sensor_text}.\nACTION: Inspect {(top[0] if top else 'the process loop')} first.\nCONFIDENCE NOTE: {confidence_note}"
        return {
            "text": text,
            "prompt_tokens": len(user_prompt.split()),
            "completion_tokens": len(text.split()),
            "reasoning_tokens": 0,
            "finish_reason": "stop",
        }


def build_backend(spec, **kwargs):
    if spec.backend == "openai_compatible":
        return OpenAICompatibleBackend(spec.model_id, provider="openai", **kwargs)
    if spec.backend == "open_weight":
        return OpenAICompatibleBackend(spec.model_id, provider="open_weight", **kwargs)
    if spec.backend == "mock":
        return MockBackend(**kwargs)
    raise ValueError(f"Unknown backend type '{spec.backend}' for model '{spec.key}'")


def generate_response(
    backend,
    spec,
    system_prompt,
    user_prompt,
    condition,
    case_id,
    abstained=False,
    abstain_reason="",
    retries=2,
):
    if abstained:
        return LLMResponse(
            model=spec.key,
            condition=condition,
            case_id=case_id,
            text=f"ABSTAINED: The system withheld an automated explanation. Reason: {abstain_reason}. Manual investigation by a qualified operator is required.",
            abstained=True,
            abstain_reason=abstain_reason,
        )
    last_error = ""
    for attempt in range(retries + 1):
        start = time.time()
        try:
            result = backend.generate(system_prompt, user_prompt)
            return LLMResponse(
                model=spec.key,
                condition=condition,
                case_id=case_id,
                text=result["text"],
                latency_s=time.time() - start,
                prompt_tokens=result.get("prompt_tokens", 0),
                completion_tokens=result.get("completion_tokens", 0),
                finish_reason=result.get("finish_reason", ""),
            )
        except RuntimeError as exc:
            last_error = str(exc)
            if attempt < retries:
                if "RATE_LIMIT" in last_error:
                    time.sleep(15.0 * 2**attempt)
                else:
                    time.sleep(2.0 * (attempt + 1))
    return LLMResponse(model=spec.key, condition=condition, case_id=case_id, text="", error=last_error)
