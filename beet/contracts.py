# beet/contracts.py
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

DETERMINATION_LABELS = Literal["RED", "AMBER", "YELLOW", "GREEN", "UNCERTAIN", "MIXED", "SKIP"]
COMPUTE_COSTS = Literal["trivial", "cheap", "moderate", "expensive"]
DOMAINS = Literal["prompt", "prose", "universal"]

@dataclass
class LayerResult:
    layer_id: str
    domain: DOMAINS
    raw_score: float
    p_llm: float                    # calibrated P(LLM), 0.0–1.0
    confidence: float               # detector's trust in its own output, 0.0–1.0
    signals: dict
    determination: DETERMINATION_LABELS
    attacker_tiers: list[str]
    compute_cost: COMPUTE_COSTS
    min_text_length: int
    spans: list[dict] = field(default_factory=list)

@dataclass
class FusionResult:
    p_llm: float
    confidence_interval: tuple[float, float]
    prediction_set: list[str]
    feature_contributions: dict
    top_contributors: list[tuple[str, float]]
    # Honest split-conformal prediction set over the binary label set.
    # One of ["LLM"], ["human"], ["LLM", "human"], or None when the
    # conformal wrapper isn't calibrated. Distinct from prediction_set,
    # which is a cosmetic severity-band mapping.
    conformal_set: list[str] | None = None
    conformal_alpha: float | None = None
    # "ebm" when the fitted model produced the result, "naive" when the
    # weighted-mean fallback did. Surfaced so the UI can caveat
    # "feature contributions" when they're really deviations-from-prior.
    fusion_mode: str = "naive"

@dataclass
class Determination:
    label: str
    p_llm: float
    confidence_interval: tuple[float, float]
    prediction_set: list[str]
    reason: str
    top_features: list[tuple[str, float]]
    override_applied: bool
    detectors_run: list[str]
    cascade_phases: list[int]
    mixed_report: dict | None
    layer_results: list[LayerResult] = field(default_factory=list)
    feature_contributions: dict = field(default_factory=dict)
    conformal_set: list[str] | None = None
    conformal_alpha: float | None = None
    fusion_mode: str = "naive"

@dataclass
class RouterDecision:
    domain: Literal["prompt", "prose", "mixed", "insufficient"]
    confidence: float
    prompt_score: float
    prose_score: float
    word_count: int
    recommended_detectors: list[str]
    skip_detectors: list[str]

@runtime_checkable
class Detector(Protocol):
    id: str
    domain: str
    compute_cost: str

    def analyze(self, text: str, config: dict) -> LayerResult: ...
    def calibrate(self, labeled_data: list) -> None: ...
