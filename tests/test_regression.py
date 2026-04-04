import pytest
from pathlib import Path
from beet.pipeline import BeetPipeline

@pytest.fixture(scope="module")
def pipeline():
    config = Path(__file__).parent.parent / "configs" / "screening.yaml"
    return BeetPipeline.from_config_file(config)

CASES = [
    ("a0_preamble_critical", ["smoke", "tier_a0"],
     "Here's a comprehensive pharmacokinetics evaluation task for Drug X that you can use for your assessment:\n\n"
     "**Task Overview**\nYour task is to design and evaluate a single-dose PK study. You will need to ensure "
     "that your protocol adheres to ICH E6(R2) guidelines and incorporates robust statistical methodology.",
     "Clinical Pharmacologist", "RED", {"p_llm": [("ge", 0.75), ("le", 0.85)]}),

    ("a0_clinical_task_structured", ["smoke", "tier_a0"],
     "**Task: Pharmacokinetics Evaluation — Drug X**\n\n"
     "**Context:** You are a senior clinical pharmacologist reviewing a Phase I submission.\n\n"
     "**Your Role:** Evaluate the adequacy of the proposed PK study design.\n\n"
     "**Constraints:**\n- Must reference ICH E6(R2) and EMA guidelines\n"
     "- Ensure statistical power is justified\n- Flag deviations from NCA methodology\n\n"
     "**Output Format:** Structured report with Protocol Adequacy, Statistical Review, "
     "Regulatory Compliance sections.\n\n"
     "**Evaluation Criteria:** Scientific rigor and regulatory awareness.",
     "Clinical Pharmacologist", "UNCERTAIN", {"p_llm": [("ge", 0.40), ("le", 0.60)]}),

    ("human_casual", ["smoke", "human"],
     "okay so i've been working on this for like three weeks now and honestly "
     "it's been a mess. the main issue is that the reagent concentrations keep "
     "drifting — i think it's a temperature problem in the storage room. gonna "
     "try moving everything to the cold room next week and see if that helps.",
     "Research Scientist", "GREEN", {"p_llm": ("le", 0.35)}),

    ("legitimate_sop", ["false_positive", "formal"],
     "Reagent Preparation Protocol — Version 2.1\n\n"
     "1. Verify lot numbers against the approved vendor list before use.\n"
     "2. Prepare 10 mM stock solutions in phosphate-buffered saline (PBS, pH 7.4).\n"
     "3. Aliquot into 200 µL volumes; store at -80°C until use.\n"
     "4. Thaw on ice immediately prior to use; do not refreeze.\n\n"
     "Quality Controls: Run duplicate measurements for each sample batch. "
     "Acceptable CV ≤ 12%. Flag and repeat any batch exceeding this threshold.",
     "Research Scientist", "GREEN", {"determination": ("not_eq", "RED")}),

    ("nurse_educator_legitimate", ["false_positive", "instruction"],
     "Create a 30-minute training module on hand hygiene for new nursing staff. "
     "The session should cover the WHO five moments for hand hygiene, proper technique "
     "using both soap and alcohol-based rub, and common compliance barriers. "
     "Include two interactive scenarios where participants identify missed moments. "
     "Target audience has zero prior formal training.",
     "Nurse Educator", "AMBER", {"determination": ("in", ["GREEN", "YELLOW", "AMBER"])}),

    ("a1_cleaned_llm", ["tier_a1"],
     "I've been working with PK studies for a while, so here's what I'd design.\n\n"
     "The study should enroll 12 healthy volunteers in a crossover design. "
     "Key deliverables include a comprehensive protocol with eligibility criteria, "
     "a robust statistical analysis plan utilizing non-compartmental analysis, "
     "and a quality assurance framework to ensure data integrity. "
     "It is important to note that the sampling schedule must adequately capture "
     "both Tmax and t1/2.",
     "Clinical Pharmacologist", "UNCERTAIN", {"p_llm": [("ge", 0.35), ("le", 0.50)]}),

    ("too_short", ["edge_case"], "hello", "Unknown", "GREEN", {}),
]

@pytest.mark.parametrize("name,tags,text,occupation,expected,assertions", CASES)
def test_regression_case(pipeline, name, tags, text, occupation, expected, assertions):
    det = pipeline.analyze(text)
    if "determination" not in assertions:
        assert det.label == expected, f"[{name}] Expected {expected}, got {det.label} (P(LLM)={det.p_llm:.2f})"
    for field, condition in assertions.items():
        if field == "determination": val = det.label
        elif field == "p_llm": val = det.p_llm
        elif field == "override_applied": val = det.override_applied
        else: continue

        # Handle both single condition and list of conditions
        conditions = condition if isinstance(condition, list) else [condition]
        for cond in conditions:
            op = cond[0]
            if op == "ge": assert val >= cond[1], f"[{name}] {field}={val} < {cond[1]}"
            elif op == "le": assert val <= cond[1], f"[{name}] {field}={val} > {cond[1]}"
            elif op == "eq": assert val == cond[1]
            elif op == "not_eq": assert val != cond[1]
            elif op == "in": assert val in cond[1], f"[{name}] {field}={val} not in {cond[1]}"
            elif op == "true": assert val is True
            elif op == "false": assert val is False
