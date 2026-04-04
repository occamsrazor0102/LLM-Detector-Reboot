# tests/fixtures/llm_samples.py
"""Confirmed LLM-generated texts for regression testing (A0–A2 tiers)."""

# A0: Raw copy-paste, no editing
A0_PREAMBLE = (
    "Here's a comprehensive pharmacokinetics evaluation task for Drug X that you "
    "can use for your assessment:\n\n"
    "**Task Overview**\n\n"
    "Your task is to design and evaluate a single-dose pharmacokinetics study. "
    "You will need to ensure that your protocol adheres to ICH E6(R2) guidelines "
    "and incorporates robust statistical methodology.\n\n"
    "**Key Deliverables**\n\n"
    "1. Study Protocol: Provide a detailed protocol including eligibility criteria, "
    "dosing schedule, and sample collection timepoints.\n"
    "2. Statistical Analysis Plan: Outline the non-compartmental analysis approach "
    "and specify primary and secondary endpoints.\n"
    "3. Quality Assurance Framework: Detail the measures you will implement to "
    "ensure data integrity throughout the study."
)

# A0: Clinical task with heavy prompt-engineering structure
A0_CLINICAL_TASK = (
    "**Task: Pharmacokinetics Evaluation — Drug X**\n\n"
    "**Context:** You are a senior clinical pharmacologist reviewing a Phase I "
    "submission for Drug X, a novel oral tablet formulation.\n\n"
    "**Your Role:** Evaluate the adequacy of the proposed PK study design and "
    "identify any gaps in the protocol.\n\n"
    "**Constraints:**\n"
    "- The evaluation must reference ICH E6(R2) and EMA bioanalytical guidelines\n"
    "- Ensure statistical power is justified (minimum 80% at α=0.05)\n"
    "- Flag any deviations from standard NCA methodology\n"
    "- Confirm that the sampling schedule adequately captures Tmax and t1/2\n\n"
    "**Output Format:** Structured report with sections for Protocol Adequacy, "
    "Statistical Review, Regulatory Compliance, and Recommendations.\n\n"
    "**Evaluation Criteria:** Your response will be assessed on scientific rigor, "
    "regulatory awareness, and clarity of recommendations."
)

# A1: Light cleanup — preamble removed, personal sentence prepended
A1_CLEANED = (
    "I've been working with PK studies for a while, so here's what I'd design.\n\n"
    "The study should enroll 12 healthy volunteers in a crossover design. "
    "Key deliverables include a comprehensive protocol with eligibility criteria, "
    "a robust statistical analysis plan utilizing non-compartmental analysis, "
    "and a quality assurance framework to ensure data integrity. "
    "It is important to note that the sampling schedule must adequately capture "
    "both Tmax and t1/2. The protocol should also ensure compliance with ICH E6(R2) "
    "and EMA bioanalytical guidelines to facilitate regulatory approval."
)

# A2: Prompt-coached — "write casually with typos, avoid numbered lists"
A2_COACHED = (
    "ok so for drug x pk study — you'd probably want like 12 healthy volunteers, "
    "crossover design with a washout period of at least 7 days. the sampling "
    "schedule needs to capture tmax and t1/2 properly so you'd want points at "
    "0.5, 1, 2, 4, 6, 8, 12, 24 hours post-dose. analytical method would be "
    "lc-ms/ms validated per the ema guidelines. for statistical analysis you'd "
    "use non-compartmental analysis in phoenix — that's the standard. it's worth "
    "mentioning that the protocol needs to align with ich e6 r2 for regulatory "
    "purposes and you'll want to ensure your power calculation justifies the "
    "sample size."
)
