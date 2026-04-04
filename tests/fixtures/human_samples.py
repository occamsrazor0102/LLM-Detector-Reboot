# tests/fixtures/human_samples.py
"""Confirmed human-authored texts for regression testing."""

CASUAL_SHORT = (
    "okay so i've been working on this for like three weeks now and honestly "
    "it's been a mess. the main issue is that the reagent concentrations keep "
    "drifting — i think it's a temperature problem in the storage room. gonna "
    "try moving everything to the cold room next week and see if that helps."
)

FORMAL_SOP = (
    "Reagent Preparation Protocol — Version 2.1\n\n"
    "1. Verify lot numbers against the approved vendor list before use.\n"
    "2. Prepare 10 mM stock solutions in phosphate-buffered saline (PBS, pH 7.4).\n"
    "3. Aliquot into 200 µL volumes; store at -80°C until use.\n"
    "4. Thaw on ice immediately prior to use; do not refreeze.\n\n"
    "Quality Controls: Run duplicate measurements for each sample batch. "
    "Acceptable CV ≤ 12%. Flag and repeat any batch exceeding this threshold."
)

NURSE_EDUCATOR = (
    "Create a 30-minute training module on hand hygiene for new nursing staff. "
    "The session should cover the WHO five moments for hand hygiene, proper technique "
    "using both soap and alcohol-based rub, and common compliance barriers. "
    "Include two interactive scenarios where participants identify missed moments. "
    "Target audience has zero prior formal training."
)

CLINICAL_PHARMACOLOGIST_HUMAN = (
    "Design a single-dose pharmacokinetics study for Drug X (oral, 50 mg tablet) "
    "in healthy volunteers. Primary endpoints: Cmax, Tmax, AUC0-inf. Include "
    "sampling schedule (pre-dose; 0.5, 1, 2, 4, 6, 8, 12, 24, 48 h), "
    "analytical method (LC-MS/MS validated per EMA guideline), and statistical "
    "analysis plan (non-compartmental analysis using Phoenix WinNonlin). "
    "n=12 subjects, fed/fasted crossover with 7-day washout."
)
