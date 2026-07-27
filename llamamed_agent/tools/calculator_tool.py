"""A small set of well-established clinical calculators.

These are plain arithmetic implementations of published formulas, not
model inference -- they exist so the agent doesn't have to (and shouldn't
have to) do arithmetic in free text, which is a common source of error in
LLM-generated clinical reasoning. Each function documents its formula and
source so it can be checked against the reference at any time.

None of this output is a diagnosis or a treatment recommendation, and it
is not validated for real clinical use -- see the top-level disclaimer.
"""

from __future__ import annotations

from typing import Callable, Dict

from .base import Tool


def bmi(weight_kg: float, height_cm: float) -> Dict:
    """Body Mass Index = weight_kg / height_m^2."""
    height_m = height_cm / 100.0
    value = weight_kg / (height_m ** 2)
    return {"value": round(value, 1), "unit": "kg/m^2", "formula": "weight_kg / height_m^2"}


def egfr_ckd_epi_2021(creatinine_mg_dl: float, age_years: float, sex: str) -> Dict:
    """2021 CKD-EPI creatinine equation (race-free).
    Source: Inker LA et al., NEJM 2021;385:1737-1749.
    """
    sex = sex.strip().lower()
    if sex not in ("male", "female"):
        raise ValueError("sex must be 'male' or 'female'")

    kappa = 0.7 if sex == "female" else 0.9
    alpha = -0.241 if sex == "female" else -0.302
    sex_factor = 1.012 if sex == "female" else 1.0

    scr_over_k = creatinine_mg_dl / kappa
    min_term = min(scr_over_k, 1.0) ** alpha
    max_term = max(scr_over_k, 1.0) ** -1.200

    value = 142 * min_term * max_term * (0.9938 ** age_years) * sex_factor
    return {
        "value": round(value, 1),
        "unit": "mL/min/1.73m^2",
        "formula": "2021 CKD-EPI creatinine equation (race-free)",
    }


def anion_gap(sodium: float, chloride: float, bicarbonate: float) -> Dict:
    """Serum anion gap = Na - (Cl + HCO3). Normal range roughly 8-12 mEq/L."""
    value = sodium - (chloride + bicarbonate)
    return {"value": round(value, 1), "unit": "mEq/L", "formula": "Na - (Cl + HCO3)"}


def corrected_calcium(measured_calcium: float, albumin: float) -> Dict:
    """Corrected calcium for hypoalbuminemia = Ca + 0.8 * (4.0 - albumin)."""
    value = measured_calcium + 0.8 * (4.0 - albumin)
    return {"value": round(value, 2), "unit": "mg/dL", "formula": "Ca + 0.8 * (4.0 - albumin)"}


def mean_arterial_pressure(systolic: float, diastolic: float) -> Dict:
    """MAP = DBP + 1/3 * (SBP - DBP)."""
    value = diastolic + (systolic - diastolic) / 3.0
    return {"value": round(value, 1), "unit": "mmHg", "formula": "DBP + (SBP - DBP) / 3"}


CALCULATIONS: Dict[str, Callable] = {
    "bmi": bmi,
    "egfr_ckd_epi_2021": egfr_ckd_epi_2021,
    "anion_gap": anion_gap,
    "corrected_calcium": corrected_calcium,
    "mean_arterial_pressure": mean_arterial_pressure,
}


class ClinicalCalculatorTool(Tool):
    name = "clinical_calculator"
    description = (
        "Runs a validated clinical formula instead of doing arithmetic in free text. "
        "Available calculations: " + ", ".join(CALCULATIONS.keys()) + ". "
        "Pass the calculation name and its required numeric arguments."
    )
    parameters = {
        "calculation": {"type": "string", "description": "One of: " + ", ".join(CALCULATIONS.keys())},
        "args": {"type": "object", "description": "Named arguments for the chosen calculation"},
    }

    def run(self, calculation: str, args: Dict = None) -> str:
        args = args or {}
        if calculation not in CALCULATIONS:
            return f"Error: unknown calculation '{calculation}'. Available: {', '.join(CALCULATIONS.keys())}"
        try:
            result = CALCULATIONS[calculation](**args)
        except TypeError as e:
            return f"Error: missing or invalid arguments for '{calculation}': {e}"
        except ValueError as e:
            return f"Error: {e}"
        return f"{result['value']} {result['unit']} (formula: {result['formula']})"
