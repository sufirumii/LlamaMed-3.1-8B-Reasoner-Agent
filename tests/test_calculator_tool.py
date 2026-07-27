import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llamamed_agent.tools.calculator_tool import ClinicalCalculatorTool, bmi, egfr_ckd_epi_2021


def test_bmi():
    result = bmi(weight_kg=70, height_cm=175)
    assert abs(result["value"] - 22.9) < 0.05


def test_egfr_ckd_epi_2021_male():
    # Reference-checked against a known online CKD-EPI 2021 calculator value.
    result = egfr_ckd_epi_2021(creatinine_mg_dl=1.0, age_years=50, sex="male")
    assert 85 <= result["value"] <= 100


def test_egfr_invalid_sex_raises():
    try:
        egfr_ckd_epi_2021(creatinine_mg_dl=1.0, age_years=50, sex="unknown")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_tool_run_reports_unknown_calculation():
    tool = ClinicalCalculatorTool()
    output = tool.run(calculation="not_a_real_calc", args={})
    assert "unknown calculation" in output.lower()


def test_tool_run_reports_missing_args():
    tool = ClinicalCalculatorTool()
    output = tool.run(calculation="bmi", args={"weight_kg": 70})
    assert "error" in output.lower()
