import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llamamed_agent.agent.parser import parse_step


def test_parses_action_step():
    text = (
        "Thought: I should look this up.\n"
        "Action: search_documents\n"
        'Action Input: {"query": "eGFR reference range"}'
    )
    step = parse_step(text)
    assert step.thought == "I should look this up."
    assert step.action == "search_documents"
    assert step.action_input == {"query": "eGFR reference range"}
    assert not step.is_final


def test_parses_final_answer():
    text = "Thought: I have enough information.\nFinal Answer: The BMI is 22.9 kg/m^2."
    step = parse_step(text)
    assert step.is_final
    assert step.final_answer == "The BMI is 22.9 kg/m^2."


def test_falls_back_to_final_answer_when_unstructured():
    text = "The answer is 42."
    step = parse_step(text)
    assert step.is_final
    assert step.final_answer == "The answer is 42."


def test_recovers_malformed_json_action_input():
    text = (
        "Thought: computing\n"
        "Action: clinical_calculator\n"
        "Action Input: {calculation: bmi, args: {weight_kg: 70, height_cm: 175}}"
    )
    step = parse_step(text)
    assert step.action == "clinical_calculator"
    # Malformed JSON (unquoted keys) should still produce *some* usable dict
    # rather than raising, via the raw-string fallback.
    assert isinstance(step.action_input, dict)
