"""
tests/test_clinical_history_agent.py
-------------------------------------
Unit tests for the Clinical History Agent.

What is tested here
-------------------
- PatientData schema: valid input, optional fields, validators, helpers
- VisionFinding schema: valid input, from_vision_dict(), boundary values
- ClinicalHistoryInput: from_dicts() convenience constructor
- PromptBuilder: build(), format_for_phi3(), all 5 prompt sections,
                 edge cases (empty lists, missing fields, low-confidence finding)
- Output parser: _parse_model_output() — well-formed block, missing delimiters
                 (loose fallback), clamped confidence, invalid consistency label
- ClinicalHistoryAgent: run() with a mocked pipeline (no GPU / model download)

What is NOT tested here
-----------------------
- Actual Phi-3-mini inference — that requires a GPU and model weights.
  End-to-end inference is validated manually after the model is downloaded.

Run with:
    python -m pytest tests/test_clinical_history_agent.py -v
    # or from project root:
    pytest tests/test_clinical_history_agent.py -v
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Make sure the project root is on the path
sys.path.insert(0, "c:/Neuro Agent")

from agents.clinical_history_agent.patient_schema import (
    ClinicalHistoryInput,
    ConsistencyLabel,
    PatientData,
    Sex,
    VisionFinding,
)
from agents.clinical_history_agent.prompt_builder import PromptBuilder
from agents.clinical_history_agent.agent import (
    ClinicalHistoryAgent,
    _parse_model_output,
    FALLBACK_CONFIDENCE,
)


# ===========================================================================
# Fixtures — reusable test data
# ===========================================================================

@pytest.fixture()
def typical_patient() -> PatientData:
    """62-year-old male with headaches — typical GBM demographic."""
    return PatientData(
        patient_id="BraTS-001",
        age=62,
        sex=Sex.MALE,
        presenting_symptoms=["persistent headaches", "nausea", "blurred vision"],
        symptom_duration_weeks=8,
        neurological_history=["hypertension"],
        family_history=[],
        current_medications=["dexamethasone 4mg", "levetiracetam 500mg"],
        performance_status=1,
        prior_imaging_findings=None,
        additional_notes="Referred by GP after 2-month history of worsening headaches.",
    )


@pytest.fixture()
def minimal_patient() -> PatientData:
    """Minimal patient — only age is required."""
    return PatientData(age=45)


@pytest.fixture()
def tumour_finding() -> VisionFinding:
    """High-confidence tumour detection."""
    return VisionFinding(
        tumour_detected=True,
        confidence_score=0.912,
        tumour_volume_cc=12.4,
        tumour_volume_voxels=12400,
        requires_review=False,
        gradcam_slice=78,
    )


@pytest.fixture()
def no_tumour_finding() -> VisionFinding:
    """No tumour detected, moderate confidence."""
    return VisionFinding(
        tumour_detected=False,
        confidence_score=0.78,
        tumour_volume_cc=0.0,
        tumour_volume_voxels=0,
        requires_review=False,
    )


@pytest.fixture()
def low_confidence_finding() -> VisionFinding:
    """Tumour detected but low confidence — triggers requires_review."""
    return VisionFinding(
        tumour_detected=True,
        confidence_score=0.58,
        tumour_volume_cc=3.1,
        tumour_volume_voxels=3100,
        requires_review=True,
    )


@pytest.fixture()
def builder() -> PromptBuilder:
    return PromptBuilder()


# ===========================================================================
# 1. PatientData schema tests
# ===========================================================================

class TestPatientData:

    def test_minimal_patient_creates_successfully(self):
        """Only `age` is required — all other fields should default cleanly."""
        p = PatientData(age=30)
        assert p.age == 30
        assert p.sex == Sex.UNKNOWN
        assert p.presenting_symptoms == []
        assert p.current_medications == []
        assert p.neurological_history == []
        assert p.patient_id == "UNKNOWN"

    def test_full_patient_round_trips(self, typical_patient):
        assert typical_patient.age == 62
        assert typical_patient.sex == Sex.MALE
        assert len(typical_patient.presenting_symptoms) == 3
        assert typical_patient.performance_status == 1

    def test_age_accepts_string_input(self):
        """Validator should coerce string age to int."""
        p = PatientData(age="55")
        assert p.age == 55

    def test_age_invalid_raises(self):
        with pytest.raises(Exception):
            PatientData(age="not_a_number")

    def test_age_out_of_range_raises(self):
        with pytest.raises(Exception):
            PatientData(age=200)

    def test_sex_enum_values(self):
        assert Sex.MALE.value    == "male"
        assert Sex.FEMALE.value  == "female"
        assert Sex.OTHER.value   == "other"
        assert Sex.UNKNOWN.value == "unknown"

    def test_sex_accepts_string(self):
        p = PatientData(age=40, sex="female")
        assert p.sex == Sex.FEMALE

    def test_strip_empty_strings_validator(self):
        """Blank items in list fields should be stripped."""
        p = PatientData(
            age=50,
            presenting_symptoms=["headaches", "", "  ", "nausea"],
        )
        assert p.presenting_symptoms == ["headaches", "nausea"]

    def test_symptom_summary_empty(self, minimal_patient):
        assert minimal_patient.symptom_summary() == "no symptoms reported"

    def test_symptom_summary_nonempty(self, typical_patient):
        summary = typical_patient.symptom_summary()
        assert "headaches" in summary
        assert "nausea" in summary

    def test_medication_summary_empty(self, minimal_patient):
        assert minimal_patient.medication_summary() == "none reported"

    def test_medication_summary_nonempty(self, typical_patient):
        summary = typical_patient.medication_summary()
        assert "dexamethasone" in summary

    def test_history_summary_empty(self, minimal_patient):
        assert minimal_patient.history_summary() == "no prior neurological history"

    def test_history_summary_nonempty(self, typical_patient):
        assert "hypertension" in typical_patient.history_summary()

    def test_has_neurological_symptoms_true(self, typical_patient):
        assert typical_patient.has_neurological_symptoms() is True

    def test_has_neurological_symptoms_false(self):
        p = PatientData(age=40, presenting_symptoms=["fatigue", "back pain"])
        assert p.has_neurological_symptoms() is False

    def test_has_neurological_symptoms_empty(self, minimal_patient):
        assert minimal_patient.has_neurological_symptoms() is False

    def test_repr_contains_age_and_sex(self, typical_patient):
        r = repr(typical_patient)
        assert "62" in r
        assert "male" in r


# ===========================================================================
# 2. VisionFinding schema tests
# ===========================================================================

class TestVisionFinding:

    def test_from_vision_dict(self):
        d = {
            "tumour_detected":      True,
            "confidence_score":     0.91,
            "tumour_volume_cc":     12.4,
            "tumour_volume_voxels": 12400,
            "requires_review":      False,
            "gradcam_slice":        78,
            "inference_time_s":     4.2,   # extra field — should be ignored
        }
        f = VisionFinding.from_vision_dict(d)
        assert f.tumour_detected is True
        assert f.confidence_score == pytest.approx(0.91)
        assert f.tumour_volume_cc == pytest.approx(12.4)
        assert f.requires_review is False
        assert f.gradcam_slice == 78

    def test_from_vision_dict_minimal(self):
        """Only required fields — optional ones should default."""
        d = {"tumour_detected": False, "confidence_score": 0.80}
        f = VisionFinding.from_vision_dict(d)
        assert f.tumour_detected is False
        assert f.tumour_volume_cc == 0.0
        assert f.requires_review is False
        assert f.gradcam_slice is None

    def test_confidence_score_boundary_zero(self):
        f = VisionFinding(tumour_detected=False, confidence_score=0.0)
        assert f.confidence_score == 0.0

    def test_confidence_score_boundary_one(self):
        f = VisionFinding(tumour_detected=True, confidence_score=1.0)
        assert f.confidence_score == 1.0

    def test_confidence_score_out_of_range_raises(self):
        with pytest.raises(Exception):
            VisionFinding(tumour_detected=True, confidence_score=1.5)

    def test_requires_review_flag(self, low_confidence_finding):
        assert low_confidence_finding.requires_review is True
        assert low_confidence_finding.confidence_score < 0.75


# ===========================================================================
# 3. ClinicalHistoryInput tests
# ===========================================================================

class TestClinicalHistoryInput:

    def test_from_dicts(self):
        patient_dict = {
            "patient_id": "BraTS-002",
            "age": 45,
            "sex": "female",
            "presenting_symptoms": ["seizures"],
        }
        vision_dict = {
            "tumour_detected": True,
            "confidence_score": 0.85,
        }
        inp = ClinicalHistoryInput.from_dicts(patient_dict, vision_dict)
        assert inp.patient.patient_id == "BraTS-002"
        assert inp.patient.age == 45
        assert inp.patient.sex == Sex.FEMALE
        assert inp.vision_finding.tumour_detected is True

    def test_direct_construction(self, typical_patient, tumour_finding):
        inp = ClinicalHistoryInput(patient=typical_patient, vision_finding=tumour_finding)
        assert inp.patient.age == 62
        assert inp.vision_finding.confidence_score == pytest.approx(0.912)


# ===========================================================================
# 4. PromptBuilder tests
# ===========================================================================

class TestPromptBuilder:

    def test_build_returns_two_strings(self, builder, typical_patient, tumour_finding):
        system, user = builder.build(typical_patient, tumour_finding)
        assert isinstance(system, str)
        assert isinstance(user, str)
        assert len(system) > 100
        assert len(user) > 100

    def test_system_prompt_contains_role_keywords(self, builder):
        assert "clinical reasoning" in builder.system_prompt.lower()
        assert "---CLINICAL SUMMARY---" in builder.system_prompt
        assert "---END SUMMARY---" in builder.system_prompt
        assert "CONSISTENCY" in builder.system_prompt
        assert "CONFIDENCE" in builder.system_prompt

    def test_user_prompt_contains_patient_data(
        self, builder, typical_patient, tumour_finding
    ):
        _, user = builder.build(typical_patient, tumour_finding)
        assert "BraTS-001" in user
        assert "62" in user
        assert "male" in user
        assert "persistent headaches" in user
        assert "dexamethasone" in user
        assert "hypertension" in user

    def test_user_prompt_contains_vision_finding(
        self, builder, typical_patient, tumour_finding
    ):
        _, user = builder.build(typical_patient, tumour_finding)
        assert "TUMOUR DETECTED" in user
        assert "0.912" in user
        assert "12.4 cc" in user
        assert "high confidence" in user

    def test_user_prompt_no_tumour_case(
        self, builder, typical_patient, no_tumour_finding
    ):
        _, user = builder.build(typical_patient, no_tumour_finding)
        assert "NO TUMOUR DETECTED" in user
        assert "N/A" in user

    def test_user_prompt_low_confidence_review_flag(
        self, builder, typical_patient, low_confidence_finding
    ):
        _, user = builder.build(typical_patient, low_confidence_finding)
        assert "REVIEW REQUIRED" in user
        assert "low confidence" in user

    def test_format_for_phi3_contains_tokens(self, builder, typical_patient, tumour_finding):
        system, user = builder.build(typical_patient, tumour_finding)
        full = builder.format_for_phi3(system, user)
        assert "<|system|>" in full
        assert "<|user|>" in full
        assert "<|assistant|>" in full
        assert "<|end|>" in full
        # Assistant token must be at the very end (nothing after it)
        assert full.strip().endswith("<|assistant|>")

    def test_build_formatted_convenience(
        self, builder, typical_patient, tumour_finding
    ):
        full = builder.build_formatted(typical_patient, tumour_finding)
        # Should be identical to calling build() + format_for_phi3()
        system, user = builder.build(typical_patient, tumour_finding)
        expected = builder.format_for_phi3(system, user)
        assert full == expected

    def test_minimal_patient_prompt_has_safe_defaults(
        self, builder, minimal_patient, tumour_finding
    ):
        _, user = builder.build(minimal_patient, tumour_finding)
        assert "no symptoms reported" in user
        assert "none reported" in user              # medications
        assert "no prior neurological history" in user
        assert "not assessed" in user               # ECOG PS
        assert "none on record" in user             # prior imaging

    def test_symptom_duration_bands(self, builder, tumour_finding):
        """Test each duration band is labelled correctly in the prompt."""
        cases = [
            (0,   "acute"),
            (2,   "subacute"),
            (8,   "months"),
            (20,  "months"),
        ]
        for weeks, expected_kw in cases:
            p = PatientData(
                age=50,
                presenting_symptoms=["headache"],
                symptom_duration_weeks=weeks,
            )
            _, user = builder.build(p, tumour_finding)
            assert expected_kw in user.lower(), (
                f"Duration {weeks}w: expected '{expected_kw}' in prompt"
            )

    def test_performance_status_descriptions(self, builder, tumour_finding):
        for ps_score in [0, 1, 2, 3, 4]:
            p = PatientData(age=50, performance_status=ps_score)
            _, user = builder.build(p, tumour_finding)
            assert str(ps_score) in user

    def test_family_history_appears_in_prompt(self, builder, tumour_finding):
        p = PatientData(
            age=40,
            family_history=["father had glioblastoma"],
        )
        _, user = builder.build(p, tumour_finding)
        assert "glioblastoma" in user

    def test_additional_notes_appear_in_prompt(self, builder, tumour_finding):
        p = PatientData(
            age=55,
            additional_notes="Patient reports recent onset of aphasia.",
        )
        _, user = builder.build(p, tumour_finding)
        assert "aphasia" in user

    def test_prior_imaging_appears_in_prompt(self, builder, tumour_finding):
        p = PatientData(
            age=55,
            prior_imaging_findings="Previous CT showed 2cm hyperdense lesion.",
        )
        _, user = builder.build(p, tumour_finding)
        assert "hyperdense" in user


# ===========================================================================
# 5. Output parser tests
# ===========================================================================

class TestOutputParser:

    def _make_output(
        self,
        summary="A 62-year-old male with headaches. Profile is consistent with GBM.",
        consistency="consistent",
        confidence="0.88",
        key_factors="age 62, male sex, headaches, elevated dexamethasone",
    ) -> str:
        """Build a valid model output string with the structured block."""
        return (
            "Let me reason step by step...\n\n"
            "The patient is a 62-year-old male with chronic headaches...\n\n"
            "---CLINICAL SUMMARY---\n"
            f"SUMMARY: {summary}\n"
            f"CONSISTENCY: {consistency}\n"
            f"CONFIDENCE: {confidence}\n"
            f"KEY_FACTORS: {key_factors}\n"
            "---END SUMMARY---"
        )

    def test_well_formed_output_parsed_correctly(self):
        text = self._make_output()
        result = _parse_model_output(text)
        assert "consistent with GBM" in result["summary"]
        assert result["consistency"] == "consistent"
        assert result["confidence"] == pytest.approx(0.88)
        assert "age 62" in result["key_factors"]
        assert "male sex" in result["key_factors"]

    def test_inconsistent_label_parsed(self):
        text = self._make_output(consistency="inconsistent")
        result = _parse_model_output(text)
        assert result["consistency"] == "inconsistent"

    def test_uncertain_label_parsed(self):
        text = self._make_output(consistency="uncertain")
        result = _parse_model_output(text)
        assert result["consistency"] == "uncertain"

    def test_invalid_consistency_label_falls_back_to_uncertain(self):
        text = self._make_output(consistency="maybe")
        result = _parse_model_output(text)
        assert result["consistency"] == ConsistencyLabel.UNCERTAIN.value

    def test_confidence_clamped_above_one(self):
        text = self._make_output(confidence="1.45")
        result = _parse_model_output(text)
        assert result["confidence"] == pytest.approx(1.0)

    def test_confidence_clamped_below_zero(self):
        text = self._make_output(confidence="-0.2")
        result = _parse_model_output(text)
        assert result["confidence"] == pytest.approx(0.0)

    def test_confidence_exactly_zero(self):
        text = self._make_output(confidence="0.0")
        result = _parse_model_output(text)
        assert result["confidence"] == pytest.approx(0.0)

    def test_key_factors_parsed_as_list(self):
        text = self._make_output(key_factors="headaches, age, dexamethasone use")
        result = _parse_model_output(text)
        assert isinstance(result["key_factors"], list)
        assert len(result["key_factors"]) == 3

    def test_empty_key_factors_becomes_empty_list(self):
        # Model outputs nothing after KEY_FACTORS:
        text = (
            "---CLINICAL SUMMARY---\n"
            "SUMMARY: Some summary.\n"
            "CONSISTENCY: consistent\n"
            "CONFIDENCE: 0.80\n"
            "KEY_FACTORS: \n"
            "---END SUMMARY---"
        )
        result = _parse_model_output(text)
        assert isinstance(result["key_factors"], list)

    def test_missing_delimiters_loose_fallback(self):
        """Model drops the delimiters but keeps KEY: value lines."""
        text = (
            "Some reasoning here.\n\n"
            "SUMMARY: Patient has consistent clinical profile.\n"
            "CONSISTENCY: consistent\n"
            "CONFIDENCE: 0.82\n"
            "KEY_FACTORS: age, symptoms\n"
        )
        result = _parse_model_output(text)
        assert result["consistency"] == "consistent"
        assert result["confidence"] == pytest.approx(0.82)

    def test_no_structure_at_all_returns_fallback_defaults(self):
        """Completely garbled output — all fields should be safe fallbacks."""
        text = "I cannot determine the consistency of this patient's profile."
        result = _parse_model_output(text)
        assert result["consistency"] == ConsistencyLabel.UNCERTAIN.value
        assert result["confidence"] == FALLBACK_CONFIDENCE
        assert isinstance(result["key_factors"], list)
        assert isinstance(result["summary"], str)

    def test_case_insensitive_parsing(self):
        """Parser should handle mixed-case labels."""
        text = (
            "---clinical summary---\n"
            "summary: Consistent profile.\n"
            "consistency: CONSISTENT\n"
            "confidence: 0.90\n"
            "key_factors: headache, age\n"
            "---end summary---"
        )
        result = _parse_model_output(text)
        assert result["consistency"] == "consistent"
        assert result["confidence"] == pytest.approx(0.90)


# ===========================================================================
# 6. ClinicalHistoryAgent integration tests (mocked LLM pipeline)
# ===========================================================================

class TestClinicalHistoryAgent:
    """Test the agent's run() logic with a fully mocked HuggingFace pipeline.

    The actual Phi-3-mini model is never loaded — we patch
    transformers.pipeline so the agent class can be instantiated and
    run() called without any GPU or model download.
    """

    MOCK_MODEL_OUTPUT = (
        "The patient is a 62-year-old male with progressive headaches over 8 weeks...\n\n"
        "---CLINICAL SUMMARY---\n"
        "SUMMARY: A 62-year-old male presenting with persistent headaches and visual "
        "symptoms over 8 weeks. The clinical profile is highly consistent with an "
        "intracranial tumour, supported by age, sex, symptom duration, and current medications.\n"
        "CONSISTENCY: consistent\n"
        "CONFIDENCE: 0.91\n"
        "KEY_FACTORS: age 62 years, male sex, 8-week progressive headaches, dexamethasone use\n"
        "---END SUMMARY---"
    )

    def _make_agent_with_mock_pipeline(self) -> ClinicalHistoryAgent:
        """Create a ClinicalHistoryAgent whose pipeline is pre-mocked."""
        agent = ClinicalHistoryAgent.__new__(ClinicalHistoryAgent)
        agent.model_id       = "microsoft/Phi-3-mini-4k-instruct"
        agent.device         = MagicMock()
        agent.use_4bit       = False
        agent.max_new_tokens = 512
        agent.temperature    = 0.2
        agent.top_p          = 0.9
        agent._model         = MagicMock()
        agent._tokenizer     = MagicMock()
        agent._prompt_builder= PromptBuilder()

        # Pipeline returns the structured output
        mock_pipeline = MagicMock(
            return_value=[{"generated_text": self.MOCK_MODEL_OUTPUT}]
        )
        agent._pipeline = mock_pipeline
        return agent

    def test_run_returns_standard_dict_structure(
        self, typical_patient, tumour_finding
    ):
        agent = self._make_agent_with_mock_pipeline()
        inp = ClinicalHistoryInput(
            patient=typical_patient, vision_finding=tumour_finding
        )
        result = agent.run(inp)

        # Top-level keys
        assert "agent_name"  in result
        assert "success"     in result
        assert "output"      in result
        assert "confidence"  in result
        assert "error"       in result

    def test_run_agent_name_is_correct(self, typical_patient, tumour_finding):
        agent = self._make_agent_with_mock_pipeline()
        inp = ClinicalHistoryInput(patient=typical_patient, vision_finding=tumour_finding)
        result = agent.run(inp)
        assert result["agent_name"] == "clinical_history_agent"

    def test_run_success_is_true(self, typical_patient, tumour_finding):
        agent = self._make_agent_with_mock_pipeline()
        inp = ClinicalHistoryInput(patient=typical_patient, vision_finding=tumour_finding)
        result = agent.run(inp)
        assert result["success"] is True
        assert result["error"] is None

    def test_run_output_fields(self, typical_patient, tumour_finding):
        agent = self._make_agent_with_mock_pipeline()
        inp = ClinicalHistoryInput(patient=typical_patient, vision_finding=tumour_finding)
        result = agent.run(inp)
        out = result["output"]
        assert "clinical_summary" in out
        assert "consistency"      in out
        assert "key_factors"      in out
        assert "raw_reasoning"    in out

    def test_run_consistency_value(self, typical_patient, tumour_finding):
        agent = self._make_agent_with_mock_pipeline()
        inp = ClinicalHistoryInput(patient=typical_patient, vision_finding=tumour_finding)
        result = agent.run(inp)
        assert result["output"]["consistency"] == "consistent"

    def test_run_confidence_value(self, typical_patient, tumour_finding):
        agent = self._make_agent_with_mock_pipeline()
        inp = ClinicalHistoryInput(patient=typical_patient, vision_finding=tumour_finding)
        result = agent.run(inp)
        assert result["confidence"] == pytest.approx(0.91)

    def test_run_clinical_summary_is_nonempty_string(
        self, typical_patient, tumour_finding
    ):
        agent = self._make_agent_with_mock_pipeline()
        inp = ClinicalHistoryInput(patient=typical_patient, vision_finding=tumour_finding)
        result = agent.run(inp)
        summary = result["output"]["clinical_summary"]
        assert isinstance(summary, str)
        assert len(summary) > 10

    def test_run_key_factors_is_list(self, typical_patient, tumour_finding):
        agent = self._make_agent_with_mock_pipeline()
        inp = ClinicalHistoryInput(patient=typical_patient, vision_finding=tumour_finding)
        result = agent.run(inp)
        assert isinstance(result["output"]["key_factors"], list)
        assert len(result["output"]["key_factors"]) > 0

    def test_run_from_dicts(self):
        agent = self._make_agent_with_mock_pipeline()
        result = agent.run_from_dicts(
            patient_dict={
                "patient_id": "BraTS-003",
                "age": 55,
                "sex": "female",
                "presenting_symptoms": ["seizures", "memory loss"],
                "symptom_duration_weeks": 4,
            },
            vision_dict={
                "tumour_detected": True,
                "confidence_score": 0.88,
                "tumour_volume_cc": 7.2,
                "tumour_volume_voxels": 7200,
                "requires_review": False,
            },
        )
        assert result["success"] is True
        assert result["agent_name"] == "clinical_history_agent"

    def test_run_handles_pipeline_exception_gracefully(
        self, typical_patient, tumour_finding
    ):
        """If the LLM pipeline throws, run() must return success=False gracefully."""
        agent = self._make_agent_with_mock_pipeline()
        agent._pipeline = MagicMock(side_effect=RuntimeError("CUDA OOM"))

        inp = ClinicalHistoryInput(patient=typical_patient, vision_finding=tumour_finding)
        result = agent.run(inp)

        assert result["success"] is False
        assert result["error"] is not None
        assert "RuntimeError" in result["error"]
        assert result["output"]["consistency"] == ConsistencyLabel.UNCERTAIN.value
        assert isinstance(result["output"]["clinical_summary"], str)

    def test_pipeline_called_once_per_run(self, typical_patient, tumour_finding):
        agent = self._make_agent_with_mock_pipeline()
        inp = ClinicalHistoryInput(patient=typical_patient, vision_finding=tumour_finding)
        agent.run(inp)
        assert agent._pipeline.call_count == 1

    def test_pipeline_receives_phi3_formatted_prompt(
        self, typical_patient, tumour_finding
    ):
        """The pipeline must be called with the Phi-3 chat template tokens."""
        agent = self._make_agent_with_mock_pipeline()
        inp = ClinicalHistoryInput(patient=typical_patient, vision_finding=tumour_finding)
        agent.run(inp)

        call_args = agent._pipeline.call_args
        prompt_arg = call_args[0][0]  # first positional arg
        assert "<|system|>" in prompt_arg
        assert "<|user|>"   in prompt_arg
        assert "<|assistant|>" in prompt_arg
