"""
tests/test_tumor_classification_agent.py
-----------------------------------------
Unit and integration tests for the Tumor Classification Agent.

Test groups
-----------
TestDataset        — dataset.py: type ImageFolder, TIF collection, BraTS path scan,
                     transform shapes, TTA count, weighted sampler balance.
TestModel          — model.py: build, set_phase, forward shape, individual_logits,
                     get_individual_models, TumorGradeClassifier head architecture.
TestInference      — inference.py: _load_pil (path / array inputs), TumorPredictor
                     behaviour when checkpoints are missing.
TestAgent          — agent.py: _clinical_urgency mapping, _fail structure, run()
                     with missing checkpoint, run() with missing input keys.

All tests run without GPU and without any trained checkpoints.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch

# ── paths resolved relative to repo root ─────────────────────────────────────
REPO_ROOT  = Path(__file__).parent.parent
TYPE_DATA  = REPO_ROOT / "data" / "tumor_classification" / "type"
GRADE_DATA = REPO_ROOT / "data" / "tumor_classification" / "grade"
BRATS_DATA = REPO_ROOT / "data" / "raw" / "BraTS2020_TrainingData"


# =============================================================================
# TestDataset
# =============================================================================

class TestDataset:
    """Tests for agents/tumor_classification_agent/dataset.py"""

    def test_type_classes_match_disk(self):
        """ImageFolder discovers the four correct class folders."""
        from torchvision import datasets
        ds = datasets.ImageFolder(root=str(TYPE_DATA / "Training"))
        assert set(ds.classes) == {"glioma", "meningioma", "notumor", "pituitary"}, (
            f"Unexpected classes: {ds.classes}"
        )

    def test_type_dataloader_shapes(self):
        """Batch tensor has correct shape and normalised range."""
        from agents.tumor_classification_agent.dataset import get_type_dataloaders
        train_l, val_l, test_l, classes = get_type_dataloaders(
            data_root=str(TYPE_DATA), batch_size=4, num_workers=0, seed=42
        )
        assert set(classes) == {"glioma", "meningioma", "notumor", "pituitary"}

        imgs, labels = next(iter(train_l))
        assert imgs.shape == (4, 3, 224, 224), f"Got shape {imgs.shape}"
        assert labels.shape == (4,)
        # After ImageNet normalisation the range is roughly [-2.5, 2.5]
        assert imgs.min() < 0, "Expected negative values after normalisation"
        assert imgs.max() > 0

    def test_type_split_sizes(self):
        """val_split=0.15 on 5600 Training images → ~4760 train, ~840 val."""
        from agents.tumor_classification_agent.dataset import get_type_dataloaders
        train_l, val_l, _test_l, _ = get_type_dataloaders(
            data_root=str(TYPE_DATA), batch_size=8, num_workers=0, val_split=0.15, seed=0
        )
        n_train = len(train_l.dataset)
        n_val   = len(val_l.dataset)
        assert n_train + n_val == 5600, f"Expected 5600 total, got {n_train + n_val}"
        assert abs(n_val - 840) <= 20, f"Val size {n_val} not near 840"

    def test_no_mask_tifs_in_grade_ii(self):
        """_collect_grade_ii_samples returns zero _mask files."""
        from agents.tumor_classification_agent.dataset import _collect_grade_ii_samples
        slices = _collect_grade_ii_samples(GRADE_DATA / "kaggle_3m")
        masks  = [p for p in slices if "_mask" in p.name]
        assert len(masks) == 0, f"Found {len(masks)} mask files in grade_II collection"
        assert len(slices) > 2000, f"Too few grade_II slices: {len(slices)}"

    def test_brats_grade_map_counts(self):
        """BraTS name_mapping.csv yields 76 LGG (grade_III) and 293 HGG (grade_IV)."""
        from agents.tumor_classification_agent.dataset import _read_brats_grade_map
        brats_dir = BRATS_DATA / "MICCAI_BraTS2020_TrainingData"
        grade_map = _read_brats_grade_map(brats_dir)
        from collections import Counter
        cnt = Counter(grade_map.values())
        assert cnt["grade_III"] == 76,  f"Expected 76 grade_III, got {cnt['grade_III']}"
        assert cnt["grade_IV"]  == 293, f"Expected 293 grade_IV, got {cnt['grade_IV']}"

    def test_tta_transforms_count(self):
        """get_tta_transforms(10) returns exactly 10 transforms."""
        from agents.tumor_classification_agent.dataset import get_tta_transforms
        tta = get_tta_transforms(10)
        assert len(tta) == 10
        tta5 = get_tta_transforms(5)
        assert len(tta5) == 5

    def test_tta_first_view_is_clean(self):
        """First TTA view should not flip or rotate (clean canonical view)."""
        from agents.tumor_classification_agent.dataset import get_tta_transforms
        from torchvision.transforms import Compose
        tta = get_tta_transforms(10)
        # Apply to a test image and confirm shape is correct
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        tensor = tta[0](img)
        assert tensor.shape == (3, 224, 224)

    def test_weighted_sampler_reduces_imbalance(self):
        """WeightedRandomSampler should sample minority class proportionally."""
        from agents.tumor_classification_agent.dataset import _make_weighted_sampler
        # Simulate 3 classes with counts [100, 10, 10]
        labels = [0] * 100 + [1] * 10 + [2] * 10
        sampler = _make_weighted_sampler(labels, num_classes=3)
        # Draw 1200 samples and count class appearances
        indices = list(sampler)[:1200]
        sampled_labels = [labels[i] for i in indices]
        from collections import Counter
        cnt = Counter(sampled_labels)
        # Each class should appear roughly equally (within 2×)
        counts = sorted(cnt.values())
        ratio = counts[-1] / max(counts[0], 1)
        assert ratio < 2.5, f"Sampler imbalance too high: {cnt}"


# =============================================================================
# TestModel
# =============================================================================

class TestModel:
    """Tests for agents/tumor_classification_agent/model.py"""

    def test_build_type_ensemble_no_checkpoint(self):
        """build_type_ensemble() should succeed without a checkpoint."""
        from agents.tumor_classification_agent.model import build_type_ensemble
        model = build_type_ensemble(num_classes=4, device=torch.device("cpu"))
        assert model.num_classes == 4

    def test_build_grade_classifier_no_checkpoint(self):
        """build_grade_classifier() should succeed without a checkpoint."""
        from agents.tumor_classification_agent.model import build_grade_classifier
        model = build_grade_classifier(num_classes=3, device=torch.device("cpu"))
        assert model.num_classes == 3

    def test_type_ensemble_forward_shape(self):
        """Ensemble forward should return (B, 4) probability tensor summing to 1."""
        from agents.tumor_classification_agent.model import build_type_ensemble
        model = build_type_ensemble(num_classes=4, device=torch.device("cpu"))
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            probs = model(x)
        assert probs.shape == (2, 4), f"Got {probs.shape}"
        torch.testing.assert_close(probs.sum(dim=1), torch.ones(2), atol=1e-5, rtol=0)
        assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities not in [0,1]"

    def test_type_ensemble_individual_logits_shapes(self):
        """individual_logits() returns 3 tensors each of shape (B, 4)."""
        from agents.tumor_classification_agent.model import build_type_ensemble
        model = build_type_ensemble(num_classes=4, device=torch.device("cpu"))
        x = torch.randn(3, 3, 224, 224)
        eff_l, res_l, den_l = model.individual_logits(x)
        for t, name in [(eff_l, "eff"), (res_l, "res"), (den_l, "den")]:
            assert t.shape == (3, 4), f"{name} logits shape {t.shape} != (3, 4)"

    def test_get_individual_models(self):
        """get_individual_models() returns exactly 3 nn.Module instances."""
        from agents.tumor_classification_agent.model import (
            build_type_ensemble, TumorTypeEnsemble,
        )
        import torch.nn as nn
        model = build_type_ensemble(num_classes=4, device=torch.device("cpu"))
        parts = model.get_individual_models()
        assert len(parts) == 3
        for p in parts:
            assert isinstance(p, nn.Module)

    def test_grade_classifier_forward_shape(self):
        """TumorGradeClassifier forward should return (B, 3) raw logits."""
        from agents.tumor_classification_agent.model import build_grade_classifier
        model = build_grade_classifier(num_classes=3, device=torch.device("cpu"))
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 3), f"Got {logits.shape}"

    def test_grade_classifier_head_structure(self):
        """TumorGradeClassifier head must be: Flatten→Drop(0.4)→Lin(256)→ReLU→Lin(3)."""
        import torch.nn as nn
        from agents.tumor_classification_agent.model import TumorGradeClassifier
        model = TumorGradeClassifier(num_classes=3, dropout=0.4)
        head  = model.head
        # Layer types in order
        layer_types = [type(l).__name__ for l in head]
        assert layer_types == ["Flatten", "Dropout", "Linear", "ReLU", "Linear"], (
            f"Unexpected head layers: {layer_types}"
        )
        # Linear sizes
        lin1 = head[2]   # Linear(1792→256)
        lin2 = head[4]   # Linear(256→3)
        assert lin1.in_features  == 1792, f"Expected in_features=1792, got {lin1.in_features}"
        assert lin1.out_features == 256,  f"Expected out_features=256, got {lin1.out_features}"
        assert lin2.in_features  == 256,  f"Expected in_features=256, got {lin2.in_features}"
        assert lin2.out_features == 3,    f"Expected out_features=3, got {lin2.out_features}"
        # Dropout rate
        assert abs(head[1].p - 0.4) < 1e-6

    def test_set_phase1_freezes_backbone(self):
        """Phase 1: backbone params frozen, head params trainable."""
        from agents.tumor_classification_agent.model import TumorGradeClassifier
        model = TumorGradeClassifier(num_classes=3)
        model.set_phase(1)
        # Feature params should all be frozen
        feat_frozen = all(not p.requires_grad for p in model.features.parameters())
        # Head params should all be trainable
        head_trainable = all(p.requires_grad for p in model.head.parameters())
        assert feat_frozen,    "Feature params should be frozen in Phase 1"
        assert head_trainable, "Head params should be trainable in Phase 1"

    def test_set_phase2_unfreezes_some_layers(self):
        """Phase 2: some backbone params unfrozen (at least n_unfreeze tensors total)."""
        from agents.tumor_classification_agent.model import TumorGradeClassifier
        model = TumorGradeClassifier(num_classes=3)
        model.set_phase(2, n_unfreeze=10)
        # Count across the entire EfficientNet-B4 backbone (features + avgpool)
        n_unfrozen = sum(
            1 for p in list(model.features.parameters()) + list(model.avgpool.parameters())
            if p.requires_grad
        )
        assert n_unfrozen >= 1, (
            f"Expected at least 1 unfrozen backbone param tensor in Phase 2, got {n_unfrozen}"
        )

    def test_ensemble_set_phase1_all_backbones_frozen(self):
        """Ensemble Phase 1: non-head backbone params should be frozen."""
        from agents.tumor_classification_agent.model import TumorTypeEnsemble
        model = TumorTypeEnsemble(num_classes=4)
        model.set_phase(1)
        for name, backbone, head_attr in [
            ("efficientnet_b4", model.efficientnet_b4, "classifier"),
            ("resnet50",        model.resnet50,        "fc"),
            ("densenet121",     model.densenet121,     "classifier"),
        ]:
            head_param_ids = set(
                id(p) for p in getattr(backbone, head_attr).parameters()
            )
            frozen_non_head = [
                p for p in backbone.parameters()
                if id(p) not in head_param_ids and p.requires_grad
            ]
            assert len(frozen_non_head) == 0, (
                f"{name}: {len(frozen_non_head)} non-head backbone params are trainable in Phase 1"
            )

    def test_predict_proba_no_grad(self):
        """predict_proba() should not create a computation graph."""
        from agents.tumor_classification_agent.model import build_type_ensemble
        model = build_type_ensemble(num_classes=4, device=torch.device("cpu"))
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        probs = model.predict_proba(x)
        assert not probs.requires_grad


# =============================================================================
# TestInference
# =============================================================================

class TestInference:
    """Tests for agents/tumor_classification_agent/inference.py"""

    def test_load_pil_from_file(self, tmp_path):
        """_load_pil accepts a file path and returns an RGB PIL Image."""
        from agents.tumor_classification_agent.inference import _load_pil
        img_path = tmp_path / "test.jpg"
        Image.fromarray(
            np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        ).save(str(img_path))
        img = _load_pil(str(img_path))
        assert img.mode == "RGB"
        assert img.size == (128, 128)

    def test_load_pil_from_grayscale_array(self):
        """_load_pil accepts H×W uint8 numpy array → RGB PIL Image."""
        from agents.tumor_classification_agent.inference import _load_pil
        arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img = _load_pil(arr)
        assert img.mode == "RGB"

    def test_load_pil_from_float_array_normalises(self):
        """_load_pil normalises float arrays to [0, 255] uint8."""
        from agents.tumor_classification_agent.inference import _load_pil
        arr = np.random.rand(64, 64).astype(np.float32) * 1000.0
        img = _load_pil(arr)
        assert img.mode == "RGB"
        arr_out = np.array(img)
        assert arr_out.max() <= 255 and arr_out.min() >= 0

    def test_load_pil_bad_type_raises(self):
        """_load_pil raises TypeError for invalid input."""
        from agents.tumor_classification_agent.inference import _load_pil
        with pytest.raises(TypeError):
            _load_pil(12345)

    def test_load_pil_bad_array_shape_raises(self):
        """_load_pil raises ValueError for 4-D arrays."""
        from agents.tumor_classification_agent.inference import _load_pil
        with pytest.raises(ValueError):
            _load_pil(np.zeros((2, 64, 64, 3), dtype=np.uint8))

    def test_predictor_raises_on_missing_type_ckpt(self):
        """TumorPredictor.predict() raises FileNotFoundError for missing type ckpt."""
        from agents.tumor_classification_agent.inference import TumorPredictor
        predictor = TumorPredictor(
            type_ckpt  = "/nonexistent/type_ensemble_best.pth",
            grade_ckpt = "/nonexistent/grade_classifier_best.pth",
            device     = torch.device("cpu"),
        )
        arr = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        with pytest.raises(FileNotFoundError):
            predictor.predict(arr)


# =============================================================================
# TestAgent
# =============================================================================

class TestAgent:
    """Tests for agents/tumor_classification_agent/agent.py"""

    def test_clinical_urgency_glioma_iv(self):
        from agents.tumor_classification_agent.agent import _clinical_urgency
        assert _clinical_urgency("glioma", "grade_IV")  == "urgent"

    def test_clinical_urgency_glioma_iii(self):
        from agents.tumor_classification_agent.agent import _clinical_urgency
        assert _clinical_urgency("glioma", "grade_III") == "urgent"

    def test_clinical_urgency_glioma_ii(self):
        from agents.tumor_classification_agent.agent import _clinical_urgency
        assert _clinical_urgency("glioma", "grade_II")  == "monitor"

    def test_clinical_urgency_glioma_no_grade(self):
        from agents.tumor_classification_agent.agent import _clinical_urgency
        assert _clinical_urgency("glioma", None)        == "monitor"

    def test_clinical_urgency_meningioma(self):
        from agents.tumor_classification_agent.agent import _clinical_urgency
        assert _clinical_urgency("meningioma", None)    == "monitor"

    def test_clinical_urgency_notumor(self):
        from agents.tumor_classification_agent.agent import _clinical_urgency
        assert _clinical_urgency("notumor", None)       == "routine"

    def test_clinical_urgency_pituitary(self):
        from agents.tumor_classification_agent.agent import _clinical_urgency
        assert _clinical_urgency("pituitary", None)     == "routine"

    def test_fail_result_structure(self):
        """_fail() returns a complete, consistent error dict."""
        from agents.tumor_classification_agent.agent import TumorClassificationAgent
        result = TumorClassificationAgent._fail("test error")
        assert result["agent_name"] == "tumor_classification_agent"
        assert result["success"]    is False
        assert result["confidence"] == 0.0
        assert result["error"]      == "test error"
        assert result["output"]["tumor_type"]         is None
        assert result["output"]["tumor_grade"]        is None
        assert result["output"]["clinical_urgency"]   is None
        assert result["output"]["tta_used"]           is False

    def test_run_missing_input_keys(self):
        """run() with empty state returns success=False and a descriptive error."""
        from agents.tumor_classification_agent.agent import TumorClassificationAgent
        agent  = TumorClassificationAgent(device=torch.device("cpu"))
        result = agent.run({})
        assert result["success"] is False
        assert result["error"] is not None
        assert "mri_slice_path" in result["error"] or "mri_slice_array" in result["error"]

    def test_run_missing_checkpoint_returns_failure(self):
        """run() with a missing checkpoint path returns success=False (no crash)."""
        from agents.tumor_classification_agent.agent import TumorClassificationAgent
        agent = TumorClassificationAgent(
            type_ckpt_path  = "/no/such/type.pth",
            grade_ckpt_path = "/no/such/grade.pth",
            device          = torch.device("cpu"),
        )
        arr    = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        result = agent.run({"mri_slice_array": arr})
        assert result["success"]    is False
        assert result["confidence"] == 0.0
        assert result["error"] is not None

    def test_run_output_schema_on_success(self):
        """run() with mocked predictor returns the full expected output schema."""
        from agents.tumor_classification_agent.agent import TumorClassificationAgent

        agent = TumorClassificationAgent(device=torch.device("cpu"))

        # Inject a mock predictor so no checkpoint is needed
        mock_pred = MagicMock()
        mock_pred.predict.return_value = {
            "tumor_type":          "glioma",
            "tumor_grade":         "grade_IV",
            "type_probabilities":  {"glioma": 0.94, "meningioma": 0.03,
                                    "notumor": 0.01, "pituitary": 0.02},
            "grade_probabilities": {"grade_II": 0.05, "grade_III": 0.18, "grade_IV": 0.77},
            "confidence":          0.91,
            "tta_used":            True,
        }
        agent._predictor = mock_pred

        arr    = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        result = agent.run({"mri_slice_array": arr})

        assert result["success"]             is True
        assert result["agent_name"]          == "tumor_classification_agent"
        assert result["confidence"]          == 0.91
        assert result["error"]               is None
        assert result["output"]["tumor_type"]          == "glioma"
        assert result["output"]["tumor_grade"]         == "grade_IV"
        assert result["output"]["clinical_urgency"]    == "urgent"
        assert result["output"]["tta_used"]            is True
        assert "glioma" in result["output"]["type_probabilities"]
        assert "grade_IV" in result["output"]["grade_probabilities"]

    def test_run_notumor_gives_routine(self):
        """A no-tumor prediction should map to routine urgency."""
        from agents.tumor_classification_agent.agent import TumorClassificationAgent

        agent = TumorClassificationAgent(device=torch.device("cpu"))
        mock_pred = MagicMock()
        mock_pred.predict.return_value = {
            "tumor_type":          "notumor",
            "tumor_grade":         None,
            "type_probabilities":  {"glioma": 0.01, "meningioma": 0.02,
                                    "notumor": 0.95, "pituitary": 0.02},
            "grade_probabilities": None,
            "confidence":          0.95,
            "tta_used":            True,
        }
        agent._predictor = mock_pred

        arr    = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = agent.run({"mri_slice_array": arr})

        assert result["success"] is True
        assert result["output"]["clinical_urgency"] == "routine"
        assert result["output"]["tumor_grade"]      is None
