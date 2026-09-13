"""
tests/test_model.py – Unit tests for model architecture and forward pass.
"""
import torch
import pytest
from src.model import CNN_MLP_Model, CNN_GAP_Model


@pytest.fixture
def dummy_batch():
    """Return a (B=2, C=3, H=224, W=224) random tensor."""
    return torch.randn(2, 3, 224, 224)


class TestCNN_MLP_Model:
    def test_output_shape(self, dummy_batch):
        model = CNN_MLP_Model(num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"

    def test_parameter_count_positive(self):
        model = CNN_MLP_Model(num_classes=2)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0


class TestCNN_GAP_Model:
    def test_output_shape(self, dummy_batch):
        model = CNN_GAP_Model(num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"

    def test_gap_fewer_params_than_mlp(self):
        mlp = CNN_MLP_Model(num_classes=2)
        gap = CNN_GAP_Model(num_classes=2)
        mlp_params = sum(p.numel() for p in mlp.parameters())
        gap_params  = sum(p.numel() for p in gap.parameters())
        assert gap_params < mlp_params, (
            f"GAP model ({gap_params}) should have fewer params than MLP model ({mlp_params})"
        )
